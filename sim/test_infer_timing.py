"""Host-only contracts for B subprocess and accelerator timing."""
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "board"))

import infer_images
from infer_images import StageTimer, timing_profile
from tensor_accel import TileAccelerator


class FakeBuffer(np.ndarray):
    physical_address = 0x10000000
    freed = 0

    def freebuffer(self):
        self.freed += 1


class FakeMMIO:
    def __init__(self, address, size, core=None):
        self.address, self.size = address, size
        self.core = address == 0x43C00000 if core is None else core
        if self.core:
            self.registers = {0: 0x504C4431, 4: 0x00010002, 0x0C: 0}
        else:
            self.dma_registers = {4: 1, 0x34: 1}

    def read(self, address):
        if self.core:
            return self.registers.get(address, 0)
        return self.dma_registers.get(address, 0)

    def write(self, address, value):
        if self.core:
            return
        if address in (4, 0x34):
            self.dma_registers[address] = 0 if value & 0x7000 else 1
        elif address in (0, 0x30):
            if value & 4:
                self.dma_registers[4] = self.dma_registers[0x34] = 1
            elif value & 1:
                self.dma_registers[address + 4] = 0x1002


class FakeOverlay:
    instances = []

    def __init__(self, path, download=False):
        self.download_count = 0
        self.ip_dict = {
            "axis_tensor_accel_0": {
                "phys_addr": 0x43C00000, "addr_range": 4096,
                "type": "user:pld:axis_tensor_accel:1.0",
            },
            "axi_dma": {
                "phys_addr": 0x40400000,
                "addr_range": 4096,
                "parameters": {
                    "c_include_sg": "0", "c_micro_dma": "0",
                    "c_addr_width": "32", "c_sg_length_width": "26",
                    "c_m_axi_mm2s_data_width": "64",
                    "c_m_axi_s2mm_data_width": "64",
                    "c_m_axis_mm2s_tdata_width": "64",
                },
            },
        }
        FakeOverlay.instances.append(self)

    def download(self):
        self.download_count += 1


class FakePYNQ(types.ModuleType):
    def __init__(self):
        super().__init__("pynq")
        self.Overlay = FakeOverlay
        self.MMIO = FakeMMIO
        self.Clocks = types.SimpleNamespace(fclk0_mhz=125.0)

        def allocate(shape, dtype, cacheable=False):
            buffer = np.zeros(shape, dtype=dtype).view(FakeBuffer)
            buffer.physical_address = 0x10000000 + FakeBuffer.freed * 0x20000
            return buffer

        self.allocate = allocate


class StageTimerTests(unittest.TestCase):
    def test_records_successful_and_failing_stages(self):
        timer = StageTimer()
        with timer.stage("successful"):
            pass
        with self.assertRaisesRegex(ValueError, "expected"):
            with timer.stage("failing"):
                raise ValueError("expected")
        self.assertGreaterEqual(timer.stages["successful"], 0.0)
        self.assertGreaterEqual(timer.stages["failing"], 0.0)

    def test_rejects_duplicate_and_nested_names(self):
        timer = StageTimer()
        with timer.stage("same"):
            pass
        with self.assertRaisesRegex(ValueError, "duplicate"):
            with timer.stage("same"):
                pass
        with timer.stage("outer"):
                with self.assertRaisesRegex(RuntimeError, "nested"):
                    with timer.stage("inner"):
                        pass
        self.assertIn("outer", timer.stages)
        self.assertNotIn("inner", timer.stages)

    def test_profile_keeps_startup_accelerator_and_frame_scopes_separate(self):
        startup = StageTimer()
        with startup.stage("model_init_and_manifest_validation"):
            pass
        frame = StageTimer()
        with frame.stage("jpeg_decode"):
            pass
        accelerator = {
            "init_stages": {"pynq_import": 0.1},
            "close_stages": {},
            "elapsed": {"tile_accelerator_init": 0.2},
        }
        profile = timing_profile(startup, accelerator, frame)
        self.assertEqual(profile["timing_clock"], "time.perf_counter wall seconds")
        self.assertIn("model_init_and_manifest_validation", profile["startup_stage_seconds"])
        self.assertIn("pynq_import", profile["accelerator_init_stage_seconds"])
        self.assertIn("jpeg_decode", profile["frame_stage_seconds"])
        self.assertIn("online_b_seconds", profile["legacy_fields"])


class AcceleratorTimingTests(unittest.TestCase):
    def setUp(self):
        FakeOverlay.instances.clear()
        self.pynq = FakePYNQ()
        self.old_pynq = sys.modules.get("pynq")
        sys.modules["pynq"] = self.pynq
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name) / "dma_top"
        base.with_suffix(".bit").write_bytes(b"bit")
        base.with_suffix(".hwh").write_text(
            '<?xml version="1.0"?><root><PARAMETER NAME='
            '"PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ" VALUE="125"/></root>',
            encoding="utf-8",
        )
        self.base = base

    def tearDown(self):
        sys.modules["pynq"] = self.old_pynq
        if self.old_pynq is None:
            del sys.modules["pynq"]
        self.temp.cleanup()

    def make_device(self):
        device = TileAccelerator(self.base.with_suffix(".bit"))
        optional = {"native_control_init", "xrt_import_and_tx_sync_init",
                    "native_copy_init", "native_control_bind"}
        self.assertFalse(optional & set(device.timing["init_stages"]))
        actual = {
            "pynq_import", "hwh_clock_parse", "overlay_metadata_parse",
            "bitstream_download", "attach_setup", "initial_recovery",
            "dma_buffer_allocation",
        }
        self.assertEqual(set(device.timing["init_stages"]), actual)
        for value in device.timing["init_stages"].values():
            self.assertGreaterEqual(value, 0.0)
        self.assertGreaterEqual(device.timing["elapsed"]["tile_accelerator_init"], 0.0)
        self.assertEqual(FakeOverlay.instances[-1].download_count, 1)
        return device

    def test_cold_init_and_normal_close_stages(self):
        device = self.make_device()
        init_stages = dict(device.timing["init_stages"])
        tx, rx = device.tx, device.rx
        device.close()
        self.assertEqual(device.timing["init_stages"], init_stages)
        self.assertEqual(tx.freed, 1)
        self.assertEqual(rx.freed, 1)
        self.assertIn("close_dma_and_core_reset", device.timing["close_stages"])
        self.assertIn("close_dma_buffer_free", device.timing["close_stages"])
        self.assertNotIn("close_error_overlay_download", device.timing["close_stages"])
        self.assertGreaterEqual(device.timing["elapsed"]["device_close_total"], 0.0)

    def test_close_reset_failure_still_redownloads_and_frees(self):
        device = self.make_device()
        init_stages = dict(device.timing["init_stages"])
        overlay = device.overlay
        tx, rx = device.tx, device.rx
        device.dma.reset = mock.Mock(side_effect=RuntimeError("reset failed"))
        device.close()
        self.assertEqual(overlay.download_count, 2)
        self.assertEqual(tx.freed, 1)
        self.assertEqual(rx.freed, 1)
        self.assertTrue(device.closed)
        self.assertIn("close_error_overlay_download", device.timing["close_stages"])
        self.assertIn("close_dma_and_core_reset", device.timing["close_stages"])
        self.assertEqual(device.timing["init_stages"], init_stages)

    def test_optional_native_backends_have_separate_init_stages(self):
        control_module = types.ModuleType("native_control")
        copy_module = types.ModuleType("native_copy")
        xrt_module = types.ModuleType("pyxrt")

        class FakeNativeControl:
            def __init__(self, library):
                pass

            info = {"kind": "fake_control"}

            def bind(self, core, dma):
                pass

        class FakeNativeCopy:
            def __init__(self, library, verify_build=True):
                pass

            info = {"kind": "fake_copy"}

        control_module.NativeControl = FakeNativeControl
        copy_module.NativeCopy = FakeNativeCopy
        xrt_module.xclBOSyncDirection = types.SimpleNamespace(
            XCL_BO_SYNC_BO_TO_DEVICE=0x1
        )
        old_modules = {name: sys.modules.get(name)
                       for name in ("native_control", "native_copy", "pyxrt")}
        sys.modules.update({"native_control": control_module,
                            "native_copy": copy_module, "pyxrt": xrt_module})
        device = None
        try:
            device = TileAccelerator(
                self.base.with_suffix(".bit"), native_copy="fake-copy.so",
                tx_range_sync=True, native_control="fake-control.so",
            )
            stages = device.timing["init_stages"]
            for name in ("native_control_init", "xrt_import_and_tx_sync_init",
                         "native_copy_init", "native_control_bind"):
                self.assertGreaterEqual(stages[name], 0.0)
            self.assertEqual(device.copy_info, {"kind": "fake_copy"})
            self.assertEqual(device.control_info, {"kind": "fake_control"})
        finally:
            if device is not None:
                device.close()
            for name, module in old_modules.items():
                if module is None:
                    del sys.modules[name]
                else:
                    sys.modules[name] = module


class ProcessEntryTests(unittest.TestCase):
    def test_linux_proc_math_is_best_effort(self):
        fields = [str(value) for value in range(3, 31)]
        fields[19] = "123"
        stat = f"1 (python) {' '.join(fields)}"

        def fake_open(path, *args, **kwargs):
            content = stat if path == "/proc/self/stat" else "1000.0 2000.0\n"
            return io.StringIO(content)

        with mock.patch("builtins.open", side_effect=fake_open), \
                mock.patch.object(infer_images.os, "sysconf", create=True, return_value=100):
            self.assertAlmostEqual(infer_images._best_effort_process_entry_age(), 998.77)

    def test_malformed_proc_input_returns_none(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="malformed")):
            self.assertIsNone(infer_images._best_effort_process_entry_age())


class MainProfileTests(unittest.TestCase):
    def test_multi_image_profile_and_legacy_fields(self):
        calls = {"model": 0, "device": 0, "graph": 0, "frames": []}

        class FakeModel:
            def __init__(self, root):
                calls["model"] += 1
                self.spec_hash = "a" * 64
                self.layers = [{"input_scale": 0.5}]

        class FakeDevice:
            abi_version = 0x00010002
            copy_info = {"kind": "fake_copy"}
            control_info = {"kind": "fake_control"}
            tx_sync_info = {"kind": "fake_sync"}
            events = []

            def __init__(self, bit, **kwargs):
                calls["device"] += 1
                self.closed = False
                self.timing = {
                    "init_stages": {"pynq_import": 0.01},
                    "close_stages": {},
                    "elapsed": {"tile_accelerator_init": 0.02},
                }

            def close(self):
                self.closed = True
                self.timing["close_stages"]["close_dma_buffer_free"] = 0.001
                self.timing["elapsed"]["device_close_total"] = 0.002

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        class FakeGraph:
            def __init__(self, model, device, fuse_maps=False, fuse_decoder=False):
                calls["graph"] += 1
                self.device = device

            def run(self, data, frame, input_metadata):
                calls["frames"].append(frame)
                head = np.zeros((1, 1, 256, 256), dtype=np.int8)
                return head, {
                    "seconds": 0.25,
                    "result": {"frame_id": frame, "status": "PASS"},
                }

        def fake_decode(source):
            return types.SimpleNamespace(close=lambda: None)

        def fake_letterbox_int8(image, scale, resizer=None):
            return np.zeros((1, 3, 256, 256), dtype=np.int8), np.ones((256, 256), bool), {
                "left": 0, "top": 0, "resized_width": 256, "resized_height": 256,
            }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bit = root / "dma_top.bit"
            bit.write_bytes(b"bit")
            bit.with_suffix(".hwh").write_bytes(b"hwh")
            sources = []
            for index in range(2):
                source = root / f"source{index}.jpg"
                source.write_bytes(b"jpeg")
                sources.append(source)
            output = root / "results"
            argv = [
                str(sources[0]), str(sources[1]), "--bit", str(bit),
                "--output", str(output), "--fast-quantize",
            ]
            stdout = io.StringIO()
            with mock.patch.object(infer_images, "Model", FakeModel), \
                    mock.patch.object(infer_images, "TileAccelerator", FakeDevice), \
                    mock.patch.object(infer_images, "PLGraph", FakeGraph), \
                    mock.patch.object(infer_images, "decode_rgb", fake_decode), \
                    mock.patch.object(infer_images, "letterbox_int8", fake_letterbox_int8), \
                    redirect_stdout(stdout):
                infer_images.main(argv)

            self.assertEqual(calls["model"], 1)
            self.assertEqual(calls["device"], 1)
            self.assertEqual(calls["graph"], 1)
            self.assertEqual(calls["frames"], [1, 2])
            lines = stdout.getvalue().splitlines()
            self.assertEqual(sum(line.startswith("INFERENCE_PASS ") for line in lines), 2)
            events = [json.loads(line.removeprefix("B_TIMING_JSON "))
                      for line in lines if line.startswith("B_TIMING_JSON ")]
            self.assertEqual([event["event"] for event in events],
                             ["startup_ready", "frame_complete", "frame_complete", "device_closed"])
            for event in events[:3]:
                self.assertGreaterEqual(event["timing"]["accelerator_init_elapsed_seconds"]["tile_accelerator_init"], 0)
            for event in events[1:3]:
                self.assertIn("json_report_write", event["timing"]["frame_stage_seconds"])
                self.assertIn("decode_through_json_write", event["timing"]["frame_elapsed_seconds"])
            self.assertIn("close_dma_buffer_free", events[-1]["timing"]["accelerator_close_stage_seconds"])

            first = json.loads((output / "frame_000001.json").read_text(encoding="utf-8"))
            second = json.loads((output / "frame_000002.json").read_text(encoding="utf-8"))
            for result, frame in ((first, 1), (second, 2)):
                self.assertEqual(result["frame_id"], frame)
                self.assertGreaterEqual(result["preprocess_seconds"], 0.0)
                self.assertEqual(result["inference_seconds"], 0.25)
                self.assertAlmostEqual(result["online_b_seconds"],
                                       result["preprocess_seconds"] + 0.25)
                timing = result["b_timing"]
                self.assertIn("jpeg_decode", timing["frame_stage_seconds"])
                self.assertIn("head_file_write", timing["frame_stage_seconds"])
                self.assertNotIn("json_report_write", timing["frame_stage_seconds"])
                self.assertIn("decode_through_head_write", timing["frame_elapsed_seconds"])


if __name__ == "__main__":
    unittest.main()
