"""Shared input-to-artifacts CAMUS pipeline for CLI, web and benchmarks."""
from __future__ import annotations
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from .model import load_model, MODEL_SPEC_SHA256
from .preprocess import preprocess
from .postprocess import apply_island5_hole1
from .reference import ReferenceNetwork
from .graph import CamusGraph
from .device import EmulatedTileAccelerator
from .result import native_labels_from_head, class_areas, label_display_png, segmentation_rgb, overlay_rgb, write_html_preview
from .png import save_views


def protect_output(path, model_root):
    path, model_root = Path(path).resolve(), Path(model_root).resolve()
    protected = model_root.parent if model_root.name == 'int8_export' else model_root
    if path.is_relative_to(protected):
        raise ValueError('output must be outside the immutable model delivery')
    if path.exists(): raise ValueError('output must be a new directory: '+str(path))
    return path


class CamusPipeline:
    def __init__(self, model_root, backend='reference', runtime=None, timeout=3.):
        started = time.perf_counter()
        if backend not in ('reference', 'emulator', 'pl'): raise ValueError('unknown CAMUS backend')
        self.model = load_model(model_root)
        self.backend, self.device, self.jobs = backend, None, 0
        if backend == 'pl':
            if runtime is None: raise ValueError('PL backend requires --runtime')
            from .accelerator import CamusTileAccelerator
            self.device = CamusTileAccelerator(runtime, self.model.root, timeout)
        elif backend == 'emulator': self.device = EmulatedTileAccelerator()
        self.network = CamusGraph(self.model, self.device) if self.device else ReferenceNetwork(self.model)
        self.native = self.device.io.native_control if backend == 'pl' else None
        self.initialization_seconds = time.perf_counter()-started

    def close(self):
        if self.device is not None: self.device.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def run(self, input_path, output_dir, spacing=None, html=True):
        started = time.perf_counter(); output = protect_output(output_dir, self.model.root); stages = {}
        t = time.perf_counter(); prep = preprocess(input_path)
        stages['input_read_preprocess'] = time.perf_counter()-t
        source_spacing = prep.source.spacing_yx_mm
        if spacing is not None:
            spacing = tuple(float(v) for v in spacing)
            if len(spacing) != 2 or not all(np.isfinite(v) and v > 0 for v in spacing):
                raise ValueError('spacing must be two positive finite millimeter values')
            if source_spacing is not None and source_spacing != spacing: raise ValueError('spacing conflicts with source')
        else: spacing = source_spacing
        t = time.perf_counter(); result = self.network.forward(prep.input_int8)
        stages['network'] = time.perf_counter()-t
        t = time.perf_counter()
        restore = self.native.restore if self.native is not None else native_labels_from_head
        labels = restore(result['head'], *prep.raw_gray.shape, self.model.head_scale)
        restore_seconds = time.perf_counter()-t
        post_started = time.perf_counter()
        processed = self.native.postprocess(labels) if self.native is not None else apply_island5_hole1(labels)
        stages['native_reconstruction_postprocess'] = time.perf_counter()-t
        native_details = dict(reconstruction=restore_seconds,postprocess=time.perf_counter()-post_started)
        output.mkdir(parents=True, exist_ok=False)
        t = time.perf_counter(); (output/'head.bin').write_bytes(result['head'].tobytes())
        head_meta = dict(dtype='int8', layout='NCHW', shape=list(self.model.head_shape), scale=self.model.head_scale,
                         bytes=result['head'].nbytes, spec_sha256=MODEL_SPEC_SHA256)
        (output/'head.json').write_text(json.dumps(head_meta, indent=2)+'\n', encoding='utf-8')
        np.savez_compressed(output/'labels_native.npz', labels_raw=labels, labels_processed=processed.labels)
        stages['head_labels_write'] = time.perf_counter()-t
        t = time.perf_counter()
        png_details = save_views(output,prep.raw_gray,labels,processed.labels)
        stages['png_encoding'] = time.perf_counter()-t
        report = dict(schema='pld-camus-result-v1', backend=self.backend,
            evidence_scope='real PL execution' if self.backend == 'pl' else 'host software '+self.backend+'; no PL execution',
            model=dict(spec_sha256=MODEL_SPEC_SHA256, input_shape=self.model.input_shape, head=head_meta),
            input=dict(path=str(Path(input_path).resolve()), kind=prep.source.source_kind, native_hw=list(prep.raw_gray.shape),
                       spacing_yx_mm=spacing, clipped_below_0=prep.clipped_low, clipped_above_255=prep.clipped_high, letterbox=prep.letterbox_meta),
            output=dict(class_names=['background','lv_cavity','myocardium','left_atrium'], areas=class_areas(processed.labels, spacing),
                        png_label_values=[0,85,170,255], argmax='restore four float32 channels, strict >, ties keep lower class',
                        postprocess={k:v for k,v in processed.__dict__.items() if k != 'labels'}),
            timings_seconds=stages, session_initialization_seconds=self.initialization_seconds, session_reused=self.jobs > 0,
            timing_details=dict(native=native_details,png=png_details,png_workers=2,png_backend='sub_rle_level1',
                                native_backend=self.native.info if self.native else None),
            command_events=result.get('events', []), conv_blocks=result.get('conv_records', []))
        if html: write_html_preview(output/'preview.html', 'CAMUS 四结构分割',
            [('处理后叠加','overlay.png'),('处理后四分类','segmentation.png'),('处理后标签','label.png'),('原始标签','label_raw.png')])
        t = time.perf_counter()
        with zipfile.ZipFile(output/'result.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
            for p in sorted(output.iterdir()):
                if p.name != 'result.zip': archive.write(p, p.name)
        stages['zip_payload'] = time.perf_counter()-t
        report['seconds_to_metadata_finalize'] = time.perf_counter()-started
        report['timing_scope'] = 'sequential wall stages; PL network includes DMA/compute wait; final metadata append excluded from seconds_to_metadata_finalize'
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        with zipfile.ZipFile(output/'result.zip', 'a') as archive: archive.write(output/'report.json', 'report.json')
        self.jobs += 1
        return dict(output_dir=str(output), backend=self.backend, end_to_end_seconds=time.perf_counter()-started,
                    stages_seconds=stages, session_initialization_seconds=self.initialization_seconds, report=report)
