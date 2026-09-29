"""Board-only, one-input full PL TinyUNet validation and bounded recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import DEFAULT_MODEL_ROOT,Model,digest
from tensor_accel import TileAccelerator,AcceleratorFault
from pl_graph import PLGraph


def fault_checks(device,model):
    results=[]
    device.configure(0,1,16,1,1,1)
    def short_packet():
        device.tx[:8]=1
        device._execute(2,device.tx,8)
    try:
        device._guard(short_packet)
    except AcceleratorFault as exc:
        if not exc.recovered or exc.state["core"]["error"]!=2:
            raise
        results.append({"fault":"short_input","state":exc.state,"recovered":exc.recovered})
    else: raise RuntimeError("short packet unexpectedly accepted")
    old_timeout=device.timeout
    device.timeout=0.05
    def missing_packet():
        device._begin(2)
        device._wait(16,0)
    try:
        try: device._guard(missing_packet)
        except AcceleratorFault as exc:
            if not exc.recovered or "core completion timeout" not in str(exc): raise
            results.append({"fault":"missing_input","state":exc.state,"recovered":exc.recovered})
        else: raise RuntimeError("missing input unexpectedly completed")
    finally: device.timeout=old_timeout
    if device.core.read(0x0c)&0x7f:
        raise RuntimeError("ready/error state survived recovery")
    # Abort a real deep convolution with RX armed, before any result is ready.
    weights,bias,mult=model.parameters[7]
    source=np.pad(model.golden(7,"input"),((0,0),(0,0),(1,1),(1,1)))
    cfg=device.configure(0,34,34,64,64,3)
    device.load_parameters(weights,bias,mult)
    device.load_input(source)
    device.dma.submit(destination=device.rx,receive_nbytes=cfg["output_bytes"])
    device._begin(4)
    deadline=time.monotonic()+device.timeout
    while device.core.read(0x58)<64:
        if time.monotonic()>=deadline: raise TimeoutError("compute did not start for reset test")
    before=device.snapshot()
    before["compute_cycles"]=int(device.core.read(0x58))
    if before["core"]["status"]&7!=1 or before["core"]["sent"]!=0:
        raise RuntimeError("reset test did not catch active computation")
    device.recover()
    if device.core.read(0x0c)&0x7f or device.dma.active:
        raise RuntimeError("compute reset did not clear active state")
    results.append({"fault":"reset_during_compute","state":before,"recovered":True})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bit",required=True)
    p.add_argument("--model",default=DEFAULT_MODEL_ROOT)
    p.add_argument("--repeat",type=int,default=2)
    p.add_argument("--benchmark-frames",type=int,default=1)
    p.add_argument("--check-recovery",action="store_true")
    p.add_argument("--native-copy",help="verified native copy library")
    p.add_argument("--tx-range-sync",action="store_true",help="XRT TX prefix cache synchronization")
    p.add_argument("--native-control",help="verified native command library")
    p.add_argument("--report",required=True)
    p.add_argument("--output",default="head_int8.bin")
    p.add_argument("--input",help="host-preprocessed NCHW INT8 input, compared with frozen A input")
    p.add_argument("--input-metadata",help="matching JSON from prepare_cnn_input.py")
    p.add_argument("--fuse-maps",action="store_true",help="ABI 1.1 PL input-LUT convolution")
    p.add_argument("--fuse-decoder",action="store_true",help="ABI 1.2 PL fused decoder-entry convolution")
    args=p.parse_args()
    if args.repeat<1: p.error("repeat must be positive")
    if args.benchmark_frames<0: p.error("benchmark-frames must be nonnegative")
    if bool(args.input)!=bool(args.input_metadata): p.error("input and input-metadata must be supplied together")
    report={"passed":False,"frames":[],"scope":"one quantized first input; all convolution and non-convolution arithmetic in PL"}
    device=None
    exit_code=1
    try:
        model=Model(Path(args.model))
        bit=Path(args.bit).resolve()
        files=[bit,bit.with_suffix(".hwh"),Path(__file__),Path(__file__).with_name("tensor_accel.py"),
               Path(__file__).with_name("pl_graph.py"),Path(__file__).with_name("simple_dma.py")]
        report["sha256"]={f.name:digest(f) for f in files}
        report["model_spec_sha256"]=model.spec_hash
        input_image=model.golden(0,"input")
        input_metadata=None
        if args.input:
            prepared=np.fromfile(args.input,dtype=np.int8).reshape(input_image.shape)
            if not np.array_equal(prepared,input_image):
                raise ValueError("prepared input differs from A's frozen first input")
            input_image=prepared
            input_metadata=json.loads(Path(args.input_metadata).read_text())
        report["first_input_sha256"]=hashlib.sha256(input_image.tobytes()).hexdigest()
        device=TileAccelerator(bit,native_copy=args.native_copy,tx_range_sync=args.tx_range_sync,
                               native_control=args.native_control)
        report["copy_backend"]=device.copy_info
        report["tx_sync_backend"]=device.tx_sync_info
        report["control_backend"]=device.control_info
        report["clock_mhz"]=device.clock_mhz
        graph=PLGraph(model,device,fuse_maps=args.fuse_maps,fuse_decoder=args.fuse_decoder)
        def trace(record,output):
            record["output_sha256"]=hashlib.sha256(output.tobytes()).hexdigest()
            print(f"LAYER_PASS {record['layer']:02d} {record['name']} seconds={record['seconds']:.3f}",flush=True)
        total=args.repeat+int(args.check_recovery)
        for frame in range(1,total+1):
            if args.check_recovery and frame==total:
                report["faults"]=fault_checks(device,model)
                print("FAULT_RECOVERY_PASS; verifying a complete next frame",flush=True)
            device.events=[]
            output,record=graph.run(input_image,frame=frame,verify=True,callback=trace,input_metadata=input_metadata)
            record["commands"]=len(device.events)
            record["compute_cycles"]=sum(e["compute_cycles"] for e in device.events)
            record["dma_wait_seconds"]=sum(e["dma_seconds"] for e in device.events)
            record["command_seconds"]=sum(e["seconds"] for e in device.events)
            record["head_sha256"]=hashlib.sha256(output.tobytes()).hexdigest()
            report["frames"].append(record)
            Path(args.output).write_bytes(output.tobytes())
            Path(args.output).with_suffix(".json").write_text(json.dumps(record["result"],indent=2)+"\n",encoding="utf-8")
            print(f"FRAME_PASS id={frame} seconds={record['seconds']:.3f} commands={record['commands']}",flush=True)
        report["benchmark_frames"]=[]
        for i in range(args.benchmark_frames):
            device.events=[]
            benchmark_output,record=graph.run(input_image,frame=total+i+1,input_metadata=input_metadata)
            if not np.array_equal(benchmark_output,output):
                raise RuntimeError("benchmark head differs from verified result")
            record["commands"]=len(device.events)
            record["compute_cycles"]=sum(e["compute_cycles"] for e in device.events)
            record["dma_wait_seconds"]=sum(e["dma_seconds"] for e in device.events)
            record["b_subsystem_fps"]=1/record["seconds"]
            report["benchmark_frames"].append(record)
            print(f"BENCHMARK_PASS seconds={record['seconds']:.3f} B_fps={record['b_subsystem_fps']:.3f}",flush=True)
        report["passed"]=True
        print(f"CNN_BOARD_PASS frames={total} layers_per_frame=15 head_bytes=65536",flush=True)
        exit_code=0
    except Exception as exc:
        report["error"]=str(exc)
        if isinstance(exc,AcceleratorFault):
            report["fault_state"]=exc.state; report["recovered"]=exc.recovered
        print(f"CNN_BOARD_FAIL {exc}",file=sys.stderr,flush=True)
        exit_code=1
    finally:
        if device is not None:
            try: device.close()
            except Exception as exc:
                report["passed"]=False
                report["close_error"]=str(exc)
                exit_code=2
                print(f"CNN_CLOSE_FAIL {exc}",file=sys.stderr,flush=True)
        Path(args.report).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return exit_code


if __name__=="__main__": raise SystemExit(main())
