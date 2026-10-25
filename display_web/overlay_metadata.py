"""Per-bitstream PYNQ metadata cache; never caches hardware or DMA objects."""
import hashlib
import datetime
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import stat


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def _load_snapshot(path):
    # This verified pickle is a pure metadata graph. Repeated cyclic scans
    # during its construction are expensive; restore the caller's GC policy
    # even on an error. Reference counting and explicit hardware closes stay on.
    enabled=gc.isenabled()
    try:
        gc.disable()
        with path.open('rb') as stream:return pickle.load(stream)
    finally:
        if enabled:gc.enable()


def enable_metadata_cache(directory=None):
    import pynq
    from pynq.pl_server.device import Device
    from pynq.pl_server import embedded_device
    device = Device.active_device
    if getattr(device, '_pld_metadata_cache', False): return
    directory = Path(directory or os.environ.get('PLD_METADATA_CACHE', '/var/cache/pld/overlay-metadata'))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    original = device.get_bitfile_metadata
    sdk = _sha(Path(embedded_device.__file__))
    memory = {}
    snapshots = {}

    def get_metadata(bitfile_name, partial=False):
        bit = Path(bitfile_name).resolve(); hwh = bit.with_suffix('.hwh')
        if partial or not hwh.is_file(): return original(bitfile_name, partial=partial)
        identity = dict(bit=_sha(bit), hwh=_sha(hwh), pynq=pynq.__version__, sdk=sdk)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key in memory:
            parser = memory[key]
            if hasattr(parser, '_from_cache'): del parser._from_cache
            return parser
        cache = directory/(key+'.pkl'); manifest = directory/(key+'.json')
        if cache.exists() and manifest.exists():
            record = json.loads(manifest.read_text())
            if record.get('identity') == identity and record.get('sha256') == _sha(cache):
                parser = _load_snapshot(cache)
                # A different bitstream may have been active. Let PYNQ publish
                # the actual new global state during the real download.
                if hasattr(parser, '_from_cache'): del parser._from_cache
                print('OVERLAY_METADATA_CACHE hit', bit.name, flush=True)
                memory.clear(); memory[key] = parser
                snapshots.clear(); snapshots[id(parser)] = cache
                return parser
        parser = original(bitfile_name, partial=partial)
        temporary = directory/(key+'.'+str(os.getpid())+'.tmp')
        with temporary.open('wb') as stream: pickle.dump(parser, stream)
        temporary.replace(cache)
        temporary = manifest.with_suffix('.'+str(os.getpid())+'.tmp')
        temporary.write_text(json.dumps(dict(identity=identity, sha256=_sha(cache))))
        temporary.replace(manifest)
        print('OVERLAY_METADATA_CACHE fill', bit.name, flush=True)
        memory.clear(); memory[key] = parser
        snapshots.clear(); snapshots[id(parser)] = cache
        return parser

    device.get_bitfile_metadata = get_metadata
    # PYNQ 3.1.1 publishes a GlobalState and serializes the same ~18 MB base
    # metadata graph at every download. The source-bound snapshot above is
    # already serialized. Preserve state publication and reuse its bytes.
    if (pynq.__version__ == '3.1.1' and hasattr(device,'gen_cache')
            and hasattr(embedded_device,'GlobalState')):
        original_gen_cache = device.gen_cache
        def gen_cache(bitstream, parser=None):
            source = snapshots.get(id(parser))
            if source is None or not Path(bitstream.bitfile_name).is_file():
                return original_gen_cache(bitstream, parser)
            now = datetime.datetime.now()
            timestamp = f'{now.year}/{now.month}/{now.day} {now.hour}:{now.minute}:{now.second} +{now.microsecond}'
            state = embedded_device.GlobalState(bitfile_name=str(bitstream.bitfile_name),
                bitfile_hash=embedded_device.bitstream_hash(bitstream.bitfile_name),
                timestamp=timestamp, active_name=device.name, psddr=parser.mem_dict.get('PSDDR',{}))
            for name, details in parser.ip_dict.items():
                if details['type'] in ('xilinx.com:ip:pr_axi_shutdown_manager:1.0',
                                       'xilinx.com:ip:dfx_axi_shutdown_manager:1.0'):
                    state.add(name=name,addr=details['phys_addr'])
            embedded_device.save_global_state(state)
            if getattr(parser,'systemgraph',None) is not None:
                target = Path(embedded_device.__file__).parent/'_current_metadata.pkl'
                temporary = target.with_suffix('.'+str(os.getpid())+'.tmp')
                previous = target.stat() if target.exists() else None
                shutil.copyfile(source,temporary)
                temporary.chmod(stat.S_IMODE(previous.st_mode) if previous else 0o644)
                if previous and hasattr(os,'getuid') and os.getuid()==0:
                    os.chown(temporary,previous.st_uid,previous.st_gid)
                temporary.replace(target)
        device.gen_cache = gen_cache
    device._pld_metadata_cache = True
