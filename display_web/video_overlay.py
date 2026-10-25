"""Initialize the existing base bitstream's video and buttons only."""
from pathlib import Path
_prepared = None


def prepare_video_overlay():
    """Build the software descriptions without loading or accessing video IP."""
    global _prepared
    if _prepared is None:
        import pynq
        import pynq.lib.video
        try:from .overlay_metadata import enable_metadata_cache
        except ImportError:from overlay_metadata import enable_metadata_cache
        enable_metadata_cache()
        bit=Path(pynq.__file__).parent/'overlays/base/base.bit'
        _prepared=pynq.Overlay(str(bit),download=False)
    return _prepared


def video_overlay():
    global _prepared
    try:from .latency import Stages
    except ImportError:from latency import Stages
    trace = Stages('video_overlay')
    import pynq
    import pynq.lib.video  # Register the existing HDMI/VDMA hierarchy drivers.
    trace.mark('pynq_import')
    try:from .overlay_metadata import enable_metadata_cache
    except ImportError:from overlay_metadata import enable_metadata_cache
    enable_metadata_cache()
    overlay = prepare_video_overlay()
    _prepared = None
    trace.mark('metadata_bind')
    overlay.download()
    trace.mark('program_overlay')
    overlay.buttons = overlay.btns_gpio.channel1
    overlay.buttons.setlength(4)
    overlay.buttons.setdirection('in')
    trace.mark('buttons')
    return overlay
