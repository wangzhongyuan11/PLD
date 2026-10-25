"""Optional presentation adapter; frozen CAMUS inference files stay untouched."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from quad_display import thumbnail, enhance, compose

ROOT = Path(__file__).resolve().parent


def build_views(input_path, output):
    from camus.preprocess import load_gray_frame
    from camus.png import save_png
    output = Path(output)
    # Display-only reload after inference. Do not change preprocessing or labels.
    loaded = load_gray_frame(input_path)
    raw = np.clip(loaded.data, 0, 255).astype(np.uint8)
    original = thumbnail(raw)
    enhanced = enhance(original)
    with Image.open(output/'segmentation.png') as im:
        segmentation = thumbnail(im, nearest=True)
    with Image.open(output/'overlay.png') as im:
        overlay = thumbnail(im)
    views = [('quad_original.png', np.asarray(original)),
             ('quad_enhanced.png', np.asarray(enhanced)),
             ('quad.png', compose([original, enhanced, segmentation, overlay]))]
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='camus-quad-png') as pool:
        list(pool.map(lambda item: save_png(item[1], output/item[0]), views))


def install(web):
    original_run = web.CamusPipeline.run
    original_get = web.Handler.do_GET
    original_hdmi = web.CamusServer.start_hdmi

    def run(pipeline, input_path, output_dir, *args, **kwargs):
        result = original_run(pipeline, input_path, output_dir, *args, **kwargs)
        started = time.perf_counter()
        build_views(input_path, output_dir)
        elapsed = time.perf_counter()-started
        result['stages_seconds']['quad_display'] = elapsed
        result['end_to_end_seconds'] += elapsed
        return result

    def get(handler):
        if web.urlsplit(handler.path).path == '/':
            return handler.send_bytes((ROOT/'quad_camus.html').read_bytes(), kind='text/html; charset=utf-8')
        return original_get(handler)

    def hdmi(server, identifier, view):
        if view != 'quad' and not getattr(server,'display_agent',None):
            return original_hdmi(server, identifier, view)
        if view not in ('quad','overlay','segmentation','label','label_raw'):raise ValueError('invalid CAMUS view')
        if server.pipeline.backend != 'pl':
            raise ValueError('HDMI requires the real board PL backend')
        with server.guard:
            state = server.jobs.get(identifier, {}).get('state')
        if state != 'done':
            raise ValueError('completed CAMUS job required')
        folder = server.data/identifier/'output'
        if not (folder/'quad.png').is_file():
            raise ValueError('四宫格未生成，请重新处理图片')
        if (getattr(server,'display_agent',None) and server.hdmi is not None
                and server.hdmi.poll() is None and server.hdmi_folder==folder):
            if getattr(server,'hdmi_view',None)!=view:server.hdmi.update(view)
            server.hdmi_view=view;return
        # Same lifecycle as the original backend: no second hardware owner.
        server.stop_hdmi()
        server.pipeline.close()
        server.released = True
        server.hdmi_folder = folder
        (folder/'hdmi.ready.json').unlink(missing_ok=True)
        command=[sys.executable,'-u',str(ROOT/'quad_camus_hdmi.py'),'--result',str(folder),
                 '--source',str(Path(web.__file__).resolve().parents[1]),'--view',view]
        if getattr(server,'display_agent',None):server.hdmi=server.display_agent.start(command,folder/'hdmi.log')
        else:
            with (folder/'hdmi.log').open('w', encoding='utf-8') as log:
                server.hdmi=subprocess.Popen(command,stdout=log,stderr=log)
        server.hdmi_started = time.monotonic()
        server.hdmi_view=view

    web.ARTIFACTS.update({'quad_original.png', 'quad_enhanced.png', 'quad.png'})
    web.CamusPipeline.run = run
    web.Handler.do_GET = get
    web.CamusServer.start_hdmi = hdmi
