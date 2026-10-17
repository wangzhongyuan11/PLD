"""Bounded authenticated HTTP/PL golden and pixel check for the dual release."""
import argparse
import base64
import hashlib
import http.client
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hub-root', type=Path, required=True)
    p.add_argument('--model', choices=('skin', 'camus'), required=True)
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hdmi', action='store_true')
    p.add_argument('--resume', action='store_true', help='resume HDMI checks after an already-recorded inference pass')
    a = p.parse_args()
    if not 1 <= a.frames <= 10 or (a.output.exists() and not a.resume):
        p.error('1..10 frames and a new output path required')
    cfg = json.loads((a.hub_root/'config.json').read_text())
    installation = json.loads((a.hub_root/'installation.json').read_text())
    token = Path(cfg['token_file']).read_text().strip()
    cookie = ''

    def req(method, path, body=None, raw=False):
        c = http.client.HTTPConnection('127.0.0.1', 8765, timeout=100)
        c.request(method, path, None if body is None else json.dumps(body),
                  {'Origin': 'http://127.0.0.1:8765', 'Cookie': cookie,
                   'Content-Type': 'application/json'})
        r = c.getresponse(); data = r.read(); session = r.getheader('Set-Cookie', '').split(';')[0]
        code = r.status; c.close()
        if code not in (200, 202):
            raise RuntimeError('HTTP '+str(code)+': '+data[:300].decode(errors='replace'))
        return (data if raw else json.loads(data)), session

    _, cookie = req('POST', '/hub/login', {'token': token})
    status, _ = req('GET', '/hub/status')
    start = time.monotonic()
    deadline = time.monotonic()+200
    while status['phase'] not in ('ready', 'error'):
        if time.monotonic() > deadline:
            raise TimeoutError('Initial service startup did not become ready')
        time.sleep(.25); status, _ = req('GET', '/hub/status')
    if status['phase'] == 'error':
        raise RuntimeError('Service startup failed: '+str(status))
    if status['model'] != a.model:
        req('POST', '/hub/switch', {'model': a.model})
    deadline = time.monotonic()+200
    while status['model'] != a.model or status['phase'] != 'ready':
        if status['phase'] == 'error' or time.monotonic() > deadline:
            raise RuntimeError('Model ready deadline exceeded')
        time.sleep(.25); status, _ = req('GET', '/hub/status')
    prefix = '/'+a.model+'/g/'+str(status['generation'])
    report = {'model': a.model, 'scope': 'board loopback HTTP; original public dev images; golden equality, no quality/long-run/physical-screen acceptance',
              'switch_wait_seconds': time.monotonic()-start, 'switch': status.get('switch'), 'frames': [], 'hdmi': []}
    if a.resume:
        report = json.loads(a.output.read_text())
        if report['model'] != a.model or len(report['frames']) != a.frames:
            raise ValueError('Resume report does not match requested inference batch')
    dataset = Path('/home/xilinx/jupyter_notebooks/pld')/('final_dataset_638/dataset' if a.model == 'skin' else 'camus_dev200_20261015')
    if a.model == 'skin':
        sources = sorted((dataset/'images').glob('*.jpg'))[:a.frames]
    else:
        cases = json.loads((dataset/'dataset.json').read_text())['cases']
        selected = [cases[int(i)] for i in np.linspace(0, len(cases)-1, a.frames)]
        sources = [dataset/c['input'] for c in selected]
    if len(sources) != a.frames:
        raise RuntimeError('Fixture collection is missing images; zero-case pass forbidden')

    def infer(source):
        raw = source.read_bytes(); encoded = base64.b64encode(raw).decode()
        body = {'mode': 'infer', 'name': source.name, 'image': encoded, 'thickness': 2} if a.model == 'skin' else {
            'input_name': source.name, 'files': [{'name': source.name, 'data_base64': encoded}]}
        began = time.monotonic(); job, _ = req('POST', prefix+'/api/jobs', body)
        deadline = time.monotonic()+150
        while job['state'] not in ('done', 'error'):
            if time.monotonic() > deadline: raise TimeoutError('Job timeout; no forced hardware kill')
            time.sleep(.1); job, _ = req('GET', prefix+'/api/jobs/'+job['id'])
        if job['state'] != 'done': raise RuntimeError(str(job))
        ready = time.monotonic()-began
        archive, _ = req('GET', prefix+'/files/'+job['id']+('/results.zip' if a.model == 'skin' else '/result.zip'), raw=True)
        root = Path(installation['backends'][a.model]['data'])/job['id']
        head = root/('inference/frame_000001.bin' if a.model == 'skin' else 'output/head.bin')
        golden = dataset/(source.stem+'.npz') if a.model == 'skin' else dataset/next(c['expected'] for c in selected if dataset/c['input'] == source)
        with np.load(golden, allow_pickle=False) as expected:
            expected_head = expected['head' if a.model == 'skin' else 'logits_int8']
            if head.read_bytes() != expected_head.tobytes(): raise AssertionError('PL head differs from frozen golden: '+source.name)
            if a.model == 'camus':
                with np.load(root/'output/labels_native.npz', allow_pickle=False) as labels:
                    assert np.array_equal(labels['labels_raw'], expected['labels'])
                    assert np.array_equal(labels['labels_processed'], expected['processed'])
        pixels = {}
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert len(z.namelist()) == len(set(z.namelist())) and z.testzip() is None
            for name in z.namelist():
                if name.endswith('.png'):
                    with Image.open(io.BytesIO(z.read(name))) as image:
                        pixels[name] = {'size': list(image.size), 'pixels_sha256': hashlib.sha256(np.array(image).tobytes()).hexdigest()}
        if a.model == 'camus':
            with Image.open(root/'output/quad.png') as image: assert image.size == (640, 480)
        row = {'source': source.name, 'input_sha256': hashlib.sha256(raw).hexdigest(), 'head_sha256': hashlib.sha256(head.read_bytes()).hexdigest(),
               'golden_equal': True, 'job_id': job['id'], 'post_to_zip_ready_seconds': ready,
               'post_to_download_seconds': time.monotonic()-began, 'pngs': pixels,
               'stage_seconds': job.get('result', {}).get('stages_seconds', job.get('statistics', {}).get('final_stage_seconds', {})),
               'timing_details': job.get('result', {}).get('timing_details'), 'job': job}
        report['frames'].append(row)
        a.output.write_text(json.dumps(report, indent=2)+'\n')
        print('GOLDEN_HTTP_PASS', a.model, source.name, round(ready, 4), len(pixels), flush=True)
        return job['id']

    if a.resume:
        last = report['frames'][-1]['job_id']
    else:
        for source in sources: last = infer(source)
    if a.hdmi:
        began = time.monotonic()
        state, _ = req('POST', prefix+'/api/hdmi', {'action': 'show', 'id': last, 'view': 'quad' if a.model == 'camus' else 'outlined'})
        deadline = time.monotonic()+70
        while True:
            state, _ = req('GET', prefix+'/api/status')
            hdmi = state.get('hdmi')
            ready = hdmi == 'showing' if isinstance(hdmi, str) else bool(hdmi and (hdmi.get('ready') or hdmi.get('running')))
            if ready: break
            if time.monotonic() > deadline: raise TimeoutError('HDMI frame submission not confirmed: '+str(state))
            time.sleep(.3)
        report['hdmi'].append({'frame_submitted': True, 'seconds': time.monotonic()-began, 'physical_screen_observed': False, 'status': state})
        req('POST', prefix+'/api/hdmi', {'action': 'stop'})
        infer(sources[0])
    report['complete'] = True
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('RELEASE_BOARD_PASS', a.model, len(report['frames']), flush=True)


if __name__ == '__main__':
    main()
