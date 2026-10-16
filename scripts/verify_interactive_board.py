"""Small authenticated real-board integration check; run with PYNQ Python.

Uses one named public development image, not a segmentation quality benchmark.
Records wall time, server phases and pixel hashes without credentials or images.
"""
import argparse
import base64
import hashlib
import http.client
import io
import json
from pathlib import Path
import os
import shlex
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('skin', 'camus'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--switch', action='store_true')
    args = parser.parse_args()
    root = Path('/home/xilinx/jupyter_notebooks/PLD_DualModel')
    installation = json.loads((root/'installation.json').read_text())
    config = json.loads((root/'config.json').read_text())
    token = Path(config['token_file']).read_text().strip()
    cookie = ''

    def request(method, path, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', 8765, timeout=45)
        headers = {'Origin':'http://127.0.0.1:8765', 'Cookie':cookie,
                   'Content-Type':'application/json'}
        connection.request(method, path, None if body is None else json.dumps(body), headers)
        response = connection.getresponse()
        data = json.loads(response.read())
        session = response.getheader('Set-Cookie', '').split(';')[0]
        connection.close()
        if response.status not in (200, 202):
            raise RuntimeError(str(data.get('error', response.status)))
        return data, session

    _, cookie = request('POST', '/hub/login', {'token':token})
    status, _ = request('GET', '/hub/status')
    switch_started = time.monotonic()
    if status['model'] != args.model:
        if not args.switch:
            raise RuntimeError('Requested model is not current; use --switch explicitly')
        request('POST', '/hub/switch', {'model':args.model})
    deadline = time.monotonic()+180
    while status['model'] != args.model or status['phase'] != 'ready':
        if status['phase'] == 'error' or time.monotonic() > deadline:
            raise RuntimeError('Model not ready: '+str(status))
        time.sleep(.25)
        status, _ = request('GET', '/hub/status')
    switch_wait = time.monotonic()-switch_started
    if args.model == 'skin':
        source = Path('/home/xilinx/jupyter_notebooks/pld/stage5_dataset_20260927/dataset/images/ISIC_0000003.jpg')
        with Image.open(source) as opened:
            frame = opened.convert('RGB')
    else:
        source = Path('/home/xilinx/jupyter_notebooks/pld/camus_dev200_20261015/patient0200_2CH_ED.nii.gz')
        sys.path.insert(0, installation['backends']['camus']['source'])
        # Match the existing service's optional medical-file dependencies;
        # do not install/upgrade packages just to run a fixture check.
        environment = subprocess.check_output(['systemctl','show','pld-camus.service',
                                               '-p','Environment','--value'], text=True)
        for entry in shlex.split(environment):
            if entry.startswith('PYTHONPATH='):
                sys.path.extend(p for p in entry.split('=',1)[1].split(os.pathsep) if p)
        from camus.preprocess import load_gray_frame
        gray = load_gray_frame(source).data
        frame = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert('RGB')
    factor = min(1, 512/max(frame.size))
    frame = frame.resize(tuple(max(2, round(n*factor)) for n in frame.size), Image.BILINEAR)
    w, h = frame.size
    encoded = io.BytesIO()
    frame.save(encoded, format='PNG', compress_level=1)
    image = base64.b64encode(encoded.getvalue()).decode()
    cases = [('full', {}), ('cached_prompts', {
        'box':dict(x0=w//8, y0=h//8, x1=7*w//8, y1=7*h//8),
        'points':[dict(x=w//2, y=h//2, label='foreground'),
                  dict(x=w//4, y=h//4, label='background')]})]
    if not args.baseline:
        cases.append(('changed_roi', {'roi':dict(x0=w//8, y0=h//8, x1=7*w//8, y1=7*h//8)}))
    report = dict(model=args.model, image=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  processing_size=[w,h], switch_wait_seconds=switch_wait, switch=status.get('switch'),
                  scope='public development image; functionality and latency only, no quality/long-run/HDMI acceptance', jobs=[])
    for name, parameters in cases:
        parameters.update(backend='int8', variant='delivery' if args.model=='skin' else 'candidate', target_class=1)
        began = time.monotonic()
        job, _ = request('POST', '/interactive/jobs', dict(model=args.model, generation=status['generation'],
                                                         image=image, parameters=parameters))
        deadline = time.monotonic()+330
        while job['state'] == 'running':
            if time.monotonic() > deadline:
                raise RuntimeError('Interactive timeout; task was not killed')
            time.sleep(.1)
            job, _ = request('GET', '/interactive/jobs/'+job['id'])
        elapsed = time.monotonic()-began
        if job['state'] != 'done':
            raise RuntimeError(job.get('error', 'Interactive failed'))
        result = job['result']
        outputs = {}
        labels = None
        for key, value in result.pop('outputs').items():
            raw = base64.b64decode(value.split(',')[1])
            with Image.open(io.BytesIO(raw)) as opened:
                pixels = np.array(opened)
                outputs[key] = dict(size=list(opened.size), bytes=len(raw),
                                    pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
            if key == 'labels_png':
                labels = pixels
        assert labels is not None and labels.shape == (h,w)
        assert set(np.unique(labels)).issubset(range(2 if args.model=='skin' else 4))
        cache = result['identity']['prediction_cache_hit']
        assert cache == (name=='cached_prompts'), (name, cache)
        if name == 'cached_prompts':
            assert labels[h//2,w//2] == 1 and labels[h//4,w//4] != 1
            assert not (labels[:h//8] == 1).any()
        if name == 'changed_roi':
            assert not labels[:h//8].any() and not labels[:,7*w//8:].any()
        row = dict(case=name, http_submit_to_result_seconds=elapsed, result=result, outputs=outputs)
        report['jobs'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps(dict(case=name, wall_seconds=elapsed, elapsed_ms=result['elapsed_ms'],
                              cache=cache, phases=result.get('timings_ms'),
                              prediction=result['identity'].get('prediction_timings_ms')), ensure_ascii=False), flush=True)
    print('INTERACTIVE_BOARD_PASS',args.model, len(cases), flush=True)


if __name__ == '__main__':
    main()
