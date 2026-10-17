'use strict';
// Drive actual page handlers with a small DOM and API fixture, without hardware.
const fs = require('fs'), vm = require('vm'), path = require('path'), assert = require('assert');
const source = fs.readFileSync(path.join(__dirname, '../interactive.js'), 'utf8');
const elements = new Map(), requests = [], actions = [];
const TOKEN = 'a'.repeat(32), SHA = 'b'.repeat(64), LABEL = 'c'.repeat(64);
const PNG = 'data:image/png;base64,AAAA';
let rejectUpload = false, rejectReplay = false, downloaded = null;

function canvasStub() {
  return {width: 100, height: 100,
    getContext: () => ({drawImage() {}, clearRect() {}, strokeRect() {}, beginPath() {},
      fill() {}, stroke() {}, arc() {}, getImageData: () => ({data: new Uint8Array(40000)})}),
    toDataURL: () => PNG,
    getBoundingClientRect: () => ({left: 0, top: 0, width: 100, height: 100}),
    setPointerCapture() {},
  };
}
function element(id) {
  if (elements.has(id)) return elements.get(id);
  const el = Object.assign(canvasStub(), {
    id, value: '1', checked: false, disabled: false, textContent: '', innerHTML: '', src: '', files: [],
    removeAttribute(name) { delete this[name]; }, querySelector() { return {}; },
    click() { const action = this.onclick?.(); if (action?.then) actions.push(action); },
  });
  elements.set(id, el); return el;
}
function result() {
  return {request_id: 'job', image: {width: 100, height: 100, sha256: SHA},
    outputs: {enhanced_png: PNG, target_mask_png: PNG, overlay_png: PNG},
    baseline: {outputs: {overlay_png: PNG}, statistics: {target_pixels: 25}, cache_hit: true},
    statistics: {target_pixels: 30, area_ratio: .003, per_class_pixels: {0: 9970, 1: 30}, changed_pixels_by_refinement: 5},
    identity: {prediction_cache_hit: true, spec_sha256: 'spec'}, execution: {pl_executed_this_request: false},
    label_hashes: {baseline: LABEL, final: LABEL}, elapsed_ms: 1, timings_ms: {}, warnings: [],
    replay: {schema: 'pld-interactive-replay-v1', image: {width: 100, height: 100, sha256: SHA}}};
}
const context = {
  document: {getElementById: element, createElement: tag => tag === 'canvas' ? canvasStub() : {click() {}}},
  fetch: async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null; requests.push({url, body});
    let response, ok = true;
    if (url === '/hub/status') response = {phase: 'ready', model: 'camus', generation: 1};
    else if (url === '/interactive/images') {
      ok = !rejectUpload;
      response = ok ? {image_token: TOKEN, sha256: SHA, width: 100, height: 100} : {error: 'upload refused'};
    } else if (url === '/interactive/replay') {
      ok = !rejectReplay;
      response = ok ? {model: 'camus', spec_sha256: 'spec', image: {width: 100, height: 100, sha256: SHA},
        original: {width: 200, height: 200, transform: {mode: 'fit_long_side_1024', processed_width: 100, processed_height: 100}},
        parameters: body.replay.parameters, expected: {baseline_labels_sha256: LABEL, final_labels_sha256: LABEL}}
        : {error: 'tampered replay'};
    } else if (url === '/interactive/jobs') response = {id: 'job', state: 'done', result: result()};
    else throw Error('unexpected API '+url);
    return {ok, json: async () => response};
  },
  Image: class {constructor() {this.naturalWidth = this.naturalHeight = 100;}
    set src(value) {this._src = value; this.onload?.();}},
  URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}},
  Blob: class {constructor(parts) {downloaded = JSON.parse(parts[0]);}},
  performance: {now: () => 1}, setTimeout: () => 0, crypto: {}, console,
};
vm.createContext(context); vm.runInContext(source, context);
const read = expression => JSON.parse(JSON.stringify(vm.runInContext(expression, context)));
const snapshot = () => read('snapshot()');
const count = url => requests.filter(r => r.url === url).length;
function pointer(x, y, endX = x, endY = y) {
  element('source').onpointerdown({clientX: x, clientY: y, pointerId: 1});
  element('source').onpointerup({clientX: endX, clientY: endY, pointerId: 1});
}

(async () => {
  await new Promise(resolve => setImmediate(resolve));
  element('threshold').value = '.45'; element('file').files = [{size: 4096}];
  await element('file').onchange(); assert.equal(count('/interactive/images'), 1);
  element('tool').value = 'roi'; pointer(10, 10, 90, 90);
  element('tool').value = 'box'; pointer(20, 20, 80, 80);
  element('tool').value = 'foreground'; pointer(40, 40);
  const prompted = snapshot();
  assert.equal(prompted.points.length, 1);
  assert.deepEqual(prompted.box, {x0: 20, y0: 20, x1: 80, y1: 80});
  element('target').value = '2'; element('target').onchange();
  assert.equal(snapshot().points.length, 0); assert.equal(snapshot().box, null);
  assert.deepEqual(snapshot().roi, prompted.roi);
  element('undo').onclick(); assert.deepEqual(snapshot(), prompted);
  element('custom').checked = true; element('custom').onchange(); const custom = snapshot();
  element('threshold').value = '.65'; element('threshold').onchange();
  element('undo').onclick(); assert.deepEqual(snapshot(), custom);
  element('undo').onclick(); assert.deepEqual(snapshot(), prompted);
  await element('run').onclick();
  let job = requests.filter(r => r.url === '/interactive/jobs').at(-1).body;
  assert.equal(job.image_token, TOKEN); assert.equal('image' in job, false);
  assert.equal(count('/interactive/images'), 1, 'prompt changes never re-upload image');
  assert.equal(element('download').disabled, false);
  vm.runInContext('setBusy(true)', context);
  const busyState = snapshot(); pointer(50, 50); element('clear').onclick(); element('undo').onclick();
  assert.deepEqual(snapshot(), busyState); assert.equal(element('target').disabled, true);
  vm.runInContext('setBusy(false)', context);
  element('restore').onclick(); await Promise.all(actions.splice(0));
  assert.deepEqual(snapshot().roi, prompted.roi);
  assert.equal(snapshot().box, null); assert.deepEqual(snapshot().points, []);
  assert.equal(snapshot().threshold, prompted.threshold);
  job = requests.filter(r => r.url === '/interactive/jobs').at(-1).body;
  assert.deepEqual(job.parameters.points, []); assert.equal(job.parameters.box, null);
  element('undo').onclick(); assert.deepEqual(snapshot(), prompted);
  assert.equal(element('download').disabled, true, 'undo invalidates stale output');
  rejectUpload = true; const beforeFailure = snapshot();
  await element('file').onchange(); rejectUpload = false;
  assert.deepEqual(snapshot(), beforeFailure); assert.equal(read('imageToken'), TOKEN);
  const record = {image_png: PNG, parameters: {backend: 'int8', variant: 'candidate', target_class: 2,
    threshold: .55, roi: null, box: null, points: [{x: 30, y: 30, label: 'foreground'}]}};
  element('replayFile').files = [{size: 4096, text: async () => JSON.stringify(record)}];
  rejectReplay = true; const uploads = count('/interactive/images');
  await element('replayFile').onchange(); rejectReplay = false;
  assert.deepEqual(snapshot(), beforeFailure); assert.equal(count('/interactive/images'), uploads);
  const jobsBeforeImport = count('/interactive/jobs');
  await element('replayFile').onchange();
  assert.equal(count('/interactive/jobs'), jobsBeforeImport, 'import never runs inference');
  assert.equal(requests.filter(r => r.url === '/interactive/images').at(-1).body.expected_sha256, SHA);
  assert.equal(snapshot().target, '2'); assert.equal(Number(snapshot().threshold), .55);
  assert.equal(snapshot().points.length, 1); assert.equal(read('sourceWidth'), 200);
  await element('run').onclick();
  assert(element('message').textContent.includes('基线哈希一致，最终哈希一致'));
  element('download').onclick();
  assert.equal(downloaded.original.width, 200, 're-export preserves original dimensions');
  assert.equal(downloaded.image_png, PNG);
  console.log('PASS full undo/restore, token-only jobs, busy guards, failed upload/import preservation, HTTP replay and re-export');
})().catch(error => {console.error(error); process.exitCode = 1;});
