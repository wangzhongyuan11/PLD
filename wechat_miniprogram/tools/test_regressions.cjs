// Execute page handlers with a small WeChat host; no cloud account or DevTools.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const api = require('../miniprogram/utils/api.js');

function loadPage(name, wx, overrides = {}) {
  let page;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../miniprogram/pages', name, name + '.js'), 'utf8'), {
    Page: value => { page = value; }, require: () => Object.assign({}, api, overrides), wx,
    setInterval: () => 1, clearInterval: () => {}, Date,
  });
  page.setData = patch => Object.assign(page.data, patch);
  return page;
}

async function main() {
  const now = Date.now();
  const board = { online: true, model: 'skin', phase: 'ready', updated_at: now };
  assert.strictEqual(api.boardSnapshot(board, now).online, true);
  assert.strictEqual(api.boardSnapshot(board, now + 60001).online, false);
  assert.strictEqual(api.boardSnapshot(null, now).online, false);

  const notices = [];
  const home = loadPage('home', { showToast: value => notices.push(value.title) });
  home.data.switchTarget = 'acdc'; home.data.switchCommand = 'command-1';
  home.applyBoard({ ...board, phase: 'error', error: 'board failure' });
  home.applyCommand({ status: 'error', error: 'model initialization failed' });
  assert.strictEqual(home.data.switchTarget, '');
  assert.strictEqual(home.data.switchCommand, '');
  assert.strictEqual(home.data.switchError, 'model initialization failed');
  assert.ok(notices.includes('切换失败，请重试'));
  home.data.switchTarget = 'camus';
  home.applyCommand({ status: 'done' });
  assert.strictEqual(home.data.switchTarget, '');
  home.applyBoard({ ...board, updated_at: 1 });
  assert.strictEqual(home.data.board.online, false);

  const uploads = [];
  let storedTask;
  global.wx = {
    showLoading() {}, hideLoading() {},
    cloud: {
      uploadFile: opts => { uploads.push(opts); },
      database: () => ({ collection: () => ({ add: opts => {
        storedTask = opts.data; opts.success({ _id: 'task-1' });
      } }) }),
    },
  };
  const submitted = api.submitTaskFlow({ type: 'acdc', files: [
    { path: 'ED-file', name: 'image.nii.gz' }, { path: 'ES-file', name: 'image.nii.gz' },
  ] });
  assert.strictEqual(new Set(uploads.map(x => x.cloudPath)).size, 2);
  // Reverse completion must not reverse ED and ES in the document.
  uploads[1].success({ fileID: 'cloud://ES' });
  uploads[0].success({ fileID: 'cloud://ED' });
  await submitted;
  assert.deepStrictEqual(storedTask.inputs.map(x => x.fileID), ['cloud://ED', 'cloud://ES']);
  assert.strictEqual(Object.hasOwn(storedTask, 'result'), false);
  assert.strictEqual(Object.hasOwn(storedTask, 'resume_job'), false);

  const retryCalls = [];
  const taskPage = loadPage('task', { redirectTo() {}, showToast() {} }, {
    createTask: opts => { retryCalls.push(opts); return Promise.resolve('new-task'); },
  });
  taskPage.data.task = { type: 'skin', inputs: [], board_completed: true,
    board_job: { model: 'skin', id: 'existing-job' }, result: { views: [{ name: 'mask.png', fileID: 'cloud://mask' }] } };
  taskPage.retry({ currentTarget: { dataset: {} } });
  taskPage.retry({ currentTarget: { dataset: { fresh: true } } });
  assert.strictEqual(retryCalls[0].resume_job.id, 'existing-job');
  assert.strictEqual(retryCalls[0].result.views[0].fileID, 'cloud://mask');
  assert.strictEqual(retryCalls[1].resume_job, null);
  assert.strictEqual(retryCalls[1].result, null);
  taskPage.applyTask({ type: 'skin', status: 'running', result: {
    views: [{ name: 'mask.png', fileID: 'cloud://mask' }], metrics: [],
  } });
  assert.strictEqual(taskPage.data.views.length, 1);

  let submittedCount = 0;
  const acdc = loadPage('acdc', { showToast: value => notices.push(value.title) }, {
    submitTaskFlow: () => { submittedCount++; return new Promise(() => {}); },
    createTask: () => { submittedCount++; return new Promise(() => {}); },
  });
  acdc.data.ed = { name: 'ed.nii', size: 20 * 1024 * 1024 };
  acdc.data.es = { name: 'es.nii', size: 20 * 1024 * 1024 };
  acdc.submitFiles();
  assert.strictEqual(submittedCount, 0);
  acdc.data.demos = [{ _id: 'demo-1', inputs: [] }, { _id: 'demo-2', inputs: [] }];
  acdc.runDemo({ currentTarget: { dataset: { id: 'demo-1' } } });
  acdc.runDemo({ currentTarget: { dataset: { id: 'demo-2' } } });
  assert.strictEqual(submittedCount, 1);
  console.log('MINIPROGRAM_REGRESSIONS_PASS');
}

main().catch(err => { console.error(err); process.exitCode = 1; });
