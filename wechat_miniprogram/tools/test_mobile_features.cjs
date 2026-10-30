const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const root = path.join(__dirname, '../miniprogram');
const api = require(root + '/utils/api.js');
const roi = require(root + '/utils/roi.js');
const album = require(root + '/utils/album.js');

function page(name, wx, overrides={}) {
  let value;
  vm.runInNewContext(fs.readFileSync(root + '/pages/' + name + '/' + name + '.js', 'utf8'), {
    wx, Page: p => { value=p; },
    require: p => p.includes('roi.js') ? roi : p.includes('album.js') ? album : Object.assign({}, api, overrides),
    setInterval: () => 1, clearInterval() {}, console,
  });
  value.setData = fields => Object.assign(value.data, fields);
  return value;
}

function albumHost() {
  const record = { scope:true, system:'authorized', saved:[], copied:[], removed:[] };
  const wx = {
    env:{USER_DATA_PATH:'/user'},
    cloud:{ downloadFile:o => o.success({tempFilePath:'/temp/actually_jpeg.png'}) },
    getImageInfo:o => o.success({type:'jpeg', width:100, height:80}),
    getAppAuthorizeSetting:() => ({albumAuthorized:record.system}),
    getSetting:o => o.success({authSetting:{'scope.writePhotosAlbum':record.scope}}),
    authorize:o => { record.scope=true; o.success({}); },
    showModal:o => o.success({confirm:true}),
    openSetting:o => { record.scope=true; o.success({}); },
    openAppAuthorizeSetting:o => { record.system='authorized'; o.success({}); },
    saveImageToPhotosAlbum:o => { record.saved.push(o.filePath); o.success({}); },
    getFileSystemManager:() => ({
      copyFile:o => { record.copied.push(o.destPath); o.success({}); },
      unlink:o => record.removed.push(o.filePath),
    }),
  };
  return {wx, record};
}

async function main() {
  let host=albumHost(); global.wx=host.wx;
  await album.saveImage('cloud://preview');
  assert(host.record.saved[0].endsWith('.jpg'), 'JPEG bytes get a JPEG extension');
  assert.deepStrictEqual(host.record.removed, host.record.saved);

  host=albumHost(); global.wx=host.wx; host.record.scope=false;
  await album.saveImage('cloud://preview');
  assert.equal(host.record.saved.length, 1, 'save resumes after settings return');

  host=albumHost(); global.wx=host.wx; host.record.system='denied';
  await album.saveImage('cloud://preview');
  assert.equal(host.record.system, 'authorized'); assert.equal(host.record.saved.length, 1);

  host=albumHost(); global.wx=host.wx;
  host.wx.saveImageToPhotosAlbum=o => o.fail({errMsg:'system failure', errno:123});
  await assert.rejects(album.saveImage('cloud://preview'), /system failure.*123/);
  assert.equal(host.record.removed.length, 1, 'failed save also removes its temporary copy');

  host=albumHost(); global.wx=host.wx;
  host.wx.getImageInfo=o => o.fail({errMsg:'invalid image'});
  await assert.rejects(album.saveImage('cloud://bad'), /检查图片失败.*invalid image/);
  assert.equal(host.record.saved.length, 0);

  let app, need, resolved=[];
  vm.runInNewContext(fs.readFileSync(root + '/app.js', 'utf8'), {
    App:a => { app=a; }, wx:{cloud:{init(){}}, onNeedPrivacyAuthorization:fn => { need=fn; }}, console,
  });
  app.onLaunch(); need(v => resolved.push(v.event)); need(v => resolved.push(v.event));
  let shown=false; app.registerPrivacy(value => { shown=value; }); assert(shown);
  app.agreePrivacy(); assert.deepStrictEqual(resolved, ['agree','agree']); assert.equal(shown,false);

  let submitted, notices=[];
  const skin=page('skin', {showToast:o => notices.push(o.title)}, {
    submitTaskFlow:o => { submitted=o; return new Promise(() => {}); },
  });
  skin.data.picked={path:'/p', name:'p.jpg'}; skin.data.roiEnabled=true;
  skin.submit(); assert(!submitted); assert(notices[0].includes('框选'));
  skin._roiRect={left:10, top:20, width:200, height:100};
  skin.roiStart({touches:[{clientX:190, clientY:110}]});
  skin.roiMove({touches:[{clientX:30, clientY:30}]});
  skin.roiEnd({changedTouches:[{clientX:30, clientY:30}]});
  skin.submit();
  assert.deepStrictEqual(submitted.params.roi, {x0:.1,y0:.1,x1:.9,y1:.9});
  skin.onPageScroll({scrollTop:50}); assert.equal(skin._roiRect.top,-30);

  const task=page('task', {});
  task.applyTask({type:'acdc',status:'running',result:{previews:[{fileID:'cloud://early'}]}});
  assert.equal(task.data.previews.length,1);
  task.applyTask({type:'acdc',status:'done',result:{previews:[{fileID:'cloud://early'}],views:[{name:'mpr.png'}]}});
  assert.equal(task.data.previews.length,0); assert.equal(task.data.views.length,1);
  console.log('MOBILE_SAVE_ROI_PREVIEW_PASS');
}
main().catch(err => { console.error(err); process.exitCode=1; });
