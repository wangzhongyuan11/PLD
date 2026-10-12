const $ = id => document.getElementById(id);
function savedAccess() {
  try { return localStorage.getItem('pld_token') || sessionStorage.getItem('pld_token') || ''; }
  catch { return ''; }
}
let token = location.hash.slice(1) || savedAccess();
function rememberAccess() {
  try { if(token) localStorage.setItem('pld_token', token); } catch { /* Private browsers may prohibit storage. */ }
}
if (location.hash) { history.replaceState(null, '', location.pathname); }
window.addEventListener('hashchange', () => {
  if(location.hash.length > 1) {
    token = location.hash.slice(1);
    history.replaceState(null, '', location.pathname);
    refreshStatus();
  }
});
let current = null, currentView = 'outlined', status = {}, busy = false;
const labels = { original: '原始图像', outlined: '轮廓描边', cutout: '切割图', mask: '二值掩膜', contour: '独立轮廓' };
let imageUrls = [];
function message(text, error=false) { $('message').textContent=text; $('message').classList.toggle('error',error); }
async function api(path, body) {
  const response = await fetch(path, {method:body ? 'POST':'GET', headers:{Authorization:'Bearer '+token, ...(body ? {'Content-Type':'application/json'}:{})}, ...(body ? {body:JSON.stringify(body)}:{})});
  const data = await response.json(); if(!response.ok) throw new Error(data.error || '请求失败'); return data;
}
function buttons() {
  $('run').disabled=busy; $('demo').disabled=busy;
  $('showHdmi').disabled=busy || !status.hdmi_available || !current || current.state!=='done';
  $('stopHdmi').disabled=!['starting','showing'].includes(status.hdmi);
  $('download').disabled=busy || !current || current.state!=='done';
  document.querySelectorAll('[data-view]').forEach(b => { b.disabled=busy || !current?.views?.includes(b.dataset.view); });
}
async function refreshStatus() {
  try {
    status=await api('/api/status');
    rememberAccess();
    if (!busy) await refreshHistory();
    $('connection').textContent=status.mode==='board' ? 'PYNQ 板端服务' : '电脑本地服务';
    $('mode').querySelector('[value=infer]').disabled=!status.inference_available;
    $('hdmiState').textContent=status.hdmi==='showing' ? '正在显示 · '+(labels[status.hdmi_view]||status.hdmi_view) : status.hdmi==='starting' ? '正在启动 HDMI…' : status.hdmi_available ? '已就绪，等待显示' : '电脑预览模式 · 未启用 HDMI';
    if(status.fault) message(status.fault,true);
    if(status.runtime_error) message('板端计算包未就绪：'+status.runtime_error,true);
    buttons();
  } catch(e) { $('connection').textContent='服务未连接'; message(e.message,true); }
}
async function fileData(file) {
  if(!file) throw new Error('请先选择文件'); if(file.size>8*1024*1024) throw new Error('单个文件不能超过 8 MB');
  return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=()=>reject(new Error('文件读取失败'));r.readAsDataURL(file);});
}
function stage(j) { $('stage').textContent=j.stage; $('progress').value=j.progress; $('percent').textContent=j.progress+'%'; }
async function start(body) {
  if(busy) throw new Error('已有任务正在处理，请稍候');
  busy=true; buttons(); message('正在处理，请稍候…');
  try {
    current=await api('/api/jobs',body); stage(current);
    while(current.state==='running') { await new Promise(r=>setTimeout(r,450)); current=await api('/api/jobs/'+current.id); stage(current); }
    if(current.state==='error') throw new Error(current.error);
    $('jobName').textContent=current.name;
    const s=current.statistics;
    $('area').replaceChildren(document.createTextNode(s.area_percent.toFixed(2)),Object.assign(document.createElement('small'),{textContent:'%'}));
    $('areaScope').textContent=s.area_scope; $('pixels').textContent=s.foreground_pixels.toLocaleString(); $('boundary').textContent=s.boundary_pixels.toLocaleString();
    $('time').textContent=(s.cpu_postprocess_seconds*1000).toFixed(1)+' ms';
    $('timingNote').textContent='后处理不含推理与显示。'+(s.b_seconds ? (s.historical_b_time?'历史 B 耗时 ':'本次 B 耗时 ')+(s.b_seconds*1000).toFixed(0)+' ms（不含显示）。':'');
    $('sourceNote').textContent=s.data_source;
    currentView='outlined'; await render();
    message(s.data_source+'。结果已生成，可切换查看或下载。');
  } catch(e) { message(e.message,true); }
  finally { busy=false; await refreshStatus(); buttons(); }
}
async function asset(name) {
  return historyAsset(current.id, name);
}
async function historyAsset(id, name) {
  const response=await fetch('/files/'+id+'/'+name,{headers:{Authorization:'Bearer '+token}});
  if(!response.ok) throw new Error('结果文件读取失败'); return response.blob();
}
let renderVersion=0;
async function render() {
  const version=++renderVersion;
  const left=current.views.includes('original')?'original':'mask';
  const blobs=await Promise.all([asset(left+'.png'),asset(currentView+'.png')]);
  if(version!==renderVersion) return;
  imageUrls.forEach(URL.revokeObjectURL); imageUrls=blobs.map(b=>URL.createObjectURL(b));
  $('leftImage').src=imageUrls[0]; $('rightImage').src=imageUrls[1];
  $('leftImage').hidden=false;$('rightImage').hidden=false;$('leftEmpty').hidden=true;$('rightEmpty').hidden=true;
  $('leftLabel').textContent=left==='original'?'原始图像':'真实板测掩膜 · 非原图'; $('rightLabel').textContent=labels[currentView];
  $('leftImage').alt=$('leftLabel').textContent;$('rightImage').alt=labels[currentView];
  document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.view===currentView)));
}
$('demo').onclick=()=>start({mode:'demo',thickness:Number($('thickness').value)});
$('run').onclick=async()=>{
  try {const mode=$('mode').value; const body={mode,thickness:Number($('thickness').value),name:$('imageFile').files[0]?.name,image:await fileData($('imageFile').files[0])};if(mode==='mask')body.mask=await fileData($('maskFile').files[0]); await start(body);}catch(e){message(e.message,true);}
};
$('mode').onchange=()=>{const infer=$('mode').value==='infer';$('maskField').hidden=infer;$('imageFile').accept=infer?'image/jpeg':'image/jpeg,image/png';$('modeNote').textContent=infer?'上传 JPG 后调用板端模型。处理前会关闭本网页启动的 HDMI。':'使用你上传的掩膜提取轮廓，不执行模型推理。';};
$('imageFile').onchange=()=>{$('fileLabel').textContent=$('imageFile').files[0]?.name||'选择原始图片';};
$('thickness').oninput=()=>{$('widthValue').textContent=$('thickness').value+' px';};
document.querySelectorAll('[data-view]').forEach(b=>{b.onclick=async()=>{currentView=b.dataset.view;try{await render();}catch(e){message(e.message,true);}};b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;const tabs=[...document.querySelectorAll('[data-view]')].filter(t=>!t.disabled);const i=tabs.indexOf(b);tabs[(i+(e.key==='ArrowRight'?1:tabs.length-1))%tabs.length].focus();e.preventDefault();};});
$('download').onclick=async()=>{try{const blob=await asset('results.zip');const url=URL.createObjectURL(blob);const a=Object.assign(document.createElement('a'),{href:url,download:'PLD_'+current.id.slice(0,8)+'.zip'});a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){message(e.message,true);}};
$('showHdmi').onclick=async()=>{try{await api('/api/hdmi',{action:'show',id:current.id,view:currentView});await refreshStatus();message('已请求显示，请查看 HDMI 屏幕。');}catch(e){message(e.message,true);}};
$('stopHdmi').onclick=async()=>{try{await api('/api/hdmi',{action:'stop'});await refreshStatus();message('HDMI 已停止。');}catch(e){message(e.message,true);}};
const drop=$('drop');drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag');};drop.ondragleave=()=>drop.classList.remove('drag');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');if(e.dataTransfer.files.length){$('imageFile').files=e.dataTransfer.files;$('imageFile').onchange();}};
if(document.modelContext?.registerTool){try{document.modelContext.registerTool({name:'load_segmentation_example',description:'加载历史板测掩膜，运行轮廓提取并更新可见预览；不启动 FPGA 或 HDMI。',inputSchema:{type:'object',properties:{},additionalProperties:false},execute:async(input)=>{if(!input||Object.keys(input).length)throw new Error('无需参数');await start({mode:'demo',thickness:Number($('thickness').value)});if(current?.state!=='done')throw new Error('处理未完成');return {id:current.id,state:current.state,statistics:current.statistics};}});}catch(e){console.warn('WebMCP unavailable',e);}}
buttons();refreshStatus();setInterval(refreshStatus,2500);

let historyKey = '', historyUrls = [], historyLoading = false;
async function downloadHistory(id, name) {
  const blob = await historyAsset(id, name);
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), {href:url, download:'PLD_'+id.slice(0,8)+'_'+name});
  document.body.append(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function refreshHistory(force=false) {
  if(historyLoading) return;
  historyLoading=true;
  try {
    const data=await api('/api/history');
    const key=data.boot_id+':'+data.items.map(j=>j.id).join(',');
    if(!force && key===historyKey) return;
    historyKey=key;
    historyUrls.forEach(URL.revokeObjectURL); historyUrls=[];
    $('historyList').replaceChildren(); $('historyEmpty').hidden=data.items.length>0;
    for(const j of data.items) {
      const card=document.createElement('article');card.className='history-card';
      const view=j.views.includes('cutout')?'cutout':'mask';
      const picture=document.createElement('img');picture.alt=view==='cutout'?'已保存的切割图':'历史回放掩膜';picture.loading='lazy';
      const title=document.createElement('strong');title.textContent=j.name;
      const meta=document.createElement('p');meta.className='hint';
      meta.textContent=new Date(j.created*1000).toLocaleString('zh-CN')+' · '+(j.mode==='demo'?'示例回放':j.mode==='infer'?'FPGA 分割':'配套掩膜处理');
      const buttons=document.createElement('div');buttons.className='history-actions';
      for(const [label,file] of [[view==='cutout'?'下载切割图':'下载掩膜',view+'.png'],['下载结果包','results.zip']]) {
        const button=document.createElement('button');button.textContent=label;
        button.onclick=()=>downloadHistory(j.id,file).catch(e=>message(e.message,true)); buttons.append(button);
      }
      card.append(picture,title,meta,buttons);$('historyList').append(card);
      try {const blob=await historyAsset(j.id,view+'.png');const url=URL.createObjectURL(blob);historyUrls.push(url);picture.src=url;}
      catch(e){picture.alt='图片暂时不可用';}
    }
  } finally {historyLoading=false;}
}
$('refreshHistory').onclick=()=>refreshHistory(true).catch(e=>message(e.message,true));
