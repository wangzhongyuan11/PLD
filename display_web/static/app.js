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
let current = null, currentView = 'outlined', status = {}, busy = false, preparing = false;
const labels = { original: '原始图像', outlined: '轮廓描边', cutout: '切割图', mask: '二值掩膜', contour: '独立轮廓' };
let imageUrls = [];
let previousWarmup=null;
function message(text, error=false) { $('message').textContent=text; $('message').classList.toggle('error',error); }
async function api(path, body) {
  const response = await fetch(path, {method:body ? 'POST':'GET', headers:{Authorization:'Bearer '+token, ...(body ? {'Content-Type':'application/json'}:{})}, ...(body ? {body:JSON.stringify(body)}:{})});
  const data = await response.json(); if(!response.ok) throw new Error(data.error || '请求失败'); return data;
}
function buttons() {
  $('run').disabled=busy || preparing || status.warmup==='running'; $('demo').disabled=busy || preparing || status.warmup==='running';
  ['imageFile','maskFile','mode','thickness'].forEach(id => { $(id).disabled=busy || preparing; });
  $('showHdmi').disabled=busy || !status.hdmi_available || !current || current.state!=='done';
  $('stopHdmi').disabled=!['starting','showing'].includes(status.hdmi);
  $('download').disabled=busy || !current || current.state!=='done';
  document.querySelectorAll('[data-view]').forEach(b => { b.disabled=!(current?.preview_ready || current?.state==='done') || !(current?.preview_ready ? current.preview_views : current?.views)?.includes(b.dataset.view); });
  $('download').textContent=busy && current?.preview_ready ? '结果包生成中…' : '下载结果包';
}
async function refreshStatus() {
  if(busy) return;
  try {
    status=await api('/api/status');
    rememberAccess();
    if (!busy && !status.active) await refreshHistory();
    $('connection').textContent=status.warmup==='running' ? '开机预热中 · 请稍候' : status.mode==='board' ? 'PYNQ 板端服务' : '电脑本地服务';
    $('mode').querySelector('[value=infer]').disabled=!status.inference_available;
    $('hdmiState').textContent=status.hdmi==='showing' ? '正在显示 · 四宫格' : status.hdmi==='starting' ? '正在启动 HDMI…' : status.hdmi_available ? '已就绪，等待显示' : '电脑预览模式 · 未启用 HDMI';
    if(status.warmup==='running') message('正在提前初始化模型与 FPGA，完成后即可处理图片。');
    else if(previousWarmup==='running' && status.warmup==='ready') message('开机预热完成，可以开始处理图片。');
    previousWarmup=status.warmup;
    if(status.fault) message(status.fault,true);
    if(status.runtime_error) message('板端计算包未就绪：'+status.runtime_error,true);
    buttons();
    if(!busy && !preparing && status.active) void resumeActive(status.active);
  } catch(e) { $('connection').textContent='服务未连接'; message(e.message,true); }
}
async function fileData(file) {
  if(!file) throw new Error('请先选择文件'); if(file.size>8*1024*1024) throw new Error('单个文件不能超过 8 MB');
  return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=()=>reject(new Error('文件读取失败'));r.readAsDataURL(file);});
}
// Read the selected bytes; image decoding, resizing and processing run on the board.
async function prepareUpload(imageFile,maskFile,mode) {
  const image=await fileData(imageFile);
  const mask=mode==='mask' ? await fileData(maskFile) : null;
  return {image,...(mask?{mask}:{}),note:'原文件已准备好，结果按板端处理尺寸生成。'};
}
function stage(j) { $('stage').textContent=j.stage; $('progress').value=j.progress; $('percent').textContent=j.progress+'%'; }
function resultStatistics(job) {
  const s=job.statistics; if(!s) return;
  $('jobName').textContent=job.name;
  $('area').replaceChildren(document.createTextNode(s.area_percent.toFixed(2)),Object.assign(document.createElement('small'),{textContent:'%'}));
  $('areaScope').textContent=s.area_scope; $('pixels').textContent=s.foreground_pixels.toLocaleString(); $('boundary').textContent=s.boundary_pixels.toLocaleString();
  const complete=job.state==='done';
  if($('timeLabel')) $('timeLabel').textContent=complete?'完整结果包总耗时':'预览就绪耗时';
  const seconds=complete ? (s.request_to_job_complete_seconds ?? s.processing_seconds) : s.preview_ready_seconds;
  $('time').textContent=Number.isFinite(seconds) ? seconds.toFixed(2)+' s' : '—';
  $('timingNote').textContent=complete ? '服务端总时：请求接收到完整结果包生成。网页与下载结果均保留上传图片的尺寸。' : '当前显示服务端预览就绪耗时。原尺寸图片与结果包正在后台生成，完成后可下载和发送到 HDMI。';
  $('sourceNote').textContent=s.data_source;
}
async function watchCurrent(clientTimes, clickStarted) {
  let rendered=false,renderTask=null,renderError=null;
  const firstRender=()=>{
    rendered=true;
    const started=performance.now();
    renderTask=render().then(()=>{
      clientTimes.preview_load=(performance.now()-started)/1000;
      if(clickStarted!==null) clientTimes.click_to_preview=(performance.now()-clickStarted)/1000;
      showTimings(current,clientTimes);
    },error=>{renderError=error;});
  };
  while(true) {
    stage(current);buttons();
    if(current.preview_ready && !rendered) {
      resultStatistics(current);firstRender();
    } else if(rendered) {
      void updateAuxiliary(current).catch(error=>message(error.message,true));
    }
    if(current.state!=='running') break;
    current=await api('/api/jobs/'+current.id+'?wait='+(rendered?'done':'preview'));
  }
  if(clickStarted!==null) clientTimes.until_done=(performance.now()-clickStarted)/1000;
  if(!rendered && current.state==='done') firstRender();
  if(renderTask) await renderTask;
  if(renderError) throw renderError;
  if(rendered) void updateAuxiliary(current).catch(error=>message(error.message,true));
  showTimings(current,clientTimes);
  if(current.state==='error') throw new Error((current.preview_ready?'预览已保留，但完整结果包生成失败：':'')+current.error);
  resultStatistics(current);
  message('完整结果包已生成，可以下载或发送到 HDMI。网页与下载文件均保留上传尺寸。');
}
async function resumeActive(id) {
  if(busy) return;
  busy=true;buttons();
  try {current=await api('/api/jobs/'+id);currentView='outlined';await watchCurrent({},null);}
  catch(e){message(e.message,true);}
  finally {busy=false;buttons();}
}
async function start(body, clickStarted=performance.now()) {
  if(busy) throw new Error('已有任务正在处理，请稍候');
  busy=true;current=null;currentView='outlined';buttons();message('正在处理，请稍候…');
  const submitStarted=performance.now();
  const clientTimes={prepare_upload:(submitStarted-clickStarted)/1000};
  $('stageTimings').textContent='先生成预览，再后台生成完整结果包；两种耗时将分别显示。';
  timingReport=null;$('downloadTimings').disabled=true;
  try {
    current=await api('/api/jobs',body);
    clientTimes.submit_roundtrip=(performance.now()-submitStarted)/1000;
    await watchCurrent(clientTimes,clickStarted);
  } catch(e){message(e.message,true);}
  finally {busy=false;await refreshStatus();buttons();}
}
async function asset(name) {
  return historyAsset(current.id, name);
}
async function historyAsset(id, name) {
  const response=await fetch('/files/'+id+'/'+name,{headers:{Authorization:'Bearer '+token}});
  if(!response.ok) throw new Error('结果文件读取失败'); return response.blob();
}
let renderVersion=0,visibleJob=null,primaryDisplayedJob=null;
const viewAssets=new Map();
function viewAsset(job,name) {
  const key=job.id+'/'+name;
  if(!viewAssets.has(key)) viewAssets.set(key,historyAsset(job.id,name).then(blob=>{
    const url=URL.createObjectURL(blob);imageUrls.push(url);return url;
  }));
  return viewAssets.get(key);
}
async function updateAuxiliary(job,version=renderVersion) {
  if(primaryDisplayedJob!==job.id) return;
  const filename=name=>(job.preview_ready && job.preview_views?.includes(name)?'preview_':'')+name+'.png';
  await Promise.all([['mask','maskImage','maskEmpty'],['cutout','cutoutImage','cutoutEmpty']].map(async([name,id,empty])=>{
    if(!(job.state==='done' || job.preview_views?.includes(name)) || !job.views.includes(name)) return;
    const url=await viewAsset(job,filename(name));
    if(version!==renderVersion || visibleJob!==job.id) return;
    const image=$(id);
    if(image.src!==url){image.src=url;await image.decode();}
    if(version!==renderVersion || visibleJob!==job.id) return;
    image.hidden=false;$(empty).hidden=true;
  }));
}
async function render() {
  const version=++renderVersion,job=current;
  if(visibleJob!==job.id) {
    imageUrls.forEach(URL.revokeObjectURL);imageUrls=[];viewAssets.clear();visibleJob=job.id;primaryDisplayedJob=null;
    for(const name of ['mask','cutout']) {$(name+'Image').hidden=true;$(name+'Empty').hidden=false;}
  }
  const left=job.views.includes('original')?'original':'mask';
  const filename=name=>(job.preview_ready && job.preview_views?.includes(name)?'preview_':'')+name+'.png';
  const urls=await Promise.all([viewAsset(job,filename(left)),viewAsset(job,filename('outlined'))]);
  if(version!==renderVersion) return;
  const images=[$('leftImage'),$('rightImage')];
  await Promise.all(images.map(async(image,index)=>{if(image.src!==urls[index]){image.src=urls[index];await image.decode();}}));
  if(version!==renderVersion) return;
  $('leftImage').hidden=false;$('rightImage').hidden=false;$('leftEmpty').hidden=true;$('rightEmpty').hidden=true;
  primaryDisplayedJob=job.id;
  $('leftLabel').textContent=left==='original'?'原始图像':'真实板测掩膜 · 非原图';$('rightLabel').textContent='轮廓叠加';
  $('leftImage').alt=$('leftLabel').textContent;$('rightImage').alt='轮廓叠加';
  document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-selected',String(button.dataset.view===currentView)));
  void updateAuxiliary(job,version).catch(error=>message(error.message,true));
}
$('demo').onclick=()=>start({mode:'demo',thickness:Number($('thickness').value)});
$('run').onclick=async()=>{
  if(busy || preparing) return;
  const clickStarted=performance.now();
  preparing=true; buttons(); $('uploadNote').textContent='正在准备上传副本…';
  try {
    const mode=$('mode').value, file=$('imageFile').files[0];
    const prepared=await prepareUpload(file,$('maskFile').files[0],mode);
    $('uploadNote').textContent=prepared.note;
    const {note,...files}=prepared;
    preparing=false;
    await start({mode,thickness:Number($('thickness').value),name:file.name,...files},clickStarted);
  } catch(e) {$('uploadNote').textContent='上传准备未完成，原文件未修改。';message(e.message,true);}
  finally {preparing=false;buttons();}
};
$('mode').onchange=()=>{const infer=$('mode').value==='infer';$('maskField').hidden=infer;$('imageFile').accept='image/jpeg,image/png';$('modeNote').textContent=infer?'上传 JPG / PNG 后调用板端模型，图片处理在板端完成。处理前会关闭本网页启动的 HDMI。':'使用你上传的掩膜提取轮廓，不执行模型推理。';};
$('imageFile').onchange=()=>{$('fileLabel').textContent=$('imageFile').files[0]?.name||'选择原始图片';if(pldAutoSegment&&$('mode').value==='infer'&&!busy&&!preparing&&$('imageFile').files.length)$('run').onclick();};
$('thickness').oninput=()=>{$('widthValue').textContent=$('thickness').value+' px';};
document.querySelectorAll('[data-view]').forEach(b=>{b.onclick=async()=>{currentView=b.dataset.view;try{await render();}catch(e){message(e.message,true);}};b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;const tabs=[...document.querySelectorAll('[data-view]')].filter(t=>!t.disabled);const i=tabs.indexOf(b);tabs[(i+(e.key==='ArrowRight'?1:tabs.length-1))%tabs.length].focus();e.preventDefault();};});
$('download').onclick=async()=>{try{const blob=await asset('results.zip');const url=URL.createObjectURL(blob);const a=Object.assign(document.createElement('a'),{href:url,download:'PLD_'+current.id.slice(0,8)+'.zip'});a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){message(e.message,true);}};
$('showHdmi').onclick=async()=>{try{await api('/api/hdmi',{action:'show',id:current.id,view:'outlined'});await refreshStatus();message('已请求显示，请查看 HDMI 屏幕。');}catch(e){message(e.message,true);}};
$('stopHdmi').onclick=async()=>{try{await api('/api/hdmi',{action:'stop'});await refreshStatus();message('HDMI 已停止。');}catch(e){message(e.message,true);}};
const drop=$('drop');drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag');};drop.ondragleave=()=>drop.classList.remove('drag');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');if(!busy && !preparing && e.dataTransfer.files.length){$('imageFile').files=e.dataTransfer.files;$('imageFile').onchange();}};
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
      const timingButton=document.createElement('button'); timingButton.textContent='查看耗时';
      timingButton.onclick=()=>{showTimings(j,{});$('timingTitle').scrollIntoView({block:'start'});};
      buttons.append(timingButton);
      card.append(picture,title,meta,buttons);$('historyList').append(card);
      try {const blob=await historyAsset(j.id,(j.preview_ready && j.preview_views?.includes(view)?'preview_':'')+view+'.png');const url=URL.createObjectURL(blob);historyUrls.push(url);picture.src=url;}
      catch(e){picture.alt='图片暂时不可用';}
    }
  } finally {historyLoading=false;}
}
$('refreshHistory').onclick=()=>refreshHistory(true).catch(e=>message(e.message,true));

let timingReport=null;
const timingLabels={request_receive_parse:'请求体接收与 JSON 解析（与接收总计重叠）',upload_decode_validate:'上传数据还原、解码与校验（与接收总计重叠）',preview_remaining:'后台生成其余预览',preview_build:'生成首屏两张预览',request_to_preview_ready:'服务端请求至预览就绪',prepare_upload:'浏览器读取上传文件',submit_roundtrip:'上传提交往返（含服务端接收校验）',until_done:'点击到获知完成（含轮询）',preview_load:'首屏图片下载、解码与显示更新',click_to_preview:'点击到预览就绪总计',request_to_job_start:'服务端接收、解析、解码及任务启动',stop_hdmi:'停止 HDMI',input_write:'写入推理输入',runtime_verify:'运行包校验',inference_launch:'推理进程启动',inference_wait:'等待推理进程结束',inference_subprocess:'推理子进程总计（含初始化与退出）',persistent_worker_startup:'推理硬件会话初始化',persistent_worker_request:'常驻进程本次请求',persistent_worker_total:'常驻进程调用总计',result_load:'读取推理结果',mask_remap:'掩膜回映射',contour_process:'轮廓提取与统计',view_composition:'生成对比视图',png_batch_wall:'剩余 PNG 保存等待',zip_payload:'ZIP 写入图片',report_write:'统计报告写入',zip_report_append:'ZIP 追加报告',zip_package:'ZIP 打包总计',job_total:'后台任务总计',request_to_job_complete:'服务端请求至 ZIP 完成总计'};
function timingText(job,clientTimes) {
  const stats=job.statistics||{};
  const stages=stats.final_stage_seconds||(job.state==='error'?job.stage_seconds:null)||stats.stage_seconds||job.stage_seconds||{};
  const lines=['浏览器测量'];
  if(!Object.keys(clientTimes).length) lines.push('历史记录未保存浏览器计时；以下为服务端测量。');
  for(const [k,v] of Object.entries(clientTimes)) if(Number.isFinite(v)) lines.push((timingLabels[k]||k)+'：'+v.toFixed(3)+' 秒');
  lines.push('','服务端测量');
  for(const [k,label] of Object.entries(timingLabels)) if(Number.isFinite(stages[k])) lines.push(label+'：'+stages[k].toFixed(3)+' 秒');
  if(!Object.keys(stages).length) lines.push('这条记录没有服务端分阶段数据，请更新后重新处理。');
  for(const [key,label] of [['model_reused','本次复用模型'],['hardware_reused','本次复用 FPGA 环境']]) if(typeof stages[key]==='boolean') lines.push(label+'：'+(stages[key]?'是':'否'));
  if(stages.runtime_check_kind) lines.push('运行包校验方式：'+(stages.runtime_check_kind==='metadata_unchanged'?'文件属性未变，复用完整校验结果':'完整 SHA256 校验'));
  if(stats.output_png_backend) lines.push('PNG 编码方式：'+stats.output_png_backend);
  if(stages.inference_backend) lines.push('推理方式：'+stages.inference_backend);
  if(stages.persistent_worker_pid) lines.push('常驻进程 PID：'+stages.persistent_worker_pid);
  if(typeof stages.persistent_worker_alive_after_reply==='boolean') lines.push('常驻进程回复后存活：'+stages.persistent_worker_alive_after_reply);
  if(Number.isFinite(stats.b_seconds) && !stats.historical_b_time) lines.push('模型程序自报 B 耗时（与推理总计重叠）：'+stats.b_seconds.toFixed(3)+' 秒');
  if(stages.png_save) {lines.push('','PNG 单图耗时（并行执行，不可相加）');for(const [k,v] of Object.entries(stages.png_save)) if(Number.isFinite(v)) lines.push((labels[k]||k)+'：'+v.toFixed(3)+' 秒');}
  if(stages.preview_images) {lines.push('','预览单图明细（已包含在预览阶段中）');for(const [k,v] of Object.entries(stages.preview_images)) lines.push((labels[k]||k)+'：缩放 '+v.resize.toFixed(3)+' 秒；转换、编码和写入 '+v.encode_write.toFixed(3)+' 秒');}
  return {text:lines.join('\n'),stages};
}
function showTimings(job,clientTimes) {
  const {text,stages}=timingText(job,clientTimes);
  timingReport={version:'preview-first-v2',id:job.id,name:job.name,mode:job.mode,state:job.state,client_seconds:{...clientTimes},server_stages:stages,statistics:job.statistics||{},notes:'总计和子项重叠，不可相加。PNG 单图并行。上传往返不是纯网络时间；预览不含后续历史列表刷新及 HDMI。'};
  $('stageTimings').textContent=text;$('downloadTimings').disabled=false;
}
$('downloadTimings').onclick=()=>{
 if(!timingReport)return;
 const url=URL.createObjectURL(new Blob([JSON.stringify(timingReport,null,2)],{type:'application/json'}));
 const a=Object.assign(document.createElement('a'),{href:url,download:'PLD_timing_'+timingReport.id.slice(0,8)+'.json'});
 document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
};


// The board entry owns preferences; iframe reloads do not reset them.
let pldAutoSegment=false;
if(typeof window!=='undefined'){
 const entryOrigin='http://'+location.hostname+':8088';
 window.addEventListener('message',event=>{
  if(event.source===window.parent&&event.origin===entryOrigin&&event.data?.type==='pld-preferences')pldAutoSegment=event.data.settings?.auto_segment===true;
 });
 if(window.parent!==window)window.parent.postMessage({type:'pld-preferences-request'},entryOrigin);
}
