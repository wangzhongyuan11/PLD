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
function message(text, error=false) { $('message').textContent=text; $('message').classList.toggle('error',error); }
async function api(path, body) {
  const response = await fetch(path, {method:body ? 'POST':'GET', headers:{Authorization:'Bearer '+token, ...(body ? {'Content-Type':'application/json'}:{})}, ...(body ? {body:JSON.stringify(body)}:{})});
  const data = await response.json(); if(!response.ok) throw new Error(data.error || '请求失败'); return data;
}
function buttons() {
  $('run').disabled=busy || preparing; $('demo').disabled=busy || preparing;
  ['imageFile','maskFile','mode','thickness'].forEach(id => { $(id).disabled=busy || preparing; });
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
// Prepare an in-memory upload copy only. Never write to the selected local files.
const UPLOAD_PIXELS = 3000000, UPLOAD_BYTES = 8*1024*1024;
function uploadSize(width, height) {
  const scale = Math.min(1, Math.sqrt(UPLOAD_PIXELS/(width*height)));
  return [Math.max(1,Math.floor(width*scale)), Math.max(1,Math.floor(height*scale))];
}
async function decodedImage(file) {
  if(!file) throw new Error('请先选择文件');
  const sig=new Uint8Array(await file.slice(0,8).arrayBuffer());
  const jpeg=sig[0]===255 && sig[1]===216;
  const png=[137,80,78,71,13,10,26,10].every((n,i)=>sig[i]===n);
  if(!jpeg && !png) throw new Error('仅支持 JPG / PNG 图片');
  try { return {bitmap:await createImageBitmap(file,{imageOrientation:'from-image'}), jpeg, png}; }
  catch { throw new Error('图片无法解码或超出浏览器可用内存，请换用较小图片'); }
}
function imageCanvas(bitmap,width,height,smooth=true) {
  const canvas=document.createElement('canvas'); canvas.width=width; canvas.height=height;
  const ctx=canvas.getContext('2d',{willReadFrequently:!smooth});
  if(!ctx) throw new Error('浏览器无法创建图像画布');
  ctx.imageSmoothingEnabled=smooth; ctx.imageSmoothingQuality='high';
  ctx.drawImage(bitmap,0,0,width,height); return canvas;
}
function canvasBlob(canvas,type,quality) {
  return new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('图片转换失败')),type,quality));
}
// Canvas PNGs are RGBA; emit a grayscale PNG so the existing mask validator
// continues to receive a single-channel binary image, including after resizing.
function pngChunk(type,data) {
  const bytes=new Uint8Array(data.length+12), view=new DataView(bytes.buffer);
  view.setUint32(0,data.length); bytes.set(new TextEncoder().encode(type),4); bytes.set(data,8);
  let crc=0xffffffff;
  for(let i=4;i<bytes.length-4;i++) { crc^=bytes[i]; for(let b=0;b<8;b++)crc=(crc>>>1)^((crc&1)?0xedb88320:0); }
  view.setUint32(bytes.length-4,(crc^0xffffffff)>>>0); return bytes;
}
async function grayscaleMask(canvas) {
  const {width,height}=canvas, rgba=canvas.getContext('2d').getImageData(0,0,width,height).data;
  const raw=new Uint8Array((width+1)*height); let hasOne=false,has255=false;
  for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
    const i=(y*width+x)*4,v=rgba[i];
    if(rgba[i+1]!==v || rgba[i+2]!==v || rgba[i+3]!==255 || ![0,1,255].includes(v)) throw new Error('掩膜须为不透明的二值图（0/1 或 0/255），不能含灰度或彩色像素');
    hasOne ||= v===1; has255 ||= v===255;
    raw[y*(width+1)+x+1]=v?255:0;
  }
  if(hasOne && has255) throw new Error('掩膜不能混用 0/1 和 0/255');
  const compressed=new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
  const header=new Uint8Array(13), hv=new DataView(header.buffer);
  hv.setUint32(0,width); hv.setUint32(4,height); header[8]=8;
  return new Blob([new Uint8Array([137,80,78,71,13,10,26,10]),pngChunk('IHDR',header),pngChunk('IDAT',compressed),pngChunk('IEND',new Uint8Array())],{type:'image/png'});
}
async function prepareUpload(imageFile,maskFile,mode) {
  let source, mask;
  try {
    source=await decodedImage(imageFile);
    const {width,height}=source.bitmap;
    if(mode==='mask') {
      mask=await decodedImage(maskFile);
      if(!mask.png) throw new Error('配套掩膜须为 PNG');
      if(mask.bitmap.width!==width || mask.bitmap.height!==height) throw new Error('配套掩膜尺寸须与原图一致（按图片方向显示后的尺寸）');
    }
    let [w,h]=uploadSize(width,height), image=imageFile;
    let converted=w!==width || h!==height || imageFile.size>UPLOAD_BYTES || (mode==='infer' && !source.jpeg);
    if(converted) {
      for(let attempt=0;attempt<12;attempt++) {
        const canvas=imageCanvas(source.bitmap,w,h);
        try { image=await canvasBlob(canvas,'image/jpeg',0.92); }
        finally {canvas.width=canvas.height=1;}
        if(image.size<=UPLOAD_BYTES) break;
        w=Math.max(1,Math.floor(w*0.8)); h=Math.max(1,Math.floor(h*0.8));
      }
      if(image.size>UPLOAD_BYTES) throw new Error('自动压缩后仍超过上传大小，请选择较小图片');
    }
    let maskData;
    if(mask) {
      // Validate at full resolution first so shrinking cannot hide invalid pixels.
      const full=imageCanvas(mask.bitmap,width,height,false);
      let binary;
      try {binary=await grayscaleMask(full);} finally {full.width=full.height=1;}
      if(w!==width || h!==height) {
        const small=imageCanvas(mask.bitmap,w,h,false);
        try {binary=await grayscaleMask(small);} finally {small.width=small.height=1;}
      }
      maskData=await fileData(binary);
    }
    const resized=w!==width || h!==height;
    const note=resized ? `已等比例缩小：${width} × ${height} → ${w} × ${h}${mask?'；配套掩膜已同步缩放':''}。原文件未修改，结果按上传尺寸生成。` : `上传尺寸：${w} × ${h}${converted?'（已转换为 JPG 上传副本）':''}。原文件未修改。`;
    return {image:await fileData(image), ...(maskData?{mask:maskData}:{}), note};
  } finally {source?.bitmap.close(); mask?.bitmap.close();}
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
    const total=Number.isFinite(s.request_to_job_complete_seconds) ? s.request_to_job_complete_seconds : s.processing_seconds;
    $('time').textContent=total.toFixed(2)+' s';
    $('timingNote').textContent=`服务端总时：HTTP 请求体开始接收到含报告的 ZIP 完成，不含浏览器预处理/渲染和历史索引提交；CPU 轮廓 ${(s.cpu_postprocess_seconds*1000).toFixed(1)} ms。`+(s.b_seconds ? (s.historical_b_time?'历史 B 耗时 ':'本次 B 耗时 ')+(s.b_seconds*1000).toFixed(0)+' ms（不含显示）。':'');
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
  if(busy || preparing) return;
  preparing=true; buttons(); $('uploadNote').textContent='正在准备上传副本…';
  try {
    const mode=$('mode').value, file=$('imageFile').files[0];
    const prepared=await prepareUpload(file,$('maskFile').files[0],mode);
    $('uploadNote').textContent=prepared.note;
    const {note,...files}=prepared;
    preparing=false;
    await start({mode,thickness:Number($('thickness').value),name:file.name,...files});
  } catch(e) {$('uploadNote').textContent='上传准备未完成，原文件未修改。';message(e.message,true);}
  finally {preparing=false;buttons();}
};
$('mode').onchange=()=>{const infer=$('mode').value==='infer';$('maskField').hidden=infer;$('imageFile').accept='image/jpeg,image/png';$('modeNote').textContent=infer?'上传 JPG / PNG 后调用板端模型，PNG 自动转为 JPG 副本。处理前会关闭本网页启动的 HDMI。':'使用你上传的掩膜提取轮廓，不执行模型推理。';};
$('imageFile').onchange=()=>{$('fileLabel').textContent=$('imageFile').files[0]?.name||'选择原始图片';};
$('thickness').oninput=()=>{$('widthValue').textContent=$('thickness').value+' px';};
document.querySelectorAll('[data-view]').forEach(b=>{b.onclick=async()=>{currentView=b.dataset.view;try{await render();}catch(e){message(e.message,true);}};b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;const tabs=[...document.querySelectorAll('[data-view]')].filter(t=>!t.disabled);const i=tabs.indexOf(b);tabs[(i+(e.key==='ArrowRight'?1:tabs.length-1))%tabs.length].focus();e.preventDefault();};});
$('download').onclick=async()=>{try{const blob=await asset('results.zip');const url=URL.createObjectURL(blob);const a=Object.assign(document.createElement('a'),{href:url,download:'PLD_'+current.id.slice(0,8)+'.zip'});a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){message(e.message,true);}};
$('showHdmi').onclick=async()=>{try{await api('/api/hdmi',{action:'show',id:current.id,view:currentView});await refreshStatus();message('已请求显示，请查看 HDMI 屏幕。');}catch(e){message(e.message,true);}};
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
      card.append(picture,title,meta,buttons);$('historyList').append(card);
      try {const blob=await historyAsset(j.id,view+'.png');const url=URL.createObjectURL(blob);historyUrls.push(url);picture.src=url;}
      catch(e){picture.alt='图片暂时不可用';}
    }
  } finally {historyLoading=false;}
}
$('refreshHistory').onclick=()=>refreshHistory(true).catch(e=>message(e.message,true));
