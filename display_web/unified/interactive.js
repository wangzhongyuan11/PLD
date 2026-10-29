'use strict';
const $=id=>document.getElementById(id),canvas=$('source'),ctx=canvas.getContext('2d');
const tools=['foreground','background','roi','box'];
let state=null,image=null,drag=null,acting=false,syncing=false,resultId=null,actionError='',waitingJob=null,completedJob=null;
async function api(path,body){const r=await fetch(path,{cache:'no-store',credentials:'same-origin',...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(data.error||'服务请求失败');return data;}
function context(){return {model:'skin',generation:state.generation,image_token:state.image_token,revision:state.revision};}
function coordinate(e,clip=false){const r=canvas.getBoundingClientRect(),scale=Math.min(r.width/canvas.width,r.height/canvas.height),ox=(r.width-canvas.width*scale)/2,oy=(r.height-canvas.height*scale)/2;const x=(e.clientX-r.left-ox)/scale,y=(e.clientY-r.top-oy)/scale;return {x:Math.max(0,Math.min(canvas.width-1,Math.floor(x))),y:Math.max(0,Math.min(canvas.height-1,Math.floor(y))),inside:clip||(x>=0&&y>=0&&x<canvas.width&&y<canvas.height)};}
function rect(a,b){return {x0:Math.min(a.x,b.x),y0:Math.min(a.y,b.y),x1:Math.max(a.x,b.x)+1,y1:Math.max(a.y,b.y)+1};}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(!image||!state)return;ctx.drawImage(image,0,0);const p=state.parameters;for(const [r,color] of [[p.roi,'#00d6ff'],[p.box,'#ffd43b'],[drag?rect(drag.start,drag.end):null,'white']])if(r){ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,canvas.width/300);ctx.strokeRect(r.x0,r.y0,r.x1-r.x0,r.y1-r.y0);}for(const point of p.points){ctx.beginPath();ctx.arc(point.x,point.y,Math.max(3,canvas.width/160),0,Math.PI*2);ctx.fillStyle=point.label==='foreground'?'#00ff70':'#ff435c';ctx.fill();ctx.strokeStyle='white';ctx.stroke();}}
function buttons(){const busy=acting||!state||state.busy||state.phase!=='ready';$('file').disabled=busy;for(const tool of tools){$(tool).disabled=busy||!image;$(tool).setAttribute('aria-pressed',String(state?.tool===tool));}$('run').disabled=busy||!image;$('clearMarks').disabled=busy||!image;$('hdmi').disabled=busy||!state?.result_ready;$('stopHdmi').disabled=busy||!state?.hdmi_requested;}
function loadImage(src){return new Promise((resolve,reject)=>{const next=new Image();next.onload=()=>resolve(next);next.onerror=()=>reject(Error('图片解码失败'));next.src=src;});}
async function render(next){state=next;if(next.image_png){image=await loadImage(next.image_png);canvas.width=next.width;canvas.height=next.height;resultId=null;}if(!next.result_ready){resultId=null;$('overlay').removeAttribute('src');$('mask').removeAttribute('src');$('stats').textContent=next.busy?'正在计算…':'提示已更新，请运行分割';}else if(resultId!==next.job_id){const job=completedJob?.id===next.job_id?completedJob:await api('/interactive/jobs/'+next.job_id);if(job.state==='done'){const r=job.result;$('overlay').src=r.outputs.overlay_png;$('mask').src=r.outputs.target_mask_png;$('stats').textContent=`病灶像素：${r.statistics.target_pixels}\n${r.parameters.roi?'ROI 内':'整图'}面积占比：${(r.statistics.area_ratio*100).toFixed(2)}%\n概率缓存：${r.identity.prediction_cache_hit?'已复用，本次未运行 FPGA':'本次运行 FPGA'}\n处理耗时：${(r.elapsed_ms/1000).toFixed(3)} 秒`;resultId=next.job_id;completedJob=null;}}$('message').textContent=next.error||actionError||(next.busy?'正在处理，必要时 HDMI 会暂时关闭…':next.hdmi_requested?'HDMI 显示已启用，可在触屏上点选或拖框。':'');draw();buttons();}
async function refresh(force=false){if(syncing||acting||drag||(waitingJob&&!force))return;syncing=true;try{const next=await api('/interactive/state?image_token='+encodeURIComponent(state?.image_token||''));if(!acting&&!drag){await render(next);if(next.busy&&next.job_id&&!waitingJob)void waitActiveJob(next.job_id);}}catch(e){$('message').textContent=e.message;}finally{syncing=false;}}
async function waitActiveJob(id){waitingJob=id;try{let job;do{job=await api('/interactive/jobs/'+id+'?wait=done');}while(job.state==='running');if(job.state==='done')completedJob=job;}catch(error){actionError=error.message;}finally{waitingJob=null;await refresh(true);}}
async function act(fn){if(acting||state?.busy)return;acting=true;buttons();try{await fn();actionError='';}catch(e){actionError=e.message;$('message').textContent=e.message;}finally{acting=false;buttons();await refresh();}}
for(const tool of tools)$(tool).onclick=()=>act(async()=>{await api('/interactive/control',{...context(),tool});});
canvas.onpointerdown=e=>{if(!image||acting||state?.busy)return;const p=coordinate(e);if(!p.inside)return;drag={start:p,end:p,tool:state.tool,revision:state.revision};canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;drag.end=coordinate(e,true);draw();};
canvas.onpointercancel=()=>{drag=null;draw();};
canvas.onpointerup=e=>{if(!drag)return;const d=drag,end=coordinate(e,true);drag=null;draw();if(d.revision!==state.revision)return;const p=JSON.parse(JSON.stringify(state.parameters));if(d.tool==='roi'||d.tool==='box'){const r=rect(d.start,end);if(r.x1-r.x0<2||r.y1-r.y0<2)return;p[d.tool]=r;if(d.tool==='roi'){p.points=[];p.box=null;}}else{if(!coordinate(e).inside)return;p.points.push({x:end.x,y:end.y,label:d.tool});}act(async()=>{await api('/interactive/control',{...context(),parameters:p});});};
$('file').onchange=()=>act(async()=>{const file=$('file').files[0];if(!file)return;if(file.size>16*1024*1024)throw Error('图片须小于16MB');const image=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);});await api('/interactive/images',{model:'skin',generation:state.generation,image,auto_segment:pldAutoSegment});$('file').value='';});
$('clearMarks').onclick=()=>act(async()=>{drag=null;await api('/interactive/control',{...context(),parameters:{...state.parameters,roi:null,box:null,points:[]}});});
$('run').onclick=()=>act(async()=>{await api('/interactive/jobs',context());});
$('hdmi').onclick=()=>act(async()=>{await api('/interactive/hdmi',{...context(),action:'show'});});
$('stopHdmi').onclick=()=>act(async()=>{await api('/interactive/hdmi',{...context(),action:'stop'});});
refresh();setInterval(refresh,750);


// The board entry owns preferences; iframe reloads do not reset them.
let pldAutoSegment=false;
if(typeof window!=='undefined'){
 const entryOrigin='http://'+location.hostname+':8088';
 window.addEventListener('message',event=>{
  if(event.source===window.parent&&event.origin===entryOrigin&&event.data?.type==='pld-preferences')pldAutoSegment=event.data.settings?.auto_segment===true;
 });
 if(window.parent!==window)window.parent.postMessage({type:'pld-preferences-request'},entryOrigin);
}
