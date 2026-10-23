/* Browser-only MPR and WebGL surfaces; viewing never requests another model inference. */
'use strict';
const $=id=>document.getElementById(id);
const state={job:null,phase:'ED',volume:null,gray:null,labels:null,pos:[0,0,0],cap:null,meshes:[],meshSets:{raw:[],smooth:[]},meshBounds:null,presentation:null,yaw:.55,pitch:-1.05,zoom:1,drag:null,loadGeneration:0};
const colors=[[0,0,0],[255,0,0],[0,255,0],[0,0,255]];
const planes=[[0,1,2],[0,2,1],[1,2,0]];
function enabled(c){return c>0&&$('class'+c).checked;}
async function json(url,options){const r=await fetch(url,options);const v=await r.json();if(!r.ok){const error=Error(v.error||r.status);error.status=r.status;throw error;}return v;}
function showError(e){$('status').textContent=String(e.message||e);$('start').disabled=!!state.cap?.read_only;$('cancel').disabled=true;}
function asset(path,job=state.job){return '/results/'+job+'/'+path;}
async function buffer(path,job=state.job){const r=await fetch(asset(path,job));if(!r.ok)throw Error('Missing result: '+path);return r.arrayBuffer();}
function matPoint(a,p){return a.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);}
function viewportTransform(i,canvas){const [u,v]=planes[i],b=state.volume.bounds;const extent=[Math.max(b[1][u]-b[0][u],1),Math.max(b[1][v]-b[0][v],1)],scale=Math.min(canvas.width/extent[0],canvas.height/extent[1])*.94;return {u,v,scale,ox:(canvas.width-extent[0]*scale)/2,oy:(canvas.height-extent[1]*scale)/2};}
function canvasWorld(i,x,y){const canvas=$('mpr'+i),t=viewportTransform(i,canvas),b=state.volume.bounds,p=[...state.pos];p[t.u]=b[0][t.u]+(x-t.ox)/t.scale;p[t.v]=b[1][t.v]-(y-t.oy)/t.scale;return p;}
const mprCache=[null,null,null];
function renderMPR(i){
 if(!state.volume)return;
 const canvas=$('mpr'+i),ctx=canvas.getContext('2d'),t=viewportTransform(i,canvas),bounds=state.volume.bounds;
 const key=JSON.stringify([state.job,state.phase,state.pos[planes[i][2]],$('level').value,$('width').value,$('alpha').value,$('contour').checked,[1,2,3].map(enabled)]);
 if(!mprCache[i]||mprCache[i].key!==key){
  const img=ctx.createImageData(canvas.width,canvas.height),alpha=+$('alpha').value,level=+$('level').value,width=Math.max(1,+$('width').value);
  const visible=[false,...[1,2,3].map(enabled)],outline=$('contour').checked,[nx,ny,nz]=state.volume.shape,xy=nx*ny;
  const inv=state.volume.inverse_affine,origin=matPoint(inv,canvasWorld(i,.5,.5));
  const dx=inv.slice(0,3).map(r=>r[t.u]/t.scale),dy=inv.slice(0,3).map(r=>-r[t.v]/t.scale);
  const gray=state.gray,labels=state.labels,out=img.data;
  for(let y=0;y<canvas.height;y++){
   let vx=origin[0]+y*dy[0],vy=origin[1]+y*dy[1],vz=origin[2]+y*dy[2];
   for(let x=0;x<canvas.width;x++,vx+=dx[0],vy+=dx[1],vz+=dx[2]){
    let intensity=0,c=0;
    if(vx>=0&&vy>=0&&vz>=0&&vx<=nx-1&&vy<=ny-1&&vz<=nz-1){
     const x0=Math.floor(vx),y0=Math.floor(vy),z0=Math.floor(vz),x1=Math.min(x0+1,nx-1),y1=Math.min(y0+1,ny-1),z1=Math.min(z0+1,nz-1);
     const fx=vx-x0,fy=vy-y0,fz=vz-z0;
     const p00=x0+nx*y0+xy*z0,p01=x0+nx*y1+xy*z0,p10=x0+nx*y0+xy*z1,p11=x0+nx*y1+xy*z1,step=x1-x0;
     const a=(gray[p00]*(1-fx)+gray[p00+step]*fx)*(1-fy)+(gray[p01]*(1-fx)+gray[p01+step]*fx)*fy;
     const b=(gray[p10]*(1-fx)+gray[p10+step]*fx)*(1-fy)+(gray[p11]*(1-fx)+gray[p11+step]*fx)*fy;
     intensity=a*(1-fz)+b*fz;
     const xx=Math.round(vx),yy=Math.round(vy),zz=Math.round(vz),idx=xx+nx*yy+xy*zz;c=labels[idx];
     if(outline&&c&&xx>0&&xx<nx-1&&yy>0&&yy<ny-1&&zz>0&&zz<nz-1&&[idx-1,idx+1,idx-nx,idx+nx,idx-xy,idx+xy].every(j=>labels[j]===c))c=0;
    }
    const val=Math.max(0,Math.min(255,(intensity-level+width/2)/width*255)),off=4*(x+canvas.width*y);
    for(let channel=0;channel<3;channel++)out[off+channel]=visible[c]?val*(1-alpha)+colors[c][channel]*alpha:val;
    out[off+3]=255;
   }
  }
  mprCache[i]={key,img};
 }
 ctx.putImageData(mprCache[i].img,0,0);
 const px=t.ox+(state.pos[t.u]-bounds[0][t.u])*t.scale,py=t.oy+(bounds[1][t.v]-state.pos[t.v])*t.scale;
 ctx.strokeStyle='#65d8e999';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,canvas.height);ctx.moveTo(0,py);ctx.lineTo(canvas.width,py);ctx.stroke();
 $('slice'+i).textContent=state.pos[planes[i][2]].toFixed(1)+' mm';
}
let scheduled=false;
function redraw(){if(scheduled)return;scheduled=true;requestAnimationFrame(()=>{scheduled=false;const start=performance.now();for(let i=0;i<3;i++)renderMPR(i);render3D();if(state.volume){$('coordinates').textContent='RAS / '+state.pos.map(v=>v.toFixed(1)).join(', ')+' mm';for(let i=0;i<3;i++)$('pos'+i).value=state.pos[i];}$('renderfps').textContent=(performance.now()-start).toFixed(0)+' ms / 更新';});}
for(let i=0;i<3;i++){const c=$('mpr'+i);c.addEventListener('pointerdown',e=>{if(!state.volume)return;const r=c.getBoundingClientRect();state.pos=canvasWorld(i,(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height);state.pos=state.pos.map((v,a)=>Math.max(state.volume.bounds[0][a],Math.min(state.volume.bounds[1][a],v)));redraw();});c.addEventListener('wheel',e=>{if(!state.volume)return;e.preventDefault();const axis=planes[i][2],b=state.volume.bounds;state.pos[axis]=Math.max(b[0][axis],Math.min(b[1][axis],state.pos[axis]+Math.sign(e.deltaY)*2));redraw();},{passive:false});}
let gl,renderer;
function initGL(){renderer=new CardiacSurfaceView($('surface'),$('orientation'));gl=renderer.gl;if(!gl)$('meshstatus').textContent='WebGL 不可用：可继续使用 MPR 与下载网格';}
function render3D(){if(!gl||!state.volume)return;renderer.render(state.meshes.filter(m=>enabled(m.label)),{
 bounds:state.meshBounds||state.volume.bounds,yaw:state.yaw,pitch:state.pitch,zoom:state.zoom,
 clip:+$('clip').value,mode:$('surfaceMode').value,opacity:+$('surfaceOpacity').value});}
function releaseMeshes(){for(const meshes of Object.values(state.meshSets))for(const m of meshes)if(gl){gl.deleteBuffer(m.p);gl.deleteBuffer(m.n);if(m.wire)gl.deleteBuffer(m.wire);}state.meshSets={raw:[],smooth:[]};state.meshes=[];}
function chooseSurface(){
 const smooth=$('surfaceGeometry').value==='smooth'&&state.meshSets.smooth.length>0;
 state.meshes=state.meshSets[smooth?'smooth':'raw'];
 $('surfaceOpacity').disabled=$('surfaceMode').value!=='transparent';
 $('meshstatus').textContent=gl?'电脑端生成 · '+state.phase+' / RAS mm / '+(smooth?'平滑展示':'原始表面')+(state.volume?.meshes.some(m=>m.fov_truncated)?' / 含视野截断':''):'WebGL 不可用：可继续使用 MPR 与下载网格';
 redraw();
}
function uploadGL(values){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);return b;}
function clearViewer(){
 state.loadGeneration++;state.volume=state.gray=state.labels=null;state.meshBounds=null;
 releaseMeshes();state.presentation=null;
 for(let i=0;i<3;i++){$('mpr'+i).getContext('2d').clearRect(0,0,$('mpr'+i).width,$('mpr'+i).height);$('slice'+i).textContent='—';}
 if(gl){gl.depthMask(true);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);}
 $('metrics').replaceChildren();$('meshstatus').textContent='当前任务处理中';$('coordinates').textContent='坐标单位 mm / RAS';
 $('timing').textContent='等待当前任务计时';$('renderfps').textContent='—';
 for(const id of ['phaseED','phaseES','snapshot'])$(id).disabled=true;
}
async function loadPhase(phase){
 if(!state.job)return;
 const loaded=state.job,generation=++state.loadGeneration,prefix='viewer/'+phase+'/';
 const v=await json(asset(prefix+'volume.json',loaded));
 const loadMesh=(m,base)=>Promise.all([fetch(base+m.name+'.vertices.f32'),fetch(base+m.name+'.normals.f32')])
  .then(async responses=>{if(responses.some(r=>!r.ok))throw Error('三维表面资源缺失');return {meta:m,raw:await Promise.all(responses.map(r=>r.arrayBuffer()))};});
 let presentationError=null;
 const [gray,labels,meshData,presentation]=await Promise.all([
  buffer(prefix+'gray.f32',loaded),buffer(prefix+'labels.u8',loaded),
  gl?Promise.all(v.meshes.map(m=>loadMesh(m,asset('meshes/'+phase+'/',loaded)))):[],
  gl?json('/api/jobs/'+loaded+'/presentation/'+phase,{method:'POST'}).catch(e=>{presentationError=e;return null;}):null
 ]);
 const smoothData=presentation?await Promise.all(presentation.meshes.map(m=>loadMesh(m,presentation.base_url))):[];
 // A slower previous phase must never replace a later selection or mix its meshes.
 if(loaded!==state.job||generation!==state.loadGeneration)return;
 const grayValues=new Float32Array(gray),labelValues=new Uint8Array(labels),count=v.shape.reduce((a,b)=>a*b,1);
 if(grayValues.length!==count||labelValues.length!==count)throw Error('Volume data length mismatch');
 for(const item of [...meshData,...smoothData])if(item.raw.some(raw=>raw.byteLength!==item.meta.vertices*3*4))throw Error('Mesh data length mismatch');
 state.phase=phase;state.volume=v;state.gray=grayValues;state.labels=labelValues;
 state.pos=v.bounds[0].map((x,i)=>(x+v.bounds[1][i])/2);
 $('level').value=((v.window[0]+v.window[1])/2).toFixed(1);$('width').value=Math.max(1,v.window[1]-v.window[0]).toFixed(1);
 for(let i=0;i<3;i++){$('pos'+i).min=v.bounds[0][i];$('pos'+i).max=v.bounds[1][i];}
 $('clip').min=v.bounds[0][2];$('clip').max=v.bounds[1][2];$('clip').value=v.bounds[1][2];
 releaseMeshes();
 const bounds=[[Infinity,Infinity,Infinity],[-Infinity,-Infinity,-Infinity]];
 const upload=item=>{const points=new Float32Array(item.raw[0]),normals=new Float32Array(item.raw[1]);for(let i=0;i<points.length;i++){const axis=i%3;bounds[0][axis]=Math.min(bounds[0][axis],points[i]);bounds[1][axis]=Math.max(bounds[1][axis],points[i]);}return {...item.meta,points,normals,p:uploadGL(points),n:uploadGL(normals)};};
 state.meshSets={raw:meshData.map(upload),smooth:smoothData.map(upload)};state.presentation=presentation;
 state.meshBounds=Number.isFinite(bounds[0][0])?bounds:null;
 $('clip').min=Math.min(v.bounds[0][2],bounds[0][2]);$('clip').max=Math.max(v.bounds[1][2],bounds[1][2]);$('clip').value=$('clip').max;
 $('surfaceGeometry').querySelector('option[value="smooth"]').disabled=!presentation;
 if(!presentation)$('surfaceGeometry').value='raw';
 $('surfaceNote').textContent=presentationError?'平滑生成失败，已保留原始表面：'+presentationError.message:'电脑端基于分割标签生成。平滑仅用于展示；体积 / EF 使用原始标签，完整结果 ZIP 保留原始表面。';
 for(const x of ['ED','ES'])$('phase'+x).classList.toggle('selected',x===phase);
 for(const id of ['phaseED','phaseES','snapshot'])$(id).disabled=false;
 chooseSurface();
}
async function loadMetrics(){
 const loaded=state.job;
 const [stats,report,manifest]=await Promise.all(['statistics.json','performance.json','manifest.json'].map(p=>json(asset(p,loaded))));
 if(loaded!==state.job)return;
 const table=document.createElement('table'),header=table.insertRow();
 for(const text of ['结构','ED mL','ES mL','EF %']){const c=document.createElement('th');c.textContent=text;header.append(c);}
 for(const name of ['LV','RV','MYO']){const row=table.insertRow(),values=[name,stats.phases.ED.classes[name].mL.toFixed(2),stats.phases.ES.classes[name].mL.toFixed(2),name==='MYO'?'—':(stats.cavities[name].EF_percent===null?'无效':stats.cavities[name].EF_percent.toFixed(1))];for(const value of values)row.insertCell().textContent=value;}
 $('metrics').replaceChildren(table);
 $('provenance').textContent=(report.synthetic?'合成任务':'Axial-only')+' · 模型 '+manifest.model_sha256.slice(0,12)+' · 原任务后端 '+manifest.backend+' · 四类 argmax';
 $('timing').textContent=report.synthetic?'合成演示耗时不计 PL FPS':report.fresh_inference_count===0?'复用完整结果：无新增 PL 样本':(state.cap.read_only?'历史':'')+'单片 P95 '+(report.slice_p95_seconds*1000).toFixed(1)+' ms';
}
let polling=null;
async function poll(){if(!state.job)return;try{const job=state.job;const s=await json('/api/jobs/'+job);if(job!==state.job)return;$('status').textContent=s.status+' · '+(s.phase||'')+(s.phase_total?' '+s.phase_completed+'/'+s.phase_total:'');$('progress').value=s.phase_total?(s.phase_completed||0)/s.phase_total:0;if(s.status==='completed'){clearInterval(polling);$('cancel').disabled=true;$('start').disabled=!!state.cap.read_only;const a=$('download');a.href=asset('result.zip');a.classList.remove('disabled');await loadPhase(state.phase);await loadMetrics();}else if(['failed','cancelled'].includes(s.status)){clearInterval(polling);showError(Error(s.error||s.status));}}catch(e){clearInterval(polling);if(e.status===404){localStorage.removeItem('acdcJob');state.job=null;$('job').textContent='';showError(Error('历史任务不在当前结果目录，请重新选择 ED/ES 开始分析。'));}else showError(e);}}
$('start').onclick=async()=>{try{const ed=$('ed').files[0],es=$('es').files[0];if(!ed||!es)throw Error('请选择 ED 和 ES 两个 3D NIfTI');if(ed.size+es.size>state.cap.max_upload_bytes)throw Error('输入文件超出上传上限');$('start').disabled=true;$('download').classList.add('disabled');const r=await json('/api/jobs?ed='+encodeURIComponent(ed.name)+'&es='+encodeURIComponent(es.name),{method:'POST',headers:{'Content-Type':'application/octet-stream','X-ED-Bytes':String(ed.size)},body:new Blob([ed,es])});state.job=r.run_id;clearViewer();localStorage.setItem('acdcJob',state.job);history.replaceState(null,'','?job='+state.job);$('job').textContent=state.job;$('cancel').disabled=false;clearInterval(polling);polling=setInterval(poll,1000);await poll();}catch(e){showError(e);}};
$('cancel').onclick=()=>json('/api/jobs/'+state.job+'/cancel',{method:'POST'}).catch(showError);
for(const phase of ['ED','ES'])$('phase'+phase).onclick=()=>loadPhase(phase).catch(showError);
for(const id of ['level','width','alpha','contour','class1','class2','class3','clip'])$(id).addEventListener('input',redraw);
for(let i=0;i<3;i++)$('pos'+i).oninput=()=>{state.pos[i]=+$('pos'+i).value;redraw();};
$('surfaceGeometry').onchange=chooseSurface;
$('surfaceMode').onchange=chooseSurface;
$('surfaceOpacity').oninput=redraw;
$('reset').onclick=()=>{state.yaw=.55;state.pitch=-1.05;state.zoom=1;if(state.volume){state.pos=state.volume.bounds[0].map((v,i)=>(v+state.volume.bounds[1][i])/2);$('clip').value=$('clip').max;}redraw();};
const surface=$('surface');surface.onpointerdown=e=>{state.drag=[e.clientX,e.clientY];surface.setPointerCapture(e.pointerId);};surface.onpointerup=()=>state.drag=null;surface.onpointermove=e=>{if(!state.drag)return;state.yaw+=(e.clientX-state.drag[0])*.01;state.pitch+=(e.clientY-state.drag[1])*.01;state.drag=[e.clientX,e.clientY];render3D();};surface.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.max(.3,Math.min(5,state.zoom*Math.exp(-e.deltaY*.001)));render3D();},{passive:false});
$('snapshot').onclick=()=>{
 const image=document.createElement('canvas');image.width=surface.width;image.height=surface.height+54;
 const ctx=image.getContext('2d');ctx.fillStyle='#070d14';ctx.fillRect(0,0,image.width,image.height);ctx.drawImage(surface,0,0);ctx.drawImage($('orientation'),10,surface.height-90);
 ctx.fillStyle='#c5d9e5';ctx.font='13px "Microsoft YaHei",sans-serif';ctx.fillText($('meshstatus').textContent,14,surface.height+21);ctx.font='11px "Microsoft YaHei",sans-serif';ctx.fillText('展示网格；原始标签与体积 / EF 未改变',14,surface.height+41);
 image.toBlob(blob=>{if(!blob){showError(Error('三维截图生成失败'));return;}const a=document.createElement('a');a.download='ACDC-'+state.phase+'-PC-3D.png';a.href=URL.createObjectURL(blob);document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),30000);},'image/png');
};
(async()=>{try{initGL();state.cap=await json('/api/capabilities');$('backend').textContent=state.cap.backend.toUpperCase()+(state.cap.synthetic?' / SYNTHETIC':' / AXIAL');$('notice').textContent=state.cap.synthetic?'合成演示：当前后端不代表生产 ACDC 模型或真实 PL 成绩。演示模式不用于精度评价。':'Axial 单方向模型 · 原空间统计 · 标签无额外后处理 · 电脑端平滑展示';$('notice').className=state.cap.synthetic?'demo':'';$('provenance').textContent='Axial-only · 模型 '+(state.cap.model_sha256||'历史结果').slice(0,12)+' · 四类 argmax';if(state.cap.read_only){$('backend').textContent='历史结果 / 不执行推理';$('notice').textContent='历史结果浏览 · 电脑端生成表面 · 浏览器渲染 · 不连接模型或板卡';for(const id of ['ed','es','start','cancel'])$(id).disabled=true;}const job=new URLSearchParams(location.search).get('job')||localStorage.getItem('acdcJob');if(job&&/^[0-9a-f]{32}$/.test(job)){state.job=job;$('job').textContent=job;polling=setInterval(poll,1000);await poll();}}catch(e){showError(e);}})();
