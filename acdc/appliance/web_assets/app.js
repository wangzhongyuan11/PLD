/* The board computes medical images and meshes. This client only displays them. */
'use strict';
const $=id=>document.getElementById(id),planes=['axial','coronal','sagittal'],axes=[[0,1,2],[0,2,1],[1,2,0]];
const state={job:null,phase:'ED',phases:{},pos:[0,0,0],yaw:.55,pitch:-1.05,zoom:1,generation:0,requests:[0,0,0],controllers:[],started:null,clientId:null,busy:false,health:null};
const renderer=new CardiacSurfaceView($('surface'),$('orientation')),gl=renderer.gl;
const indexed=gl&&gl.getExtension('OES_element_index_uint');
function asset(name,job=state.job){return '/api/v2/jobs/'+job+'/assets/'+name;}
async function request(url,options){const r=await fetch(url,options);if(!r.ok){let text;try{text=(await r.json()).error;}catch{text=r.statusText;}const e=Error(text||r.status);e.status=r.status;throw e;}return r;}
async function json(url,options){return (await request(url,options)).json();}
function post(url,body){return json(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
function error(e){$('status').textContent=e.message||String(e);}
function enabled(c){return $('class'+c).checked;}
function uploadBuffer(data,target){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,gl.STATIC_DRAW);return b;}
function freeMeshes(){for(const phase of Object.values(state.phases)){for(const item of phase.images||[])item.image.close();if(gl)for(const meshes of Object.values(phase.meshes||{}))for(const m of meshes)for(const key of ['p','n','indexBuffer','wire'])if(m[key])gl.deleteBuffer(m[key]);}}
function clearJobDisplay(){
 ++state.generation;for(const c of state.controllers)c?.abort();freeMeshes();state.phases={};state.job=null;
 for(let i=0;i<3;i++){const c=$('mpr'+i);c.getContext('2d').clearRect(0,0,c.width,c.height);$('slice'+i).textContent='—';}
 if(gl){gl.depthMask(true);gl.clearColor(.025,.039,.055,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);}
 $('orientation').getContext('2d').clearRect(0,0,88,80);$('meshstatus').textContent='等待本次完整结果';$('renderfps').textContent='—';
 $('metrics').textContent='等待本次体积与 EF';$('coordinates').textContent='RAS / mm';$('job').textContent='';
 $('download').classList.add('disabled');$('download').removeAttribute('href');
 $('timing').textContent='上传到完整可交互：测量中';history.replaceState(null,'',location.pathname);
}
function uploadMesh(result){if(gl&&!result.p){result.p=uploadBuffer(result.points,gl.ARRAY_BUFFER);result.n=uploadBuffer(result.normals,gl.ARRAY_BUFFER);if(result.indices)result.indexBuffer=uploadBuffer(result.indices,gl.ELEMENT_ARRAY_BUFFER);}return result;}
function surfaceBounds(meshes,fallback){
 const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
 for(const mesh of meshes)for(let i=0;i<mesh.points.length;i+=3)for(let a=0;a<3;a++){low[a]=Math.min(low[a],mesh.points[i+a]);high[a]=Math.max(high[a],mesh.points[i+a]);}
 return low[0]===Infinity?fallback:[low,high];
}
async function mesh(phase,row,kind,job){
 const meta=row[kind],base='meshes/'+phase+'/';
 const buffers=await Promise.all([meta.points,meta.normals,meta.indices].map(async name=>(await request(asset(base+name,job))).arrayBuffer()));
 let points=new Float32Array(buffers[0]),normals=new Float32Array(buffers[1]),indices=new Uint32Array(buffers[2]);
 if(points.length!==meta.vertex_count*3||normals.length!==points.length||indices.length!==meta.triangle_count*3)throw Error('三维数据长度不一致');
 for(const index of indices)if(index>=points.length/3)throw Error('三维索引越界');
 if(!indexed){const p=new Float32Array(indices.length*3),n=new Float32Array(p.length);for(let i=0;i<indices.length;i++){p.set(points.subarray(indices[i]*3,indices[i]*3+3),i*3);n.set(normals.subarray(indices[i]*3,indices[i]*3+3),i*3);}points=p;normals=n;indices=null;}
 const result={label:row.label,points,normals,indices,vertices:points.length/3};
 return result;
}
function draw3D(){
 const phase=state.phases[state.phase];if(!phase)return;
 const begin=performance.now(),kind=phase.meshes[$('surfaceGeometry').value]?$('surfaceGeometry').value:'smooth',meshes=phase.meshes[kind];
 renderer.render(meshes.filter(m=>enabled(m.label)),{bounds:phase.surface_bounds,yaw:state.yaw,pitch:state.pitch,zoom:state.zoom,clip:+$('clip').value,mode:$('surfaceMode').value,opacity:+$('surfaceOpacity').value});
 $('renderfps').textContent=(performance.now()-begin).toFixed(0)+' ms / 绘制';
 $('meshstatus').textContent=gl?'板卡 ARM 生成 / '+state.phase+' / RAS mm / '+(kind==='smooth'?'平滑展示':'原始表面')+(phase.volume.meshes.some(m=>m.fov_truncated)?' / 含视野截断':''):'WebGL 不可用；MPR 和结果下载仍可使用';
}
function drawPlane(i){
 const phase=state.phases[state.phase],item=phase?.images[i];if(!item)return;
 const canvas=$('mpr'+i),ctx=canvas.getContext('2d'),m=item.meta,b=phase.volume.bounds;
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(item.image,0,0,canvas.width,canvas.height);
 const x=m.ox+(state.pos[m.u]-b[0][m.u])*m.scale,y=m.oy+(b[1][m.v]-state.pos[m.v])*m.scale;
 ctx.strokeStyle='#65d8e999';ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();
 $('slice'+i).textContent=m.position[axes[i][2]].toFixed(1)+' mm';
}
function redraw(){for(let i=0;i<3;i++)drawPlane(i);draw3D();$('coordinates').textContent='RAS / '+state.pos.map(x=>x.toFixed(1)).join(', ')+' mm';for(let i=0;i<3;i++)$('pos'+i).value=state.pos[i];}
function selectPhase(phase){
 if(!state.phases[phase])return;state.phase=phase;const volume=state.phases[phase].volume;
 state.pos=[...volume.position];$('level').value=volume.window[0];$('width').value=volume.window[1];
 for(let i=0;i<3;i++){const e=$('pos'+i);e.min=volume.bounds[0][i];e.max=volume.bounds[1][i];e.value=state.pos[i];}
 const b=state.phases[phase].surface_bounds,halfZ=volume.affine?volume.affine[2].slice(0,3).reduce((s,x)=>s+Math.abs(x),0)*.5:0;
 $('clip').min=Math.floor(Math.min(b[0][2],volume.bounds[0][2]-halfZ)*10)/10;
 $('clip').max=Math.ceil(Math.max(b[1][2],volume.bounds[1][2]+halfZ)*10)/10;$('clip').value=$('clip').max;
 for(const p of ['ED','ES'])$('phase'+p).classList.toggle('selected',p===phase);
 redraw();
 if($('surfaceGeometry').value==='raw'&&!state.phases[phase].meshes.raw)$('surfaceGeometry').onchange();
 for(let i=0;i<3;i++){const m=state.phases[phase].images[i]?.meta;if(m&&(m.position[axes[i][2]]!==state.pos[axes[i][2]]||m.level!==+$('level').value||m.window!==+$('width').value||m.alpha!==+$('alpha').value||m.visible!==[1,2,3].reduce((n,c)=>n+(enabled(c)?1<<c:0),0)||m.outline!==$('contour').checked))updateMPR(i);}
}
async function updateMPR(i){
 const phase=state.phase,job=state.job,generation=state.generation,version=++state.requests[i];
 if(!state.phases[phase])return;
 state.controllers[i]?.abort();const controller=new AbortController();state.controllers[i]=controller;
 try{
  const response=await request('/api/v2/jobs/'+job+'/mpr/'+phase+'/'+planes[i],{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,body:JSON.stringify({position:[...state.pos],shape:[400,400],level:+$('level').value,window:Math.max(1,+$('width').value),alpha:+$('alpha').value,visible:[1,2,3].reduce((n,c)=>n+(enabled(c)?1<<c:0),0),outline:$('contour').checked})});
  const meta=JSON.parse(response.headers.get('X-MPR-Metadata')),image=await createImageBitmap(await response.blob());
  if(generation!==state.generation||phase!==state.phase||version!==state.requests[i]){image.close();return;}
  state.phases[phase].images[i]?.image.close();state.phases[phase].images[i]={image,meta};drawPlane(i);
 }catch(e){if(e.name!=='AbortError')error(e);}
}
let debounce;
function changeMPR(){redraw();clearTimeout(debounce);debounce=setTimeout(()=>{for(let i=0;i<3;i++)updateMPR(i);},100);}
function metrics(data){
 const table=document.createElement('table'),head=document.createElement('tr');
 for(const title of ['结构','ED mL','ES mL','EF %']){const c=document.createElement('th');c.textContent=title;head.append(c);}table.append(head);
 for(const name of ['LV','RV','MYO']){const row=document.createElement('tr'),ef=data.cavities[name]?.EF_percent;for(const value of [name,data.phases.ED.classes[name].mL.toFixed(2),data.phases.ES.classes[name].mL.toFixed(2),ef==null?'—':ef.toFixed(1)]){const c=document.createElement('td');c.textContent=value;row.append(c);}table.append(row);}
 $('metrics').replaceChildren(table);
}
async function loadJob(job){
 const generation=++state.generation;state.job=job;for(const c of state.controllers)c?.abort();
 const loaded={};
 await Promise.all(['ED','ES'].map(async phase=>{
  const volume=await json(asset('viewer/'+phase+'/volume.json',job));
  const meshes=await Promise.all(volume.meshes.map(row=>mesh(phase,row,'smooth',job)));
  const images=await Promise.all(planes.map(async plane=>({image:await createImageBitmap(await (await request(asset('mpr/'+phase+'/'+plane+'.png',job))).blob()),meta:volume.initial_mpr[plane]})));
  loaded[phase]={volume,surface_bounds:surfaceBounds(meshes,volume.bounds),meshes:{smooth:meshes},images};
 }));
 const statistics=await json(asset('statistics.json',job));
 if(generation!==state.generation){for(const phase of Object.values(loaded))for(const item of phase.images)item.image.close();return;}
 for(const phase of Object.values(loaded))for(const m of phase.meshes.smooth)uploadMesh(m);
 freeMeshes();state.phases=loaded;metrics(statistics);$('surfaceGeometry').value='smooth';$('alpha').value=.5;$('contour').checked=false;for(let c=1;c<=3;c++)$('class'+c).checked=true;selectPhase('ED');redraw();
 $('download').href=asset('result.zip',job);$('download').classList.remove('disabled');$('download').download='ACDC-'+job+'.zip';
 $('job').textContent=job;history.replaceState(null,'','?job='+job);
 await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));if(gl)gl.finish();
 if(state.started!==null){
  if(!gl)throw Error('结果已生成，但没有 WebGL，无法完成三维显示验收');
  const duration=(performance.now()-state.started)/1000;state.started=null;
  $('timing').textContent='上传到完整可交互：'+duration.toFixed(2)+' s';
  await post('/api/v2/jobs/'+job+'/client-timing',{client_id:state.clientId,interactive_seconds:duration,both_phases_loaded:true,webgl_ready:true});
 }
 $('status').textContent='两期全部完成：MPR、平滑三维、统计及 ZIP 已就绪';$('progress').value=1;
}
async function poll(job){
 while(state.busy&&state.job===job){const row=await json('/api/v2/jobs/'+job);$('status').textContent=row.status+(row.phase?' · '+row.phase:'')+(row.phase_total?' '+(row.phase_completed||0)+'/'+row.phase_total:'');
  if(row.phase_total)$('progress').value=(row.phase==='ES'?.5:0)+.5*(row.phase_completed||0)/row.phase_total;
  if(row.status==='assets_ready'){await loadJob(job);return;}
  if(['failed','cancelled','interrupted','upload_expired'].includes(row.status))throw Error(row.error||row.status);
  await new Promise(resolve=>setTimeout(resolve,250));
 }
}
async function refreshHealth(){
 try{const health=await json('/api/v2/health');state.health=health;
  $('backend').textContent=health.status==='ready'?'PYNQ-Z2 / PL 就绪':health.status==='busy'?'PYNQ-Z2 / 正在处理':health.status==='fault'?'板卡会话故障':'板卡初始化中';
  $('start').disabled=state.busy||!health.ready||health.backend!=='pl'||health.synthetic||!!health.active_job;
 }catch(e){$('backend').textContent='连接中';}
}
async function resume(job){state.job=job;state.busy=true;$('cancel').disabled=false;try{await poll(job);}finally{state.busy=false;$('cancel').disabled=true;await refreshHealth();}}
$('start').onclick=async()=>{
 const ed=$('ed').files[0],es=$('es').files[0];if(!ed||!es){error(Error('请选择 ED 和 ES'));return;}
 if(ed.size+es.size>(state.health?.max_upload_bytes||0)){error(Error('上传文件超过板卡限制'));return;}
 clearJobDisplay();
 state.busy=true;$('start').disabled=true;$('cancel').disabled=true;$('progress').value=0;$('status').textContent='上传到板卡';
 state.clientId=Array.from(crypto.getRandomValues(new Uint8Array(16)),x=>x.toString(16).padStart(2,'0')).join('');state.started=performance.now();
 try{const result=await json('/api/v2/jobs?ed='+encodeURIComponent(ed.name)+'&es='+encodeURIComponent(es.name),{method:'POST',headers:{'Content-Type':'application/octet-stream','X-ED-Bytes':String(ed.size)},body:new Blob([ed,es])});state.job=result.run_id;history.replaceState(null,'','?job='+state.job);$('cancel').disabled=false;$('job').textContent=state.job;await poll(state.job);}
 catch(e){state.started=null;error(e);}finally{state.busy=false;$('cancel').disabled=true;await refreshHealth();}
};
$('cancel').onclick=()=>post('/api/v2/jobs/'+state.job+'/cancel',{}).catch(error);
for(const p of ['ED','ES'])$('phase'+p).onclick=()=>selectPhase(p);
for(const id of ['level','width','alpha','contour','class1','class2','class3'])$(id).addEventListener('input',changeMPR);
for(let i=0;i<3;i++){
 $('pos'+i).addEventListener('input',()=>{state.pos[i]=+$('pos'+i).value;changeMPR();});
 $('mpr'+i).addEventListener('pointerdown',event=>{const phase=state.phases[state.phase],item=phase?.images[i];if(!item)return;const c=$('mpr'+i),r=c.getBoundingClientRect(),m=item.meta,b=phase.volume.bounds;state.pos[m.u]=b[0][m.u]+((event.clientX-r.left)*400/r.width-m.ox)/m.scale;state.pos[m.v]=b[1][m.v]-((event.clientY-r.top)*400/r.height-m.oy)/m.scale;state.pos=state.pos.map((v,a)=>Math.max(b[0][a],Math.min(b[1][a],v)));changeMPR();});
 $('mpr'+i).addEventListener('wheel',event=>{if(!state.phases[state.phase])return;event.preventDefault();const a=axes[i][2],b=state.phases[state.phase].volume.bounds;state.pos[a]=Math.max(b[0][a],Math.min(b[1][a],state.pos[a]+Math.sign(event.deltaY)*2));changeMPR();},{passive:false});
}
$('surfaceMode').onchange=()=>{$('surfaceOpacity').disabled=$('surfaceMode').value!=='transparent';draw3D();};$('surfaceOpacity').oninput=draw3D;$('clip').oninput=draw3D;
$('surfaceGeometry').onchange=async()=>{const kind=$('surfaceGeometry').value,phase=state.phase,job=state.job,generation=state.generation;try{const data=state.phases[phase];if(!data)return;if(!data.meshes[kind])data.meshes[kind]=await Promise.all(data.volume.meshes.map(row=>mesh(phase,row,kind,job)));if(job===state.job&&generation===state.generation){for(const m of data.meshes[kind])uploadMesh(m);if(phase===state.phase)draw3D();}}catch(e){error(e);}};
let drag=null;$('surface').onpointerdown=event=>{drag=[event.clientX,event.clientY];$('surface').setPointerCapture(event.pointerId);};$('surface').onpointerup=()=>drag=null;$('surface').onpointermove=event=>{if(!drag)return;state.yaw+=(event.clientX-drag[0])*.008;state.pitch+=(event.clientY-drag[1])*.008;drag=[event.clientX,event.clientY];draw3D();};$('surface').addEventListener('wheel',event=>{event.preventDefault();state.zoom=Math.max(.3,Math.min(4,state.zoom*Math.exp(-event.deltaY*.001)));draw3D();},{passive:false});
$('reset').onclick=()=>{state.yaw=.55;state.pitch=-1.05;state.zoom=1;selectPhase(state.phase);};
$('snapshot').onclick=()=>{if(!state.phases[state.phase])return;const canvas=document.createElement('canvas');canvas.width=720;canvas.height=680;const ctx=canvas.getContext('2d');ctx.fillStyle='#071018';ctx.fillRect(0,0,720,680);ctx.drawImage($('surface'),0,0);ctx.fillStyle='#a9e5ee';ctx.font='14px sans-serif';ctx.fillText('板卡 ARM 生成 / 浏览器渲染 / '+state.phase+' / '+$('surfaceGeometry').value+' / '+state.job,12,665);const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download='ACDC-'+state.phase+'-3D.png';a.click();};
refreshHealth().then(async()=>{const job=new URLSearchParams(location.search).get('job')||state.health?.active_job;if(state.health&&job&&/^[0-9a-f]{32}$/.test(job))try{await resume(job);}catch(e){error(e);}});setInterval(refreshHealth,1500);
