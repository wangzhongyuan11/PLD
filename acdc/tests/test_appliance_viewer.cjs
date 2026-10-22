/* Actual V2 loader: indexed assets, phase preload and stale-job ownership. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const elements=new Map();let buffers=0,closed=0,holdA=false,releaseA;
const gl={ARRAY_BUFFER:1,ELEMENT_ARRAY_BUFFER:2,STATIC_DRAW:3,getExtension(){return {};},createBuffer(){return ++buffers;},bindBuffer(){},bufferData(){},deleteBuffer(){},finish(){}};
function element(id){if(!elements.has(id))elements.set(id,{value:({surfaceGeometry:'smooth',surfaceMode:'solid',surfaceOpacity:.45,alpha:.5,level:100,width:200,clip:1})[id]??0,checked:!['contour'].includes(id),hidden:false,files:[],classList:{toggle(){},remove(){},add(){}},addEventListener(){},append(){},replaceChildren(){},getContext(){return {clearRect(){},drawImage(){},beginPath(){},moveTo(){},lineTo(){},stroke(){}};}});return elements.get(id);}
const kinds=Object.fromEntries(['raw','smooth'].map(kind=>[kind,{points:'RV.'+kind+'.points.f32',normals:'RV.'+kind+'.normals.f32',indices:'RV.'+kind+'.indices.u32',vertex_count:3,triangle_count:1}]));
const volume=()=>({bounds:[[0,0,0],[1,1,1]],position:[.5,.5,.5],window:[100,200],meshes:[{name:'RV',label:1,...kinds}],initial_mpr:Object.fromEntries(['axial','coronal','sagittal'].map((p,i)=>[p,{position:[.5,.5,.5],u:i===2?1:0,v:i===0?1:2,scale:300,ox:50,oy:50,level:100,window:200,alpha:.5,visible:14,outline:false}]))});
const statistics={phases:{ED:{classes:{}},ES:{classes:{}}},cavities:{}};for(const name of ['LV','RV','MYO'])for(const phase of ['ED','ES'])statistics.phases[phase].classes[name]={mL:10};
const context=vm.createContext({console,Float32Array,Uint32Array,Uint8Array,Math,JSON,Map,Promise,Error,URLSearchParams,AbortController,
 document:{getElementById:element,createElement:()=>element('created'+Math.random())},
 CardiacSurfaceView:class{constructor(){this.gl=gl;}render(){}},
 history:{replaceState(){}},location:{search:''},performance:{now:()=>1000},crypto:{},
 requestAnimationFrame:fn=>fn(),setTimeout,clearTimeout,setInterval(){},
 createImageBitmap:async()=>({close(){closed++;}}),
 fetch:async url=>{
  if(url==='/api/v2/health')return {ok:true,json:async()=>({ready:true,status:'ready',backend:'pl',synthetic:false,max_upload_bytes:1e8})};
  if(holdA&&url.includes('/'+'a'.repeat(32)+'/')&&url.includes('/ED/volume.json'))await new Promise(resolve=>releaseA=resolve);
  if(url.endsWith('volume.json'))return {ok:true,json:async()=>volume()};
  if(url.endsWith('statistics.json'))return {ok:true,json:async()=>statistics};
  if(url.endsWith('.png'))return {ok:true,blob:async()=>({})};
  return {ok:true,arrayBuffer:async()=>url.endsWith('.u32')?new Uint32Array([0,1,2]).buffer:new Float32Array([0,0,0,1,0,0,0,1,0]).buffer};
 }});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../appliance/web_assets/app.js'),'utf8')+'\nthis.api={loadJob,state,selectPhase};',context);
(async()=>{
 await context.api.loadJob('c'.repeat(32));
 assert.equal(buffers,6);assert.equal(context.api.state.phases.ES.meshes.smooth[0].indices.length,3);
 context.api.selectPhase('ES');assert.equal(context.api.state.phase,'ES');assert.equal(buffers,6);
 holdA=true;const a=context.api.loadJob('a'.repeat(32));await new Promise(resolve=>setImmediate(resolve));
 const b=context.api.loadJob('b'.repeat(32));await b;const before=buffers;releaseA();await a;
 assert.equal(context.api.state.job,'b'.repeat(32));assert.equal(buffers,before);assert.ok(closed>=6);
 console.log('PASS: V2 indexed preload, phase reuse and stale-job GPU ownership');
})().catch(e=>{console.error(e);process.exitCode=1;});
