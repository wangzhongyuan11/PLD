/* Exercise the actual async phase loader with out-of-order responses. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web_assets/app.js'),'utf8');
const elements=new Map();let releaseED,uploads=0;
const context=vm.createContext({console,Float32Array,Uint8Array,Map,Promise,Error,Math,JSON,
 document:{getElementById(id){if(!elements.has(id))elements.set(id,{addEventListener(){},classList:{toggle(){}},querySelector(){return {};},value:id==='surfaceGeometry'?'smooth':0});return elements.get(id);}},
 __fakeGl:{ARRAY_BUFFER:1,STATIC_DRAW:2,createBuffer(){return ++uploads;},bindBuffer(){},bufferData(){},deleteBuffer(){}},
 fetch:async url=>{
  const phase=/\/ED(?:\/|$)/.test(url)?'ED':'ES';
  if(url.includes('/presentation/'))return {ok:true,json:async()=>({base_url:'/presentations/test/'+phase+'/',meshes:[{name:'LV',label:3,vertices:3,smoothed:true}]})};
  if(url.endsWith('volume.json')){
   if(phase==='ED')await new Promise(resolve=>{releaseED=resolve;});
   return {ok:true,json:async()=>({phase,shape:[1,1,1],bounds:[[0,0,0],[1,1,1]],window:[0,1],meshes:[{name:'LV',label:3,vertices:3}]})};
  }
  const length=url.endsWith('gray.f32')?4:url.endsWith('labels.u8')?1:36;
  return {ok:true,arrayBuffer:async()=>new ArrayBuffer(length)};
 }});
// Load the render/loader definitions; omit the application boot and live event bindings below them.
vm.runInContext(source.slice(0,source.indexOf('async function loadMetrics'))+
 '\ngl=__fakeGl;redraw=()=>{};globalThis.subject={state,loadPhase,chooseSurface};',context);
(async()=>{
 const {state,loadPhase,chooseSurface}=context.subject;state.job='first';
 const old=loadPhase('ED');await loadPhase('ES');releaseED();await old;
 assert.equal(state.phase,'ES');assert.equal(state.volume.phase,'ES');assert.equal(state.meshes.length,1);
 assert.equal(state.meshes[0].smoothed,true);
 assert.equal(uploads,4,'stale phase must not allocate or mix either geometry variant');
 elements.get('surfaceGeometry').value='raw';chooseSurface();
 assert.notEqual(state.meshes[0].smoothed,true);
 elements.get('surfaceGeometry').value='smooth';elements.get('surfaceMode').value='transparent';chooseSurface();
 assert.equal(state.meshes[0].smoothed,true);assert.equal(elements.get('surfaceOpacity').disabled,false);
 assert.equal(uploads,4,'display controls must reuse meshes without new inference or uploads');
 const staleJob=loadPhase('ED');state.job='second';releaseED();await staleJob;
 assert.equal(state.phase,'ES');assert.equal(uploads,4);
 console.log('PASS: latest phase/job owns the complete volume and GPU mesh set');
})().catch(error=>{console.error(error);process.exitCode=1;});
