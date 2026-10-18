/* Exercise the actual async phase loader with out-of-order responses. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../web_assets/app.js'),'utf8');
const elements=new Map();let releaseED,uploads=0;
const context=vm.createContext({console,Float32Array,Uint8Array,Map,Promise,Error,Math,JSON,
 document:{getElementById(id){if(!elements.has(id))elements.set(id,{addEventListener(){},classList:{toggle(){}},value:0});return elements.get(id);}},
 __fakeGl:{ARRAY_BUFFER:1,STATIC_DRAW:2,createBuffer(){return ++uploads;},bindBuffer(){},bufferData(){},deleteBuffer(){}},
 fetch:async url=>{
  const phase=url.includes('/ED/')?'ED':'ES';
  if(url.endsWith('volume.json')){
   if(phase==='ED')await new Promise(resolve=>{releaseED=resolve;});
   return {ok:true,json:async()=>({phase,shape:[1,1,1],bounds:[[0,0,0],[1,1,1]],window:[0,1],meshes:[{name:'LV',label:3,vertices:3}]})};
  }
  const length=url.endsWith('gray.f32')?4:url.endsWith('labels.u8')?1:36;
  return {ok:true,arrayBuffer:async()=>new ArrayBuffer(length)};
 }});
// Load the render/loader definitions; omit the application boot and live event bindings below them.
vm.runInContext(source.slice(0,source.indexOf('async function loadMetrics'))+
 '\ngl=__fakeGl;redraw=()=>{};globalThis.subject={state,loadPhase};',context);
(async()=>{
 const {state,loadPhase}=context.subject;state.job='first';
 const old=loadPhase('ED');await loadPhase('ES');releaseED();await old;
 assert.equal(state.phase,'ES');assert.equal(state.volume.phase,'ES');assert.equal(state.meshes.length,1);
 assert.equal(uploads,2,'stale phase must not allocate or mix GPU meshes');
 const staleJob=loadPhase('ED');state.job='second';releaseED();await staleJob;
 assert.equal(state.phase,'ES');assert.equal(uploads,2);
 console.log('PASS: latest phase/job owns the complete volume and GPU mesh set');
})().catch(error=>{console.error(error);process.exitCode=1;});
