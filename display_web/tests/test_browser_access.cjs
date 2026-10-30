const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const full=fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8');
const source=full.split('let current =')[0];
const helperSource=full.slice(0,full.indexOf('function stage'));
const saved=new Map();
const storage={getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)};
function page(hash, store=storage){
 const events={};
 const c=vm.createContext({document:{getElementById:()=>null},location:{hash,pathname:'/'},localStorage:store,sessionStorage:{getItem:()=>null},history:{replaceState(){c.location.hash='';}},window:{addEventListener:(n,f)=>events[n]=f},refreshStatus:()=>{c.refreshed=true;}});
 vm.runInContext(source,c);return {c,events};
}
const first=page('#stable-code');
assert.equal(vm.runInContext('token',first.c),'stable-code');
assert.equal(saved.size,0); // Not persisted until authentication succeeds.
vm.runInContext('rememberAccess()',first.c);
const next=page('');
assert.equal(vm.runInContext('token',next.c),'stable-code');
next.c.location.hash='#replacement-code';next.events.hashchange();
assert.equal(vm.runInContext('token',next.c),'replacement-code');
assert.equal(next.c.refreshed,true);
const blocked=page('#works-in-memory',{getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}});
vm.runInContext('rememberAccess()',blocked.c);
assert.equal(vm.runInContext('token',blocked.c),'works-in-memory');


async function main(){
 const canvasFailure=()=>{throw Error('browser image processing');};
 const c=vm.createContext({document:{getElementById:()=>null,createElement:canvasFailure},
  location:{hash:'',pathname:'/'},localStorage:storage,sessionStorage:{getItem:()=>null},
  history:{replaceState(){}},window:{addEventListener(){}},createImageBitmap:canvasFailure,
  FileReader:class{readAsDataURL(file){this.result='data:'+file.type+';base64,'+file.bytes.toString('base64');this.onload();}}});
 vm.runInContext(helperSource,c);
 const png={size:8,type:'image/png',bytes:Buffer.from([137,80,78,71,13,10,26,10])};
 const result=await c.prepareUpload(png,null,'infer');assert.equal(result.image,png.bytes.toString('base64'));
 const mask={size:2,type:'image/png',bytes:Buffer.from([0,255])};
 const both=await c.prepareUpload(png,mask,'mask');assert.equal(both.mask,mask.bytes.toString('base64'));
 const html=fs.readFileSync(path.join(__dirname,'../static/index.html'),'utf8');
 assert.match(html,/板端处理/);assert(!html.includes('enhancedImage'));assert(html.includes('cutoutImage'));
 console.log('PASS: access code, exact original upload bytes, board-side image processing and result panels');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
