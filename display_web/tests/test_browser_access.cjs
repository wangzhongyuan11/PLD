const fs=require('node:fs');
const vm=require('node:vm');
const zlib=require('node:zlib');
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

function helperContext(maskRgba){
 const canvas={width:2,height:2,getContext(){return {getImageData(){return {data:maskRgba};}};}};
 const events={};
 const c=vm.createContext({
   document:{getElementById:()=>null,createElement(tag){if(tag==='canvas')return canvas;throw new Error('unexpected element '+tag);}},
   location:{hash:'',pathname:'/'},localStorage:storage,sessionStorage:{getItem:()=>null},
   history:{replaceState(){}},window:{addEventListener:(name,fn)=>{events[name]=fn;}},refreshStatus(){},
   TextEncoder,CompressionStream,Blob,Response
 });
 vm.runInContext(helperSource,c);c.canvas=canvas;return c;
}
const helper=helperContext(new Uint8Array([0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,255]));
assert.deepEqual(Array.from(helper.uploadSize(4000,3000)),[2000,1500]);
assert.deepEqual(Array.from(helper.uploadSize(1022,767)),[1022,767]);

async function checkGrayscalePng(){
 const blob=await vm.runInContext('grayscaleMask(canvas)',helper);
 const png=Buffer.from(await blob.arrayBuffer());
 assert.deepEqual(png.subarray(0,8),Buffer.from([137,80,78,71,13,10,26,10]));
 let offset=8,idat=[];
 while(offset<png.length){
   const length=png.readUInt32BE(offset),type=png.subarray(offset+4,offset+8).toString('ascii');
   const data=png.subarray(offset+8,offset+8+length),crc=png.readUInt32BE(offset+8+length);
   assert.equal(crc,zlib.crc32(png.subarray(offset+4,offset+8+length)));
   if(type==='IDAT')idat.push(data);
   if(type==='IHDR'){
     assert.equal(data.readUInt32BE(0),2);assert.equal(data.readUInt32BE(4),2);
     assert.equal(data[8],8);assert.equal(data[9],0);assert.equal(data[10],0);assert.equal(data[11],0);assert.equal(data[12],0);
   }
   offset+=12+length;
   if(type==='IEND')break;
 }
 assert.deepEqual(zlib.inflateSync(Buffer.concat(idat)),Buffer.from([0,0,255,0,255,0]));
}
async function main(){
 await checkGrayscalePng();
 const invalid=helperContext(new Uint8Array(16).fill(0));
 await assert.rejects(()=>vm.runInContext('grayscaleMask(canvas)',invalid),/灰度/);
 const mixed=helperContext(new Uint8Array([0,0,0,255,1,1,1,255,255,255,255,255,0,0,0,255]));
 await assert.rejects(()=>vm.runInContext('grayscaleMask(canvas)',mixed),/混用/);
 const html=fs.readFileSync(path.join(__dirname,'../static/index.html'),'utf8');
 assert.match(html,/id="uploadNote"/);
 assert.match(html,/大图上传前自动缩小/);
assert.match(html,/id="timeLabel"[^<]*服务端处理耗时/);
assert.match(full,/request_to_job_complete_seconds/);
 console.log('PASS: access code, upload sizing, grayscale PNG, and page hint');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
