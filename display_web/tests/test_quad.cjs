const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8');
const nodes=new Map(),requests=[];
function $(id){if(!nodes.has(id))nodes.set(id,{hidden:true,async decode(){}});return nodes.get(id);}
let releaseMask;
const c=vm.createContext({$,document:{querySelectorAll:()=>[]},currentView:'outlined',
 current:{id:'a',state:'running',preview_ready:true,preview_views:['original','outlined'],views:['original','mask','outlined','cutout']},
 historyAsset:async(id,name)=>{requests.push(name);if(name.includes('mask'))await new Promise(r=>{releaseMask=r;});return name;},
 imageUrls:[],URL:{createObjectURL:v=>'blob:'+v,revokeObjectURL(){}},message:()=>{}});
vm.runInContext(source.slice(source.indexOf('let renderVersion='),source.indexOf("$('demo').onclick")),c);
(async()=>{
 await c.render();assert.equal($('maskImage').hidden,true);assert.equal($('leftImage').hidden,false);assert.equal($('rightImage').hidden,false);
 assert.deepEqual(requests,['preview_original.png','preview_outlined.png']);
 c.current.preview_views.push('mask','cutout');await c.render();
 assert.equal(requests.filter(n=>n==='preview_original.png').length,1);
 assert.equal(requests.filter(n=>n==='preview_outlined.png').length,1);
 assert.equal($('maskImage').hidden,true);assert(releaseMask);releaseMask();
 await new Promise(setImmediate);assert.equal($('maskImage').hidden,false);assert.equal($('cutoutImage').hidden,false);
 assert.equal($('maskImage').src,'blob:preview_mask.png');
 console.log('PASS: primary views display before auxiliary images, image reuse, board-generated cutout, no browser image processing');
})().catch(e=>{console.error(e);process.exitCode=1;});
