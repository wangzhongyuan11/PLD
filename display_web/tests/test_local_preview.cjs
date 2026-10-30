/* Exercise the actual renderer: upload originals stay local and job-bound. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8');
const nodes=new Map(),requests=[];let colored;
const element=id=>{if(!nodes.has(id))nodes.set(id,{hidden:true,decode:async()=>{}});return nodes.get(id);};
const context=vm.createContext({current:null,currentView:'outlined',localOriginal:null,imageUrls:[],
 Blob,Uint8Array,atob,
 $:element,document:{querySelectorAll:()=>[],createElement:kind=>kind==='img'?{decode:async()=>{}}:{getContext:()=>({drawImage(){},getImageData:()=>({data:new Uint8ClampedArray([0,0,0,255,255,255,255,255])}),putImageData:p=>colored=p.data}),toBlob:callback=>callback('composed')}},
 createImageBitmap:async()=>({width:2,height:1,close(){}}),URL:{createObjectURL:blob=>'blob:'+blob,revokeObjectURL(){}},
 historyAsset:async(id,name)=>{requests.push([id,name]);return id+'/'+name;},message(){}});
vm.runInContext(source.slice(source.indexOf('let renderVersion='),source.indexOf("$('demo').onclick")),context);
(async()=>{
 context.current={id:'upload',views:['original','outlined'],preview_ready:true,preview_views:['original','outlined']};
 context.localOriginal={id:'upload',url:'blob:local-original'};
 await context.render();
 assert.equal(element('leftImage').src,'blob:local-original');
 assert.deepEqual(requests,[['upload','preview_outlined.png']]);
 context.current={...context.current,id:'history'};await context.render();
 assert.equal(element('leftImage').src,'blob:history/preview_original.png');
 assert(requests.some(([id,name])=>id==='history'&&name==='preview_original.png'));
 requests.length=0;context.current={...context.current,id:'composite',client_composite:true,preview_views:['contour']};
 context.localOriginal={id:'composite',url:'blob:local-original'};await context.render();
 assert.deepEqual(requests,[['composite','preview_contour.png']]);
 assert.deepEqual([...colored],[255,220,0,0,255,220,0,255]);
 assert.equal(element('rightImage').src,'blob:composed');
 requests.length=0;context.current={...context.current,id:'inline',preview_contour_base64:btoa('mask')};
 context.localOriginal={id:'inline',url:'blob:local-original'};await context.render();assert.deepEqual(requests,[]);
 console.log('PASS: local source, contour-only transfer and exact yellow overlay; job-bound history fallback');
})().catch(e=>{console.error(e);process.exitCode=1;});
