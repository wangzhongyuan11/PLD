const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8');
const nodes=new Map(),requests=[];
function $(id){
 if(!nodes.has(id))nodes.set(id,{hidden:true,naturalWidth:2,naturalHeight:2,
  async decode(){},getContext(){return{drawImage(){},getImageData(){return{data:new Uint8ClampedArray([30,40,50,255,200,210,220,255,60,70,80,255,90,100,110,255])};},putImageData(frame){assert.equal(frame.data.length,16);}};}});
 return nodes.get(id);
}
const c=vm.createContext({$,document:{querySelectorAll:()=>[]},currentView:'outlined',labels:{outlined:'轮廓叠加'},
 current:{id:'a',state:'running',preview_ready:true,preview_views:['original','outlined'],views:['original','mask','outlined']},
 historyAsset:async(id,name)=>{requests.push(name);return name;},imageUrls:[],URL:{createObjectURL:v=>'blob:'+v,revokeObjectURL(){}},
 requestAnimationFrame:cb=>cb(),Uint32Array,Math});
vm.runInContext(source.slice(source.indexOf('function enhancePreview('),source.indexOf("$('demo').onclick")),c);
(async()=>{
 await c.render();assert.equal($('maskImage').hidden,true);assert.equal($('enhancedImage').hidden,false);
 assert.deepEqual(requests,['preview_original.png','preview_outlined.png']);
 c.current.preview_views.push('mask');await c.render();assert.equal($('maskImage').hidden,false);
 assert.equal($('maskImage').src,'blob:preview_mask.png');
 c.current={id:'b',state:'done',views:['original','mask','outlined']};await c.render();
 assert.equal($('maskImage').src,'blob:mask.png');assert.equal($('rightLabel').textContent,'轮廓叠加');
 console.log('PASS: four-panel progressive previews, display enhancement, background mask and legacy completed result');
})().catch(e=>{console.error(e);process.exitCode=1;});
