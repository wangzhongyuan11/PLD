// Run with node tests/test_ui.cjs. No packages, browser, or board required.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{hidden:true,disabled:true,value:'',textContent:'',style:{},classList:{toggle(){}},setAttribute(){}});return elements.get(id);}
const ready={phase:'ready',model:'skin',target:'skin',generation:1,memory:{},switch:{}};
let state=ready,calls=[];
const context=vm.createContext({document:{getElementById:element},location:{hash:'',pathname:'/'},history:{replaceState(){}},localStorage:{getItem(){return null},setItem(){}},sessionStorage:{getItem(){return null}},setInterval(){},ResizeObserver:class{observe(){}disconnect(){}},fetch:async(url,opts)=>{calls.push([url,opts]);return{ok:true,status:200,json:async()=>state}},console});
vm.runInContext(source,context);
(async()=>{
 await new Promise(setImmediate);
 assert.equal(element('workspace').src,'/skin/g/1/');assert.equal(element('skin').disabled,false);
 vm.runInContext("render({phase:'loading',model:null,target:'camus',generation:2})",context);
 assert.equal(element('workspace').src,'about:blank');assert.equal(element('workspace').hidden,true);assert.equal(element('skin').disabled,true);
 vm.runInContext("render({phase:'ready',model:'camus',target:'camus',generation:2})",context);
 assert.equal(element('workspace').src,'/camus/g/2/');assert.equal(element('workspace').hidden,false);
 state={phase:'waiting',target:'skin',generation:3};await element('skin').onclick();
 const call=calls.find(c=>c[0]==='/hub/switch');assert.equal(JSON.parse(call[1].body).model,'skin');assert.equal(element('camus').disabled,true);
 vm.runInContext("render({phase:'error',error:'release failed',generation:3})",context);
 assert.equal(element('error').textContent,'release failed');assert.equal(element('workspace').hidden,true);
 element('compareMode').checked=true;
 vm.runInContext("render({phase:'ready',model:'skin',generation:4,evaluation_busy:true})",context);
 assert.equal(element('workspace').src,'/evaluation');assert.equal(element('skin').disabled,true);
 element('compareMode').checked=false;
 vm.runInContext("render({phase:'ready',model:'skin',generation:4,evaluation_busy:false})",context);
 assert.equal(element('workspace').src,'/skin/g/4/');assert.equal(element('skin').disabled,false);
 console.log('PASS: ready iframe, loading blanking, target routing, switch payload, error lockout, evaluation selection');
})().catch(e=>{console.error(e);process.exitCode=1});
