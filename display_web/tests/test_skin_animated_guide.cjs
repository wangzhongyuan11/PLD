const vm=require('vm'),fs=require('fs'),assert=require('assert');
const nodes={},listeners={},windows={};let loop;
function make(id=''){const n={id,style:{},attrs:{},handlers:{},files:[],value:'infer',hidden:false,disabled:false,isConnected:true,offsetWidth:320,offsetHeight:180,textContent:'',parentElement:{append(n){nodes[n.id]=n}},append(n){nodes[n.id]=n},setAttribute(k,v){this.attrs[k]=v},addEventListener(k,v){this.handlers[k]=v},contains(x){return x===this||(this.id==='drop'&&x===nodes.imageFile)},focus(){ctx.document.activeElement=this},scrollIntoView(){},getBoundingClientRect(){return {left:30,top:100,width:240,height:50,bottom:150}}};Object.defineProperty(n,'innerHTML',{set(){for(const id of ['pldTourSpot','pldTourCard','pldTourArrow','pldTourTitle','pldTourText','pldTourDone'])nodes[id]=make(id)}});return n}
for(const id of ['run','imageFile','download','showHdmi','stopHdmi','demo','drop','mode','progress'])nodes[id]=make(id);
let resolveDownload;nodes.download.onclick=()=>new Promise(r=>resolveDownload=r);nodes.run.onclick=async()=>{throw Error('original error')};
let backend={inference_available:true},requests=0;
const ctx={document:{head:{append(){}},body:{append(n){nodes[n.id]=n}},createElement:()=>make(),getElementById:id=>nodes[id],querySelector:()=>({append(n){nodes[n.id]=n}}),addEventListener(t,f){listeners[t]=f},activeElement:nodes.run},window:{addEventListener(t,f){windows[t]=f}},innerWidth:1000,innerHeight:800,busy:false,preparing:false,status:{},current:null,api:async()=>{requests++;return backend},setInterval(fn){loop=fn;return 1},clearInterval(){},setTimeout(){return 1},clearTimeout(){}};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(require('path').join(__dirname,'../static/app.js'),'utf8').split('/* PLD_ANIMATED_GUIDE_BEGIN */')[1].split('/* PLD_ANIMATED_GUIDE_END */')[0],ctx);
const click=()=>nodes.pldTourToggle.handlers.click(),title=()=>nodes.pldTourTitle.textContent;
(async()=>{
 ctx.busy=true;await click();assert(nodes.pldTourRoot.hidden);assert.equal(requests,0);ctx.busy=false;
 backend={inference_available:true,active:'other'};await click();assert(nodes.pldTourRoot.hidden);
 backend={inference_available:true};await click();assert(!nodes.pldTourRoot.hidden);assert(title().includes('Step 1'));
 let blocked=false;listeners.click({target:nodes.demo,preventDefault(){blocked=true},stopImmediatePropagation(){}});assert(blocked);
 nodes.imageFile.files=[{}];loop();assert(title().includes('Step 2'));
 ctx.busy=true;ctx.current={id:'new',state:'running'};loop();assert(title().includes('Step 3'));await click();assert(!nodes.pldTourRoot.hidden);
 ctx.current.preview_ready=true;loop();assert(nodes.pldTourText.textContent.includes('仍在生成'));
 ctx.current.state='done';ctx.busy=false;loop();assert(title().includes('Step 4'));
 const d=nodes.download.onclick();await click();assert(!nodes.pldTourRoot.hidden);resolveDownload();await d;
 nodes.pldTourDone.handlers.click();assert(nodes.pldTourRoot.hidden);
 await click();assert(title().includes('Step 2')); // old completed result is not this guide's completion
 await assert.rejects(nodes.run.onclick(),/original error/);loop();assert(title().includes('Step 2'));
 ctx.current={id:'error-new',state:'error'};loop();assert(nodes.pldTourRoot.hidden);
 nodes.mode.value='mask';await click();assert(title().includes('准备'));await click();assert(nodes.pldTourRoot.hidden);
 console.log('PASS: busy rejection, remote active rejection, step transitions, unrelated click blocked, preview vs package, download exclusion, final dismissal, old-result isolation, original error propagation, failed task recovery, mode preparation.');
})().catch(e=>{console.error(e);process.exitCode=1});
