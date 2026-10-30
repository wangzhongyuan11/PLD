const vm=require('vm'),fs=require('fs'),assert=require('assert');
async function test(kind){
 const nodes={};let tick;function make(id=''){const n={id,style:{},dataset:{},handlers:{},children:[],files:[],disabled:false,hidden:false,tabIndex:0,textContent:'',offsetWidth:320,offsetHeight:200,classList:{contains(){return false}},append(x){nodes[x.id]=x;this.children.push(x)},setAttribute(){},addEventListener(t,f){this.handlers[t]=f},contains(x){return x===this||this.children.includes(x)},focus(){},scrollIntoView(){},getBoundingClientRect(){return {left:20,top:100,width:200,height:40,bottom:140}},querySelectorAll(){return this.children}};Object.defineProperty(n,'innerHTML',{set(){if(n.id!=='pldTourRoot')return;for(const id of ['pldTourSpot','pldTourCard','pldTourArrow','pldTourTitle','pldTourText','pldTourDiagram','pldTourChoices','pldTourSpeak','pldTourDone','pldTourBack'])nodes[id]=make(id)}});return n}
 for(const id of ['files','ed','es','file','run','start','status','progress','message','result','download','error','source','foreground','background','roi','box','showHdmi','stopHdmi','hdmi','clearMarks','snapshot'])nodes[id]=make(id);
 nodes.result.hidden=true;let release;nodes[kind==='acdc'?'start':'run'].onclick=()=>new Promise(r=>release=r);
 const remote={ready:true,inference_available:true,backend:'pl',phase:'ready'};
 const ctx={document:{head:make(),body:make(),createElement:()=>make(),getElementById:id=>nodes[id],querySelector:()=>make(),addEventListener(){},activeElement:null},window:{addEventListener(){}},ready:true,latestJob:null,state:{phase:'ready',busy:false,health:remote,phases:{},job:null,result_ready:false},image:null,acting:false,drag:null,actionError:'',api:async(path,body)=>{ctx.lastBody=body;return remote},json:async()=>remote,innerWidth:1000,innerHeight:800,setInterval(f){tick=f;return 1},clearInterval(){},setTimeout(){return 1},clearTimeout(){}};
 if(kind==='roi')for(const t of ['foreground','background','roi','box'])nodes[t].onclick=async()=>{ctx.state.tool=t};
 vm.createContext(ctx);vm.runInContext(fs.readFileSync(require('path').join(__dirname,({camus:'../unified/quad_camus.html',acdc:'../../acdc/appliance/web_assets/app.js',roi:'../unified/interactive.js'})[kind]),'utf8').split('/* PLD_MULTI_GUIDE_BEGIN */')[1].split('/* PLD_MULTI_GUIDE_END */')[0],ctx);
 const click=()=>nodes.pldTourToggle.handlers.click(),title=()=>nodes.pldTourTitle.textContent;
 await click();assert(!nodes.pldTourRoot.hidden);assert(title().includes('Step 1'));
 if(kind==='roi'){ctx.image={};await ctx.api('/interactive/images',{auto_segment:true});assert.equal(ctx.lastBody.auto_segment,false);}else for(const n of kind==='acdc'?['ed','es']:['files'])nodes[n].files=[{}];tick();assert(title().includes('Step 2'));if(kind==='roi'){nodes.roi.handlers.click();tick();assert(title().includes('ROI'));nodes.pldTourBack.onclick();assert(title().includes('Step 2'));nodes.foreground.handlers.click();tick();await nodes.pldTourDone.onclick();assert(title().includes('自由标记'));await nodes.pldTourDone.onclick();assert(title().includes('Step 3'));nodes.pldTourBack.onclick();assert(title().includes('自由标记'));await nodes.pldTourDone.onclick();assert(title().includes('Step 3'));}
 const work=nodes[kind==='acdc'?'start':'run'].onclick();tick();await click();assert(!nodes.pldTourRoot.hidden);assert(title().includes('等待'));
 if(kind==='roi'){ctx.state.job_id='new';ctx.state.result_ready=true}else if(kind==='acdc'){ctx.state.job='new';ctx.state.phases={ED:{},ES:{}}}else{ctx.latestJob='new';nodes.result.hidden=false}
 release();await work;tick();
 if(kind==='roi'){
  assert(title().includes('完成'));nodes.pldTourBack.onclick();assert(title().includes('自由标记'));nodes.pldTourBack.onclick();assert(title().includes('Step 2'));nodes.roi.handlers.click();tick();assert(title().includes('ROI'));assert(!nodes.pldTourDiagram.hidden);assert(nodes.pldTourDiagram.dataset.tool==='roi');
  nodes.background.handlers.click();tick();assert(title().includes('背景'));nodes.pldTourSpeak.onclick();assert(nodes.pldTourNotice.textContent.includes('不支持'));
  await click();assert(nodes.pldTourRoot.hidden);
 }else{assert(title().includes('下载'));nodes.pldTourDone.onclick();assert(nodes.pldTourRoot.hidden)}
 remote.active=true;if(kind==='acdc')remote.active_job='other';if(kind==='roi')remote.busy=true;await click();assert(nodes.pldTourRoot.hidden);
 console.log('PASS',kind,'idle/busy guard, step progression, completion, rejection'+(kind==='roi'?', tool illustration and speech fallback':''));
}
(async()=>{for(const kind of ['camus','acdc','roi'])await test(kind)})().catch(e=>{console.error(e);process.exitCode=1});
