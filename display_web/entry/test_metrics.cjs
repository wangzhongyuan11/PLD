const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(__dirname+'/metrics.html','utf8'),elements=new Map();
function fresh(){return {value:'',textContent:'',children:[],append(...nodes){this.children.push(...nodes);},replaceChildren(...nodes){this.children=[...nodes];},set innerHTML(value){throw Error('untrusted HTML insertion');}};}
function element(id){if(!elements.has(id))elements.set(id,fresh());return elements.get(id);}
const metric={processing_fps:0,median_ms:150,p95_ms:195,throughput_fps:2.5,gmac_s:null,power_w:null,lut:100,ff:200,bram:2.5,dsp:8,ddr_MB_s:null,dice:0,iou:0,dice_delta_pp:0,iou_delta_pp:0};
const record={model:'skin',model_sha256:'a'.repeat(64),runtime_sha256:'b'.repeat(64),dataset_sha256:'c'.repeat(64),synthetic:true,label_scope:'raw-grid',cpu:{metrics:metric,frames:2,baseline:'<img src=x onerror=alert(1)>',platform:'fixture',power_method:'unmeasured'},fpga:{metrics:metric,frames:2,platform:'fixture',power_method:'Vivado chip estimate'},accuracy_scope:'synthetic developer labels',checks:{integer_head_all_exact:true,inference_acceleration_cpu_int8_over_pl:null},sources:[]};
let response={status:'ready',records:[record]};
const context=vm.createContext({document:{getElementById:element,createElement:fresh},fetch:async()=>({ok:true,json:async()=>response})});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
(async()=>{
 await new Promise(setImmediate);
 assert.equal(element('metrics').children.length,10);
 assert.equal(element('metrics').children[0].children[1].textContent,'0.00');
 assert.equal(element('metrics').children[3].children[2].textContent,'未测量');
 assert(element('metrics').children[5].children[1].textContent.includes('不适用'));
 assert(element('metrics').children[8].children[1].textContent.includes('0.0000'));
 assert(element('notice').textContent.includes('合成测试'));
 assert(element('scope').textContent.includes('<img src=x onerror=alert(1)>'));
 response={status:'invalid_evidence',records:[]};await vm.runInContext('refresh()',context);
 assert(element('notice').textContent.includes('校验失败'));
 assert(element('metrics').children[0].children[1].textContent.includes('未测量'));
 assert.equal(element('sources').textContent,'没有报告。');
 console.log('PASS: all metric groups, zero versus missing, software resources, synthetic label, text-only rendering, invalid evidence');
})().catch(e=>{console.error(e);process.exitCode=1;});
