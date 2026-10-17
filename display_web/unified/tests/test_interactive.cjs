'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync(require('path').join(__dirname,'../interactive.js'),'utf8');
const fn=code.slice(code.indexOf('function coordinate('),code.indexOf('canvas.onpointerdown'));
const context={canvas:{width:1000,height:500,getBoundingClientRect:()=>({left:20,top:30,width:600,height:450})}};
vm.createContext(context);vm.runInContext(fn,context);
let p=context.coordinate({clientX:320,clientY:255});assert.equal(p.x,500);assert.equal(p.y,250);assert.equal(p.inside,true);
p=context.coordinate({clientX:320,clientY:40});assert.equal(p.inside,false);assert.equal(p.y,0);
context.canvas.width=400;context.canvas.height=800;
p=context.coordinate({clientX:320,clientY:255});assert.equal(p.x,200);assert.equal(p.y,400);
p=context.coordinate({clientX:25,clientY:255});assert.equal(p.inside,false);assert.equal(p.x,0);
console.log('PASS interactive original-coordinate mapping including letterbox and portrait');
vm.runInContext(code.slice(code.indexOf('function formatStats('),code.indexOf("$('run').onclick")),context);
const text=context.formatStats({statistics:{target_pixels:42,area_ratio:.25,per_class_pixels:{0:126,1:42}},
 elapsed_ms:2000,timings_ms:{prediction:1000,result_png:200},
 identity:{prediction_cache_hit:true,spec_sha256:'spec',prediction_timings_ms:{cache_lookup_copy:25}}},2200);
assert(text.includes('服务端合计：2.000 秒'));assert(text.includes('提交至结果接收：2.200 秒'));
assert(text.includes('缓存查询与复制：0.025 秒'));assert(text.includes('本次未运行 FPGA'));
assert(text.includes('请勿重复相加'));assert(!text.includes('"target_pixels"'));
console.log('PASS readable phase timings, cache identity and nested timing scope');
