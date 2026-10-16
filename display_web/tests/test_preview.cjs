const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const code=fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8');
const nodes=new Map(),tabs=['outlined','cutout','mask','contour'].map(view=>({dataset:{view}}));
const $=id=>{if(!nodes.has(id))nodes.set(id,{replaceChildren(){}});return nodes.get(id);};
const events=[],queue=[];
const context={$,document:{querySelectorAll:()=>tabs,createTextNode:t=>t,createElement:()=>({})},
 current:null,currentView:'outlined',busy:true,preparing:false,status:{hdmi_available:true},
 performance:{now:()=>1000},setTimeout:f=>f(),message:(m,e)=>events.push(['message',m,e]),
 showTimings:(j,t)=>events.push(['timings',j.state,{...t}]),
 api:async()=>{events.push(['poll']);assert(queue.length);return queue.shift();},refreshStatus:async()=>{}};
vm.createContext(context);
vm.runInContext(code.slice(code.indexOf('function buttons()'),code.indexOf('async function refreshStatus()')),context);
vm.runInContext(code.slice(code.indexOf('function stage(j)'),code.indexOf('async function asset(name)')),context);
context.render=async()=>events.push(['render',context.current.state,context.currentView]);
const stats={area_percent:25,area_scope:'test',foreground_pixels:2,boundary_pixels:1,preview_ready_seconds:1.2,request_to_job_complete_seconds:5.6};
const preview={id:'a',state:'running',preview_ready:true,preview_views:['original','outlined','cutout','mask','contour'],views:['original','outlined','cutout','mask','contour'],statistics:stats};
const done={...preview,state:'done',package_ready:true};
(async()=>{
 context.current={...preview,preview_views:['original','outlined']};context.buttons();
 assert(!tabs[0].disabled);assert(tabs.slice(1).every(t=>t.disabled));
 context.current=preview;context.buttons();
 assert(tabs.every(t=>!t.disabled));assert($('download').disabled);assert($('showHdmi').disabled);
 queue.push(done);const times={};await context.watchCurrent(times,0);
 assert.equal(events.filter(e=>e[0]==='render').length,1);
 assert(events.findIndex(e=>e[0]==='render')<events.findIndex(e=>e[0]==='poll'));
 assert.equal(events.find(e=>e[0]==='render')[1],'running');
 assert('click_to_preview' in times);assert('until_done' in times);
 context.busy=false;context.buttons();assert(!$('download').disabled);assert(!$('showHdmi').disabled);
 events.length=0;context.busy=true;context.current=preview;queue.push({...preview,state:'error',error:'disk full'});
 await assert.rejects(context.watchCurrent({},0),/完整结果包生成失败/);
 context.busy=false;context.buttons();assert($('download').disabled);assert(tabs.every(t=>!t.disabled));
 events.length=0;context.current=preview;queue.push(done);const resumed={};await context.watchCurrent(resumed,null);
 assert(!('click_to_preview' in resumed));assert(!('until_done' in resumed));
 events.length=0;context.current={...done,preview_ready:false};await context.watchCurrent({},null);
 assert.equal(events.filter(e=>e[0]==='render').length,1);
 console.log('PASS: preview renders before completion, tabs enabled, ZIP/HDMI gated, failed package retains preview, resumed timing and legacy jobs');
})().catch(e=>{console.error(e);process.exitCode=1;});
