"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const model = document.body.dataset.model;
  const api = `/api/${model}`;
  const isCamus = model === "camus";
  if(!isCamus) $("upload").accept=".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff";
  const canvas = $("input-canvas"), ctx = canvas.getContext("2d");
  const state = { image: null, source: null, name: "", tool: "foreground", points: [], roi: null, box: null,
    history: [], drag: null, revision: 0, result: null, busy: false, timer: null, loadRevision: 0 };
  document.title = isCamus ? "CAMUS · 心脏超声交互分割" : "皮肤病灶 · 交互分割";
  $("title").textContent = isCamus ? "CAMUS 心脏超声" : "皮肤病灶分割";
  $("subtitle").textContent = isCamus ? "独立接口 · 左室腔 / 心肌 / 左房" : "独立接口 · 冻结病灶分割模型";
  function message(text, error=false) { $("message").textContent=text; $("message").classList.toggle("error",error); }
  function invalidate() {
    state.revision++; state.result=null;
    $("statistics").hidden=true; $("output-image").hidden=true; $("output-placeholder").hidden=false;
    $("result-state").textContent="参数已更新";
    message("参数已更新，等待处理。");
    ["download-mask","download-labels","download-json"].forEach(x=>$(x).disabled=true);
  }
  function parameters() {
    return { variant:$("variant").value, backend:$("backend").value, target_class:Number($("target").value),
      threshold:isCamus&&!$("threshold-enabled").checked?null:Number($("threshold").value),
      roi:state.roi, box:state.box, points:state.points.map(p=>({...p})) };
  }
  function remember() { state.history.push(JSON.stringify({points:state.points,roi:state.roi,box:state.box})); }
  function controls() {
    $("run").disabled=!state.source||state.busy; $("undo").disabled=!state.history.length;
    $("clear").disabled=!state.points.length&&!state.roi&&!state.box;
    $("prompt-count").textContent=[state.points.length?`${state.points.length} 个提示点`:"",state.box?"目标框":"",state.roi?"ROI":""].filter(Boolean).join(" · ")||"无交互标记";
    $("image-size").textContent=state.image?`${canvas.width} × ${canvas.height} · 原图像素坐标`:"提示坐标始终对应原图";
  }
  function draw() {
    if(!state.image)return;
    ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(state.image,0,0);
    const unit=Math.max(1,canvas.width/400);
    if(state.roi){ const r=state.roi; ctx.fillStyle="rgba(15,26,20,.45)";ctx.beginPath();ctx.rect(0,0,canvas.width,canvas.height);ctx.rect(r.x0,r.y0,r.x1-r.x0,r.y1-r.y0);ctx.fill("evenodd"); }
    function rect(r,color,dashed){if(!r)return;ctx.strokeStyle=color;ctx.lineWidth=2*unit;ctx.setLineDash(dashed?[5*unit,3*unit]:[]);ctx.strokeRect(r.x0,r.y0,r.x1-r.x0,r.y1-r.y0);ctx.setLineDash([]);}
    rect(state.roi,"#e9ba61",true);rect(state.box,"#8c99f5",false);
    for(const p of state.points){ctx.beginPath();ctx.arc(p.x,p.y,4*unit,0,2*Math.PI);ctx.fillStyle=p.label==="foreground"?"#24cf8e":"#f17e72";ctx.fill();ctx.strokeStyle="white";ctx.lineWidth=1.5*unit;ctx.stroke();}
    if(state.drag)rect(rectangle(state.drag.start,state.drag.end),state.tool==="roi"?"#e9ba61":"#8c99f5",true);
    controls();
  }
  function rectangle(a,b){ return {x0:Math.floor(Math.min(a.x,b.x)),y0:Math.floor(Math.min(a.y,b.y)),x1:Math.ceil(Math.max(a.x,b.x)),y1:Math.ceil(Math.max(a.y,b.y))}; }
  function position(event){const r=canvas.getBoundingClientRect();return{x:Math.max(0,Math.min(canvas.width,(event.clientX-r.left)*canvas.width/r.width)),y:Math.max(0,Math.min(canvas.height,(event.clientY-r.top)*canvas.height/r.height))};}
  function add(mark){
    if(!state.source)return;
    remember();
    if(state.tool==="roi"){state.roi=mark;state.box=null;state.points=[];message("ROI 已更新，旧提示已清除。仅处理所选区域。");}
    else if(state.tool==="box"){state.box=mark;state.points=state.points.filter(p=>p.label!=="foreground"||(p.x>=mark.x0&&p.x<mark.x1&&p.y>=mark.y0&&p.y<mark.y1));}
    else state.points.push(mark);
    invalidate();draw();schedule();
  }
  canvas.addEventListener("pointerdown",e=>{if(!state.image)return;e.preventDefault();const p=position(e);if(state.tool==="foreground"||state.tool==="background"){add({x:Math.min(canvas.width-1,Math.floor(p.x)),y:Math.min(canvas.height-1,Math.floor(p.y)),label:state.tool});}else{canvas.setPointerCapture(e.pointerId);state.drag={start:p,end:p};draw();}});
  canvas.addEventListener("pointermove",e=>{if(state.drag){state.drag.end=position(e);draw();}});
  canvas.addEventListener("pointerup",e=>{if(!state.drag)return;const r=rectangle(state.drag.start,position(e));state.drag=null;if(r.x1-r.x0>=2&&r.y1-r.y0>=2)add(r);else{message("框选区域至少为2×2像素。",true);draw();}});
  canvas.addEventListener("pointercancel",()=>{state.drag=null;draw();});
  document.querySelectorAll("[data-tool]").forEach(button=>button.addEventListener("click",()=>{state.tool=button.dataset.tool;document.querySelectorAll("[data-tool]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));$("tool-help").textContent={foreground:"在目标内部单击，标记前景点。",background:"在不属于目标的位置单击，标记背景点。",box:"拖出包住整个目标的框，轮廓由算法提取。",roi:"拖出需要处理的区域，结果映射回原图。"}[state.tool];}));
  $("add-coordinate").addEventListener("click",()=>{if(!state.source)return;const r=Object.fromEntries(["x0","y0","x1","y1"].map(k=>[k,Number($(k).value)]));if(Object.values(r).some(v=>!Number.isFinite(v))){message("请输入有效坐标。",true);return;}if(state.tool==="foreground"||state.tool==="background"){add({x:r.x0,y:r.y0,label:state.tool});}else add(r);});
  $("undo").addEventListener("click",()=>{if(!state.history.length)return;Object.assign(state,JSON.parse(state.history.pop()));invalidate();draw();schedule();});
  $("clear").addEventListener("click",()=>{remember();state.points=[];state.roi=null;state.box=null;invalidate();draw();schedule();});
  async function request(path,body){const response=await fetch(api+path,{method:body?"POST":"GET",headers:body?{"Content-Type":"application/json"}:{},body:body?JSON.stringify(body):undefined});let value;try{value=await response.json();}catch{throw new Error("服务没有返回有效结果。");}if(!response.ok)throw new Error(value.error||"处理失败");return value;}
  async function setImage(data,name,loadRevision){
    const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=reject;image.src=data;});
    if(loadRevision!==state.loadRevision)return false;
    invalidate();state.source=data;state.name=name;state.points=[];state.roi=null;state.box=null;state.history=[];state.drag=null;
    state.image=image;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;canvas.hidden=false;$("input-placeholder").hidden=true;$("filename").textContent=name;
    $("threshold").value=isCamus?"0.50":"0.45";$("threshold-enabled").checked=false;thresholdUI();draw();controls();message("图像已载入，可点击前景/背景、拖框或直接开始分割。");return true;
  }
  $("upload").addEventListener("change",async event=>{const file=event.target.files[0];if(!file)return;const loadRevision=++state.loadRevision;invalidate();try{message("正在读取图像…");const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);});const normalized=await request("/image",{image:data,filename:file.name});await setImage(normalized.image,file.name,loadRevision);}catch(error){if(loadRevision===state.loadRevision)message(error.message||"图像读取失败",true);}});
  $("sample").addEventListener("click",async()=>{const loadRevision=++state.loadRevision;invalidate();try{$("sample").disabled=true;message("正在载入开发样例…");const data=await request("/sample");if(await setImage(data.image,data.name+"（开发样例）",loadRevision))await run();}catch(error){if(loadRevision===state.loadRevision)message(error.message,true);}finally{$("sample").disabled=false;}});
  function thresholdUI(){const active=!isCamus||$("threshold-enabled").checked;$("threshold").disabled=!active;$("threshold-value").textContent=active?Number($("threshold").value).toFixed(2):"默认";$("threshold-help").textContent=isCamus?(active?"仅调整所选结构的概率门槛，其他结构按模型概率重新归类。":"保持原来的多类别判断；需要时可启用所选结构的独立阈值。 "):"前景概率达到此值时判为病灶。原模型默认阈值为0.45。";}
  $("threshold-toggle-row").hidden=!isCamus;
  ["threshold","threshold-enabled"].forEach(id=>$(id).addEventListener(id==="threshold"?"input":"change",()=>{thresholdUI();invalidate();schedule();}));
  ["variant","backend","target"].forEach(id=>$(id).addEventListener("change",()=>{state.points=[];state.box=null;state.history=[];invalidate();draw();schedule();}));
  function schedule(){clearTimeout(state.timer);if(state.source)state.timer=setTimeout(run,350);}
  async function run(){
    clearTimeout(state.timer);if(!state.source)return;
    if(state.busy){state.timer=setTimeout(run,450);return;}
    const revision=state.revision;const body={image:state.source,filename:"normalized.png",parameters:parameters()};
    state.busy=true;controls();$("run").firstChild.textContent="正在分割 ";message("正在处理当前图像与提示…");
    try{const result=await request("/predict",body);if(revision!==state.revision)return;state.result=result;$("output-image").src=result.outputs.overlay_png;$("output-image").hidden=false;$("output-placeholder").hidden=true;$("result-state").textContent="当前参数结果";$("statistics").hidden=false;$("pixels").textContent=result.statistics.target_pixels.toLocaleString();$("area").textContent=(result.statistics.area_ratio*100).toFixed(2)+"%";$("area-label").textContent=state.roi?"目标占 ROI 面积":"目标面积占比";$("elapsed").textContent=result.elapsed_ms>=1000?(result.elapsed_ms/1000).toFixed(2)+" 秒":result.elapsed_ms.toFixed(0)+" 毫秒";$("method").textContent=result.refinement?"模型 + 交互修正":state.roi?"ROI 模型推理":"模型自动分割";["download-mask","download-labels","download-json"].forEach(x=>$(x).disabled=false);message(["分割完成。",...result.warnings].join("\n"));}
    catch(error){if(revision===state.revision){invalidate();$("result-state").textContent="本次未成功";message(error.message,true);}}
    finally{state.busy=false;$("run").firstChild.textContent="开始分割 ";controls();}
  }
  $("run").addEventListener("click",run);
  function download(url,name){
    let temporary=false;
    if(url.startsWith("data:")){
      const split=url.indexOf(","),mime=url.slice(5,url.indexOf(";")),raw=atob(url.slice(split+1));
      const bytes=Uint8Array.from(raw,character=>character.charCodeAt(0));
      url=URL.createObjectURL(new Blob([bytes],{type:mime}));temporary=true;
    }
    const a=document.createElement("a");a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();
    if(temporary)setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  $("download-mask").addEventListener("click",()=>{if(state.result)download(state.result.outputs.target_mask_png,model+"_mask.png");});
  $("download-labels").addEventListener("click",()=>{if(state.result)download(state.result.outputs.labels_png,model+"_labels.png");});
  $("download-json").addEventListener("click",()=>{if(!state.result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state.result,null,2)],{type:"application/json"}));download(url,model+"_result.json");setTimeout(()=>URL.revokeObjectURL(url),1000);});
  request("/capabilities").then(data=>{for(const [value,label] of Object.entries(data.variants))$("variant").add(new Option(label,value));for(const [value,label] of Object.entries(data.classes))$("target").add(new Option(label,value));thresholdUI();}).catch(error=>message("接口未连接："+error.message,true));
})();
