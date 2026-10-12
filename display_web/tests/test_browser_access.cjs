const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8').split('let current =')[0];
const saved=new Map();
const storage={getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)};
function page(hash, store=storage){
 const events={};
 const c=vm.createContext({document:{getElementById:()=>null},location:{hash,pathname:'/'},localStorage:store,sessionStorage:{getItem:()=>null},history:{replaceState(){c.location.hash='';}},window:{addEventListener:(n,f)=>events[n]=f},refreshStatus:()=>{c.refreshed=true;}});
 vm.runInContext(source,c);return {c,events};
}
const first=page('#stable-code');
assert.equal(vm.runInContext('token',first.c),'stable-code');
assert.equal(saved.size,0); // Not persisted until authentication succeeds.
vm.runInContext('rememberAccess()',first.c);
const next=page('');
assert.equal(vm.runInContext('token',next.c),'stable-code');
next.c.location.hash='#replacement-code';next.events.hashchange();
assert.equal(vm.runInContext('token',next.c),'replacement-code');
assert.equal(next.c.refreshed,true);
const blocked=page('#works-in-memory',{getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}});
vm.runInContext('rememberAccess()',blocked.c);
assert.equal(vm.runInContext('token',blocked.c),'works-in-memory');
console.log('PASS: remembered access, new page, same-page link, storage-disabled fallback');
