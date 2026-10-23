/* Camera-correct transparency order across classes, not a per-object draw order. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const context=vm.createContext({Float32Array,Math});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../web_assets/surface_view.js'),'utf8')+'\nthis.Surface=CardiacSurfaceView;',context);
const Surface=context.Surface;
const triangle=(label,z)=>({label,points:new Float32Array([0,0,z,1,0,z,0,1,z])});
const near=triangle(1,3),far=triangle(3,-2),middle=triangle(2,0);
const sorted=Surface.sortedTriangles([near,far,middle],Surface.rotation(0,0));
assert.deepEqual(Array.from(sorted,t=>t.mesh.label),[3,2,1]);
const reversed=Surface.sortedTriangles([near,far,middle],Surface.rotation(Math.PI,0));
assert.deepEqual(Array.from(reversed,t=>t.mesh.label),[1,2,3]);
console.log('PASS: transparent triangles sort across classes in camera depth order');
