/* Browser rendering of board-generated indexed meshes. No medical volume computation. */
'use strict';
class CardiacSurfaceView {
 constructor(canvas,orientation){
  this.canvas=canvas;this.orientation=orientation;
  this.gl=canvas.getContext('webgl',{alpha:false,antialias:true,preserveDrawingBuffer:true});
  if(!this.gl)return;
  const gl=this.gl;
  const compile=(kind,source)=>{const shader=gl.createShader(kind);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));return shader;};
  this.program=gl.createProgram();
  gl.attachShader(this.program,compile(gl.VERTEX_SHADER,`
   attribute vec3 p;attribute vec3 n;attribute vec4 tint;
   uniform mat3 r;uniform vec3 center;uniform vec2 scale;
   varying vec3 normal;varying float worldZ;varying vec4 material;
   void main(){vec3 v=r*(p-center);gl_Position=vec4(v.xy*scale,-v.z*min(scale.x,scale.y)*.32,1.);
    normal=r*n;worldZ=p.z;material=tint;}`));
  gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,`
   precision mediump float;uniform float clipZ;uniform float wire;
   varying vec3 normal;varying float worldZ;varying vec4 material;
   void main(){if(worldZ>clipZ)discard;
    vec3 n=normalize(normal);if(!gl_FrontFacing)n=-n;
    vec3 key=normalize(vec3(-.45,.65,1.));vec3 fill=normalize(vec3(.8,-.3,.6));
    float diffuse=.24+.59*max(dot(n,key),0.)+.17*max(dot(n,fill),0.);
    float spec=.22*pow(max(dot(n,normalize(key+vec3(0.,0.,1.))),0.),36.);
    vec3 shaded=material.rgb*diffuse+vec3(spec);
    gl_FragColor=vec4(mix(shaded,material.rgb,wire),material.a);}`));
  gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));
  this.loc={};for(const id of ['r','center','scale','clipZ','wire'])this.loc[id]=gl.getUniformLocation(this.program,id);
  for(const id of ['p','n','tint'])this.loc[id]=gl.getAttribLocation(this.program,id);
  this.transparentBuffer=gl.createBuffer();
 }
 static color(label){return [[0,0,0],[.90,.29,.34],[.43,.76,.38],[.26,.51,.93]][label];}
 static rotation(yaw,pitch){const ca=Math.cos(yaw),sa=Math.sin(yaw),cb=Math.cos(pitch),sb=Math.sin(pitch);return new Float32Array([ca,sa*sb,-sa*cb,0,cb,sb,sa,-ca*sb,ca*cb]);}
 static sortedTriangles(meshes,r){
  const triangles=[];
  for(const mesh of meshes)for(let offset=0;offset<(mesh.indices?mesh.indices.length:mesh.points.length/3);offset+=3){
   const p=mesh.points;let depth=0;
   for(let k=0;k<3;k++){const j=(mesh.indices?mesh.indices[offset+k]:offset+k)*3;depth+=r[2]*p[j]+r[5]*p[j+1]+r[8]*p[j+2];}
   triangles.push({mesh,offset,depth:depth/3});
  }
  // gl_Position.z is minus view Z: increasing view Z is back to front.
  triangles.sort((a,b)=>a.depth-b.depth);return triangles;
 }
 render(meshes,{bounds,yaw,pitch,zoom,clip,mode,opacity}){
  const gl=this.gl;if(!gl)return;
  const r=CardiacSurfaceView.rotation(yaw,pitch),center=bounds[0].map((v,i)=>(v+bounds[1][i])/2);
  const diameter=Math.max(Math.hypot(...bounds[1].map((v,i)=>v-bounds[0][i])),1),base=1.8/diameter*zoom;
  gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.depthMask(true);gl.clearColor(.025,.039,.055,1);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
  gl.useProgram(this.program);gl.uniformMatrix3fv(this.loc.r,false,r);gl.uniform3fv(this.loc.center,center);
  const aspect=this.canvas.width/this.canvas.height;
  gl.uniform2f(this.loc.scale,base/Math.max(aspect,1),base*Math.min(aspect,1));
  gl.uniform1f(this.loc.clipZ,clip);gl.uniform1f(this.loc.wire,mode==='wire'?1:0);
  if(mode==='transparent'){
   const triangles=CardiacSurfaceView.sortedTriangles(meshes,r),data=new Float32Array(triangles.length*30);
   let j=0;
   for(const {mesh,offset} of triangles){const color=CardiacSurfaceView.color(mesh.label);
    for(let k=0;k<3;k++){
     const index=(mesh.indices?mesh.indices[offset+k]:offset+k)*3;
     for(let a=0;a<3;a++)data[j++]=mesh.points[index+a];
     for(let a=0;a<3;a++)data[j++]=mesh.normals[index+a];
     for(const c of color)data[j++]=c;
     data[j++]=opacity;
    }
   }
   gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
   gl.bindBuffer(gl.ARRAY_BUFFER,this.transparentBuffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);
   for(const [id,size,offset] of [['p',3,0],['n',3,12],['tint',4,24]]){
    gl.enableVertexAttribArray(this.loc[id]);gl.vertexAttribPointer(this.loc[id],size,gl.FLOAT,false,40,offset);
   }
   gl.drawArrays(gl.TRIANGLES,0,triangles.length*3);gl.depthMask(true);
  }else{
   gl.disable(gl.BLEND);gl.disableVertexAttribArray(this.loc.tint);
   for(const mesh of meshes){
    gl.vertexAttrib4fv(this.loc.tint,[...CardiacSurfaceView.color(mesh.label),1]);
    gl.enableVertexAttribArray(this.loc.p);
    if(mode==='wire'){
     if(!mesh.wire){const count=mesh.indices?mesh.indices.length:mesh.points.length/3,points=new Float32Array(count*6);let j=0;
      for(let i=0;i<count;i+=3)for(const k of [0,1,1,2,2,0]){const index=(mesh.indices?mesh.indices[i+k]:i+k)*3;for(let a=0;a<3;a++)points[j++]=mesh.points[index+a];}
      mesh.wire=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,mesh.wire);gl.bufferData(gl.ARRAY_BUFFER,points,gl.STATIC_DRAW);
     }
     gl.bindBuffer(gl.ARRAY_BUFFER,mesh.wire);gl.vertexAttribPointer(this.loc.p,3,gl.FLOAT,false,0,0);
     gl.disableVertexAttribArray(this.loc.n);gl.vertexAttrib3f(this.loc.n,0,0,1);gl.drawArrays(gl.LINES,0,(mesh.indices?mesh.indices.length:mesh.vertices)*2);
    }else{
     gl.bindBuffer(gl.ARRAY_BUFFER,mesh.p);gl.vertexAttribPointer(this.loc.p,3,gl.FLOAT,false,0,0);
     gl.bindBuffer(gl.ARRAY_BUFFER,mesh.n);gl.enableVertexAttribArray(this.loc.n);gl.vertexAttribPointer(this.loc.n,3,gl.FLOAT,false,0,0);
     if(mesh.indexBuffer){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,mesh.indexBuffer);gl.drawElements(gl.TRIANGLES,mesh.indices.length,gl.UNSIGNED_INT,0);}
     else gl.drawArrays(gl.TRIANGLES,0,mesh.vertices);
    }
   }
  }
  const ctx=this.orientation.getContext('2d');ctx.clearRect(0,0,88,80);ctx.lineWidth=2;ctx.font='bold 12px sans-serif';
  for(const [axis,label,color] of [[0,'R','#ef7a83'],[1,'A','#78d5ac'],[2,'S','#6aacf3']]){
   const x=36+26*r[axis*3],y=44-26*r[axis*3+1];ctx.strokeStyle=ctx.fillStyle=color;
   ctx.beginPath();ctx.moveTo(36,44);ctx.lineTo(x,y);ctx.stroke();ctx.fillText(label,x+3,y-3);
  }
 }
}
