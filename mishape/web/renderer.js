/* Original minimal WebGL2 viewer. No CDN, npm bundle or third-party JS required.
 * Display transforms are isolated here; the geometry/export API never sees them. */
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const scale=(a,s)=>a.map(v=>v*s);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>scale(a,1/(Math.hypot(...a)||1));
const clamp=(v,a,b)=>Math.max(a,Math.min(v,b));
function mul(a,b){const c=new Float32Array(16);for(let col=0;col<4;col++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)c[col*4+row]+=a[k*4+row]*b[col*4+k];return c;}
function perspective(fov,aspect,near,far){const s=1/Math.tan(fov/2);return new Float32Array([s/aspect,0,0,0,0,s,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);}
function lookAt(eye,target){const z=unit(sub(eye,target)),x=unit(cross([0,0,1],z)),y=cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);}
function rgb(hex){return [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255);}
function bounds(v){const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(let i=0;i<v.length;i++) {const k=i%3;lo[k]=Math.min(lo[k],v[i]);hi[k]=Math.max(hi[k],v[i]);}return [lo,hi];}
const VS=`#version 300 es
precision highp float;
in vec3 aPosition; in vec3 aNormal; in float aShift;
uniform mat4 uVP; uniform vec3 uOffset;
out vec3 vNormal; out vec3 vWorld; out float vShift;
void main(){vNormal=aNormal;vWorld=aPosition+uOffset;vShift=aShift;gl_Position=uVP*vec4(vWorld,1.);gl_PointSize=8.;}`;
const FS=`#version 300 es
precision highp float;
in vec3 vNormal; in vec3 vWorld; in float vShift;
uniform vec3 uColor; uniform vec3 uEye; uniform float uAlpha; uniform float uMaxShift;
uniform float uRoughness; uniform float uMetallic; uniform float uEmission;
uniform bool uField; uniform float uFieldMin; uniform float uFieldMax;
uniform bool uReview; uniform bool uMask; uniform bool uUnlit; uniform bool uHeat; uniform bool uPoint; uniform bool uShadow;
out vec4 outColor;
void main(){
 if(uShadow){float d=pow(vWorld.x/2.63,2.)+pow(vWorld.y/1.17,2.);outColor=vec4(.12,.15,.13,.30*exp(-2.1*d)+.19*exp(-8.*d));return;}
 if(uPoint&&length(gl_PointCoord-vec2(.5))>.5)discard;
 vec3 color=uColor;
 if(uHeat){float t=clamp(vShift/max(uMaxShift,.000001),0.,1.);color=mix(vec3(.12,.4,.55),vec3(1.,.48,.23),t);}
 if(uReview){float q=clamp(vShift,0.,1.);color=mix(vec3(.85,.34,.29),vec3(.24,.73,.64),q);}
 if(uMask){float w=clamp(vShift,0.,1.);color=w<.00001?vec3(.19,.24,.28):(w>.999?vec3(.93,.71,.31):mix(vec3(.19,.36,.41),vec3(.23,.78,.71),sqrt(w)));}
 if(uField){float t=clamp((vShift-uFieldMin)/max(uFieldMax-uFieldMin,.000001),0.,1.);
  vec3 a=vec3(.141,.341,.698),b=vec3(.275,.663,.769),c=vec3(.890,.937,.910),d=vec3(.953,.741,.412),e=vec3(.773,.267,.208);
  color=t<.25?mix(a,b,t*4.):t<.5?mix(b,c,(t-.25)*4.):t<.75?mix(c,d,(t-.5)*4.):mix(d,e,(t-.75)*4.);
 }
 if(!uUnlit&&!uField){
  vec3 n=normalize(vNormal);if(!gl_FrontFacing)n=-n;
  vec3 v=normalize(uEye-vWorld),r=reflect(-v,n);
  float nv=max(dot(n,v),0.);
  vec3 key=normalize(vec3(-.45,-.60,1.6)),fill=normalize(vec3(.45,.8,.8));
  float rough=clamp(uHeat?.7:uRoughness,.08,.98),met=uHeat?0.:uMetallic;
  vec3 base=pow(max(color,vec3(.001)),vec3(2.2));
  float hemi=mix(.28,.76,clamp(n.z*.5+.5,0.,1.));
  float diffuse=hemi+.58*max(dot(n,key),0.)+.27*max(dot(n,fill),0.);
  float gloss=mix(210.,9.,rough*rough);
  float spec1=pow(max(dot(n,normalize(key+v)),0.),gloss);
  float spec2=pow(max(dot(n,normalize(fill+v)),0.),gloss*.55);
  // Analytic softbox reflections: broad overhead panel and long side strips.
  // This needs no network HDRI and gives stable readable surface curvature.
  float overhead=exp(-pow((r.x+.12)/(.38+.5*rough),4.)-pow((r.y+.28)/(.28+.5*rough),4.))*smoothstep(.15,.75,r.z);
  float strip=exp(-pow((r.z-.29)/(.035+.16*rough),2.))*smoothstep(-.1,.4,-r.y);
  float opposite=exp(-pow((r.z-.40)/(.06+.2*rough),2.))*smoothstep(.05,.5,r.y);
  vec3 environment=mix(vec3(.13,.15,.13),vec3(.56,.62,.63),smoothstep(-.35,.8,r.z));
  environment+=vec3(1.00,.98,.91)*overhead*.75+vec3(.85,.91,.96)*strip*.64+vec3(.80,.85,.81)*opposite*.35;
  vec3 f0=mix(vec3(.045),base*.72+.13,met);
  vec3 fres=f0+(1.-f0)*pow(1.-nv,5.);
  color=base*diffuse*(1.-.28*met)+environment*fres*(.48-.20*rough);
  color+=mix(vec3(.16),base*.5+.12,met)*(spec1+.35*spec2)*(1.-.60*rough);
  color+=base*uEmission;
  color=pow(clamp(color,0.,1.),vec3(1./2.2));
 }
 outColor=vec4(color,uAlpha);
}`;
export class Viewer {
  constructor(host){
    this.canvas=document.createElement('canvas');host.append(this.canvas);this.host=host;
    this.gl=this.canvas.getContext('webgl2',{antialias:true,alpha:false,preserveDrawingBuffer:true});
    if(!this.gl)throw new Error('浏览器未能启动 WebGL2。请启用硬件加速或使用支持 WebGL2 的浏览器。');
    const gl=this.gl;
    const shader=(type,text)=>{const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
    this.program=gl.createProgram();gl.attachShader(this.program,shader(gl.VERTEX_SHADER,VS));gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,FS));gl.linkProgram(this.program);
    if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
    this.u={};['VP','Offset','Color','Eye','Alpha','MaxShift','Unlit','Heat','Point','Roughness','Metallic','Emission','Shadow','Review','Mask','Field','FieldMin','FieldMax'].forEach(n=>this.u[n]=gl.getUniformLocation(this.program,'u'+n));
    this.a={};['Position','Normal','Shift'].forEach(n=>this.a[n]=gl.getAttribLocation(this.program,'a'+n));
    this.yaw=-2.32;this.pitch=1.20;this.radius=8.;this.target=[0,0,.7];this.fov=.65;
    this.groups=[];this.selected=new Set();this.explode=0;this.wire=false;this.heat=false;this.isolate=false;this.showOriginal=false;this.cage=null;this.materials=true;
    this.scalar=null;this.scalarRange=[0,1];this.region=null;this.influence=null;this.mask=false;this.issueLines=[];this.aux=gl.createBuffer();this.grid=this.makeGrid();this.bindEvents();
    new ResizeObserver(()=>this.draw()).observe(host);
  }
  makeGrid(){const a=[];for(let x=-8;x<=8;x+=.5){a.push(x,-8,-.018,x,8,-.018,-8,x,-.018,8,x,-.018);}return a;}
  bindEvents(){
    const c=this.canvas;c.addEventListener('contextmenu',e=>e.preventDefault());
    c.addEventListener('pointerdown',e=>{c.setPointerCapture(e.pointerId);const r=c.getBoundingClientRect();const regional=e.button===0&&this.pickRegionHandle(e.clientX-r.left,e.clientY-r.top);const handle=e.shiftKey&&!regional?this.pickHandle(e.clientX-r.left,e.clientY-r.top):null;this.drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,button:e.button,handle,regional,axisPixels:regional?this.regionAxisPixels():null};if(regional)this.onRegionStart?.();if(handle)this.onHandlePick?.(handle);});
    c.addEventListener('pointermove',e=>{if(!this.drag)return;const d=this.drag,dx=e.clientX-d.x,dy=e.clientY-d.y;d.x=e.clientX;d.y=e.clientY;
      if(d.regional){const ax=d.axisPixels;const length=ax[0]*ax[0]+ax[1]*ax[1];if(length>4)this.onRegionDrag?.(((e.clientX-d.startX)*ax[0]+(e.clientY-d.startY)*ax[1])/length*.30);}
      else if(d.handle){const eye=this.eye(),forward=unit(sub(this.target,eye)),right=unit(cross(forward,[0,0,1])),up=cross(right,forward);const s=this.radius*2*Math.tan(this.fov/2)/c.clientHeight;this.onHandleDrag?.(d.handle,add(scale(right,dx*s),scale(up,-dy*s)));}
      else if(d.button===2){const fw=unit(sub(this.target,this.eye())),rt=unit(cross(fw,[0,0,1])),up=cross(rt,fw);this.target=add(this.target,add(scale(rt,-dx*this.radius/800),scale(up,dy*this.radius/800)));}
      else{this.yaw-=dx*.006;this.pitch=clamp(this.pitch-dy*.006,.05,Math.PI-.05);}this.draw();});
    c.addEventListener('pointerup',e=>{const d=this.drag;this.drag=null;if(!d)return;if(d.regional){this.onRegionEnd?.();return;}if(Math.hypot(e.clientX-d.startX,e.clientY-d.startY)<4){const r=c.getBoundingClientRect();const x=e.clientX-r.left,y=e.clientY-r.top;const h=this.pickHandle(x,y);if(h&&e.shiftKey)this.onHandlePick?.(h);else{const hit=this.pick(x,y);if(hit)this.onPick?.(hit,e);}}});
    c.addEventListener('pointercancel',()=>{if(this.drag?.regional)this.onRegionEnd?.();this.drag=null;});
    c.addEventListener('wheel',e=>{e.preventDefault();this.radius=clamp(this.radius*Math.exp(e.deltaY*.001),.15,200);this.draw();},{passive:false});
    c.addEventListener('webglcontextlost',e=>{e.preventDefault();this.onError?.('WebGL 上下文丢失，请重新加载页面。');});
  }
  eye(){return add(this.target,[this.radius*Math.sin(this.pitch)*Math.cos(this.yaw),this.radius*Math.sin(this.pitch)*Math.sin(this.yaw),this.radius*Math.cos(this.pitch)]);}
  fit(){if(!this.vertices)return;const [lo,hi]=bounds(this.vertices);this.target=scale(add(lo,hi),.5);this.radius=Math.hypot(...sub(hi,lo))*1.50;this.yaw=-2.32;this.pitch=1.20;this.draw();}
  view(name){if(name==='iso'){this.fit();return;}if(name==='top'){this.pitch=.001;this.yaw=-Math.PI/2;}if(name==='side'){this.pitch=Math.PI/2;this.yaw=-Math.PI/2;}if(name==='front'){this.pitch=Math.PI/2;this.yaw=Math.PI;}if(name==='rear'){this.pitch=1.25;this.yaw=.80;}this.draw();}
  setModel(m){this.model=m;this.vertices=Float64Array.from(m.vertices);this.reference=this.vertices.slice();this.faces=Int32Array.from(m.faces);this.labels=Int32Array.from(m.face_labels);this.materialIDs=m.face_materials;this.reviewMode=false;this.reviewScores=null;this.influence=null;this.region=null;this.issueLines=[];this.buildGroups();this.fit();}
  setVertices(v){this.vertices=Float64Array.from(v);this.buildGroups();this.draw();}
  setScalar(values,range){
    if(values&&(values.length!==this.vertices.length/3||!values.every(Number.isFinite)))throw Error('场数据必须与显示顶点逐项对应。');
    this.scalar=values?Float32Array.from(values):null;this.scalarRange=range||[0,1];this.buildGroups();this.draw();
  }
  point(i,original=false){const v=original?this.reference:this.vertices;return [v[i*3],v[i*3+1],v[i*3+2]];}
  buildGroups(){
    const gl=this.gl;for(const g of this.groups) {gl.deleteBuffer(g.buffer);gl.deleteBuffer(g.edgeBuffer);}this.groups=[];this.maxShift=0;
    const all=new Map();for(let i=0;i<this.labels.length;i++){const key=this.labels[i]+':'+(this.materialIDs?.[i]??-1);if(!all.has(key))all.set(key,[]);all.get(key).push(i);}
    const [lo,hi]=bounds(this.reference);this.center=scale(add(lo,hi),.5);
    const normals=new Float64Array(this.vertices.length),faceNormals=new Float32Array(this.faces.length);
    for(let fi=0;fi<this.faces.length/3;fi++){
      const ia=this.faces[fi*3],ib=this.faces[fi*3+1],ic=this.faces[fi*3+2];
      const a=this.point(ia),bb=this.point(ib),c=this.point(ic),n=cross(sub(bb,a),sub(c,a));
      faceNormals.set(unit(n),fi*3);
      for(const i of [ia,ib,ic])for(let k=0;k<3;k++)normals[i*3+k]+=n[k];
    }
    for(let i=0;i<normals.length;i+=3){const len=Math.hypot(normals[i],normals[i+1],normals[i+2])||1;for(let k=0;k<3;k++)normals[i+k]/=len;}
    const partCenters=new Map();
    for(let fi=0;fi<this.labels.length;fi++){const id=this.labels[fi];if(!partCenters.has(id))partCenters.set(id,{sum:[0,0,0],n:0});const rec=partCenters.get(id);for(let j=0;j<3;j++){rec.sum=add(rec.sum,this.point(this.faces[3*fi+j],true));rec.n++;}}
    for(const [key,ids] of all){const [id,materialID]=key.split(':').map(Number);const data=new Float32Array(ids.length*3*7),edgeData=new Float32Array(ids.length*6*7);let off=0,eo=0;let centroid=[0,0,0];
      for(const fi of ids){const indices=[this.faces[fi*3],this.faces[fi*3+1],this.faces[fi*3+2]];const pts=indices.map(i=>this.point(i));const normal=unit(cross(sub(pts[1],pts[0]),sub(pts[2],pts[0])));
        const rows=indices.map((idx,k)=>{const shift=Math.hypot(...sub(pts[k],this.point(idx,true)));this.maxShift=Math.max(this.maxShift,shift);centroid=add(centroid,this.point(idx,true));const smooth=[normals[idx*3],normals[idx*3+1],normals[idx*3+2]];return [...pts[k],...(dot(smooth,normal)>.57?smooth:normal),this.scalar?this.scalar[idx]:this.mask&&this.influence?this.influence[idx]:(this.reviewMode&&this.reviewScores?this.reviewScores[fi]:shift)];});
        for(const row of rows){data.set(row,off);off+=7;}for(const j of [0,1,1,2,2,0]){edgeData.set(rows[j],eo);eo+=7;}}
      centroid=scale(partCenters.get(id).sum,1/partCenters.get(id).n);let direction=sub(centroid,this.center);direction[2]*=1.5;
      if(Math.hypot(...direction)<.05)direction=[0,0,1];direction=unit(direction);
      const part=this.model.parts.find(p=>p.id===id)||{color:'#8ea4ba'};
      const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);
      const edgeBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,edgeBuffer);gl.bufferData(gl.ARRAY_BUFFER,edgeData,gl.DYNAMIC_DRAW);
      this.groups.push({id,buffer,edgeBuffer,count:ids.length*3,color:rgb(part.color),material:this.model.metadata.material_palette?.[materialID]||part.material||{base_color:part.color,roughness:.65,metallic:.05},direction:part.explode_direction||direction,centroid});
    }
  }
  bind(buffer){const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(const [name,size,offset] of [['Position',3,0],['Normal',3,12],['Shift',1,24]]){gl.enableVertexAttribArray(this.a[name]);gl.vertexAttribPointer(this.a[name],size,gl.FLOAT,false,28,offset);}}
  lines(coords,color,points=false){if(!coords.length)return;const gl=this.gl;const d=new Float32Array(coords.length/3*7);for(let i=0;i<coords.length/3;i++)d.set([coords[i*3],coords[i*3+1],coords[i*3+2],0,0,1,0],i*7);gl.bindBuffer(gl.ARRAY_BUFFER,this.aux);gl.bufferData(gl.ARRAY_BUFFER,d,gl.DYNAMIC_DRAW);this.bind(this.aux);gl.uniform1i(this.u.Field,0);gl.uniform1i(this.u.Review,0);gl.uniform1i(this.u.Mask,0);gl.uniform3fv(this.u.Offset,[0,0,0]);gl.uniform3fv(this.u.Color,color);gl.uniform1i(this.u.Unlit,1);gl.uniform1i(this.u.Heat,0);gl.uniform1i(this.u.Point,points?1:0);gl.drawArrays(points?gl.POINTS:gl.LINES,0,coords.length/3);gl.uniform1i(this.u.Point,0);}
  cagePoints(){if(!this.cage)return [];const {bounds:b,grid=[4,3,3],controls=[]}=this.cage;const pts=[];for(let i=0;i<grid[0];i++)for(let j=0;j<grid[1];j++)for(let k=0;k<grid[2];k++){const index=[i,j,k];const p=index.map((v,a)=>b[0][a]+(b[1][a]-b[0][a])*v/(grid[a]-1));const control=controls.find(c=>c.index.every((v,a)=>v===index[a]));pts.push({index,position:add(p,control?.displacement||[0,0,0])});}return pts;}
  draw(){
    const gl=this.gl,dpr=Math.min(window.devicePixelRatio||1,1.75),w=Math.max(this.host.clientWidth,1),h=Math.max(this.host.clientHeight,1);if(this.canvas.width!==Math.round(w*dpr)||this.canvas.height!==Math.round(h*dpr)){this.canvas.width=Math.round(w*dpr);this.canvas.height=Math.round(h*dpr);}gl.viewport(0,0,this.canvas.width,this.canvas.height);
    const background=this.backgroundColor||[.962,.965,.970];gl.clearColor(background[0],background[1],background[2],1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.useProgram(this.program);
    const eye=this.eye();this.vp=mul(perspective(this.fov,w/h,.015,500),lookAt(eye,this.target));gl.uniformMatrix4fv(this.u.VP,false,this.vp);gl.uniform3fv(this.u.Eye,eye);gl.uniform1f(this.u.Alpha,1);gl.uniform1f(this.u.MaxShift,this.maxShift||1);gl.uniform1i(this.u.Point,0);gl.uniform1i(this.u.Shadow,0);
    this.lines(this.grid,this.gridColor||[.835,.85,.865]);
    this.drawShadow();
    for(const g of this.groups){if(this.isolate&&!this.selected.has(g.id))continue;this.bind(g.buffer);gl.uniform1i(this.u.Field,this.scalar?1:0);gl.uniform1f(this.u.FieldMin,this.scalarRange[0]);gl.uniform1f(this.u.FieldMax,this.scalarRange[1]);gl.uniform3fv(this.u.Offset,scale(g.direction,this.explode));const base=this.materials?rgb(g.material.base_color):g.color;gl.uniform3fv(this.u.Color,this.selected.has(g.id)?base.map((v,i)=>Math.min(1,v+(i===0?.07:.035))):base);gl.uniform1f(this.u.Roughness,this.materials?(g.material.roughness??.6):.7);gl.uniform1f(this.u.Metallic,this.materials?(g.material.metallic??0):0);gl.uniform1f(this.u.Emission,this.materials?(g.material.emission??0):0);gl.uniform1i(this.u.Unlit,0);gl.uniform1i(this.u.Review,this.reviewMode&&!(this.mask&&this.influence)?1:0);gl.uniform1i(this.u.Heat,this.heat&&!(this.mask&&this.influence)&&!this.reviewMode?1:0);gl.uniform1i(this.u.Mask,this.mask&&this.influence?1:0);gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1,1);gl.drawArrays(gl.TRIANGLES,0,g.count);gl.disable(gl.POLYGON_OFFSET_FILL);
      if(this.wire){gl.uniform1i(this.u.Field,0);gl.uniform1i(this.u.Review,0);this.bind(g.edgeBuffer);gl.uniform1i(this.u.Unlit,1);gl.uniform1i(this.u.Heat,0);gl.uniform1i(this.u.Mask,0);gl.uniform3fv(this.u.Color,this.wireColor||[.19,.28,.35]);gl.drawArrays(gl.LINES,0,g.edgeCount??g.count*2);}}
    if(this.cage&&!this.explode){gl.disable(gl.DEPTH_TEST);const pts=this.cagePoints();const lines=[];const grid=this.cage.grid||[4,3,3];for(const p of pts){for(let a=0;a<3;a++){const ix=[...p.index];ix[a]++;if(ix[a]>=grid[a])continue;const other=pts.find(t=>t.index.every((v,k)=>v===ix[k]));lines.push(...p.position,...other.position);}}
      this.lines(lines,this.cageLineColor||[.46,.38,.25]);this.lines(pts.flatMap(p=>p.position),this.cagePointColor||[.9,.68,.37],true);
      if(this.activeHandle){const p=pts.find(t=>t.index.every((v,k)=>v===this.activeHandle[k]));if(p)this.lines(p.position,[1,.96,.77],true);}}
    if(!this.explode){
      if(this.issueLines?.length){gl.disable(gl.DEPTH_TEST);this.lines(this.issueLines,[1,.32,.25]);}
      if(this.region){gl.disable(gl.DEPTH_TEST);const r=this.region;const base=add(r.center,scale(r.direction,r.amount||0));const tip=add(base,scale(r.direction,.30));this.lines([...base,...tip],[.94,.79,.46]);this.lines(tip,[1,.89,.59],true);this.lines(base,[.4,.96,.83],true);
        const fw=unit(sub(this.eye(),base));let across=unit(cross(r.direction,fw));if(Math.hypot(...across)<.01)across=[1,0,0];const back=sub(tip,scale(r.direction,.04));this.lines([...tip,...add(back,scale(across,.022)),...tip,...sub(back,scale(across,.022))],[1,.85,.53]);
      }
    }
    gl.enable(gl.DEPTH_TEST);
  }
  setRegion(data,amount=0){this.region=data?{center:data.center,direction:data.direction,amount}:null;this.influence=data?Float32Array.from(data.weights):null;this.buildGroups();this.draw();}
  regionAxisPixels(){if(!this.region)return [0,0];const r=this.region,base=add(r.center,scale(r.direction,r.amount||0));const a=this.project(base),b=this.project(add(base,scale(r.direction,.30)));return [b[0]-a[0],b[1]-a[1]];}
  pickRegionHandle(x,y){if(!this.region||this.explode)return false;const r=this.region;const p=this.project(add(r.center,scale(r.direction,(r.amount||0)+.30)));return p[2]<1&&Math.hypot(x-p[0],y-p[1])<20;}
  drawShadow(){
    if(!this.vertices||this.isolate)return;
    const gl=this.gl,pts=[-4,-2,-.013,4,-2,-.013,4,2,-.013,-4,-2,-.013,4,2,-.013,-4,2,-.013];
    const d=new Float32Array(42);for(let i=0;i<6;i++)d.set([...pts.slice(i*3,i*3+3),0,0,1,0],i*7);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.aux);gl.bufferData(gl.ARRAY_BUFFER,d,gl.DYNAMIC_DRAW);this.bind(this.aux);
    gl.uniform3fv(this.u.Offset,[0,0,0]);gl.uniform1i(this.u.Shadow,1);gl.uniform1i(this.u.Mask,0);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
    gl.drawArrays(gl.TRIANGLES,0,6);gl.depthMask(true);gl.disable(gl.BLEND);gl.uniform1i(this.u.Shadow,0);
  }
  project(p){const m=this.vp;const out=[0,0,0,0];for(let r=0;r<4;r++)out[r]=m[r]*p[0]+m[4+r]*p[1]+m[8+r]*p[2]+m[12+r];return [(out[0]/out[3]+1)*this.host.clientWidth/2,(1-out[1]/out[3])*this.host.clientHeight/2,out[2]/out[3]];}
  pickHandle(x,y){if(!this.cage||this.explode||this.cage.kind!=='ffd')return null;let best=null,dist=15;for(const p of this.cagePoints()){const s=this.project(p.position);const d=Math.hypot(x-s[0],y-s[1]);if(d<dist&&s[2]<1){best=p.index;dist=d;}}return best;}
  pick(x,y){if(!this.vertices)return null;const eye=this.eye(),fw=unit(sub(this.target,eye)),right=unit(cross(fw,[0,0,1])),up=cross(right,fw);const tan=Math.tan(this.fov/2);const dx=(2*x/this.host.clientWidth-1)*tan*this.host.clientWidth/this.host.clientHeight,dy=(1-2*y/this.host.clientHeight)*tan;const ray=unit(add(fw,add(scale(right,dx),scale(up,dy))));let nearest=Infinity,best=null;const offsets=new Map(this.groups.map(g=>[g.id,scale(g.direction,this.explode)]));
    for(let fi=0;fi<this.labels.length;fi++){const label=this.labels[fi];if(this.isolate&&!this.selected.has(label))continue;const offset=offsets.get(label);const a=add(this.point(this.faces[3*fi]),offset),b=add(this.point(this.faces[3*fi+1]),offset),c=add(this.point(this.faces[3*fi+2]),offset);const e1=sub(b,a),e2=sub(c,a),p=cross(ray,e2),det=dot(e1,p);if(Math.abs(det)<1e-10)continue;const t=sub(eye,a),u=dot(t,p)/det;if(u<0||u>1)continue;const q=cross(t,e1),v=dot(ray,q)/det;if(v<0||u+v>1)continue;const distance=dot(e2,q)/det;if(distance>0&&distance<nearest){nearest=distance;best={faceIndex:fi,sourceFace:this.model.source_face_ids?.[fi]??fi,part:label,position:add(eye,scale(ray,distance)),referencePosition:add(scale(this.point(this.faces[3*fi],true),1-u-v),add(scale(this.point(this.faces[3*fi+1],true),u),scale(this.point(this.faces[3*fi+2],true),v)))};}}
    return best;
  }
}
