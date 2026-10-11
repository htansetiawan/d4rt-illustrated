/* Explicit synthetic geometry and loss arithmetic; no D4RT weights. */
(() => {
'use strict';
const $=id=>document.getElementById(id);
if(!$('geometry-lab'))return;
const state={src:8,tgt:30,cam:18,point:0,predX:-.35,scale:1,confidence:.8,missing:false};
const colors=['#647a59','#b28d66','#7e8ca2'];
const fmt=x=>x.toFixed(4), vec=x=>'['+x.map(fmt).join(', ')+']';
const points=(id,t)=>id===0?[-.9+.039*t,.24*Math.sin(t/12),4.8]:id===1?[1.05-.007*t,-.45,5.5+.005*t]:[-.6,.6-.006*t,6.3];
const camera=t=>({center:[.48*Math.sin(t/24),.08*Math.sin(t/18),0],yaw:.12*Math.sin(t/30)});
function toCamera(p,t){const c=camera(t),x=p[0]-c.center[0],y=p[1]-c.center[1],z=p[2];return [Math.cos(c.yaw)*x-Math.sin(c.yaw)*z,y,Math.sin(c.yaw)*x+Math.cos(c.yaw)*z];}
function uv(p){return [.5+.9*p[0]/p[2],.5+.9*p[1]/p[2]];}
const sl=x=>Math.sign(x)*Math.log1p(Math.abs(x));
function compute(s=state){
 const world=Array.from({length:3},(_,i)=>points(i,s.tgt));
 const target=world.map(p=>toCamera(p,s.cam));
 // Fixed, intentionally imperfect values. They never read target geometry.
 const base=[[-.35,.15,4.1],[.4,-.2,5.3],[.6,.3,6.5]];
 base[s.point][0]=s.predX;
 const pred=base.map(p=>p.map(x=>x*s.scale));
 const valid=base.map((_,i)=>!(s.missing&&i===s.point)), ids=valid.flatMap((v,i)=>v?[i]:[]),n=ids.length;
 const zp=ids.reduce((a,i)=>a+pred[i][2],0)/n,zt=ids.reduce((a,i)=>a+target[i][2],0)/n;
 const pn=pred.map(p=>p.map(x=>x/zp)),tn=target.map(p=>p.map(x=>x/zt));
 const pl=pn.map(p=>p.map(sl)),tl=tn.map(p=>p.map(sl));
 const errors=pl.map((p,i)=>p.reduce((a,x,k)=>a+Math.abs(x-tl[i][k]),0));
 const l1=ids.reduce((a,i)=>a+errors[i],0)/n, loss=s.confidence*l1-.2*Math.log(s.confidence);
 // Analytic gradient includes the dependence of every normalized prediction
 // on the predicted set's mean Z. Missing points do not enter this demo's set.
 const gradient=pred.map(()=>[0,0,0]);
 for(const j of ids)for(let k=0;k<3;k++){
  const g=s.confidence/n*Math.sign(pl[j][k]-tl[j][k])/(1+Math.abs(pn[j][k]));
  gradient[j][k]+=g/zp;
  for(const i of ids)gradient[i][2]-=g*pred[j][k]/(zp*zp*n);
 }
 const source=Array.from({length:3},(_,i)=>uv(toCamera(points(i,s.src),s.src)));
 return {state:{...s},world,target,pred,valid,zp,zt,pn,tn,pl,tl,errors,l1,loss,gradient,source,
 query:[...source[s.point],s.src,s.tgt,s.cam],camera:camera(s.cam)};
}
function frame(id,physicalTime,cameraTime,selected){
 const canvas=$(id),ctx=canvas.getContext('2d'),w=256;
 ctx.fillStyle='#f1f0eb';ctx.fillRect(0,0,w,w);
 ctx.strokeStyle='#deded7';ctx.lineWidth=1;
 for(let i=0;i<=8;i++){ctx.beginPath();ctx.moveTo(i*32,0);ctx.lineTo(i*32,w);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*32);ctx.lineTo(w,i*32);ctx.stroke();}
 for(let i=0;i<3;i++){
  const p=toCamera(points(i,physicalTime),cameraTime),v=uv(p),x=v[0]*w,y=v[1]*w;
  ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(x,y,10,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#fff';ctx.textAlign='center';ctx.font='10px sans-serif';ctx.fillText('ABC'[i],x,y+3);
  if(i===selected){ctx.strokeStyle='#222';ctx.lineWidth=1.5;ctx.strokeRect(x-16,y-16,32,32);}
 }
 ctx.fillStyle='#777';ctx.font='10px monospace';ctx.textAlign='left';ctx.fillText(`state ${physicalTime} / camera ${cameraTime}`,10,244);
}
function render(){
 const r=compute(),s=state;
 for(const name of ['src','tgt','cam'])$('geo-'+name+'-out').textContent=s[name];
 $('geo-point-out').textContent='ABC'[s.point];
 frame('geo-source',s.src,s.src,s.point);frame('geo-target',s.tgt,s.tgt,s.point);frame('geo-reference',s.tgt,s.cam,s.point);
 $('geo-query').textContent=`q = (${r.source[s.point].map(fmt).join(', ')}, ${s.src}, ${s.tgt}, ${s.cam})\n\nFourier(u,v) → FC\n+ E_source[${s.src}]\n+ E_target[${s.tgt}]\n+ E_camera[${s.cam}]\n+ MLP(9 × 9 source RGB crop)\n→ query MLP → decoder token`;
 $('geo-world').textContent=vec(r.world[s.point]);$('geo-coordinate').textContent=vec(r.target[s.point]);
 $('geo-camera').textContent=`Camera center = ${vec(r.camera.center)}; yaw = ${fmt(r.camera.yaw)} radians.`;
 $('geo-times').textContent=`Point ${'ABC'[s.point]} · source ${s.src} → target state ${s.tgt} → camera coordinates ${s.cam}`;
 $('geo-pred-out').textContent=fmt(s.predX);$('geo-scale-out').textContent=s.scale.toFixed(2)+'×';$('geo-confidence-out').textContent=fmt(s.confidence);
 $('geo-loss-rows').innerHTML=r.target.map((p,i)=>`<tr data-selected="${i===s.point}" data-valid="${r.valid[i]}"><th>${'ABC'[i]}${r.valid[i]?'':' · masked'}</th><td>${vec(r.pred[i])}</td><td>${vec(p)}</td><td>${vec(r.pl[i])}</td><td>${vec(r.tl[i])}</td><td>${r.valid[i]?fmt(r.errors[i]):'not applied'}</td></tr>`).join('');
 $('geo-depth-pred').textContent=fmt(r.zp);$('geo-depth-label').textContent=fmt(r.zt);$('geo-loss').textContent=fmt(r.loss);
 $('geo-loss-equation').textContent=`L_demo = ${fmt(s.confidence)} × ${fmt(r.l1)} − 0.2 × log(${fmt(s.confidence)}) = ${fmt(r.loss)}`;
 $('geo-gradient').textContent=`∂L_demo / ∂P̂_${'ABC'[s.point]} = ${vec(r.gradient[s.point])}`;
 $('geo-gradient-note').textContent=s.missing?'This point has no 3D supervision in this exercise, so its 3D loss and direct output gradient are zero. Other labeled queries still train the shared network.':'All predicted depths contribute to the normalization. Changing one Z coordinate can therefore change errors for other queries in the training loss, even though decoder queries are independent in the forward pass.';
 $('geo-validity').textContent=`${r.valid.filter(Boolean).length} of 3 points supervise this isolated geometry objective. Same query times as the decoder workshop above.`;
}
for(const name of ['src','tgt','cam'])$('geo-'+name).oninput=e=>{state[name]=+e.target.value;render();};
$('geo-point').onchange=e=>{state.point=+e.target.value;state.predX=[-.35,.4,.6][state.point];$('geo-pred').value=state.predX;render();};
for(const [id,key] of [['pred','predX'],['scale','scale'],['confidence','confidence']])$('geo-'+id).oninput=e=>{state[key]=+e.target.value;render();};
$('geo-missing').onchange=e=>{state.missing=e.target.checked;render();};
$('geo-depth-task').onclick=()=>{state.tgt=state.cam=state.src;for(const name of ['tgt','cam'])$('geo-'+name).value=state[name];render();};
$('geo-track-task').onclick=()=>{state.cam=0;$('geo-cam').value=0;render();};
function volume(){
 const s=window.D4RTDeepDive.snapshot().state;
 let cells='';
 const addresses=[];
 for(let t=0;t<2;t++){
  const ox=40+t*30,oy=170-t*95;
  cells+=`<g transform="translate(${ox},${oy})"><path d="M0 0 L170 0 L225 -45 L55 -45Z" fill="#f2f3ed" stroke="#bdbfb5"/>`;
  for(let j=1;j<8;j++)cells+=`<path d="M${j*170/8} 0 l55 -45 M${j*55/8} ${-j*45/8} h170" stroke="#d9dbd2" fill="none"/>`;
  const x=170*s.col/16+55*(15-s.row)/16,y=-45*(15-s.row)/16;
  cells+=`<path d="M${x} ${y} h${170/16} l${55/16} ${-45/16} h${-170/16}Z" fill="#687d5d" stroke="#32442e"/><text x="232" y="-20" class="tiny">frame ${s.pair*2+t+1}</text></g>`;
  addresses.push([ox+x+7,oy+y-2]);
 }
 cells+=`<path d="M${addresses[0].join(' ')} L${addresses[1].join(' ')}" stroke="#738367" stroke-dasharray="3 4" fill="none"/>`;
 $('geo-volume').innerHTML=`${cells}<path d="M345 100 H410" fill="none" stroke="#a3aa9b"/><rect x="414" y="55" width="175" height="93" rx="6" fill="#eff1eb" stroke="#c6cbbc"/><text x="428" y="81">2 × 16 × 16 × 3</text><text x="428" y="105">1,536 RGB values</text><text x="428" y="128" class="tiny">shared projection W</text><path d="M589 100 H633" fill="none" stroke="#a3aa9b"/><rect x="637" y="65" width="130" height="75" rx="6" fill="#fafaf7" stroke="#c6cbbc"/><text x="652" y="96">one vector</text><text x="652" y="119" class="tiny">C channels</text>`;
 $('geo-volume-address').textContent=`Highlighted address: column ${s.col}, row ${s.row}; frames ${s.pair*2+1}–${s.pair*2+2}. A coarse grid is drawn for legibility; the real frame contains 16 × 16 patches.`;
}
['dd-time','dd-col','dd-row'].forEach(id=>$(id).addEventListener('input',volume));
window.D4RTGeometry={snapshot:()=>compute(),evaluate:overrides=>compute({...state,...overrides})};
render();volume();
})();
