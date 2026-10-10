/* Original teaching geometry. No learned model or reconstruction is run here. */
(() => {
  'use strict';
  const add=(a,b)=>a.map((v,i)=>v+b[i]);
  const sub=(a,b)=>a.map((v,i)=>v-b[i]);
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const unit=a=>{const n=Math.hypot(...a);return a.map(v=>v/n);};
  const camera=(t)=>({position:[-2.1+4.2*t/11,3.0,7.4],target:[0,.35,0]});
  function cameraPoint(p,c){const f=unit(sub(c.target,c.position)),r=unit(cross(f,[0,1,0])),u=cross(r,f),d=sub(p,c.position);return [dot(d,r),-dot(d,u),dot(d,f)];}
  function project(p,c,w,h){const q=cameraPoint(p,c),f=Math.min(w,h)*1.18;return {x:w/2+f*q[0]/q[2],y:h/2+f*q[1]/q[2],z:q[2]};}
  const specs=[];
  for(let a=0;a<=10;a++) for(let b=0;b<22;b++) specs.push({id:`sphere-${a}-${b}`,type:'sphere',local:[.57*Math.sin(a*Math.PI/10)*Math.cos(b*Math.PI/11),.57*Math.cos(a*Math.PI/10),.57*Math.sin(a*Math.PI/10)*Math.sin(b*Math.PI/11)]});
  for(let face=0;face<6;face++)for(let a=0;a<7;a++)for(let b=0;b<7;b++){const p=[0,0,0],axis=Math.floor(face/2);p[axis]=(face%2?1:-1)*.4;p[(axis+1)%3]=-.4+a*.8/6;p[(axis+2)%3]=-.4+b*.8/6;specs.push({id:`cube-${face}-${a}-${b}`,type:'cube',local:p});}
  for(let x=-10;x<=10;x++)for(let z=-8;z<=8;z++)specs.push({id:`floor-${x}-${z}`,type:'floor',local:[x*.25,-.1,z*.25]});
  function position(spec,t){if(spec.type==='floor')return [...spec.local];if(spec.type==='sphere')return add(spec.local,[-1.75+3.5*t/11,.72+.22*Math.sin(t/11*Math.PI),.35*Math.sin(t/11*Math.PI*2)]);const a=t*.045,[x,y,z]=spec.local;return [1.25+x*Math.cos(a)-z*Math.sin(a),.33+y,-1.15+x*Math.sin(a)+z*Math.cos(a)];}
  const presets={sphere:'sphere-4-5',cube:'cube-5-6-6',floor:'floor-8-4'};
  const byId=new Map(specs.map(s=>[s.id,s]));
  function prepare(canvas){const b=canvas.getBoundingClientRect(),w=b.width,h=b.height,dpr=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);return {ctx,w,h};}
  function line(ctx,points,c,w,h,color,width=1,dash=[]){ctx.beginPath();points.forEach((p,i)=>{const q=project(p,c,w,h);if(i===0)ctx.moveTo(q.x,q.y);else ctx.lineTo(q.x,q.y);});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
  function render(canvas,{time=6,selected=presets.sphere,source=false,camTime=0,yaw=.65,path=true,hero=false}={}){
    const {ctx,w,h}=prepare(canvas);if(!w||!h)return [];
    const c=source?camera(camTime):{position:[Math.sin(yaw)*8.5,4.5,Math.cos(yaw)*8.5],target:[0,.25,0]};
    const selectedSpec=byId.get(selected);
    for(let i=-3;i<=3;i++){line(ctx,[[i,-.13,-2.2],[i,-.13,2.2]],c,w,h,'#e8e9e3');line(ctx,[[-3,-.13,i*.7],[3,-.13,i*.7]],c,w,h,'#e8e9e3');}
    if(path&&!source){const pts=Array.from({length:67},(_,i)=>position(selectedSpec,i*11/66));line(ctx,pts,c,w,h,'#8e9b8e',1.3,[3,4]);[0,11].forEach(t=>{const p=project(position(selectedSpec,t),c,w,h);ctx.fillStyle='#929b8d';ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fill();ctx.font='9px ui-monospace,monospace';ctx.fillText(t===0?'t₁':'t₁₂',p.x+8,p.y-5);});}
    const pts=specs.map(s=>({...project(position(s,time),c,w,h),spec:s})).filter(p=>p.z>0).sort((a,b)=>b.z-a.z);
    for(const p of pts){ctx.fillStyle=p.spec.type==='sphere'?'#879385':p.spec.type==='cube'?'#b0b0a5':'#d8d9d0';ctx.globalAlpha=p.spec.type==='floor'?.55:.84;ctx.beginPath();ctx.arc(p.x,p.y,p.spec.type==='floor'?.8:1.35,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;
    const p=project(position(selectedSpec,time),c,w,h);ctx.beginPath();ctx.arc(p.x,p.y,7.5,0,Math.PI*2);ctx.fillStyle='#ffffff';ctx.fill();ctx.lineWidth=1;ctx.strokeStyle='#51614f';ctx.stroke();ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fillStyle='#465b44';ctx.fill();
    if(!source&&!hero){const cc=camera(camTime),cp=project(cc.position,c,w,h);const endpoint=project(cc.target,c,w,h);ctx.strokeStyle='#c4c9bd';ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(cp.x,cp.y);ctx.lineTo(endpoint.x,endpoint.y);ctx.stroke();ctx.setLineDash([]);if(cp.x>5&&cp.x<w-40&&cp.y>10&&cp.y<h-20){ctx.fillStyle='#737b6f';ctx.fillRect(cp.x-4,cp.y-3,8,6);ctx.font='9px ui-monospace,monospace';ctx.fillText(`C${camTime+1}`,cp.x+8,cp.y+3);}
      [['x',[.55,0,0]],['y',[0,-.55,0]],['z',[0,0,.55]]].forEach(([name,v])=>{const origin=[-2.3,.05,1.6];line(ctx,[origin,add(origin,v)],c,w,h,'#939b8d',1);const a=project(add(origin,v),c,w,h);ctx.fillStyle='#7f877b';ctx.font='9px ui-monospace,monospace';ctx.fillText(name,a.x+3,a.y);});
    }
    if(source){ctx.strokeStyle='#e7e7e1';ctx.strokeRect(.5,.5,w-1,h-1);ctx.fillStyle='#92928a';ctx.font='9px ui-monospace,monospace';ctx.fillText('u →',w-36,h-12);ctx.fillText('v ↓',10,17);}
    return pts;
  }
  window.D4RTScene={camera,cameraPoint,project,position,presets,byId,render};
})();
