/* Original teaching experiments. No trained D4RT parameters are loaded. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  if (!$('model-lifetime')) return;
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const txt=(x,y,s,cl='')=>`<text x="${x}" y="${y}" class="${cl}">${esc(s)}</text>`;
  const box=(x,y,w,h,cl='paper')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" class="${cl}"/>`;
  const wire=(x,y,a,b)=>`<path d="M${x} ${y} L${a} ${b}" class="wire"/><path d="M${a-5} ${b-4} L${a} ${b} L${a-5} ${b+4}" class="wire"/>`;
  const fmt=(n,p=3)=>(Math.abs(n)<.5*10**(-p)?0:n).toFixed(p).replace('-','−');
  const state={tab:'pixels',mask:'local',op:'attention',route:'global',train:0,features:[],pixel:0,query:null,target:null,visibility:true};
  function tab(name,focus=false){
    state.tab=name;
    document.querySelectorAll('[data-wb]').forEach(b=>{const on=b.dataset.wb===name;b.setAttribute('aria-selected',on);b.tabIndex=on?0:-1;});
    ['pixels','encoder','labels'].forEach(n=>{$('wb-'+n).hidden=n!==name;});
    if(focus)$('wb-tab-'+name).focus({preventScroll:true});
    if(name==='pixels')patch(); else if(name==='encoder'){block();route();} else labels();
  }
  document.querySelectorAll('[data-wb]').forEach(b=>{
    b.addEventListener('click',()=>tab(b.dataset.wb));
    b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();e.stopPropagation();const names=['pixels','encoder','labels'];let i=names.indexOf(state.tab);i=e.key==='Home'?0:e.key==='End'?2:(i+(e.key==='ArrowRight'?1:2))%3;tab(names[i],true);});
  });
  document.querySelectorAll('[data-wb-next]').forEach(b=>b.addEventListener('click',()=>{tab(b.dataset.wbNext,true);$('model-lifetime').scrollIntoView({block:'start',behavior:'auto'});}));
  function patch(){
    const pair=Number($('tube-t').value),col=Number($('tube-x').value),row=Number($('tube-y').value),j=Number($('wb-scalar').value);
    [['pair',pair,`${2*pair+1}–${2*pair+2}`],['column',col,`${col+1} / 16`],['row',row,`${row+1} / 16`]].forEach(([id,v,label])=>{$('wb-'+id).value=v;$('wb-'+id+'-out').value=label;});
    const data=['a','b'].map(s=>$('crop-'+s).getContext('2d').getImageData(0,0,16,16).data);
    const rgb=Array.from({length:1536},(_,i)=>data[Math.floor(i/768)][Math.floor((i%768)/3)*4+i%3]);
    const x=rgb.map(v=>v/127.5-1),weight=(i,k)=>Math.sin((i+1)*(k+1)*.017)/Math.sqrt(1536);
    state.features=Array.from({length:6},(_,k)=>x.reduce((sum,v,i)=>sum+v*weight(i,k),0));state.pixel=j;
    const slice=Math.floor(j/768),sp=Math.floor((j%768)/3),r=Math.floor(sp/16),c=sp%16,ch=j%3;
    let g=txt(20,24,'RGB TUBELET','label')+txt(275,24,'FLATTEN','label')+txt(492,24,'SHARED W','label')+txt(714,24,'FEATURES','label');
    ['a','b'].forEach((s,i)=>{
      const y=46+i*119;
      g+=box(20,y,190,105)+`<image x="28" y="${y+8}" width="89" height="89" href="${$('crop-'+s).toDataURL()}" style="image-rendering:pixelated"/>`;
      if(slice===i)g+=`<rect x="${28+c*89/16}" y="${y+8+r*89/16}" width="${89/16}" height="${89/16}" fill="none" stroke="#fff" stroke-width="2"/><rect x="${28+c*89/16-1}" y="${y+8+r*89/16-1}" width="${89/16+2}" height="${89/16+2}" fill="none" stroke="#26371c"/>`;
      g+=txt(128,y+31,`frame ${pair*2+i+1}`)+txt(128,y+51,'16 × 16','sub')+txt(128,y+69,'3 channels','sub');
    });
    g+=wire(212,151,260,151)+box(275,47,140,226);
    const start=Math.max(0,Math.min(1529,j-3));
    for(let i=0;i<7;i++){
      const n=start+i,yy=74+i*27;
      if(n===j)g+=`<rect x="282" y="${yy-16}" width="125" height="23" rx="3" fill="#e2eadb"/>`;
      g+=txt(290,yy,`[${String(n).padStart(4,'0')}]`,'sub')+txt(353,yy,String(rgb[n]));
    }
    g+=txt(287,264,'1,536 scalars','sub')+wire(423,151,474,151);
    for(let r0=0;r0<12;r0++)for(let k=0;k<6;k++){
      const v=weight(r0*128,k),dark=Math.round(206-v*1050);
      g+=`<rect x="${492+k*23}" y="${53+r0*16}" width="20" height="13" rx="2" fill="rgb(${dark},${dark+3},${dark-4})"/>`;
    }
    g+=txt(493,264,'[1536, C]','sub')+wire(640,151,698,151);
    state.features.forEach((v,k)=>{const y=70+k*33;g+=box(712,y-19,125,27)+txt(722,y,`${k}: ${fmt(v)}`);});
    g+=txt(712,282,'C = 6 in demo','sub');
    $('wb-patch-svg').innerHTML=g;
    $('wb-scalar-out').value=`j = ${j}`;
    $('wb-pixel-address').textContent=`Frame ${pair*2+slice+1}, pixel (${col*16+c}, ${row*16+r}), channel ${'RGB'[ch]} → ${rgb[j]}.`;
    $('wb-projection').textContent=`z[0] = Σⱼ x[j] W[j,0] = ${fmt(state.features[0],5)}\nselected term: ${fmt(x[j],5)} × ${fmt(weight(j,0),5)} = ${fmt(x[j]*weight(j,0),6)}`;
    $('wb-projection-detail').textContent='All 1,536 inputs contribute to each feature. Moving the pixel inspector changes the highlighted term, not the projected vector. Moving the patch changes the inputs and therefore the vector.';
    $('wb-token-address').textContent=`Token ${pair*256+row*16+col} of 6,144 · zero-based address`;
  }
  [['pair','tube-t'],['column','tube-x'],['row','tube-y']].forEach(([a,b])=>$('wb-'+a).addEventListener('input',e=>{$(b).value=e.target.value;$(b).dispatchEvent(new Event('input',{bubbles:true}));}));
  $('wb-scalar').addEventListener('input',patch);
  window.addEventListener('d4rt:tubelet',patch);
  const operations={
    norm:['Normalize channels within each token.','Layer normalization rescales a token’s feature vector. It does not average across frames or combine different token addresses. Its scale and shift can be learned.','X: [N,C] → LN(X): [N,C]'],
    qkv:['One feature stream, three learned projections.','A query asks what evidence is relevant; a key makes evidence addressable; a value carries the information to mix. Encoder Q, K and V all come from the current video-token features. They use different weight matrices.','Q = X̃WQ; K = X̃WK; V = X̃WV\nper head: [N,dₕ]'],
    attention:['Mix tokens using content-dependent weights.','Compute query–key scores, apply the local or global mask, and normalize each row with softmax. Each output is a weighted sum of readable values. Multiple heads provide separate learned mixtures before an output projection.','A = softmax(QKᵀ / √dₕ + mask)\nMSA(X̃) = concat(A₁V₁,…,AₕVₕ)WO'],
    residual:['Add an update; retain the incoming features.','The attention branch produces a vector at each address. Residual addition combines it with the feature vector already at that same address. Addition requires matching feature widths.','Y = X + MSA(LN(X))\n[N,C] + [N,C] → [N,C]'],
    mlp:['Change features without talking to another token.','After normalization, a feed-forward network transforms each token separately. It typically expands the feature width, applies a nonlinearity, and projects back. The parameters are shared across token addresses within this layer.','MLP(LN(Y)): [N,C] → [N,Cff] → [N,C]\nCff and activation are left unspecified for D4RT.'],
    output:['Pass contextual features to the next layer.','A second residual addition preserves Y alongside its feed-forward update. Deeper layers receive features already influenced by earlier attention. The final encoder features become the decoder’s shared scene memory F.','Z = Y + MLP(LN(Y))\n40-layer encoder → F: [N,C]']
  };
  function block(){
    const op=state.op,active=k=>op===k?'selected':'paper';
    let g=txt(25,28,'ONE REPRESENTATIVE ViT BLOCK','label')+txt(25,53,'Token count preserved; batch and metadata omitted.','sub');
    const steps=[['norm',95,80,'LN'],['qkv',195,110,'Q / K / V'],['attention',325,155,'Multi-head'],['residual',500,48,'+'],['mlp',575,140,'LN → MLP'],['output',735,48,'+']];
    g+=txt(24,164,'X')+wire(42,158,87,158);
    steps.forEach(([k,x,w,label],i)=>{g+=box(x,125,w,66,active(k))+txt(x+12,153,label)+(k==='attention'?txt(x+12,175,'self-attention','sub'):'');if(i<steps.length-1)g+=wire(x+w+2,158,steps[i+1][1]-8,158);});
    g+=wire(788,158,830,158)+txt(828,143,'Z');
    g+='<path d="M57 158 V86 H523 V125 M556 158 V222 H759 V191" class="wire"/>';
    g+=txt(258,78,'residual X','sub')+txt(614,241,'residual Y','sub');
    g+=txt(30,280,state.mask==='local'?'LOCAL ATTENTION DOMAIN':'GLOBAL ATTENTION DOMAIN','label');
    g+=txt(30,306,state.mask==='local'?'24 groups × 256 tokens':'1 sequence × 6,144 tokens')+txt(400,306,state.mask==='local'?'one head: 24 × [256,256] scores':'one head: [6144,6144] scores');
    g+=txt(30,330,state.mask==='local'?'Mix image positions within one token-time slice.':'Mix positions and times across the observed clip.','sub');
    $('wb-block-svg').innerHTML=g;
    const copy=operations[op];$('wb-op-title').textContent=copy[0];$('wb-op-copy').textContent=copy[1];$('wb-op-shape').textContent=copy[2];
    document.querySelectorAll('[data-wb-mask]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.wbMask===state.mask));
  }
  $('wb-layer-bars').innerHTML=Array.from({length:40},(_,i)=>`<i title="Encoder layer ${i+1}"></i>`).join('');
  $('wb-operation').addEventListener('change',e=>{state.op=e.target.value;block();});
  document.querySelectorAll('[data-wb-mask]').forEach(b=>b.addEventListener('click',()=>{state.mask=b.dataset.wbMask;block();}));
  function route(){
    const centers=(t,s)=>[90+t*230+(s%4)*29,62+Math.floor(s/4)*29];
    let g='';
    for(let t=0;t<3;t++){
      g+=txt(79+t*230,25,`TIME ${t+1}`,'label');
      for(let s=0;s<16;s++){
        const [x,y]=centers(t,s),target=t===2&&s===10,source=t===0&&s===1,relay=t===0&&s===10;
        g+=`<rect x="${x-10}" y="${y-10}" width="22" height="22" rx="3" fill="${target?'#4a5c3d':source?'#a0ad92':relay&&state.route==='composed'?'#ced9c3':'#e9ede3'}"/>`;
        if(target||source||(relay&&state.route==='composed'))g+=`<text x="${x-6}" y="${y+5}" style="fill:${target?'#fff':'#394a2b'}">${target?'Q':source?'A':'B'}</text>`;
      }
    }
    const a=centers(0,1),b=centers(0,10),q=centers(2,10);
    if(state.route==='global')g+=`<path d="M${a} Q360 -10 ${q}" class="wire" stroke-dasharray="4 3"/>`+txt(313,47,'one direct edge','sub');
    else if(state.route==='temporal')g+=`<path d="M${b} L${q}" class="wire"/>`+txt(306,112,'same address only','sub');
    else g+=`<path d="M${a} L${b} L${q}" class="wire"/>`+txt(99,202,'A → B: spatial; B → Q: causal temporal','sub');
    const copies={global:'A can contribute directly to Q through D4RT’s global attention. Global attention also permits later-to-earlier evidence: the whole input clip is already observed.',temporal:'In Genie’s temporal sublayer alone, Q reads the same spatial address in times 1–3. A is at a different address, so it has no direct edge to Q.',composed:'Spatial attention at time 1 can first mix A into B. Temporal attention can then carry B to Q at time 3. A different spatial address can therefore influence Q through the composed block. Causality still excludes future frames.'};
    $('wb-route-svg').innerHTML=g;$('wb-path-copy').textContent=copies[state.route];
    document.querySelectorAll('[data-wb-path]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.wbPath===state.route));
  }
  document.querySelectorAll('[data-wb-path]').forEach(b=>b.addEventListener('click',()=>{state.route=b.dataset.wbPath;route();}));
  const point=t=>[-.8+1.6*t/11,.18*Math.sin(Math.PI*t/11),3.2];
  const camera=t=>({x:.55*Math.sin(1.5*t/11),yaw:.12*Math.sin(1.2*t/11)});
  function cameraPoint(p,t){const c=camera(t),x=p[0]-c.x,z=p[2];return [Math.cos(c.yaw)*x-Math.sin(c.yaw)*z,p[1],Math.sin(c.yaw)*x+Math.cos(c.yaw)*z];}
  function project(p,t){const [x,y,z]=cameraPoint(p,t);return [.5+.9*x/z,.5+.9*y/z];}
  function visible(t){const p=point(t),c=camera(t),s=2.4/p[2],x=c.x+s*(p[0]-c.x),y=s*p[1];return !(Math.abs(x)<.22&&Math.abs(y)<.6);}
  function frame(id,t){
    const c=$(id),ctx=c.getContext('2d'),w=c.width,h=c.height,to2=p=>{const uv=project(p,t);return [uv[0]*w,uv[1]*h];};
    ctx.fillStyle='#f5f6f0';ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='#e0e4d8';ctx.lineWidth=1;
    for(let i=-5;i<=5;i++){const a=to2([i*.3,.7,1.6]),b=to2([i*.3,.7,6]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
    for(let z=2;z<=6;z+=.5){ctx.beginPath();ctx.moveTo(...to2([-1.5,.7,z]));ctx.lineTo(...to2([1.5,.7,z]));ctx.stroke();}
    const p=to2(point(t));ctx.fillStyle='#889d72';ctx.beginPath();ctx.ellipse(p[0],p[1],22,20,0,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#c2c9b6';ctx.strokeStyle='#a4ae95';ctx.beginPath();[[-.22,-.6,2.4],[.22,-.6,2.4],[.22,.6,2.4],[-.22,.6,2.4]].forEach((v,i)=>{const p=to2(v);if(i)ctx.lineTo(...p);else ctx.moveTo(...p);});ctx.closePath();ctx.fill();ctx.stroke();
    ctx.strokeStyle='#394d2b';ctx.lineWidth=1.7;ctx.setLineDash(visible(t)?[]:[3,3]);ctx.beginPath();ctx.arc(...p,6,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
    ctx.font='10px monospace';ctx.fillStyle='#72815d';ctx.fillText(visible(t)?'● visible surface point':'◌ hidden point · ground-truth overlay',14,h-15);
    c.setAttribute('aria-label',`Frame ${t+1}, tracked point ${visible(t)?'visible':'occluded behind panel; dashed marker shows ground truth'}.`);
  }
  function labels(){
    const src=Number($('wb-src').value),tgt=Number($('wb-tgt').value),cam=Number($('wb-cam').value);
    ['src','tgt','cam'].forEach(k=>$('wb-'+k+'-out').value=`${Number($('wb-'+k).value)+1} / 12`);
    const uv=project(point(src),src),p=cameraPoint(point(tgt),cam);
    state.query=[...uv,src+1,tgt+1,cam+1];state.target=p;state.visibility=visible(tgt);
    frame('wb-source-view',src);frame('wb-target-view',tgt);
    $('wb-source-label').textContent=`frame ${src+1}`;$('wb-target-label').textContent=`frame ${tgt+1}`;
    $('wb-query-value').textContent=`q = (${uv.map(v=>fmt(v,2)).join(', ')}, ${src+1}, ${tgt+1}, ${cam+1})`;
    $('wb-target-value').textContent=`P* = (${p.map(v=>fmt(v)).join(', ')})`;
    $('wb-visibility-value').textContent=`Target visibility = ${visible(tgt)?1:0} · measured from camera ${tgt+1}`;
    $('wb-target-explain').textContent=(visible(src)?'':'The source point is currently hidden; this is a coordinate-transform exercise with known point identity, not a valid visible-source sampling example. ')+(visible(tgt)?'The target point is visible. ':'The target point is hidden behind the panel, but our synthetic scene still supplies its 3D position. ')+`Changing camera time re-expresses the same physical target in camera ${cam+1}; it does not change its target-time visibility.`;
    train();
  }
  const trainSteps=['Target','Forward','Compare','Backward'];
  $('wb-training-flow').innerHTML=trainSteps.map((n,i)=>`<li><button type="button" data-wb-train="${i}" aria-current="false"><span>0${i+1}</span>${n}</button></li>`).join('');
  function train(){
    const geo=$('wb-has-geometry').checked,vis=$('wb-has-visibility').checked;
    const active=geo||vis;
    const text=[
      ['Keep the answer separate from the question.','The renderer or labeled dataset supplies point trajectories and camera geometry. Those let us construct the supervision target. The network’s forward inputs remain video pixels and the point query.','target = R_camᵀ (P_world(t_tgt) − C_cam)'],
      ['Encode once; answer sampled point queries.','The full clip becomes scene features F. Each query embedding reads F through the decoder and produces its own prediction. The geometric target is not passed into the encoder or the decoder.','F = Eθ(video); prediction = Dφ(query, F)'],
      ['Apply losses only where labels exist.',`3D loss: ${geo?'active':'masked'}. Visibility loss: ${vis?'active, with target '+(state.visibility?1:0):'masked'}. Missing supervision and occlusion are different: a hidden point can have a known 3D label and visibility target zero. The full model also has auxiliary losses; this switch isolates two families.`,`L_demo = ${geo?'geometry(predictions, targets)':'0'} + ${vis?'BCE(v̂, visibility)':'0'}\nGeometry compares normalized point sets across the sampled batch.`],
      [active?'Backpropagate through both learned paths.':'No active label, no supervised gradient.',active?'Loss gradients reach the heads, decoder, query embeddings, and encoder through F. AdamW then updates parameters. A reused scene feature cache would be stale after that update.':'With both demonstrated loss families masked and all other auxiliaries excluded from this exercise, this example supplies zero data-loss gradient. Weight decay is a separate optimizer effect.','loss → heads → decoder\n├ query embeddings\n└ F → encoder\n      → patch projection']
    ][state.train];
    $('wb-training-title').textContent=text[0];$('wb-training-copy').textContent=text[1];$('wb-training-equation').textContent=text[2];
    $('wb-train-next').textContent=['Follow forward pass →','Compare with target →','Follow gradients →','Start with target ↺'][state.train];
    document.querySelectorAll('[data-wb-train]').forEach(b=>b.setAttribute('aria-current',Number(b.dataset.wbTrain)===state.train?'step':'false'));
  }
  ['src','tgt','cam'].forEach(k=>$('wb-'+k).addEventListener('input',labels));
  ['geometry','visibility'].forEach(k=>$('wb-has-'+k).addEventListener('change',train));
  $('wb-train-next').addEventListener('click',()=>{state.train=(state.train+1)%4;train();});
  document.querySelectorAll('[data-wb-train]').forEach(b=>b.addEventListener('click',()=>{state.train=Number(b.dataset.wbTrain);train();}));
  const lr=step=>step<=2500?1e-4*step/2500:1e-6+(1e-4-1e-6)*.5*(1+Math.cos(Math.PI*(step-2500)/497500));
  function schedule(){
    if(!$('wb-train-step'))return;
    const step=Number($('wb-train-step').value),val=lr(step);
    $('wb-train-step-out').value=step.toLocaleString();$('wb-lr-out').textContent=val.toExponential(2);
    const sx=n=>45+690*n/500000,sy=n=>170-140*n/1e-4;
    const pts=[0,500,1000,1500,2000,2500,...Array.from({length:100},(_,i)=>2500+(i+1)*4975)];
    $('wb-lr-plot').innerHTML=`<path d="M45 20V170H735" fill="none" stroke="#d0d6c6"/><path d="${pts.map((s,i)=>`${i?'L':'M'}${sx(s)} ${sy(lr(s))}`).join(' ')}" fill="none" stroke="#798f62" stroke-width="2"/><line x1="${sx(step)}" y1="20" x2="${sx(step)}" y2="170" stroke="#b2bea5" stroke-dasharray="4 4"/><circle cx="${sx(step)}" cy="${sy(val)}" r="5" fill="#40552e"/>`+txt(45,194,'0','sub')+txt(635,194,'500,000 steps','sub')+txt(60,20,'peak 10⁻⁴ after warmup','sub');
    $('wb-schedule-phase').textContent=step<2500?'Warmup: increasing the learning rate. The linear interpolation shown is illustrative; the paper specifies the duration and peak.':'Cosine decay: smaller optimizer steps as training progresses.';
  }
  if($('wb-train-step'))$('wb-train-step').addEventListener('input',schedule);
  window.D4RTWorkbench={snapshot:()=>JSON.parse(JSON.stringify(state)),point,cameraPoint,project,visible,lr};
  patch();block();route();labels();schedule();
})();
