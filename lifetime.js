/* One executable clip, shared with the training miniature in tour.js. */
(() => {
'use strict';
const $=id=>document.getElementById(id), api=window.D4RTMiniature;
if(!$('lifetime')||!api)return;
const state={step:1,t:4,x:1,y:1,query:0};
let current=api.trace(),previous=current;
const stages=[
 ['Video','RGB is the input, geometry is the target.','The eight images below are the actual 4 × 4 RGB tensors fed to this miniature. Select a pixel, then change its red channel. Selection outlines are drawn afterward; they never become model inputs.','V: [8,4,4,3] → 384 RGB scalars','The reported D4RT training input is 48 × 256 × 256 × 3, plus original-aspect-ratio metadata.','pixels[t,y,x,channel]','https://arxiv.org/html/2512.08924v2#S2.SS2'],
 ['Tubelets','A volume of pixels becomes one continuous vector.','Gather two consecutive 2 × 2 RGB patches at a fixed image address, flatten their 24 scalars, and apply the same learned projection to every tubelet. Add a small fixed position feature. Moving objects can cross these fixed addresses.','[8,4,4,3] → [16,24] → [16,4]','D4RT: 2 × 16 × 16 × 3 = 1,536 values per tubelet; (48/2) × (256/16)² = 6,144 video tokens. Aspect-ratio metadata is additional.','X = patches @ Wpatch + position\nWpatch: [24,4]','https://arxiv.org/html/2512.08924v2#S4'],
 ['Local encoder','First, exchange evidence inside a time slice.','Each token creates a query, key and value. It reads four tokens in its own frame-pair group. The attention-weighted values are added back to its incoming feature. The count of tokens stays sixteen.','Q,K,V: [16,4] · A_local: [16,16]','D4RT’s frame-local operation mixes spatial tokens within a temporal group; global layers mix groups. The paper reports 40 total encoder layers.','Q = X Wq; K = X Wk; V = X Wv\nA = softmax(mask(QKᵀ / √4))\nL = X + A V','https://arxiv.org/html/2512.08924v2#S2.SS2'],
 ['Global encoder','Now, every token can read the whole clip.','The second attention block reads L, the output of the local block. Every token can read all sixteen addresses, including later observed frames. Its output F is the shared scene representation that the decoder reads.','L: [16,4] → F: [16,4]','D4RT uses a 40-layer ViT-g encoder (~1B parameters). Global attention spans the observed clip; it is not Genie’s causal temporal mask.','Q = L Wq; K = L Wk; V = L Wv\nF = L + softmax(QKᵀ / √4) V','https://arxiv.org/html/2512.08924v2#S2.SS2'],
 ['Decoder','A point query reads the scene through cross-attention.','Here Q comes from the requested target time and source RGB; K and V come from F. Two independent query rows read the same scene. Change query 2’s time: F and query 1’s output stay exactly the same.','Q: [2,4] · K,V: [16,4] · A_cross: [2,16]','D4RT uses 8 decoder layers (~144M parameters). The full query combines Fourier(u,v), three time embeddings and a 9 × 9 source RGB crop. No query-to-query self-attention.','Q = query_features @ Wquery\nK = F Wkey; V = F Wvalue\nR = softmax(QKᵀ / √4) V','https://arxiv.org/html/2512.08924v2#S2.SS2'],
 ['Prediction & loss','The label only enters after prediction.','The miniature predicts a single x coordinate per query. Its target comes from the known path of the synthetic point. A mean-squared error trains this demonstration. D4RT instead has geometric and auxiliary objectives, detailed below.','[2,4] → [2,1] → scalar loss','D4RT projects each decoded query into XYZ, UV, visibility, displacement, surface normal and confidence. Each loss uses its available ground-truth labels.','prediction = R @ Whead + bias\nloss = mean((prediction − target)²)','https://arxiv.org/html/2512.08924v2#S2.SS3'],
 ['Backward & update','A point-level error can teach the video encoder.','Reverse-mode autodiff carries dL/dF back through the global block, the local block and tubelet projection. Inspect a pixel’s derivative below. Pixels are fixed input data; SGD updates the learned parameters. Freezing the encoder stops its updates while keeping its forward path.','loss → gradients → 249 parameter values','D4RT trains end-to-end with AdamW: 500k steps, 2,500-step warmup, cosine decay, gradient clipping. This small model uses clipped SGD to make one update inspectable.','loss.backward()\nθ ← θ − 0.06 · clipped_gradient\n# recompute F after encoder weights change','https://arxiv.org/html/2512.08924v2#A1']
];
const token=()=>Math.floor(state.t/2)*4+Math.floor(state.y/2)*2+Math.floor(state.x/2);
const f=n=>Number(n).toFixed(5), vec=a=>'['+a.map(f).join(', ')+']';
const norm=a=>Math.sqrt(a.flat(Infinity).reduce((s,v)=>s+v*v,0));
const diff=(a,b)=>a.map((v,i)=>Array.isArray(v)?diff(v,b[i]):v-b[i]);
const changed=(a,b)=>a.filter((row,i)=>norm(diff(row,b[i]))>1e-12).length;
$('life-steps').innerHTML=stages.map((s,i)=>`<button type="button" data-life-step="${i}" aria-pressed="false"><small>0${i+1}</small>${s[0]}</button>`).join('');
$('life-film').innerHTML=Array.from({length:8},(_,t)=>`<figure><canvas id="life-frame-${t}" width="96" height="96" tabindex="0" role="img" aria-label="Frame ${t}; click to select a pixel, or use arrow keys when focused"></canvas><figcaption>frame ${t}</figcaption></figure>`).join('');
function drawFrames(){
 for(let t=0;t<8;t++){
  const ctx=$('life-frame-'+t).getContext('2d');
  for(let y=0;y<4;y++)for(let x=0;x<4;x++){
   ctx.fillStyle=`rgb(${current.pixels[t][y][x].map(v=>Math.round(v*255)).join(',')})`;ctx.fillRect(x*24,y*24,24,24);
   ctx.strokeStyle='#ffffff55';ctx.lineWidth=1;ctx.strokeRect(x*24,y*24,24,24);
  }
  if(Math.floor(t/2)===Math.floor(state.t/2)){
   ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.strokeRect(Math.floor(state.x/2)*48+2,Math.floor(state.y/2)*48+2,44,44);
  }
  if(t===state.t){ctx.strokeStyle='#222';ctx.lineWidth=3;ctx.strokeRect(state.x*24+4,state.y*24+4,16,16);}
 }
}
function vectorRows(rows){return rows.map(([label,values])=>`<div class="life-vector"><b>${label}</b><code>${vec(values)}</code></div>`).join('');}
function attentionBars(a,kind){return `<p class="eyebrow">${kind} · ROW SUM ${a.reduce((s,v)=>s+v,0).toFixed(6)}</p><div class="life-attention">${a.map((v,i)=>`<div title="Token ${i}: ${v.toFixed(8)}" class="${v===0?'masked':''}"><span>${(v*100).toFixed(1)}%</span><i style="height:${Math.max(1,Math.sqrt(v)*105)}px"></i><b>${i}</b></div>`).join('')}</div>`;}
function render(){
 const i=token(),s=stages[state.step],q=state.query;
 drawFrames();
 document.querySelectorAll('[data-life-step]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.lifeStep===state.step));
 $('life-title').textContent=s[1];$('life-copy').textContent=s[2];$('life-shape').textContent=s[3];$('life-paper').textContent=s[4];$('life-code').textContent=s[5];$('life-source').href=s[6];
 $('life-prev').disabled=state.step===0;$('life-next').disabled=state.step===6;
 $('life-pixel').textContent=`frame ${state.t} · (x=${state.x}, y=${state.y}) · token ${i}`;
 $('life-red').value=current.pixels[state.t][state.y][state.x][0];$('life-red-value').textContent=f(current.pixels[state.t][state.y][state.x][0]);
 $('life-time').value=current.queryTimes[1];$('life-time-value').textContent=current.queryTimes[1];
 $('life-freeze').checked=current.freeze;
 ['life-red','life-time','life-freeze','life-train','life-reset'].forEach(id=>$(id).disabled=current.busy);
 $('life-status').textContent=`${current.parameterCount} parameters · ${current.iteration} updates · MSE ${current.loss.toFixed(6)}`;
 const values=state.step<=1?current.x:state.step===2?current.l:current.f;
 const before=state.step<=1?previous.x:state.step===2?previous.l:previous.f;
 $('life-bank-label').textContent=state.step<=1?'X · projected tokens':state.step===2?'L · after local attention':'F · after global attention';
 $('life-tokens').innerHTML=values.map((row,j)=>`<button type="button" data-life-token="${j}" aria-pressed="${i===j}" aria-label="Inspect token ${j} at temporal group ${Math.floor(j/4)}, spatial cell ${j%4}"><b>${j}</b><span class="life-channels">${row.map(v=>`<i style="background:${v>=0?'#65795a':'#a19587'};opacity:${.18+.82*Math.min(1,Math.abs(v))}" title="${v.toFixed(7)}"></i>`).join('')}</span><small>${norm(diff(row,before[j]))>1e-12?'changed':'—'}</small></button>`).join('');
 document.querySelectorAll('[data-life-token]').forEach(b=>b.onclick=()=>{const n=+b.dataset.lifeToken;state.t=Math.floor(n/4)*2;state.y=Math.floor((n%4)/2)*2;state.x=(n%2)*2;render();});
 let detail='';
 if(state.step===0)detail=vectorRows([['Selected RGB',current.pixels[state.t][state.y][state.x]]])+`<p>The source appearance sample is the RGB pixel at frame 0, x=0, y=2. It takes a second route directly into the query embedding.</p>`;
 if(state.step===1){detail=`<p class="eyebrow">TOKEN ${i} · 24 RAW RGB VALUES, IN ORDER</p><div class="life-scalars">${current.patches[i].map((v,j)=>`<span title="scalar ${j}; offset frame ${Math.floor(j/12)}, row ${Math.floor(j%12/6)}, column ${Math.floor(j%6/3)}, ${'RGB'[j%3]}">${v.toFixed(3)}</span>`).join('')}</div>`+vectorRows([['Projection',current.projected[i]],['+ position → X',current.x[i]]]);}
 if(state.step===2||state.step===3){const a=state.step===2?current.local:current.global;detail=attentionBars(a.a[i],`TOKEN ${i} READS TOKENS 0–15`)+vectorRows([['Q',a.q[i]],['K (same token)',a.k[i]],['V (same token)',a.v[i]],['Weighted read',a.read[i]],['Residual output',state.step===2?current.l[i]:current.f[i]]]);}
 if(state.step===4){detail=`<label class="life-query-choice">Inspect query <select id="life-query"><option value="0" ${q===0?'selected':''}>1 · time ${current.queryTimes[0]}</option><option value="1" ${q===1?'selected':''}>2 · time ${current.queryTimes[1]}</option></select></label>`+attentionBars(current.cross.a[q],`QUERY ${q+1} READS F`)+vectorRows([['Q · from query',current.cross.q[q]],['K · selected F token',current.cross.k[i]],['V · selected F token',current.cross.v[i]],['Read → head',current.cross.read[q]]]);}
 if(state.step===5)detail=`<div class="life-predictions">${current.pred.map((p,j)=>`<div><span>Query ${j+1} · time ${current.queryTimes[j]}</span><b>${f(p)}</b><small>target ${f(current.targets[j])} · squared error ${f((p-current.targets[j])**2)}</small></div>`).join('')}</div><p>Mean squared error: <strong>${current.loss.toFixed(7)}</strong></p><p>The targets never enter X, F or the query vector. An untrained network can predict outside the target range.</p>`;
 if(state.step===6){const gs=current.gradients.groups,max=Math.max(...Object.values(gs),1e-12);detail=`<p>Selected input red-channel derivative<br><code>∂L/∂pixel = ${current.gradients.pixels[state.t][state.y][state.x][0].toExponential(6)}</code></p>`+Object.entries(gs).map(([name,v])=>`<div class="life-gradient"><span>${name}${name==='encoder'&&current.freeze?' · frozen':''}</span><i><b style="width:${100*v/max}%"></b></i><code>${v.toExponential(3)}</code></div>`).join('')+`<p class="small">These are derivatives at the current weights. The previous encoder update norm was ${current.deltaEncoder.toExponential(3)}. Frozen parameters can have nonzero mathematical derivatives while their updates remain zero.</p>`;}
 $('life-detail').innerHTML=detail;
 if($('life-query'))$('life-query').onchange=e=>{state.query=+e.target.value;render();};
 $('life-changes').innerHTML=`<span>Since last edit or update</span><b>X: ${changed(current.x,previous.x)}/16 changed</b><b>L: ${changed(current.l,previous.l)}/16</b><b>F: ${changed(current.f,previous.f)}/16</b><b>Δ prediction 1: ${(current.pred[0]-previous.pred[0]).toExponential(2)}</b>`;
}
for(let t=0;t<8;t++){
 const c=$('life-frame-'+t);
 c.onclick=e=>{const b=c.getBoundingClientRect();state.t=t;state.x=Math.min(3,Math.floor((e.clientX-b.left)/b.width*4));state.y=Math.min(3,Math.floor((e.clientY-b.top)/b.height*4));render();};
 c.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();state.t=t;state.x=Math.max(0,Math.min(3,state.x+(e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0)));state.y=Math.max(0,Math.min(3,state.y+(e.key==='ArrowDown'?1:e.key==='ArrowUp'?-1:0)));render();};
}
document.querySelectorAll('[data-life-step]').forEach(b=>b.onclick=()=>{state.step=+b.dataset.lifeStep;render();});
$('life-next').onclick=()=>{state.step=Math.min(6,state.step+1);render();};$('life-prev').onclick=()=>{state.step=Math.max(0,state.step-1);render();};
$('life-red').oninput=e=>api.editPixel(state.t,state.y,state.x,+e.target.value);
$('life-time').oninput=e=>api.setQueryTime(+e.target.value);
$('life-train').onclick=()=>api.train();$('life-reset').onclick=()=>api.reset();$('life-freeze').onchange=e=>api.freeze(e.target.checked);
document.addEventListener('d4rt-miniature-updated',()=>{previous=current;current=api.trace();render();});
window.D4RTLifetime={snapshot:()=>({step:state.step,token:token(),current,previous}),selectPixel(t,y,x){state.t=t;state.y=y;state.x=x;render();}};
render();
})();
