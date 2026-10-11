/* Independent teaching diagrams and a tiny differentiable network. No D4RT weights. */
(() => {
'use strict';
const $=id=>document.getElementById(id), nf=n=>n.toLocaleString('en-US');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text=(x,y,s,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${esc(s)}</text>`;
const rect=(x,y,w,h,fill='#fff',stroke='#ddd')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${stroke}"/>`;
const line=(x,y,a,b)=>`<path d="M${x} ${y} L${a} ${b}" class="wire"/>`;
const box=(x,y,w,title,subtitle)=>rect(x,y,w,64)+text(x+14,y+25,title)+text(x+14,y+47,subtitle,'tiny');
const state={stage:0,pair:6,col:6,row:7,mask:'local',backward:false};
const source='https://arxiv.org/html/2512.08924v2';
const steps=[
{name:'Video',title:'Start with a clip, preserve its context.',shape:'V: [B,48,256,256,3]',copy:'The encoder receives RGB video resized to a square. A separate aspect-ratio token retains information about the original frame shape. Both earlier and later observed frames can inform geometry.',change:'Original video → sampled and resized frames, with aspect-ratio metadata retained.',notice:'This is reconstruction from an observed clip. Global attention can use later input frames; it is not an autoregressive next-frame predictor.',code:'video = sample_clip(source_video)\nV = resize_frames(video, height=256, width=256)\naspect_token = embed(original_width / original_height)\n# Training augmentation must also transform geometric labels.',boundary:'48 frames and 256² are the reported training configuration, not a universal statement about every possible inference input.',link:source+'#S2.SS2',label:'Input preparation · §2.2',draw:'video'},
{name:'Tubelets',title:'Two frames at one address become one token.',shape:'[48,256,256,3] → [6144,1536] → [6144,C]',copy:'Cut the video into 2 × 16 × 16 volumes. Flatten the RGB values inside each volume, then use the same learned projection for every volume. The result is a continuous feature vector.',change:'24 time groups × 16 rows × 16 columns = 6,144 video tokens. Each begins with 2 × 16 × 16 × 3 = 1,536 scalar inputs.',notice:'A tubelet does not follow an object. It gathers the same image coordinates in two frames. Tokens are feature vectors, not discrete vocabulary IDs.',code:'patches = rearrange(V,\n  "b (t pt) (h ph) (w pw) c -> b (t h w) (pt ph pw c)",\n  pt=2, ph=16, pw=16)\nX = patches @ Wpatch + bias + position_encoding\n# Wpatch: [1536,C]; append aspect metadata for the encoder.',boundary:'The rearrangement is illustrative pseudocode. Feature width and the exact positional encoding are left symbolic; token counts here exclude metadata.',link:source+'#S4',label:'Tubelet specification · §4',draw:'tubelets'},
{name:'Encoder',title:'Give every token context from other tokens.',shape:'local: [B·24,256,C]  /  global: [B,6144,C]',copy:'The ViT-g encoder interleaves local frame-wise and global self-attention. Inspect the attention domain below: dark cells are addresses the selected query token is allowed to read.',change:'Q, K and V are different projections of the same video-feature stream. Attention mixes addresses; the feed-forward network transforms channels at each address.',notice:'Local means a whole spatial token slice here, not a small sliding window. Global attention connects different locations across time. Metadata routing and the exact layer schedule are not specified by this diagram.',code:'Q = LN(X) @ Wq; K = LN(X) @ Wk; V = LN(X) @ Wv\nA = softmax(Q @ K.T / sqrt(head_width) + mask)\nY = X + multihead_output(A @ V)\nZ = Y + MLP(LN(Y))\n# Repeat through the encoder, interleaving local/global domains.\nF = encoder_output   # shared scene features',boundary:'This is the standard pre-normalization ViT block template, not verified D4RT implementation code. Normalization variant, head count, MLP width and layer order are intentionally unspecified.',link:source+'#S2.SS2',label:'Attention domains · §2.2',draw:'encoder'},
{name:'Point query',title:'Ask where one point is, in one reference frame.',shape:'q = (u,v,t_source,t_target,t_camera)',copy:'The source location identifies a point. Target time selects its physical state. Camera time selects the coordinate system for expressing that state. A separate 9 × 9 source RGB crop carries local appearance.',change:'Coordinate Fourier features, learned time embeddings and an RGB patch embedding form the decoder query token.',notice:'The 9 × 9 source crop is a second input path. It is not one of the encoder’s 2 × 16 × 16 tubelets. Camera time can differ from both source and target time.',code:'q_token = Fourier(u,v)\n        + Esource[t_source] + Etarget[t_target] + Ecamera[t_camera]\n        + embed(source_rgb_crop_9x9)\n# Supervision target, when available:\nP_star = world_to_camera(P_world[t_target], camera[t_camera])\n# P_star is a label, never a decoder input.',boundary:'Embedding addition follows the model overview. The coordinate transform explains label construction when the required geometry is available.',link:source+'#A1',label:'Two input paths · Figure 7',draw:'query'},
{name:'Decoder',title:'Read the scene with independent point queries.',shape:'Q: [B,M,d] · K,V: [B,N,d] → scores: [B,M,N]',copy:'Across eight decoder layers, each query reads the shared scene features through cross-attention. Queries do not attend to each other. Linear heads turn each decoded feature into geometry and auxiliary outputs.',change:'Decoder Q comes from the evolving query. K and V come from F, the encoder’s scene representation. Each query receives its own weighted read.',notice:'At inference, encode the video once and reuse F across batches of queries. “Independent” means no query-to-query attention; all predictions still share the encoder and learned weights.',code:'F = encoder(video)\nfor layer in decoder_layers:\n    Q = project_query(query_features)\n    K,V = project_scene(F)\n    read = softmax(Q @ K.T / sqrt(d)) @ V\n    query_features = decoder_block(query_features, read)\nprediction = linear_heads(query_features)\n# No [M,M] query-self-attention matrix.',boundary:'The diagram isolates cross-attention routing. Residual and feed-forward details are not presented as a recovered implementation.',link:source+'#S2.SS2',label:'Independent decoding · §2.2',draw:'decoder'},
{name:'Learning',title:'Send prediction error back through both paths.',shape:'loss → heads → decoder → F → encoder',copy:'Training compares predictions with available geometric labels. Backpropagation differentiates through the shared scene features, so point-query supervision can improve the video encoder.',change:'Forward computation creates activations; backward computation creates parameter gradients; AdamW updates the parameters. These are three different operations.',notice:'F is reusable while the video and model weights stay fixed. After an encoder weight update, its old features are stale. An occluded point can still have a known 3D target; missing labels and invisibility are different.',code:'F = encoder(video)\npred = decoder(queries, F)\nloss = masked_geometry_and_auxiliary_losses(pred, labels)\nloss.backward()  # includes dL/dF and encoder parameter gradients\nclip_grad_norm(max_norm=10)\nAdamW.step()\n# Next training iteration recomputes F with updated weights.',boundary:'The paper normalizes predicted and target point sets by their respective mean depths before signed-log 3D error. The training miniature below uses MSE instead to isolate the gradient path.',link:source+'#S2.SS3',label:'End-to-end training · §2.3',draw:'learning'}];
function frame(x,y,t,size=115,grid=false){
 let s=rect(x,y,size,size,'#eeefea');
 s+=`<path d="M${x} ${y+size*.75} H${x+size}" stroke="#d6d8d1"/>`;
 const px=x+size*(.2+.55*t/47),py=y+size*(.43+.1*Math.sin(t/9));
 s+=`<circle cx="${px}" cy="${py}" r="${size*.12}" fill="#8c9883"/>`;
 if(grid)for(let k=1;k<16;k++)s+=`<path d="M${x+k*size/16} ${y} V${y+size} M${x} ${y+k*size/16} H${x+size}" stroke="#fff" opacity=".65" stroke-width=".7"/>`;
 return s;
}
function controls(){
 let html='';
 if(state.stage===1)html=`<label>Frame pair <input id="pair" type="range" min="0" max="23" value="${state.pair}"></label><label>Column <input id="col" type="range" min="0" max="15" value="${state.col}"></label><label>Row <input id="row" type="range" min="0" max="15" value="${state.row}"></label><output id="address" class="control-description"></output>`;
 if(state.stage===2)html='<label>Attention domain <select id="domain"><option value="local">D4RT · local</option><option value="global">D4RT · global</option><option value="temporal">Genie · causal temporal only</option><option value="composed">Genie · spatial then temporal</option></select></label><output id="domain-count" class="control-description"></output>';
 if(state.stage===5)html=`<button type="button" id="direction" aria-pressed="${state.backward}">${state.backward?'Show forward pass →':'Follow the gradient ←'}</button><span class="control-description">Targets enter the loss, not the encoder.</span>`;
 $('stage-controls').innerHTML=html;
 if(state.stage===1)for(const k of ['pair','col','row'])$(k).oninput=e=>{state[k]=+e.target.value;draw();};
 if(state.stage===2){$('domain').value=state.mask;$('domain').onchange=e=>{state.mask=e.target.value;draw();};}
 if(state.stage===5)$('direction').onclick=()=>{state.backward=!state.backward;controls();draw();};
}
function draw(){
 const stage=steps[state.stage];let s='<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="#999"/></marker></defs>';
 if(stage.draw==='video'){
  s+=text(34,36,'A SAMPLED CLIP · SYNTHETIC FRAMES','tiny');
  [0,8,16,24,32,47].forEach((t,i)=>{s+=frame(34+i*151,65,t,134)+text(34+i*151,219,`frame ${t+1}`,'tiny');});
  s+=line(480,237,480,266)+box(280,278,400,'RGB clip + aspect-ratio metadata','batch axis is omitted from the drawings');
 }else if(stage.draw==='tubelets'){
  const pair=state.pair*2,index=state.pair*256+state.row*16+state.col;
  [0,1].forEach((dt,i)=>{s+=frame(40+i*205,65,pair+dt,180,true)+`<rect x="${40+i*205+state.col*11.25}" y="${65+state.row*11.25}" width="11.25" height="11.25" fill="#303e26" stroke="#111"/>`+text(40+i*205,270,`Frame ${pair+dt+1} · same (x,y)`,'tiny');});
  s+=line(440,157,485,157)+box(498,126,182,'Gather RGB values','2 × 16 × 16 × 3 = 1536')+line(687,157,725,157)+box(738,126,177,'Learned projection','1536 scalars → C features');
  s+=text(40,320,`Token ${nf(index)} · frames ${pair+1}–${pair+2} · x ${state.col*16}–${state.col*16+15} · y ${state.row*16}–${state.row*16+15}`);
  $('address').textContent=`token ${nf(index)} / 6,144 (zero-based)`;
 }else if(stage.draw==='encoder'){
  const queryT=2,queryS=6,mask=state.mask;let count=0;
  s+=text(30,29,'ILLUSTRATIVE DOMAIN · 4 TIME SLICES × 4 × 4 SPATIAL TOKENS','tiny');
  for(let t=0;t<4;t++){
   s+=text(52+t*225,63,`Token-time ${t+1}`,'tiny');
   for(let p=0;p<16;p++){
    const allowed=mask==='global'||(mask==='local'&&t===queryT)||(mask==='temporal'&&p===queryS&&t<=queryT)||(mask==='composed'&&t<=queryT),q=t===queryT&&p===queryS;
    if(allowed)count++;
    const x=52+t*225+(p%4)*35,y=85+Math.floor(p/4)*35;
    s+=rect(x,y,29,29,q?'#242c22':allowed?'#a9b39f':'#efefec',q?'#242c22':'#deded9');if(q)s+=text(x+8,y+20,'Q','white');
   }
  }
  const descriptions={local:['16 readable addresses','Q reads image positions in its own slice.'],global:['64 readable addresses','Q reads every position and every observed time.'],temporal:['3 readable addresses','The temporal sublayer alone reads this address at times 1–3.'],composed:['48 reachable addresses','A spatial read then a temporal read can carry any past position to Q.']};
  s+=text(52,279,descriptions[mask][0],'svg-title')+text(52,307,descriptions[mask][1])+text(52,340,mask==='composed'?'Two-hop reachability, not a dense attention matrix.':mask==='temporal'?'Genie comparison only; D4RT global layers are not causal.':'40 D4RT encoder layers; exact local/global ordering is left unspecified.','tiny');
  $('domain-count').textContent=`${count} / 64 ${mask==='composed'?'reachable':'readable'} toy addresses`;
 }else if(stage.draw==='query'){
  s+=frame(34,89,9,164)+text(34,279,'Source frame: identify a point','tiny');
  s+=`<rect x="104" y="156" width="20" height="20" fill="none" stroke="#222"/>`;
  s+=line(212,172,260,172)+box(280,50,235,'Fourier(u,v)','continuous source coordinates')+box(280,137,235,'Three time embeddings','source / target / camera')+box(280,224,235,'Source RGB patch','9 × 9 × 3 · local appearance');
  s+='<path d="M520 82 H600 V168 H656 M520 169 H656 M520 256 H600 V168" class="wire"/>';
  s+=box(675,136,245,'One query token','embedded ingredients combined')+text(280,338,'Keep the query location distinct from the target position it asks the model to infer.','tiny');
 }else if(stage.draw==='decoder'){
  s+=box(40,43,218,'Scene memory F','K and V · shared across queries');
  for(let i=0;i<3;i++){
   const y=157+i*68;s+=box(40,y,155,`Point query ${i+1}`,'Q · own query stream')+line(200,y+32,350,y+32)+box(365,y,222,'Cross-attention read','softmax(QKᵀ / √d) V')+line(590,y+32,680,y+32)+box(695,y,224,'Decoded feature → heads','XYZ + auxiliary predictions');
   s+=`<path d="M260 76 H315 V${y+32} H356" class="wire"/>`;
  }
  s+=text(420,73,'8 decoder layers','svg-title')+text(420,99,'No arrows between query rows.','tiny');
 }else{
  const labels=[['Encoder','pixels → scene features F'],['Decoder','queries read F'],['Heads','geometry + auxiliary values'],['Loss','predictions versus labels']];
  labels.forEach(([a,b],i)=>{s+=box(25+i*236,148,202,a,b);if(i<3)s+=state.backward?line(254+i*236,180,231+i*236,180):line(231+i*236,180,254+i*236,180);});
  s+=box(733,31,202,'Ground-truth labels','only available losses apply')+line(834,97,834,141);
  s+=text(30,51,state.backward?'BACKWARD · HOW AN ERROR CHANGES THE MODEL':'FORWARD · HOW A VIDEO BECOMES A PREDICTION','tiny');
  s+=text(30,282,state.backward?'∂L/∂F flows through both K and V back to the encoder.':'All predictions depend on the same encoded scene features.','svg-title');
  s+=text(30,316,state.backward?'Optimizer updates follow gradient computation; they do not happen during the forward pass.':'The input video and query describe the question. The target coordinates supply the answer.','tiny');
 }
 $('diagram').innerHTML=s;$('diagram').setAttribute('aria-label',`${stage.title} ${stage.notice}`);
}
function activate(index,focus=false,updateHash=true){
 state.stage=index;const d=steps[index];
 $('stage-number').textContent=`0${index+1} / 06`;$('stage-title').textContent=d.title;$('stage-shape').textContent=d.shape;
 for(const [id,value] of [['copy',d.copy],['change',d.change],['notice',d.notice],['code',d.code],['boundary',d.boundary]])$('stage-'+id).textContent=value;
 $('stage-link').textContent=d.label+' ↗';$('stage-link').href=d.link;
 document.querySelectorAll('[data-stage]').forEach(b=>{const active=+b.dataset.stage===index;b.setAttribute('aria-selected',active);b.tabIndex=active?0:-1;});
 $('stage-panel').setAttribute('aria-labelledby','tab-'+index);$('prev').disabled=index===0;$('next').disabled=index===5;
 if(focus)$('tab-'+index).focus();controls();draw();
 if(updateHash)history.replaceState(null,'','#stage-'+index);
}
$('stages').innerHTML=steps.map((s,i)=>`<button id="tab-${i}" type="button" data-stage="${i}" role="tab" aria-controls="stage-panel" aria-selected="false"><span>0${i+1}</span>${s.name}</button>`).join('');
document.querySelectorAll('[data-stage]').forEach(b=>{b.onclick=()=>activate(+b.dataset.stage);b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();activate(e.key==='Home'?0:e.key==='End'?5:(state.stage+(e.key==='ArrowRight'?1:5))%6,true);};});
$('prev').onclick=()=>activate(Math.max(0,state.stage-1));$('next').onclick=()=>activate(Math.min(5,state.stage+1));
const initial=location.hash.match(/^#stage-([0-5])$/);activate(initial?+initial[1]:0,false,false);

// Scalar reverse-mode autodiff. Gradients are computed through attention, not painted on.
class Value {
 constructor(data,parents=[],backward=()=>{}){this.data=data;this.grad=0;this.parents=parents;this.backward=backward;}
 add(other){other=asValue(other);const out=new Value(this.data+other.data,[this,other],()=>{this.grad+=out.grad;other.grad+=out.grad;});return out;}
 mul(other){other=asValue(other);const out=new Value(this.data*other.data,[this,other],()=>{this.grad+=other.data*out.grad;other.grad+=this.data*out.grad;});return out;}
 exp(){const out=new Value(Math.exp(this.data),[this],()=>{this.grad+=out.data*out.grad;});return out;}
 inv(){const out=new Value(1/this.data,[this],()=>{this.grad-=out.grad/(this.data*this.data);});return out;}
}
const asValue=x=>x instanceof Value?x:new Value(x),sum=xs=>xs.reduce((a,b)=>a.add(b),new Value(0));
const dot=(a,b)=>sum(a.map((x,i)=>asValue(x).mul(b[i])));
const mm=(a,w)=>a.map(row=>Array.from({length:w[0].length},(_,j)=>dot(row,w.map(r=>r[j]))));
function softmax(xs){const max=Math.max(...xs.map(x=>x.data)),es=xs.map(x=>x.add(-max).exp()),den=sum(es).inv();return es.map(e=>e.mul(den));}
function attention(q,k,v){return q.map(row=>{const weights=softmax(k.map(key=>dot(row,key).mul(.5)));return Array.from({length:v[0].length},(_,j)=>dot(weights,v.map(r=>r[j])));});}
function backprop(loss){const seen=new Set(),nodes=[];function visit(v){if(seen.has(v))return;seen.add(v);v.parents.forEach(visit);nodes.push(v);}visit(loss);nodes.forEach(v=>v.grad=0);loss.grad=1;for(let i=nodes.length-1;i>=0;i--)nodes[i].backward();}
let seed=123,groups,weights,iteration=0,losses=[],lastNorms={},deltaEncoder=0;
function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
function matrix(rows,cols,group){return Array.from({length:rows},()=>Array.from({length:cols},()=>{const v=new Value((random()-.5)*2/Math.sqrt(rows));groups[group].push(v);return v;}));}
const position=t=>.15+.7*t/7;
const pixels=Array.from({length:8},(_,t)=>Array.from({length:4},(_,y)=>Array.from({length:4},(_,x)=>{const d=(x/3-position(t))**2+(y/3-.5)**2,b=Math.exp(-d*12);return [.15+.7*b,.2+.5*b,.25+.3*b];})));
const patches=[];
for(let t=0;t<4;t++)for(let y=0;y<2;y++)for(let x=0;x<2;x++){
 const patch=[];for(let dt=0;dt<2;dt++)for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++)patch.push(...pixels[t*2+dt][y*2+dy][x*2+dx]);patches.push(patch);
}
const queries=[[2/7,1],[6/7,1]],targets=[position(2),position(6)];
function forward(){
 const x=mm(patches,weights.patch),read=attention(mm(x,weights.q),mm(x,weights.k),mm(x,weights.v));
 const f=x.map((row,i)=>row.map((v,j)=>v.add(read[i][j])));
 const r=attention(mm(queries,weights.query),mm(f,weights.key),mm(f,weights.value));
 const pred=mm(r,weights.head).map(row=>row[0].add(weights.bias));
 const loss=sum(pred.map((p,i)=>{const e=p.add(-targets[i]);return e.mul(e);})).mul(.5);
 return {pred,loss};
}
function initialize(){seed=123;groups={encoder:[],query:[],decoder:[],head:[]};weights={patch:matrix(24,4,'encoder'),q:matrix(4,4,'encoder'),k:matrix(4,4,'encoder'),v:matrix(4,4,'encoder'),query:matrix(2,4,'query'),key:matrix(4,4,'decoder'),value:matrix(4,4,'decoder'),head:matrix(4,1,'head'),bias:new Value(0)};groups.head.push(weights.bias);iteration=0;lastNorms={};deltaEncoder=0;losses=[forward().loss.data];renderMini();}
function train(){
 const {loss}=forward();backprop(loss);
 const frozen=$('freeze').checked;lastNorms={};
 for(const [name,params] of Object.entries(groups))lastNorms[name]=Math.sqrt(params.reduce((s,p)=>s+p.grad*p.grad,0));
 const active=Object.entries(groups).filter(([name])=>!(frozen&&name==='encoder')).flatMap(([,p])=>p),norm=Math.sqrt(active.reduce((s,p)=>s+p.grad*p.grad,0)),clip=Math.min(1,1/(norm||1));
 deltaEncoder=frozen?0:.06*clip*lastNorms.encoder;
 for(const p of active)p.data-=.06*clip*p.grad;
 iteration++;losses.push(forward().loss.data);
}
function renderMini(){
 const {pred,loss}=forward();$('parameter-count').textContent=Object.values(groups).flat().length;
 $('predictions').innerHTML=pred.map((p,i)=>`<div class="prediction"><span>Query ${i+1} · target time ${i?7:3}</span><strong>${p.data.toFixed(4)} → ${targets[i].toFixed(4)}</strong></div>`).join('')+`<div class="prediction"><span>Mean squared error</span><strong id="mini-loss">${loss.data.toFixed(6)}</strong></div>`;
 const max=Math.max(...losses,1e-8),n=Math.max(1,losses.length-1),path=losses.map((l,i)=>`${i?'L':'M'}${35+390*i/n} ${135-100*l/max}`).join(' ');
 $('loss-chart').innerHTML=`<path d="M35 28 V135 H430" fill="none" stroke="#ddd"/><path d="${path}" fill="none" stroke="#64745a" stroke-width="2"/>`+text(35,160,'0','tiny')+text(340,160,`${iteration} SGD steps`,'tiny')+text(40,24,`max ${max.toFixed(4)}`,'tiny');
 const maxNorm=Math.max(...Object.values(lastNorms),1e-8),labels={encoder:'Video encoder',query:'Query embedding',decoder:'Cross-attention',head:'Output head'};
 $('gradient-bars').innerHTML=Object.keys(groups).map(name=>{const frozen=name==='encoder'&&$('freeze').checked,norm=lastNorms[name]||0;return `<div class="grad-row ${frozen?'frozen':''}"><span>${labels[name]}${frozen?' · frozen':''}</span><div class="grad-track"><div class="grad-fill" style="width:${norm/maxNorm*100}%"></div></div><b>${lastNorms[name]===undefined?'—':norm.toExponential(2)}</b></div>`;}).join('')+`<p class="small" id="encoder-update">Encoder parameter update norm: ${deltaEncoder.toExponential(3)}${iteration?'':' · no update yet'}</p>`;
 $('train-status').textContent=`Step ${iteration} · deterministic miniature`;
}
$('train-one').onclick=()=>{train();renderMini();};
$('train-many').onclick=async()=>{const buttons=['train-one','train-many','reset','freeze'];buttons.forEach(id=>$(id).disabled=true);try{for(let i=0;i<20;i++){train();if(i%4===3){renderMini();await new Promise(resolve=>requestAnimationFrame(resolve));}}}finally{buttons.forEach(id=>$(id).disabled=false);renderMini();}};
$('reset').onclick=initialize;
$('freeze').onchange=()=>{lastNorms={};deltaEncoder=0;renderMini();};
initialize();
// Read-only diagnostics for reproducible gradient and rendering checks.
window.D4RTTourChecks={gradientCheck(){const {loss}=forward();backprop(loss);const entries=[];for(const [name,params] of Object.entries(groups)){const ranked=[...params].sort((a,b)=>Math.abs(b.grad)-Math.abs(a.grad));const p=ranked[0],analytic=p.grad,old=p.data,eps=1e-5;p.data=old+eps;const plus=forward().loss.data;p.data=old-eps;const minus=forward().loss.data;p.data=old;entries.push({name,analytic,numeric:(plus-minus)/(2*eps)});}return entries;},snapshot(){return {iteration,loss:forward().loss.data,encoder:groups.encoder.map(p=>p.data),head:groups.head.map(p=>p.data),patches:patches.map(p=>p.length),parameterCount:Object.values(groups).flat().length};}};
})();
