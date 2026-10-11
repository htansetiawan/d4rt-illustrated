/* Computed teaching examples. No model weights or inference claims. */
(() => {
'use strict';
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString('en-US');
const state = {pair:6, col:6, row:7, scalar:0, query:6, domain:'local', operation:0};
let patch = [], features = [], block;
// A deterministically rendered moving scene. Pixel values are kept separate
// from the selection outline, so the annotation never enters a tubelet.
function rgb(t,x,y) {
  const ground=y>184, cx=40+3.7*t, cy=123+24*Math.sin(t/10);
  const ball=(x-cx)**2+(y-cy)**2<27**2;
  if(ball) return [Math.round(103+(x-cx)/2), Math.round(120+(y-cy)/3), 93];
  if(x>177 && x<225 && y>112 && y<191) return [155,153,145];
  const shade=Math.round((x+y)/30);
  return ground ? [203-shade,204-shade,195-shade] : [239-shade,238-shade,230-shade];
}
function fullFrame(t,id) {
  const c=$(id), ctx=c.getContext('2d'), image=ctx.createImageData(256,256);
  for(let y=0;y<256;y++)for(let x=0;x<256;x++)image.data.set([...rgb(t,x,y),255],4*(256*y+x));
  ctx.putImageData(image,0,0);
  ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.strokeRect(state.col*16,state.row*16,16,16);
  ctx.strokeStyle='#222';ctx.lineWidth=1.5;ctx.strokeRect(state.col*16,state.row*16,16,16);
}
function updatePatch() {
  patch=[];
  for(let dt=0;dt<2;dt++) {
    const ctx=$('dd-crop-'+(dt?'b':'a')).getContext('2d'), im=ctx.createImageData(16,16);
    for(let y=0;y<16;y++)for(let x=0;x<16;x++) {
      const p=rgb(state.pair*2+dt,state.col*16+x,state.row*16+y);
      im.data.set([...p,255],4*(16*y+x));patch.push(...p);
    }
    ctx.putImageData(im,0,0);
    fullFrame(state.pair*2+dt,'dd-frame-'+(dt?'b':'a'));
    $('dd-frame-'+(dt?'b':'a')+'-label').textContent=`Frame ${state.pair*2+dt+1} · x ${state.col*16}–${state.col*16+15}`;
  }
  features=Array.from({length:6},(_,j)=>patch.reduce((s,p,k)=>s+(p/255)*Math.cos((k+1)*(j+1)*.013),0)/Math.sqrt(1536));
  $('dd-features').innerHTML=features.map((v,j)=>`<span title="Illustrative channel ${j+1}"><i style="height:${Math.min(40,Math.abs(v)*22)}px;opacity:${v<0?.45:.9}"></i>${v.toFixed(2)}</span>`).join('');
  $('dd-time-out').textContent=`${state.pair*2+1}–${state.pair*2+2}`;
  $('dd-col-out').textContent=state.col;$('dd-row-out').textContent=state.row;
  $('dd-token-address').textContent=`Token ${fmt(state.pair*256+state.row*16+state.col)} / 0…6,143`;
  updateScalar();
}
function updateScalar() {
  const k=state.scalar, dt=Math.floor(k/768), within=k%768, y=Math.floor(within/48), x=Math.floor(within%48/3), channel='RGB'[k%3];
  $('dd-scalar-out').textContent=`${fmt(k)} / 1,535`;
  $('dd-pixel-address').textContent=`Frame ${state.pair*2+dt+1}, pixel (${state.col*16+x}, ${state.row*16+y}), channel ${channel}`;
  $('dd-pixel-value').innerHTML=`<strong>${patch[k]}</strong> on [0,255] → <strong>${(patch[k]/255).toFixed(4)}</strong> on [0,1]`;
}
for(const [name,key] of [['time','pair'],['col','col'],['row','row'],['scalar','scalar']])$('dd-'+name).oninput=e=>{state[key]=+e.target.value;if(key==='scalar')updateScalar();else updatePatch();};

const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const weight=(inWidth,outWidth,phase)=>Array.from({length:inWidth},(_,i)=>Array.from({length:outWidth},(_,j)=>Math.sin((i+1)*(j+2)+phase)/Math.sqrt(inWidth)));
const project=(x,w)=>w[0].map((_,j)=>x.reduce((s,v,i)=>s+v*w[i][j],0));
const norm=x=>{const m=x.reduce((a,b)=>a+b,0)/x.length,v=x.reduce((a,b)=>a+(b-m)**2,0)/x.length;return x.map(k=>(k-m)/Math.sqrt(v+1e-5));};
const input=Array.from({length:16},(_,i)=>[Math.sin(i*.6),Math.cos(i*.4),i%4/3,Math.floor(i/4)/3]);
const Wq=weight(4,4,.2),Wk=weight(4,4,1.4),Wv=weight(4,4,2.6),Wo=weight(4,4,3.8),W1=weight(4,8,4.1),W2=weight(8,4,5.3);
function allowed(i,j,domain=state.domain) {
  if(domain==='global')return true;
  if(domain==='local')return Math.floor(i/4)===Math.floor(j/4);
  return i%4===j%4 && Math.floor(j/4)<=Math.floor(i/4);
}
function computeBlock() {
  const u=input.map(norm), q=u.map(x=>project(x,Wq)),k=u.map(x=>project(x,Wk)),v=u.map(x=>project(x,Wv));
  const scores=q.map((x,i)=>k.map((y,j)=>allowed(i,j)?dot(x,y)/2:-Infinity));
  const a=scores.map(row=>{const mx=Math.max(...row),e=row.map(s=>Math.exp(s-mx)),sum=e.reduce((x,y)=>x+y,0);return e.map(s=>s/sum);});
  const read=a.map(row=>Array.from({length:4},(_,c)=>row.reduce((s,p,j)=>s+p*v[j][c],0)));
  const y=read.map((x,i)=>add(input[i],project(x,Wo)));
  const output=y.map(x=>add(x,project(project(norm(x),W1).map(Math.tanh),W2)));
  return {u,q,k,v,scores,a,read,y,output};
}
const operations=[
 ['Normalize','Normalize channels at each token.','The four channels of each token are centered and scaled independently of other tokens. The miniature omits learned normalization gain and bias. This does not mix token addresses.','U = LayerNorm(X)\nX, U: [16,4]','u'],
 ['Q · K · V','Three views of the same feature.','Separate learned projections create queries, keys and values. In encoder self-attention all three originate from the same normalized video-token stream.','Q = U Wq   K = U Wk   V = U Wv\nQ, K, V: [16,4]','q'],
 ['Scores','Compare a question with every allowed key.','Dot products measure compatibility. Divide by √4 = 2 in this one-head example. A mask removes disallowed edges before softmax. Hatching denotes a masked cell.','S[i,j] = dot(Q[i],K[j]) / √4\nS[i,j] = −∞ where masked','q'],
 ['Softmax','Turn scores into a distribution.','Each query row is normalized independently. Its allowed weights sum to one; masked entries are zero. These numbers are computed from the miniature’s features and weights.','A[i,:] = softmax(S[i,:])\nΣⱼ A[i,j] = 1','q'],
 ['Read + skip','Mix values; preserve the incoming token.','Weights choose a blend of value vectors. An output projection maps the read into an update, then the first residual connection adds the original token. The token count is unchanged.','read[i] = Σⱼ A[i,j] V[j]\nY = X + read Wo','y'],
 ['MLP + skip','Transform channels, keep the second residual.','A feed-forward network acts independently on each token. This example expands four channels to eight, uses tanh, then contracts to four. These are teaching choices, not D4RT’s reported MLP specification.','Z = Y + tanh(LayerNorm(Y) W1) W2\n[16,4] → [16,8] → [16,4]','output']
];
$('dd-token-grid').innerHTML=Array.from({length:4},(_,t)=>`<div class="token-slice"><span>Time slice ${t+1}</span><div>${Array.from({length:4},(_,p)=>`<button type="button" data-dd-token="${t*4+p}" aria-pressed="false" aria-label="Select token ${t*4+p}, time ${t+1}, spatial position ${p}">${t*4+p}</button>`).join('')}</div></div>`).join('');
$('dd-block-tabs').innerHTML=operations.map((o,i)=>`<button type="button" id="dd-op-${i}" role="tab" aria-controls="dd-block-panel" aria-selected="false" data-dd-op="${i}">${o[0]}</button>`).join('');
function vector(label,values){return `<div class="vector-row"><b>${label}</b>${values.map(v=>`<span>${v.toFixed(3)}</span>`).join('')}</div>`;}
function renderBlock(){
  block=computeBlock();const i=state.query,op=operations[state.operation];
  document.querySelectorAll('[data-dd-token]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.ddToken===i));
  document.querySelectorAll('[data-dd-op]').forEach(b=>{const active=+b.dataset.ddOp===state.operation;b.setAttribute('aria-selected',active);b.tabIndex=active?0:-1;});
  $('dd-block-panel').setAttribute('aria-labelledby','dd-op-'+state.operation);
  $('dd-op-number').textContent=`OPERATION ${state.operation+1} / 6 · TOKEN ${i}`;
  $('dd-op-title').textContent=op[1];$('dd-op-copy').textContent=op[2];$('dd-op-formula').textContent=op[3];
  let vectors=vector('X',input[i]);
  if(state.operation===1) vectors+=vector('Q',block.q[i])+vector('K',block.k[i])+vector('V',block.v[i]);
  else if(state.operation===4)vectors+=vector('read',block.read[i])+vector('Y',block.y[i]);
  else vectors+=vector(state.operation===5?'Z':state.operation===0?'U':'Q',block[op[4]][i]);
  $('dd-vector-readout').innerHTML=vectors;
  const scores=state.operation===2;
  $('dd-matrix-title').textContent=scores?'SCORES / BEFORE SOFTMAX':'ATTENTION WEIGHTS / A';
  $('dd-mass').textContent=scores?'mask → −∞':`row sum ${block.a[i].reduce((a,b)=>a+b,0).toFixed(4)}`;
  let html='';
  for(let r=0;r<16;r++)for(let c=0;c<16;c++){
    const ok=allowed(r,c),value=scores?block.scores[r][c]:block.a[r][c],shade=Math.round(247-200*(scores?1/(1+Math.exp(-value)):Math.sqrt(value)));
    html+=`<i class="${ok?'':'masked'} ${r===i?'selected-row':''}" ${ok?`style="background:rgb(${shade},${shade},${shade})"`:''} title="query ${r} reads ${c}: ${ok?value.toFixed(5):'masked'}"></i>`;
  }
  $('dd-matrix').innerHTML=html;
  $('dd-matrix').setAttribute('aria-label',`16 by 16 ${scores?'score':'attention'} matrix. Selected token ${i} has ${block.a[i].filter(v=>v>0).length} readable tokens.`);
  $('dd-matrix-caption').textContent=state.domain==='global'?'Global attention has an edge between every pair of displayed tokens, including later observed time slices.':state.domain==='local'?'Four diagonal blocks: a token reads all four spatial positions in its own time slice.':'Genie comparison: the temporal sublayer reads the same spatial address at the current and earlier times. A preceding spatial sublayer already mixes locations within each frame.';
  const total=block.a.flat().filter(v=>v>0).length, count=block.a[i].filter(v=>v>0).length;
  $('dd-edge-count').textContent=`Token ${i}: ${count} readable / 16 · whole matrix: ${total} allowed edges / 256`;
}
document.querySelectorAll('[data-dd-token]').forEach(b=>b.onclick=()=>{state.query=+b.dataset.ddToken;renderBlock();});
document.querySelectorAll('[data-dd-op]').forEach(b=>{
  b.onclick=()=>{state.operation=+b.dataset.ddOp;renderBlock();};
  b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();state.operation=e.key==='Home'?0:e.key==='End'?5:(state.operation+(e.key==='ArrowRight'?1:5))%6;renderBlock();$('dd-op-'+state.operation).focus();};
});
$('dd-domain').onchange=e=>{state.domain=e.target.value;renderBlock();};
document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>{$('tab-'+b.dataset.jump).click();$('tour').scrollIntoView({block:'start'});$('tab-'+b.dataset.jump).focus({preventScroll:true});});
$('genie-route').onclick=()=>{$('tab-2').click();$('domain').value='composed';$('domain').dispatchEvent(new Event('change'));$('tour').scrollIntoView({block:'start'});$('domain').focus({preventScroll:true});};
// Preserve older deep links to the full reference workbench.
const oldSections=['model-lifetime','architecture','execution','query-lab','recipes','efficiency','understanding','gradient-lab','attention-math','tubelet-volume'];
function routeHash(){
  if(oldSections.includes(location.hash.slice(1))){location.replace('reference.html'+location.hash);return;}
  const stage=location.hash.match(/^#stage-([0-5])$/);
  if(stage){$('tab-'+stage[1]).click();$('tour').scrollIntoView({block:'start'});}
}
window.addEventListener('hashchange',routeHash);
routeHash();
window.D4RTDeepDive={snapshot:()=>({state:{...state},patch:[...patch],features:[...features],input:input.map(x=>[...x]),attention:block.a.map(x=>[...x]),output:block.output.map(x=>[...x])}),rgb,allowed};
updatePatch();renderBlock();
})();
