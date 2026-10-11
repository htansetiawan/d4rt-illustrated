/* Original numerical teaching experiments. No D4RT weights or inference. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const fmt = (x, n = 3) => (Math.abs(x) < 0.5 * 10 ** -n ? 0 : x).toFixed(n);
  const vector = x => `[${x.map(v => fmt(v)).join(', ')}]`;
  const label = (x,y,text,size=11,color='#6b6e68') => `<text x="${x}" y="${y}" font-size="${size}" fill="${color}">${text}</text>`;

  // The same synthetic RGB frames and selection feed the crop and volume views.
  let volume;
  function drawVolume() {
    if (!volume) return;
    const {t,px,py,frames,rgb} = volume;
    const gap = $('volume-explode').checked ? 115 : 35;
    const base = [40,88], next = [40+gap,88+gap*.45];
    const project = (origin,x,y) => [origin[0]+x*.79+y*.40,origin[1]-x*.22+y*.43];
    const corners = origin => [[px*16,py*16],[(px+1)*16,py*16],[(px+1)*16,(py+1)*16],[px*16,(py+1)*16]].map(([x,y])=>project(origin,x,y));
    const a=corners(base), b=corners(next);
    const poly = points => points.map(p=>p.join(',')).join(' ');
    let svg='<title>A two-frame tubelet in a video volume</title><g font-family="system-ui,sans-serif">';
    const frame = (origin,image,index) => {
      let result=`<g transform="matrix(.79 -.22 .40 .43 ${origin[0]} ${origin[1]})"><image href="${image}" width="256" height="256"/><rect width="256" height="256" fill="none" stroke="#a6aba2" stroke-width="1.3"/>`;
      for(let k=16;k<256;k+=16)result+=`<path d="M${k} 0V256 M0 ${k}H256" stroke="#fff" stroke-opacity=".4" stroke-width=".7"/>`;
      result+=`<rect x="${px*16}" y="${py*16}" width="16" height="16" fill="#395342" fill-opacity=".75" stroke="#fff" stroke-width="2.5"/></g>`;
      return result+label(origin[0],origin[1]+125,`FRAME ${String(index+1).padStart(2,'0')}`,10);
    };
    svg+=frame(base,frames[0],t*2);
    for(let k=0;k<4;k++)svg+=`<polygon points="${poly([a[k],a[(k+1)%4],b[(k+1)%4],b[k]])}" fill="#6f9076" fill-opacity=".13" stroke="#6f9076" stroke-width=".7"/>`;
    svg+=frame(next,frames[1],t*2+1);
    for(let k=0;k<4;k++)svg+=`<path d="M${a[k].join(' ')}L${b[k].join(' ')}" stroke="#43624b" stroke-dasharray="3 3"/>`;
    svg+=label(380,27,'2 FRAMES × 16 × 16',11,'#334e3c');
    svg+=label(380,47,'RGB in both slices',10);
    svg+=label(40,277,`Patch row ${py+1}, column ${px+1} · video token ${t*256+py*16+px}`,11);
    svg+=label(40,295,'Illustrative flattening order: time → row → column',9);
    $('tube-volume').innerHTML=svg+'</g>';
    $('tube-volume').setAttribute('aria-label',`Tubelet at row ${py+1}, column ${px+1}, spanning frames ${t*2+1} and ${t*2+2}. Token index ${t*256+py*16+px}.`);
    $('pixel-values').innerHTML=rgb.map((values,i)=>`<div><i style="background:rgb(${values.join(',')})" aria-hidden="true"></i><span>Frame ${t*2+i+1}</span><code>RGB ${vector(values).replaceAll('.000','')}</code></div>`).join('');
  }
  window.addEventListener('d4rt:tubelet',e=>{volume=e.detail;drawVolume();});
  $('volume-explode').addEventListener('change',drawVolume);
  $('tube-t').dispatchEvent(new Event('input'));

  // Explicit, small, hand-chosen features and matrices make every number auditable.
  // These are NOT features produced by D4RT or by the synthetic RGB projection.
  const features=Array.from({length:64},(_,i)=>{
    const time=Math.floor(i/16),row=Math.floor(i%16/4),col=i%4;
    return [col/1.5-1,row/1.5-1,time/1.5-1,Math.sin(i*.63)];
  });
  const wq=[[1,.2],[.3,1],[.6,-.4],[.2,.5]];
  const wk=[[.8,-.2],[.1,.9],[.5,.5],[-.3,.4]];
  const wv=[[.7,.1],[-.2,.6],[.3,-.5],[.4,.2]];
  const project=(x,w)=>w[0].map((_,j)=>x.reduce((v,xi,i)=>v+xi*w[i][j],0));
  const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
  let mask='local', query=37, mathStage=0;
  const mathCopy=[
    ['Make three views of the same features.','Each current encoder token is projected into a query, key and value. For self-attention, all three come from the same token array X. The selected query will compare with every allowed key.','Q = XW_Q;  K = XW_K;  V = XW_V'],
    ['Compare one query with every key.','A dot product measures compatibility. Dividing by √dₕ controls its scale. These raw scores can be negative; they are not probabilities. Here dₕ = 2.','scoreⱼ = (qᵢ · kⱼ) / √2'],
    ['Exclude forbidden keys. Normalize the rest.','Masked scores become −∞ before softmax, giving exactly zero weight. Permitted entries are exponentiated and normalized to sum to one. Change the mask to see both the support and the weights change.','aⱼ = exp(scoreⱼ) / Σₖ∈allowed exp(scoreₖ)'],
    ['Read a weighted mixture of values.','Use those weights to combine value vectors, rather than key vectors. Multiple heads then combine through an output projection. A token-wise MLP processes the updated feature; residual paths preserve previous information.','oᵢ = Σⱼ aⱼ vⱼ']
  ];
  $('math-heatmap').innerHTML=Array.from({length:64},(_,i)=>`<button type="button" data-math-key="${i}" aria-label="Inspect key token ${i}" aria-pressed="false"></button>`).join('');
  function attentionMath() {
    const tq=Math.floor(query/16),sq=query%16;
    const q=project(features[query],wq),keys=features.map(x=>project(x,wk)),values=features.map(x=>project(x,wv));
    const scores=keys.map(k=>dot(q,k)/Math.sqrt(2));
    const allowed=scores.map((_,i)=>mask==='global'||(mask==='local'?Math.floor(i/16)===tq:i%16===sq&&Math.floor(i/16)<=tq));
    const max=Math.max(...scores.filter((_,i)=>allowed[i]));
    const exp=scores.map((x,i)=>allowed[i]?Math.exp(x-max):0),sum=exp.reduce((a,b)=>a+b,0);
    const weights=exp.map(x=>x/sum),out=[0,1].map(j=>weights.reduce((v,w,i)=>v+w*values[i][j],0));
    const inspected=Number($('math-key').value);
    const [title,description,equation]=mathCopy[mathStage];
    $('math-title').textContent=title;$('math-description').textContent=description;$('math-equation').textContent=equation;
    $('math-selection').textContent=`Query token ${query} · time ${tq+1} · row ${Math.floor(sq/4)+1}, column ${sq%4+1}`;
    $('math-vectors').innerHTML=`<div><span>xᵢ · 4 input features</span><code>${vector(features[query])}</code></div><div><span>qᵢ · 2 head features</span><code>${vector(q)}</code></div><div><span>Head output oᵢ</span><code>${vector(out)}</code></div>`;
    $('math-heat-label').textContent=mathStage===1?'SCORES BEFORE MASKING':mathStage===0?'KEY FEATURE MAGNITUDES':'NORMALIZED ATTENTION WEIGHTS';
    $('math-mass').textContent=`${allowed.filter(Boolean).length} readable / 64`;
    const display=mathStage===0?keys.map(k=>Math.hypot(...k)):mathStage===1?scores:weights;
    const lo=Math.min(...display),hi=Math.max(...display);
    document.querySelectorAll('[data-math-key]').forEach((b,i)=>{
      const blocked=mathStage>=2&&!allowed[i];
      const a=hi===lo?.5:(display[i]-lo)/(hi-lo);
      b.style.backgroundColor=`rgb(${Math.round(237-160*a)},${Math.round(240-150*a)},${Math.round(235-158*a)})`;
      b.classList.toggle('masked',blocked);b.classList.toggle('inspected',inspected===i);
      b.setAttribute('aria-pressed',inspected===i);
      b.setAttribute('aria-label',`Key ${i}, time ${Math.floor(i/16)+1}, ${blocked?'masked':`${mathStage<2?'value':'weight'} ${fmt(display[i])}`}`);
      b.title=`Key ${i}: score ${fmt(scores[i])}, weight ${fmt(weights[i],4)}${allowed[i]?'':' (masked)'}`;
    });
    $('math-key-out').value=String(inspected);
    $('math-key-detail').textContent=`k${inspected} = ${vector(keys[inspected])}; v${inspected} = ${vector(values[inspected])}\nq·k / √2 = ${fmt(scores[inspected])} → ${allowed[inspected]?'allowed':'masked (−∞)'} → weight ${fmt(weights[inspected],4)}`;
    $('math-result').textContent=`Σ attention weights = ${fmt(weights.reduce((a,b)=>a+b,0))} · output ${vector(out)} · ${mask==='temporal'?'Genie temporal':mask+' self-attention'}`;
    $('math-panel').setAttribute('aria-labelledby',`math-tab-${mathStage}`);
    document.querySelectorAll('[data-math-stage]').forEach(b=>{const selected=Number(b.dataset.mathStage)===mathStage;b.setAttribute('aria-selected',selected);b.tabIndex=selected?0:-1;});
    document.querySelectorAll('[data-math-mask]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mathMask===mask));
    $('math-next').textContent=mathStage===3?'Start with Q, K, V ↻':'Next operation →';
  }
  window.addEventListener('d4rt:attention',e=>{mask=e.detail.mode;query=e.detail.token;attentionMath();});
  document.querySelectorAll('[data-math-mask]').forEach(b=>b.addEventListener('click',()=>document.querySelector(`[data-attn="${b.dataset.mathMask}"]`).click()));
  document.querySelectorAll('[data-math-stage]').forEach(b=>b.addEventListener('click',()=>{mathStage=Number(b.dataset.mathStage);attentionMath();}));
  document.querySelectorAll('[data-math-key]').forEach(b=>b.addEventListener('click',()=>{$('math-key').value=b.dataset.mathKey;attentionMath();}));
  $('math-next').addEventListener('click',()=>{mathStage=(mathStage+1)%4;attentionMath();});
  $('math-key').addEventListener('input',attentionMath);
  attentionMath();

  // Geometry objective: average over queries of the per-point L1 coordinate sum.
  // delta changes only x, so the mean depth is independent of delta.
  const truth=[[-.3,.2,2],[.4,-.1,4],[.7,.3,6]];
  const signedLog=x=>Math.sign(x)*Math.log1p(Math.abs(x));
  let delta=1.2, steps=0;
  function geometry() {
    const scale=Number($('geometry-scale').value),valid=$('geometry-label').checked;
    const prediction=truth.map((p,i)=>p.map((v,j)=>scale*(v+(i===2&&j===0?delta:0))));
    const mean=prediction.reduce((s,p)=>s+p[2],0)/3;
    let raw=0,normalized=0;
    prediction.forEach((p,i)=>p.forEach((v,j)=>{
      raw+=Math.abs(v-truth[i][j])/3;
      normalized+=Math.abs(signedLog(v/mean)-signedLog(truth[i][j]/4))/3;
    }));
    const x=(.7+delta)/4;
    const gradient=valid&&Math.abs(delta)>1e-12?Math.sign(delta)/(12*(1+Math.abs(x))):0;
    return {scale,valid,prediction,mean,raw,normalized:valid?normalized:0,gradient};
  }
  function drawGeometry() {
    const r=geometry();
    $('geometry-scale-out').value=`${fmt(r.scale,2)}×`;$('geometry-error-out').value=fmt(delta,3);
    $('predicted-mean').textContent=fmt(r.mean);$('raw-error').textContent=fmt(r.raw);$('normalized-error').textContent=r.valid?fmt(r.normalized,4):'Masked';
    const rows=[['Ground truth',.7,4],['Prediction',r.prediction[2][0],r.mean]];
    $('geometry-rows').innerHTML=rows.map(([name,x,mean])=>`<tr><th>${name}</th><td data-label="Raw x">${fmt(x)}</td><td data-label="Normalized x">${fmt(x/mean,4)}</td><td data-label="Signed log">${fmt(signedLog(x/mean),4)}</td></tr>`).join('');
    $('geometry-derivative').textContent=`∂L/∂δ = ${fmt(r.gradient,5)}\nδ′ = δ − 4 · ∂L/∂δ = ${fmt(delta-4*r.gradient,5)}`;
    $('geometry-insight').textContent=!r.valid?'No 3D label means no gradient from this objective. Other labeled objectives could still train the real model.':Math.abs(delta)<1e-12?'Every predicted point has the same geometry up to a common scale. Separate mean-depth normalization removes that scale; the geometric error is zero.':`Only the third point has a local inconsistency after scale normalization. The gradient ${delta>0?'decreases':'increases'} δ. Changing the common scale does not change this normalized error.`;
    $('geometry-steps').textContent=`${steps} illustrative update${steps===1?'':'s'} · full precision retained`;
    $('geometry-step').disabled=!r.valid||Math.abs(delta)<1e-12;
  }
  $('geometry-scale').addEventListener('input',drawGeometry);
  $('geometry-error').addEventListener('input',()=>{delta=Number($('geometry-error').value);steps=0;drawGeometry();});
  $('geometry-label').addEventListener('change',drawGeometry);
  $('geometry-step').addEventListener('click',()=>{const r=geometry();if(!r.valid)return;delta-=4*r.gradient;steps++;$('geometry-error').value=String(delta);drawGeometry();});
  $('geometry-reset').addEventListener('click',()=>{delta=1.2;steps=0;$('geometry-error').value='1.2';$('geometry-scale').value='1';$('geometry-label').checked=true;drawGeometry();});
  drawGeometry();
})();
