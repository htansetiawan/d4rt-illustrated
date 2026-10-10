/* Original teaching diagrams. No trained weights or actual attention maps. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const comma = n => n.toLocaleString('en-US');
  const stages = [
    ['Video', 'THE OBSERVATIONS', 'Start with the entire clip.', 'The input contains RGB frames, not precomputed depth or camera poses. Square resizing standardizes the encoder input; a separate token remembers the original aspect ratio.', 'V: 48 × 256 × 256 × 3', 'The three query times later index original frames, even though the encoder groups frames in pairs.'],
    ['Tubelets', 'PATCHIFICATION', 'Two frames become one temporal slice.', 'At every spatial cell, gather a 16 × 16 region from each of two adjacent frames. A learned projection creates a continuous token, then positional encoding supplies location information.', '24 × 16 × 16 = 6,144 video tokens', 'Each tubelet contains 1,536 RGB scalars. C is the token feature width. The aspect-ratio token is additional.'],
    ['Encoder', 'SELF-ATTENTION · 40 LAYERS', 'Let evidence travel through the clip.', 'The ViT-g encoder interleaves local frame-wise attention and global attention, with MLPs. Local processing connects a time slice; global processing can link any video token to another.', '6,144 spatial-temporal tokens → contextual features', 'The encoder has about 1 billion parameters. Scene geometry is learned through the downstream losses.'],
    ['Scene F', 'THE SHARED REPRESENTATION', 'Keep the expensive result.', 'F is a collection of contextual feature vectors. During inference, the same encoded clip can serve many different point queries without rerunning the encoder.', 'F ∈ ℝᴺˣᶜ', 'F is neither a finished point cloud nor a single scene vector. During training, gradients still pass through F into the encoder.'],
    ['Query', 'A SECOND INPUT PATH', 'Name the point and the question.', 'Combine Fourier features of (u, v), three learned time embeddings, and an embedded 9 × 9 RGB source patch. An MLP prepares the resulting feature for the decoder.', 'q = (u, v, t_src, t_tgt, t_cam) + local appearance', 'Source time identifies the physical point; target time asks when; camera time chooses its coordinate system.'],
    ['Decoder', 'CROSS-ATTENTION · 8 LAYERS', 'Read F on behalf of this point.', 'Query features produce attention queries Q; scene features F produce keys K and values V. Each requested point independently reads the shared representation.', 'B × d queries attend to N scene tokens', 'There is no self-attention across point queries. A 144-million-parameter decoder can process many query batches against one F.'],
    ['Outputs', 'LEARNED PROJECTIONS', 'Return geometry, not a rendered video.', 'Decoder features feed linear prediction heads: XYZ, image UV, visibility, motion, surface normal and confidence. XYZ supplies the shared geometric interface.', 'B × 13 = B × (3 + 2 + 1 + 3 + 3 + 1)', 'Sweep target time for a trajectory. Match all three times and read Z for depth. These are choices of query, not different encoders.'],
    ['Learn / reuse', 'AFTER THE FORWARD PASS', 'Reuse F for another question.', 'At inference there is no optimizer step. Keep F and issue more point queries. For long videos, the paper uses overlapping clips and aligns their reconstructions.', 'New q → D(q, F); E(V) is reused', 'Long-video alignment uses Sim(3) on overlapping predictions. A single fixed-size clip encoder does not provide unlimited context.']
  ];
  let stage = 0, mode = 'inference';
  $('lifetime-steps').innerHTML = stages.map((s,i) => `<button type="button" role="tab" id="life-tab-${i}" aria-controls="life-panel" aria-selected="${i===0}" tabindex="${i===0?0:-1}" data-life="${i}"><small>${String(i+1).padStart(2,'0')}</small>${s[0]}</button>`).join('');
  const svgText = (x,y,s,size=14,fill='#333') => `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}">${s}</text>`;
  function modelMap() {
    const box = (i,x,y,w,h,title,sub) => `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${stage===i?'#e8eeea':'#fafaf9'}" stroke="${stage===i?'#52695c':'#d5d8d3'}" stroke-width="${stage===i?2:1}"/>${svgText(x+14,y+27,title,15)}${svgText(x+14,y+49,sub,11,'#727870')}</g>`;
    const arrow=(d,grad=false)=>`<path d="${d}" fill="none" stroke="${grad?'#917452':'#a4aba2'}" stroke-width="${grad?2:1.4}" ${grad?'stroke-dasharray="5 5"':''} marker-end="url(#${grad?'grad-arrow':'model-arrow'})"/>`;
    let g = `<title>D4RT model anatomy</title><desc>Video tubelets and aspect ratio feed a 40-layer encoder. The scene representation supplies keys and values to an 8-layer decoder. Independent point queries supply queries. Prediction heads return 13 values. Training gradients return through both networks.</desc><defs><marker id="model-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10" fill="#a4aba2"/></marker><marker id="grad-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10" fill="#917452"/></marker></defs><g font-family="system-ui,sans-serif">`;
    g += svgText(26,27,'VIDEO PATH · compute once per clip',11,'#737970');
    g += box(0,26,48,165,73,'RGB video','48 × 256 × 256 × 3')+arrow('M191 84H220');
    g += box(1,225,48,180,73,'Tubelets + position','2 × 16 × 16 · 6,144 tokens')+arrow('M405 84H437');
    g += box(2,442,48,210,105,'ViT-g encoder','local ↔ global self-attention');
    g += svgText(457,131,'40 layers · 1 B parameters',11,'#727870')+arrow('M652 84H687');
    g += box(3,692,48,221,73,'Scene representation F','N context vectors, width C');
    g += svgText(31,153,'Original W/H → aspect-ratio token',11,'#737970')+arrow('M235 149H337V135H435');
    g += svgText(26,217,'POINT PATH · independent queries',11,'#737970');
    g += box(4,26,234,250,103,'Point query + appearance','φ(u,v) + e_src + e_tgt + e_cam');
    g += svgText(40,315,'+ embedded 9 × 9 source RGB patch',11,'#727870')+arrow('M276 273H330');
    g += box(4,335,239,105,67,'MLP','query Q')+arrow('M440 273H478');
    g += box(5,483,234,205,103,'Cross-attention + MLP','8 layers · 144 M parameters');
    g += svgText(498,315,'No query ↔ query attention',11,'#727870');
    g += arrow('M800 122V189H585V231')+svgText(670,180,'F → keys K + values V',11,'#697769');
    g += arrow('M688 273H724')+box(6,729,234,184,103,'Output heads','XYZ · UV · visibility');
    g += svgText(743,315,'motion · normal · confidence',10,'#727870');
    if(mode==='training') {g += arrow('M821 339V380H550V342',true)+arrow('M550 380H307V184H547V158',true)+arrow('M307 380H150V342',true)+svgText(400,405,'Labeled targets → masked losses → gradients → AdamW',13,'#866743');}
    else {g += arrow('M820 340V373H150V342')+svgText(335,395,'New questions reuse F. No weight update.',13,'#737970');}
    return g+'</g>';
  }
  function setLife(i) {
    stage=i;
    document.querySelectorAll('[data-life]').forEach(b=>{const on=Number(b.dataset.life)===stage;b.setAttribute('aria-selected',on);b.tabIndex=on?0:-1;});
    let a=[...stages[i]];
    if(i===7&&mode==='training')a=['Learn / reuse','THE BACKWARD PASS','Teach the entire network.', 'Compare sampled predictions with available ground truth, mask missing labels, and backpropagate the weighted losses. AdamW updates both networks and their learned embeddings.', '∂L/∂D and ∂L/∂E → parameter update', 'The encoder is not frozen just because inference caches F. The next optimizer step recomputes features with the updated parameters.'];
    ['life-kicker','life-title','life-text','life-shape','life-detail'].forEach((id,j)=>$(id).textContent=a[j+1]);
    $('life-panel').setAttribute('aria-labelledby',`life-tab-${i}`);$('life-status').textContent=`${mode==='training'?'Training':'Inference'} · stage ${i+1} / ${stages.length}`;
    $('life-prev').disabled=i===0;$('life-next').disabled=i===stages.length-1;
    $('anatomy-map').innerHTML=modelMap();
  }
  document.querySelectorAll('[data-life]').forEach(b=>b.addEventListener('click',()=>setLife(Number(b.dataset.life))));
  $('life-prev').addEventListener('click',()=>setLife(Math.max(0,stage-1)));$('life-next').addEventListener('click',()=>setLife(Math.min(7,stage+1)));
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(a=>a.setAttribute('aria-pressed',a===b));setLife(stage);}));
  $('diagram-download').addEventListener('click',()=>{const svg=$('anatomy-map').cloneNode(true);svg.setAttribute('xmlns','http://www.w3.org/2000/svg');svg.setAttribute('width','940');svg.setAttribute('height','420');const blob=new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`d4rt-anatomy-${mode}.svg`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});

  // A reproducible, deliberately simple RGB video. Tubelet crops use its actual pixels.
  const raw=[document.createElement('canvas'),document.createElement('canvas')];raw.forEach(c=>{c.width=256;c.height=256;});
  function drawVideo(c,t) {
    const x=c.getContext('2d');x.fillStyle='#ededeb';x.fillRect(0,0,256,256);
    const bg=x.createLinearGradient(0,60,0,256);bg.addColorStop(0,'#f7f7f4');bg.addColorStop(1,'#c6c8c2');x.fillStyle=bg;x.fillRect(0,0,256,256);
    x.strokeStyle='#d0d2cc';x.lineWidth=1;for(let i=0;i<9;i++){x.beginPath();x.moveTo(128,90);x.lineTo(i*52-80,256);x.stroke();}for(let y=136;y<256;y+=28){x.beginPath();x.moveTo(0,y);x.lineTo(256,y);x.stroke();}
    x.fillStyle='#858b80';x.fillRect(164,92,42,57);x.fillStyle='#a4aa9b';x.beginPath();x.moveTo(164,92);x.lineTo(182,77);x.lineTo(224,77);x.lineTo(206,92);x.fill();x.fillStyle='#6f7869';x.beginPath();x.moveTo(206,92);x.lineTo(224,77);x.lineTo(224,132);x.lineTo(206,149);x.fill();
    const cx=40+t*3.4,cy=132-14*Math.sin(t/47*Math.PI*2);x.fillStyle='#535b5040';x.beginPath();x.ellipse(cx+4,cy+33,29,7,0,0,Math.PI*2);x.fill();const ball=x.createRadialGradient(cx-9,cy-10,3,cx,cy,28);ball.addColorStop(0,'#acb9a7');ball.addColorStop(1,'#485c47');x.fillStyle=ball;x.beginPath();x.arc(cx,cy,27,0,Math.PI*2);x.fill();
  }
  function tubelet() {
    const t=Number($('tube-t').value),px=Number($('tube-x').value),py=Number($('tube-y').value);
    [0,1].forEach(i=>{drawVideo(raw[i],t*2+i);const name=i?'b':'a',c=$(`tube-${name}`),ctx=c.getContext('2d');ctx.drawImage(raw[i],0,0);ctx.strokeStyle='#ffffff62';ctx.lineWidth=.6;for(let k=0;k<=16;k++){ctx.beginPath();ctx.moveTo(k*16,0);ctx.lineTo(k*16,256);ctx.moveTo(0,k*16);ctx.lineTo(256,k*16);ctx.stroke();}ctx.fillStyle='#344b351f';ctx.fillRect(px*16,py*16,16,16);ctx.strokeStyle='#243d27';ctx.lineWidth=2;ctx.strokeRect(px*16+1,py*16+1,14,14);ctx.strokeStyle='white';ctx.lineWidth=1;ctx.strokeRect(px*16-1,py*16-1,18,18);$(`crop-${name}`).getContext('2d').drawImage(raw[i],px*16,py*16,16,16,0,0,16,16);$(`tube-frame-${name}`).textContent=`Frame ${String(t*2+i+1).padStart(2,'0')} / 48`;});
    $('tube-t-out').value=`${t+1} / 24`;$('tube-x-out').value=`${px+1} / 16`;$('tube-y-out').value=`${py+1} / 16`;
    $('tube-index').textContent=`Token index = (${t} × 16 + ${py}) × 16 + ${px} = ${comma(t*256+py*16+px)}`;
    $('tube-pixels').textContent=`Zero-based example ordering · rows ${py*16}–${py*16+15}, columns ${px*16}–${px*16+15}`;
  }
  ['tube-t','tube-x','tube-y'].forEach(id=>$(id).addEventListener('input',tubelet));
  ['tube-a','tube-b'].forEach(id=>$(id).addEventListener('click',e=>{const b=e.target.getBoundingClientRect();$('tube-x').value=Math.min(15,Math.floor((e.clientX-b.left)/b.width*16));$('tube-y').value=Math.min(15,Math.floor((e.clientY-b.top)/b.height*16));tubelet();}));

  let attn='local',token=2*16+5;
  $('attention-grids').innerHTML=Array.from({length:4},(_,t)=>`<div class="time-grid"><span>Time group ${t+1}</span><div>${Array.from({length:16},(_,s)=>`<button type="button" data-token="${t*16+s}" aria-label="Time group ${t+1}, row ${Math.floor(s/4)+1}, column ${s%4+1}" aria-pressed="false"></button>`).join('')}</div></div>`).join('');
  function attention() {
    const t=Math.floor(token/16),s=token%16;let count=0;
    document.querySelectorAll('[data-token]').forEach(b=>{const n=Number(b.dataset.token),nt=Math.floor(n/16),ns=n%16,on=attn==='global'||(attn==='local'?nt===t:ns===s&&nt<=t);if(on)count++;b.classList.toggle('readable',on);b.classList.toggle('selected',n===token);b.setAttribute('aria-pressed',n===token);b.title=`${b.getAttribute('aria-label')} · ${n===token?'selected query':on?'readable':'masked'}`;});
    const copy={local:['Local self-attention',`All 16 tokens in time group ${t+1} can contribute to the selected token. Other time groups wait for a global layer to exchange evidence.`],global:['Global self-attention','All 64 tokens can contribute, including later times and different image locations. This is full-clip context for reconstruction, not a causal next-frame predictor.'],temporal:['Genie’s causal temporal attention',`Only spatial address (${Math.floor(s/4)+1}, ${s%4+1}) in time groups 1 through ${t+1} can contribute in this layer. Its separate spatial layer exchanges information within a time step.`]};
    $('attn-title').textContent=copy[attn][0];$('attn-text').textContent=copy[attn][1];$('attn-count').textContent=count;
  }
  document.querySelectorAll('[data-token]').forEach(b=>b.addEventListener('click',()=>{token=Number(b.dataset.token);attention();}));
  document.querySelectorAll('[data-attn]').forEach(b=>b.addEventListener('click',()=>{attn=b.dataset.attn;document.querySelectorAll('[data-attn]').forEach(a=>a.setAttribute('aria-pressed',a===b));attention();}));

  const ingredients={uv:'(u, v) identifies a normalized source-image location. Fourier features encode multiple spatial frequencies; a learned projection puts them into the query feature space. This allows coordinates between encoder patch centers.',src:'t_src selects the source frame in which (u, v) identifies the physical point. Its learned embedding has a different role from target and camera time. The local RGB patch comes from this frame.',tgt:'t_tgt selects the moment whose 3D position you want. Varying this index while preserving point identity lets you trace a trajectory through the observed clip.',cam:'t_cam chooses which camera’s coordinates express XYZ. Changing it can change the output even for a stationary object. It is a separate learned embedding, not a supplied camera pose.',patch:'A 9 × 9 RGB neighborhood around the source point goes through an MLP. Its 243 RGB scalars provide appearance detail that normalized coordinates alone cannot describe. This patch is separate from the encoder’s tubelets.'};
  document.querySelectorAll('[data-ingredient]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-ingredient]').forEach(a=>a.setAttribute('aria-pressed',a===b));$('ingredient-detail').textContent=ingredients[b.dataset.ingredient];}));
  function queryCost(){const b=[1,256,2048,8192,32768][Number($('query-batch').value)];$('query-batch-out').value=comma(b);$('cross-count').textContent=comma(b*6144);}
  $('query-batch').addEventListener('input',queryCost);

  const training=[
    ['Sample','01 / DATA → CLIP','Choose a clip and valid geometric labels.','Mix videos from datasets with different supervision. Select a temporal stride and apply consistent augmentations. Camera geometry, image coordinates and target tracks must agree after any crop or resize.','V: 48 × 256 × 256 × 3 + available targets'],
    ['Encode','02 / FORWARD · VIDEO PATH','Compute the shared features once.','Patchify the augmented clip and run the encoder. The same F supports every query sampled for this clip. VideoMAE initialization is evaluated in the paper; end-to-end training then adapts the representation to geometry.','F = Eθ(V)'],
    ['Query','03 / FORWARD · POINT PATH','Sample 2,048 questions.','Sample locations from available point trajectories, with extra weight on boundaries. Choose source, target and camera times; build query tokens and decode them independently against F.','P̂ᵢ, auxiliaryᵢ = Dφ(qᵢ, F)'],
    ['Compare','04 / TARGETS → ERRORS','Compare only what is labeled.','Normalize each predicted and target point set by its own mean depth, apply the signed-log transform, and measure 3D error. Add the available auxiliary errors. An absent visibility label is masked; it is not a label meaning “invisible.”','L = weighted sum of valid supervised errors'],
    ['Update','05 / BACKWARD · BOTH NETWORKS','Send the lesson all the way back.','Differentiate the loss through the output heads, cross-attention, scene representation and video encoder. AdamW updates θ and φ, including learned query embeddings. The next training step uses new weights.','∇θ L, ∇φ L → clip gradients → AdamW']
  ];
  let train=0;
  $('train-steps').innerHTML=training.map((t,i)=>`<button type="button" role="tab" id="train-tab-${i}" aria-controls="train-panel" aria-selected="${i===0}" tabindex="${i===0?0:-1}" data-train="${i}"><span>${i+1}</span>${t[0]}</button>`).join('');
  function trainGraphic(i){
    let g='<g font-family="system-ui,sans-serif">';
    if(i===0||i===2){for(let t=0;t<3;t++){g+=`<rect x="${25+t*30}" y="${66-t*18}" width="185" height="120" rx="6" fill="#f7f8f5" stroke="#c8cec5"/>`;if(i===0){g+=`<path d="M${43+t*30} ${152-t*18}H${192+t*30}" stroke="#d3dace"/><circle cx="${88+t*39}" cy="${120-t*18}" r="22" fill="#97a68d"/><rect x="${147+t*30}" y="${112-t*18}" width="30" height="40" fill="#c2c8b9"/><path d="M${87+t*39} ${115-t*18}l6 5 -6 5" fill="none" stroke="#edf1e9"/>`; }if(i===2)for(let k=0;k<22;k++){const x=35+t*30+(k*47%161),y=80-t*18+(k*31%91);g+=`<circle cx="${x}" cy="${y}" r="2.8" fill="${k<7?'#485e4b':'#b2b9ad'}"/>`;}}g+=svgText(30,213,i===0?'VIDEO + GEOMETRIC TARGETS':'SAMPLED QUERY LOCATIONS',10,'#788170');}
    if(i===1){for(let r=0;r<6;r++)for(let c=0;c<12;c++)g+=`<rect x="${23+c*22}" y="${33+r*24}" width="15" height="17" rx="2" fill="${(r+c)%3?'#c1cbbd':'#768e76'}"/>`;g+=svgText(65,210,'F · CONTEXTUAL FEATURES',10,'#788170');}
    if(i===3){['3D','2D','Normal','Visibility','Motion'].forEach((l,k)=>{g+=svgText(10,38+k*36,l,11,'#687260');g+=`<rect x="84" y="${24+k*36}" width="${[125,85,105,0,71][k]}" height="13" rx="2" fill="#9caa93"/>`;if(k===3)g+=svgText(84,38+k*36,'NO LABEL → MASKED (EXAMPLE)',10,'#949a90');});}
    if(i===4){[['Encoder θ',23],['Decoder φ',119],['Loss L',215]].forEach(([s,x])=>{g+=`<rect x="${x}" y="73" width="76" height="65" rx="5" fill="#f1f4ef" stroke="#9aa895"/>`+svgText(x+6,110,s,11);});g+='<path d="M253 143V175H60V143M154 175V143" fill="none" stroke="#937958" stroke-width="2"/><path d="M55 150 60 143 65 150M149 150 154 143 159 150" fill="none" stroke="#937958" stroke-width="2"/>';g+=svgText(65,203,'GRADIENTS → BOTH NETWORKS',10,'#937958');}
    return g+'</g>';
  }
  function setTrain(i){train=i;const t=training[i];['train-kicker','train-title','train-detail','train-code'].forEach((id,j)=>$(id).textContent=t[j+1]);document.querySelectorAll('[data-train]').forEach(b=>{const on=Number(b.dataset.train)===i;b.setAttribute('aria-selected',on);b.tabIndex=on?0:-1;});$('train-panel').setAttribute('aria-labelledby',`train-tab-${i}`);$('train-status').textContent=`Stage ${i+1} / 5 · ${t[0]}`;$('train-next').textContent=i===4?'Start next training step ↻':'Next stage →';$('train-diagram').innerHTML=trainGraphic(i);$('train-diagram').setAttribute('aria-label',t[2]);}
  document.querySelectorAll('[data-train]').forEach(b=>b.addEventListener('click',()=>setTrain(Number(b.dataset.train))));$('train-next').addEventListener('click',()=>setTrain((train+1)%5));
  function confidence(){
    const e=Number($('error-value').value),c=Number($('confidence-value').value),value=x=>x*e-.2*Math.log(x),opt=Math.min(1,.2/e),loss=value(c);
    $('error-out').value=e.toFixed(2);$('confidence-out').value=c.toFixed(2);$('loss-total').textContent=loss.toFixed(3);
    $('loss-insight').textContent=`Weighted error: ${(c*e).toFixed(3)}. Log penalty: ${(-.2*Math.log(c)).toFixed(3)}. On this slider’s interval, the minimum is at c ≈ ${opt.toFixed(2)}. Reducing confidence discounts the error, but the log penalty grows as c approaches zero.`;
    const ymax=Math.max(1,e*1.12),sx=x=>43+x*314,sy=y=>190-y/ymax*155;
    let g='<g font-family="system-ui,sans-serif"><path d="M43 20V190H357" fill="none" stroke="#d0d3cb"/>';
    [0,.5,1].forEach(x=>{g+=svgText(sx(x)-4,210,String(x),10,'#838b7d');});g+=svgText(15,25,'ℓ',13,'#626f5c')+svgText(362,194,'c',12,'#626f5c');
    const pts=Array.from({length:100},(_,i)=>{const x=.01+i*.01;return `${i?'L':'M'}${sx(x).toFixed(1)} ${sy(value(x)).toFixed(1)}`;}).join(' ');
    g+=`<path d="${pts}" fill="none" stroke="#8b9c83" stroke-width="2"/><path d="M${sx(c)} 190V${sy(loss)}" stroke="#a6b3a0" stroke-dasharray="3 4"/><circle cx="${sx(opt)}" cy="${sy(value(opt))}" r="4" fill="white" stroke="#768669"/><circle cx="${sx(c)}" cy="${sy(loss)}" r="5" fill="#405840"/>`+svgText(155,24,'● selected   ○ minimum',10,'#738269')+'</g>';
    $('loss-plot').innerHTML=g;$('loss-plot').setAttribute('aria-label',`Loss curve for error ${e.toFixed(2)}. Selected confidence ${c.toFixed(2)} gives loss ${loss.toFixed(3)}; minimum on (0,1] is at ${opt.toFixed(2)}.`);
  }
  ['error-value','confidence-value'].forEach(id=>$(id).addEventListener('input',confidence));
  setLife(0);tubelet();attention();queryCost();setTrain(0);confidence();$('ingredient-detail').textContent=ingredients.uv;
})();
