/* Routing diagrams: exact masks on a reduced lattice, no trained activations. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const order = ['video','patch','local','global','scene','query','decoder','heads','loss'];
  const labels = ['Video','Tubelets','Local','Global','Memory','Query','Decoder','Heads','Training'];
  const state = {node:'video', mode:'train', token:5, comparison:false, group:4, row:7, col:8};
  const ink = '#3f493e', light = '#e7ede2', muted = '#888b83', line = '#cdd0c7';
  const txt = (x,y,s,size=13,fill=ink) => `<text x="${x}" y="${y}" style="font-size:${size}px;fill:${fill}">${s}</text>`;
  const box = (x,y,w,h,fill='#fff',stroke=line) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${stroke}"/>`;
  const arrow = (x1,y1,x2,y2,color=muted,dash='') => `<path d="M${x1},${y1} L${x2},${y2}" fill="none" stroke="${color}" stroke-width="1.4" ${dash ? 'stroke-dasharray="4 4"' : ''} marker-end="url(#cut-arrow)"/>`;
  function block(x,y,w,title,sub,fill='#fff') {return box(x,y,w,51,fill)+txt(x+10,y+21,title)+txt(x+10,y+40,sub,10,muted);}
  function frame(x,y,t,grid=false) {
    let s=box(x,y,104,83,'#f2f2ed');
    s+=`<path d="M${x+6},${y+65} L${x+43},${y+35} L${x+98},${y+61}" fill="none" stroke="#b9bcb3"/>`;
    s+=`<circle cx="${x+23+(t%12)*4}" cy="${y+31}" r="10" fill="#adbba4"/>`;
    if(grid) for(let j=1;j<4;j++) s+=`<path d="M${x+j*26},${y} v83 M${x},${y+j*20.75} h104" stroke="#d1d3cd" fill="none"/>`;
    return s+txt(x,y+101,`frame ${t}`,11);
  }
  function allowed(node, token, comparison=false) {
    return Array.from({length:12},(_,i)=>i).filter(i => comparison
      ? i%4===token%4 && Math.floor(i/4)<=Math.floor(token/4)
      : node==='local' ? Math.floor(i/4)===Math.floor(token/4) : true);
  }
  function attention() {
    const keys = allowed(state.node,state.token,state.comparison);
    let s=txt(8,22,state.comparison?'Genie comparison: causal temporal sublayer':state.node==='local'?'D4RT encoder: within one time group':'D4RT encoder: across the observed clip',13);
    for(let t=0;t<3;t++) {
      s+=txt(25+t*128,53,`time group ${t}`,11,muted);
      s+=box(16+t*128,64,112,90,'#fafaf8');
      for(let p=0;p<4;p++) {
        const i=t*4+p,x=28+t*128+(p%2)*49,y=75+Math.floor(p/2)*35;
        s+=box(x,y,36,25,i===state.token?ink:keys.includes(i)?light:'#fff',keys.includes(i)?'#77816d':line);
        s+=txt(x+8,y+17,`${i}`,12,i===state.token?'#fff':ink);
      }
    }
    const xq=46+Math.floor(state.token/4)*128+(state.token%2)*49;
    s+=txt(10,183,`Q comes from token ${state.token}. Readable K,V tokens:`,12);
    s+=txt(10,205,keys.join(' · '),13);
    s+=block(10,230,160,'softmax(QKᵀ / √d)','weights over allowed keys',light)+arrow(173,255,200,255)+block(207,230,181,'weighted values → MLP','update the querying token');
    // A route, not a depiction of learned attention magnitudes.
    s+=arrow(xq,157,xq,169);
    return {svg:s,caption:state.comparison
      ? 'The same spatial address can read the present and past. Genie places a spatial sublayer before this temporal sublayer; their composition also carries evidence between different addresses.'
      : `${keys.length} of 12 teaching tokens are readable. Dark = selected query; pale green = allowed evidence. These are connectivity masks, not attention weights. At D4RT scale: ${state.node==='local'?'256 spatial tokens per group':'6,144 video tokens'}, excluding metadata.`};
  }
  function draw() {
    let s='',caption='';
    const g=state.group,r=state.row,c=state.col, token=g*256+r*16+c;
    switch(state.node) {
      case 'video':
        s=frame(10,25,0)+frame(143,25,1)+frame(276,25,47);
        s+=txt(129,66,'…',14)+txt(263,66,'…',14);
        s+=arrow(65,139,65,176)+arrow(330,139,330,176);
        s+=block(5,183,180,'Resize for the encoder','48 × 256 × 256 × 3',light);
        s+=block(208,183,186,'Source RGB for queries','9 × 9 crop around a point');
        s+=txt(10,274,'Original aspect ratio → a separate metadata token',12);
        caption='The video enters two paths. The large encoder sees the whole clip; each point query also gets local appearance from its source frame. Frames above are drawn teaching scenes.';
        break;
      case 'patch': {
        s=frame(12,12,g*2,true)+frame(140,35,g*2+1,true);
        const px=12+c/16*104,py=12+r/16*83;
        s+=`<rect x="${px}" y="${py}" width="6.5" height="5.1875" fill="${ink}"/><rect x="${px+128}" y="${py+23}" width="6.5" height="5.1875" fill="${ink}"/>`;
        s+=arrow(px+3,py+6,px+131,py+25);
        s+=txt(266,55,'2 frames',13)+txt(266,78,'× 16 × 16',13)+txt(266,101,'× 3 RGB',13);
        s+=arrow(199,146,199,165)+block(14,170,165,'1,536 pixel values','one flattened tubelet');
        s+=arrow(184,196,208,196)+block(217,170,170,'Learned projection','1,536 → C channels',light);
        s+=txt(14,254,`token index = ${g} × 256 + ${r} × 16 + ${c} = ${token}`,12);
        s+=txt(14,279,'Add position information; repeat at every address.',12);
        caption='Move the tubelet address below. Two patches at a fixed image address make one token. The coarse grid is schematic; the selected region has the correct 16/256 side ratio. It does not follow the moving object.';
        break; }
      case 'local': case 'global': ({svg:s,caption}=attention());break;
      case 'scene':
        s=txt(10,26,'Same token addresses, contextual feature vectors',13);
        for(let i=0;i<6;i++) {
          s+=txt(10,62+i*25,i===5?'…':`token ${i}`,11,muted);
          for(let j=0;j<9;j++) s+=box(95+j*31,46+i*25,25,17,(i+j)%3===0?light:'#f5f5f0');
        }
        s+=txt(95,214,'C feature channels per token →',12);
        s+=block(15,239,370,'F = encoder(video)','Reuse this activation tensor for every query.',light);
        caption='F is a bank of 6,144 contextual vectors, plus metadata handling. Swatches show tensor layout, not numeric features. F is neither a single pooled vector nor a mesh. New weights or video require recomputing it.';
        break;
      case 'query':
        s=block(8,12,178,'Source pixel (u,v)','Fourier coordinate features')+block(214,12,178,'Source RGB crop','9 × 9 × 3 → embedding');
        s+=block(8,92,118,'t_source','which point?')+block(141,92,118,'t_target','when?')+block(274,92,118,'t_camera','which coordinates?');
        s+=arrow(95,65,194,177)+arrow(306,65,206,177);
        for(const x of [67,200,333])s+=arrow(x,146,200,180);
        s+=box(167,183,66,33,light)+txt(181,205,'SUM',13);
        s+=arrow(200,219,200,237)+block(73,241,254,'MLP → initial query feature','All embeddings map to compatible widths.',light);
        caption='The same five-coordinate query can request depth, a track or another camera frame. Camera poses and target XYZ are labels for training; they are not inserted into this forward query.';
        break;
      case 'decoder':
        s=block(8,12,159,'Query features H','Q = H Wq',light)+block(202,12,191,'Scene features F','K = F Wk; V = F Wv');
        s+=arrow(86,65,86,97)+arrow(265,65,265,97);
        s+=block(8,103,280,'A = softmax(QKᵀ / √d)','M queries × N scene tokens',light);
        s+=arrow(346,65,346,186)+arrow(146,157,146,186);
        s+=block(8,192,385,'A V → output projection → query update → MLP','Each query reads F. Queries do not read one another.');
        s+=txt(8,279,'Repeat through 8 decoder layers → per-query features',13);
        caption='Q and K/V come from different streams: this is cross-attention. F stays the shared memory as query features evolve. Diagram omits residual and normalization wiring; widths and head counts remain symbolic.';
        break;
      case 'heads': {
        s=block(70,8,260,'One decoded query feature','shared input to output projections',light);
        const hs=[['XYZ','3'],['Image UV','2'],['Visibility','1'],['Motion','3'],['Normal','3'],['Confidence','1']];
        hs.forEach(([name,n],i)=>{const x=10+(i%3)*132,y=108+Math.floor(i/3)*103;s+=arrow(200,63,x+60,y-6)+block(x,y,120,name,`${n} channels`);});
        caption='3 + 2 + 1 + 3 + 3 + 1 = 13 output channels for each query. These are learned predictions. Camera matrices are recovered from predicted point sets, rather than appearing as an extra head here.';
        break; }
      case 'loss':
        s=block(8,10,176,'Predicted outputs','from the forward model',light)+block(213,10,178,'Available target labels','from training data');
        s+=arrow(90,65,164,93)+arrow(302,65,235,93);
        s+=block(56,98,290,'Masked, weighted geometric losses','normalize → signed log → 3D L1 + auxiliaries');
        s+=arrow(200,151,200,176)+block(71,182,258,'Backward → gradient clipping','AdamW updates learned parameters',light);
        s+=arrow(141,236,96,263,ink,'dash')+arrow(265,236,311,263,ink,'dash');
        s+=txt(10,284,'encoder ← F ← decoder',12)+txt(245,284,'query embeddings',12);
        caption='Training updates both branches end-to-end. Missing labels mask their losses. At inference, this entire label/loss/update path is absent. The detailed training workshop below opens the geometry targets and derivatives.';
        break;
    }
    $('atlas-cutaway-svg').innerHTML=`<defs><marker id="cut-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10" fill="${muted}"/></marker></defs>${s}`;
    $('atlas-cutaway-svg').setAttribute('aria-label',`${labels[order.indexOf(state.node)]} cutaway. ${caption}`);
    $('atlas-cutaway-caption').textContent=caption;
    $('atlas-cutaway-step').textContent=`${order.indexOf(state.node)+1} / ${state.mode==='train'?9:8} · ${labels[order.indexOf(state.node)]}`;
    $('atlas-cutaway-controls').hidden=!['local','global','patch'].includes(state.node);
    $('atlas-route-controls').hidden=!['local','global'].includes(state.node);
    $('atlas-tube-controls').hidden=state.node!=='patch';
    $('atlas-comparison').checked=state.comparison;
    for(const b of document.querySelectorAll('[data-cut-token]')) b.setAttribute('aria-pressed',+b.dataset.cutToken===state.token);
    for(const b of document.querySelectorAll('[data-cut-stage]')) {
      b.setAttribute('aria-current',b.dataset.cutStage===state.node?'step':'false');
      b.disabled=state.mode==='infer' && b.dataset.cutStage==='loss';
    }
    $('atlas-tube-index').textContent=`Frames ${2*g}–${2*g+1} · row ${r} · column ${c} → token ${token.toLocaleString('en-US')}`;
  }
  $('atlas-cutaway-stages').innerHTML=order.map((key,i)=>`<button type="button" data-cut-stage="${key}" aria-label="Stage ${i+1}: ${labels[i]}"><span>${String(i+1).padStart(2,'0')}</span>${labels[i]}</button>`).join('');
  $('atlas-token-buttons').innerHTML=Array.from({length:12},(_,i)=>`<button type="button" data-cut-token="${i}" aria-label="Query token ${i}, time group ${Math.floor(i/4)}, spatial address ${i%4}">${i}</button>`).join('');
  document.querySelectorAll('[data-cut-stage]').forEach(b=>b.addEventListener('click',()=>{
    window.D4RTAtlas.select(b.dataset.cutStage);
    if(window.innerWidth<=650){$('atlas-inspector').focus({preventScroll:true});$('atlas-inspector').scrollIntoView({block:'start',behavior:'instant'});}
  }));
  document.querySelectorAll('[data-cut-token]').forEach(b=>b.addEventListener('click',()=>{state.token=+b.dataset.cutToken;draw();}));
  $('atlas-comparison').addEventListener('change',e=>{state.comparison=e.target.checked;draw();});
  for(const key of ['group','row','col']) $('atlas-tube-'+key).addEventListener('input',e=>{state[key]=+e.target.value;draw();});
  document.addEventListener('atlas-select',e=>{const previous=state.node;Object.assign(state,e.detail);if(previous!==state.node)state.comparison=false;draw();});
  Object.assign(state,window.D4RTAtlas.snapshot());
  draw();
  window.D4RTCutaways={allowed,snapshot:()=>({...state,readable:allowed(state.node,state.token,state.comparison)})};
})();
