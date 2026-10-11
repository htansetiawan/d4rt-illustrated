/* Source-grounded data-flow drawings; these are not learned D4RT activations. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const svg = $('cutaway-svg');
  if (!svg) return;
  let state = {node:'video', mode:'training', index:0, code:''};
  const escape = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text = (x,y,s,c='') => `<text x="${x}" y="${y}" class="${c}">${escape(s)}</text>`;
  const rect = (x,y,w,h,c='group') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" class="${c}"/>`;
  const line = (x,y,a,b,back=false) => `<path d="M${x},${y} L${a},${b}" class="wire ${back?'gradient':''}" marker-end="url(#cutaway-arrow${back?'-back':''})"/>`;
  const box = (x,y,w,h,title,sub='',active=false) => rect(x,y,w,h,active?'active':'group')+text(x+13,y+25,title,'title')+(sub?text(x+13,y+46,sub,'small'):'');
  const grid = (x,y,rows,cols,size,selected=-1) => Array.from({length:rows*cols},(_,i)=>`<rect x="${x+(i%cols)*size}" y="${y+Math.floor(i/cols)*size}" width="${size-3}" height="${size-3}" rx="2" class="${i===selected?'ink':'faint'}"/>`).join('');
  const matrix = (x,y,label,sub,rows=4,cols=5) => text(x,y,label)+grid(x,y+10,rows,cols,11)+text(x,y+rows*11+34,sub,'small');
  const defs = '<defs><marker id="cutaway-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L8 4L0 8" fill="none" stroke="#858f7d"/></marker><marker id="cutaway-arrow-back" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#8d715a"/></marker></defs>';
  function frames(x,y) {
    let s='';
    for(let i=2;i>=0;i--) s+=rect(x+i*15,y-i*15,110,100)+grid(x+10+i*15,y+10-i*15,5,5,18,8+i);
    return s;
  }
  function render(){
    let d='',title='',caption='';
    const n=state.node, back=['backward','update'].includes(n);
    if(n==='video'){
      title='Start with pixels, preserve the geometry';
      d=frames(30,75)+text(30,205,'raw video')+line(180,120,245,120)+box(250,65,210,100,'Sample + resize','48 × 256 × 256 × 3',true)+line(460,115,520,115)+box(525,65,335,100,'RGB tensor → encoder','original W/H → separate metadata token');
      d+=line(353,167,353,217)+box(255,220,530,55,'Training only: transform labels with the crop');
      caption='The clip has space, time and color axes. Labels remain a separate training branch; camera geometry and point coordinates must agree with any spatial augmentation.';
    } else if(n==='patch'){
      title='Patchification groups pixels; projection learns features';
      for(let i=0;i<2;i++){
        const canvas=$('crop-'+(i?'b':'a'));d+=rect(25+i*78,60,70,70);
        if(canvas)d+=`<image x="${30+i*78}" y="65" width="60" height="60" href="${canvas.toDataURL()}" style="image-rendering:pixelated"/>`;
      }
      d+=text(27,154,'2 × 16 × 16 × RGB')+text(27,176,'one fixed image address','small')+line(181,98,230,98);
      d+=matrix(238,66,'flatten','1,536 scalars',7,3)+line(285,105,345,105)+box(350,65,210,95,'Learned projection','[1536, C] weights + bias',true)+line(560,105,613,105)+matrix(623,70,'one token zᵢ','C continuous features',1,9);
      d+=box(350,210,210,56,'+ position encoding')+line(560,237,670,161)+text(620,214,'separate aspect-ratio token','small')+text(26,256,'Repeat over 24 × 16 × 16 locations → 6,144 video tokens.');
      caption='The two crops are linked to the tubelet selector below. The same projection is shared across all positions. Grouping 2 frames is not object tracking: an object may leave this patch.';
    } else if(n==='encoder'){
      const global=state.code.includes('global');title=global?'Global self-attention changes the available evidence':'Local self-attention mixes a temporal slice';
      for(let t=0;t<3;t++){const x=25+t*133;d+=rect(x,42,113,128)+text(x+9,61,'slice '+(t+1),'small')+grid(x+15,74,4,4,22,t===1?5:-1);}
      d+=text(28,25,'3 of 24 groups; grids reduced for clarity','small');
      d+=line(416,110,455,110)+box(460,45,180,120,'Q, K, V ← X',global?'flatten: [6144, C]':'reshape: [24, 256, C]',true)+line(640,109,676,109)+box(681,45,193,120,'Attention → MLP','same token count');
      if(global){for(const x of [83,216,349])d+=line(x,173,543,226);d+=text(369,253,'All slices share one attention domain.');}
      else {d+=text(28,238,'24 separate sequences → 256 keys per video-token query.');d+=text(28,260,'No cross-slice attention edges in this local operation.','small');}
      caption=global?'One global layer can connect different image locations in different time groups, including later frames. This is full-clip context, unlike Genie’s causal temporal stage. The paper reports 40 encoder layers with interleaved local/global processing.':'A 2-frame tubelet supplies a temporal slice in this token-grid view. Each local attention operation mixes spatial tokens inside its group. The MLP processes each token’s features; it does not create new token-to-token edges.';
    } else if(n==='scene'||n==='reuse'){
      title='A reusable scene representation, not a list of 3D points';
      d=box(30,80,215,115,'Encoder output F','[N, C] scene features',true)+line(245,120,330,120)+matrix(342,74,'cached activations','same clip, same weights',7,8)+line(445,98,535,59)+line(445,155,535,201)+box(542,30,305,65,'Query batch A','track selected points')+box(542,173,305,65,'Query batch B','depth or dense reconstruction');
      caption='At inference, encode once and reuse F for new query batches. During end-to-end training, gradients still flow through F into the encoder. A weight update makes a previously cached F stale.';
    } else if(n==='query'){
      title='A point query is assembled along a second input path';
      const labels=[['(u, v)','Fourier + FC'],['t_src','source embedding'],['t_tgt','target embedding'],['t_cam','camera embedding'],['9 × 9 RGB','local patch MLP']];
      labels.forEach(([a,b],i)=>{d+=box(20+i*175,27,160,65,a,b);d+=line(100+i*175,94,440,156);});
      d+=box(323,161,239,63,'Sum → MLP','one query feature h',true)+line(562,193,638,193)+box(643,161,235,63,'Q = h Wq','decoder input');
      caption='Three time indices have different jobs: identify the point, choose its temporal state, and choose the output camera coordinates. The 9 × 9 appearance crop is separate from the encoder’s 2 × 16 × 16 tubelets.';
    } else if(n==='decoder'){
      title='Cross-attention: point queries read video evidence';
      d=box(22,25,180,65,'F: scene','N tokens')+box(22,183,180,65,'H: point queries','M query features');
      d+=line(202,54,266,54)+box(272,20,160,56,'K = F Wk')+line(202,69,269,120)+box(272,93,160,56,'V = F Wv')+line(202,213,266,213)+box(272,184,160,56,'Q = H Wq');
      d+=line(432,48,495,110)+line(432,122,495,136)+line(432,210,495,160)+box(502,74,203,120,'softmax(QKᵀ/√dₕ)V','M × N scores per head',true)+line(705,136,742,136)+box(746,74,133,120,'MLP','× 8 layers');
      caption='Each layer has learned projections; K and V read the shared F. There is no point-query self-attention. Adding another query does not change the decoded answer for an existing query when F and weights are fixed.';
    } else if(n==='heads'){
      title='One decoder feature, six prediction heads';
      d=box(20,102,190,76,'Decoded h','M × d features')+line(210,140,295,140);
      [['XYZ','3'],['UV','2'],['Visibility','1'],['Motion','3'],['Normal','3'],['Confidence','1']].forEach(([a,b],i)=>{const x=320+(i%3)*187,y=35+Math.floor(i/3)*124;d+=line(295,140,x,y+40)+box(x,y,164,78,a,b+' output values',i===0);});
      caption='Learned linear heads map features to 13 values per query. XYZ is the principal prediction; the auxiliary tasks teach complementary properties of the same point. Camera poses are derived from queried geometry, not an extra camera head here.';
    } else if(n==='loss'){
      title='Where the labels enter: after prediction';
      d=box(23,22,185,77,'Predicted points','P̂: M × 3')+box(23,177,185,77,'Target points','P: M × 3');
      d+=line(208,60,252,60)+line(208,212,252,212)+box(258,22,240,77,'Normalize by mean Z','use predicted-set mean')+box(258,177,240,77,'Normalize by mean Z','use target-set mean');
      d+=line(498,60,545,111)+line(498,212,545,159)+box(551,94,152,90,'Signed log','then L1',true)+line(703,140,745,140)+box(751,94,128,90,'Loss','+ masked aux');
      caption='The predicted and target sets are normalized separately. Missing labels disable their loss terms. Confidence weights the 3D error with a −log(c) penalty; the other heads add geometric, visibility and motion supervision.';
    } else if(back){
      title=n==='backward'?'Backpropagation trains both branches':'AdamW changes the weights, not the input clip';
      d=box(23,50,183,84,'Video encoder','patch projection + ViT',true)+box(23,176,183,72,'Query embedding','time, UV, local RGB',true)+box(292,50,157,84,'F','clip activations')+box(509,101,161,84,'Decoder','cross-attention',true)+box(731,101,145,84,'Heads / loss','supervision',true);
      d+=line(206,90,289,90)+line(449,90,506,127)+line(206,212,506,157)+line(670,140,728,140);
      d+=line(728,164,674,164,true)+line(510,106,449,110,true)+line(289,115,207,115,true)+line(510,178,207,235,true);
      caption=n==='backward'?'The decoder’s K/V path carries gradients into F and the encoder. Its Q path carries gradients into query embeddings. Backward computes derivatives; the optimizer updates weights afterward. No stop-gradient separates the encoder here.':'Apply the gradients to the trainable parameters, then encode the next batch with the new weights. The reported run used AdamW for 500k steps. VideoMAE initialization precedes this supervised geometry training.';
    }
    svg.innerHTML=defs+d;svg.setAttribute('aria-label',title+'. '+caption);
    $('cutaway-title').textContent=title;$('cutaway-caption').textContent=caption;
    $('cutaway-phase').textContent=state.mode==='training'?(back?'BACKWARD / UPDATE':'FORWARD PASS'):'INFERENCE / FIXED WEIGHTS';
    document.querySelectorAll('[data-cutaway-go]').forEach(b=>{const match=b.dataset.cutawayGo;b.disabled=state.mode==='inference'&&['loss =','backward'].includes(match);b.setAttribute('aria-pressed',state.code.includes(match));});
  }
  window.addEventListener('d4rt:execution',e=>{state=e.detail;render();});
  window.addEventListener('d4rt:tubelet',()=>{if(state.node==='patch')render();});
  document.querySelectorAll('[data-cutaway-go]').forEach(b=>b.addEventListener('click',()=>{
    const target=[...document.querySelectorAll('[data-exec-step]')].find(row=>row.textContent.includes(b.dataset.cutawayGo));
    if(target)target.click();
  }));
  render();
})();
