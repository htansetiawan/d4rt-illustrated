/* A navigable computational trace. It does not execute neural-network weights. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  if (!$('execution-code')) return;
  const stages = [
    {code:'video = sample_and_resize(clip)', title:'Keep the clip and its geometry together.', shape:'video: [48, 256, 256, 3]\noriginal aspect ratio: W / H', text:'Sampling chooses the video window. Resizing makes the encoder input square; original aspect ratio remains an input. During training, spatial augmentation must transform the associated point coordinates too.', node:'video', link:'tokenizer', learn:[]},
    {code:'z = tokenize(video, aspect_ratio, position)', title:'A pixel block becomes one feature vector.', shape:'[48, 256, 256, 3]\n→ [24, 16, 16, 1536]\n→ [24, 16, 16, C]\n+ position encoding; separate aspect-ratio token', text:'The first reshape only groups pixels. A learned projection then maps 1,536 RGB scalars into C features. The same projection serves every tubelet. Positional encoding supplies addresses; a separate token preserves the original aspect ratio.', node:'patch', link:'tubelet-volume', learn:['encoder']},
    {code:'z = encoder.local_attention_and_mlp(z)', title:'Mix space within a time group.', shape:'video-token view: [24, 256, C]\n24 separate sequences of 256 tokens', text:'For this representative local layer, queries, keys and values all come from the video features. Attention mixes token addresses within a group; the MLP transforms each resulting feature independently. Standard residual connections are omitted from this pseudocode.', node:'encoder', link:'encoder', learn:['encoder']},
    {code:'z = encoder.global_attention_and_mlp(z)', title:'Now connect places across times.', shape:'video-token view: [6144, C]\nQ, K, V are projected from z', text:'The global layer can connect a source location to a different location in another time group, including later frames. Local and global processing are interleaved across the 40-layer encoder. These two lines illustrate the pattern, not the exact source-code loop.', node:'encoder', link:'attention-math', learn:['encoder']},
    {code:'F = encoded_scene_features(z)', title:'The encoder returns context, not XYZ.', shape:'F: [N, C]\n6144 video tokens + metadata context', text:'Each vector now carries contextual evidence. It still needs a query to become a point prediction. “Fixed F” means reuse for decoding this clip; it does not mean that gradients stop at F during training.', node:'scene', link:'decoder', learn:[]},
    {code:'h = query_embed(uv, src, tgt, cam, rgb9)', title:'Construct the second input path.', shape:'one query: 2 coordinates + 3 time indices\nRGB patch: [9, 9, 3] → learned feature\ntraining batch: [2048, d]', text:'The normalized location gets Fourier features. Separate learned time embeddings and the embedded RGB crop are added into the query representation. The query crop is sampled around a point; the encoder tubelets tile the entire clip.', node:'query', link:'decoder', learn:['query']},
    {code:'h = decoder.cross_attention_and_mlp(h, F)', title:'Eight layers read the shared scene.', shape:'Q = h Wq       K = F Wk       V = F Wv\nattention per head: [M, N]\nupdated queries: [M, d]', text:'Each decoder layer has its own learned projections. Queries exchange information with F, but never with other point queries through self-attention. Increasing M asks more questions of the same encoded evidence.', node:'decoder', link:'decoder', learn:['decoder']},
    {code:'predictions = linear_heads(h)', title:'Turn the query features into predictions.', shape:'[M, d] → [M, 13]\nXYZ 3 · UV 2 · visibility 1\nmotion 3 · normal 3 · confidence 1', text:'The output is geometry and point properties. To obtain a trajectory, repeat the point identity with different target times. To obtain depth, match the three times and read Z. These are different queries to the same network.', node:'heads', link:'recipes', learn:['decoder']},
    {code:'loss = masked_geometry_and_aux(pred, labels)', title:'Compare against available supervision.', shape:'labeled trajectories → query targets\nnormalize each point set by its own mean depth\nsigned log → geometric L1 + auxiliary losses', text:'Ground truth does not go into the forward prediction. It supplies the training comparison afterward. A missing label masks that loss term. Independent query decoding still permits coupled losses: mean-depth normalization uses the sampled point set.', node:'loss', link:'gradient-lab', learn:[], training:true},
    {code:'gradients = backward(loss)', title:'The error reaches both input paths.', shape:'loss → heads → decoder\n             ↙       ↘\nquery embeddings    F → encoder → patch projection', text:'Cross-attention is differentiable with respect to the query path and its scene keys and values. Thus geometry supervision trains the video representation too. This pass computes gradients; model weights change in the following optimizer step.', node:'backward', link:'training', learn:['encoder','query','decoder'], training:true},
    {code:'weights = AdamW(weights, gradients)', title:'Update parameters, then start a new pass.', shape:'clip gradients → optimizer update\nencoder + query embeddings + decoder/heads', text:'The next training batch is encoded with the updated weights. A cached F from before that update is stale. Activations describe this particular clip; learned weights are reused across clips. The paper reports 500,000 optimizer steps.', node:'update', link:'training', learn:['encoder','query','decoder'], training:true},
    {code:'next_points = decode(new_queries, cached_F)', title:'At inference, reuse the scene.', shape:'same clip + same weights → reuse F\nnew queries → new point predictions', text:'No labels, backward pass or optimizer are needed. You can decode another query batch without rerunning the encoder. A different clip needs a new F; changed weights also invalidate the old representation.', node:'reuse', link:'efficiency', learn:[], inference:true}
  ];
  let mode = 'training', index = 0;
  const path = () => stages.filter(s => mode === 'training' ? !s.inference : !s.training);
  const weightNames = {encoder:'Video encoder', query:'Query embeddings', decoder:'Decoder + heads'};
  const graph = [['video','RGB'],['patch','Tubelets'],['encoder','Encoder'],['scene','F'],['decoder','Decoder'],['heads','Heads']];
  function render(focus = false) {
    const list = path();
    index = Math.min(index, list.length - 1);
    const s = list[index];
    $('execution-code').innerHTML = list.map((entry,i) => `<button type="button" class="execution-line" data-exec-step="${i}" aria-current="${i===index?'step':'false'}" aria-label="Step ${i+1}: ${entry.title}"><span>${String(i+1).padStart(2,'0')}</span><code></code></button>`).join('');
    [...$('execution-code').children].forEach((b,i) => { b.querySelector('code').textContent=list[i].code; b.addEventListener('click',()=>{index=i;render(true);}); });
    $('execution-kicker').textContent = `${mode.toUpperCase()} · STEP ${index+1} / ${list.length}`;
    $('execution-title').textContent = s.title;
    $('execution-description').textContent = s.text;
    $('execution-shape').textContent = s.shape;
    $('execution-deeper').href = '#'+s.link;
    $('execution-weights').innerHTML = Object.entries(weightNames).map(([key,label]) => {
      const active = s.learn.includes(key);
      const status = mode === 'inference' ? 'fixed weights' : s.node === 'backward' ? 'gradient computed' : s.node === 'update' ? 'weights updated' : active ? 'weights used' : 'waiting / shared';
      return `<div class="${active?'receives-gradient':''}">${label}<small>${status}</small></div>`;
    }).join('');
    const back = ['backward','update'].includes(s.node);
    $('execution-connections').innerHTML = graph.map(([key,label],i)=>`${i?'<b>'+(back?'←':'→')+'</b>':''}<span class="${back||s.node===key?'active':''}">${label}</span>`).join('') + '<div class="execution-branch"><span class="'+(back||s.node==='query'?'active':'')+'">Point query + RGB crop</span><b> '+(back?'←':'→')+' </b><span class="'+(back||s.node==='decoder'?'active':'')+'">Decoder</span></div>';
    $('execution-status').textContent = s.node === 'backward' ? 'BACKWARD · errors flow through F into the encoder' : s.node === 'update' ? 'UPDATE · weights change; old F becomes stale' : mode === 'inference' ? 'INFERENCE · weights stay fixed' : 'FORWARD · weights are used, not yet updated';
    $('execution-prev').disabled=index===0;
    $('execution-next').disabled=index===list.length-1;
    $('execution-position').textContent=`${index+1} / ${list.length}`;
    document.querySelectorAll('[data-exec-mode]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.execMode===mode));
    window.dispatchEvent(new CustomEvent('d4rt:execution', {detail: {node:s.node, mode, index, code:s.code}}));
    const b=$('execution-code').children[index];
    if(matchMedia('(max-width:750px)').matches) $('execution-code').scrollTop=b.offsetTop-$('execution-code').clientHeight/2+b.clientHeight/2;
    if (focus) {
      b.focus({preventScroll:true});
    }
  }
  document.querySelectorAll('[data-exec-mode]').forEach(b=>b.addEventListener('click',()=>{const selected=path()[index].code;mode=b.dataset.execMode;index=Math.max(0,path().findIndex(s=>s.code===selected));render();}));
  $('execution-prev').addEventListener('click',()=>{index--;render();});
  $('execution-next').addEventListener('click',()=>{index++;render();});
  $('execution-code').addEventListener('keydown',e=>{if(!['ArrowUp','ArrowDown','Home','End'].includes(e.key))return;e.preventDefault();index=e.key==='Home'?0:e.key==='End'?path().length-1:Math.max(0,Math.min(path().length-1,index+(e.key==='ArrowDown'?1:-1)));render(true);});
  const tubeAddress=()=>{
    const t=Number($('tube-t').value), x=Number($('tube-x').value), y=Number($('tube-y').value);
    $('execution-token').textContent=`Linked tubelet: frames ${2*t+1}–${2*t+2}, row ${y+1}, column ${x+1} → token ${t*256+y*16+x}`;
  };
  window.addEventListener('d4rt:tubelet',tubeAddress);
  render();tubeAddress();
})();
