/* Functional anatomy. Shapes are symbolic where the paper does not specify widths. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const paper = 'https://arxiv.org/html/2512.08924v2';
  const entries = {
    tubelet: ['01 / TOKENIZATION', 'A token is a learned pixel mixture.',
      'Flatten a fixed patch across two frames. A shared projection maps its RGB values to a continuous feature vector. The patch grid stays fixed even when an object moves.',
      'V: [48, 256, 256, 3]\npatches: [24, 16, 16, 1536]\nX_video = patches.reshape(6144,1536) @ W + b\nX_video += position\nX = append(X_video, aspect_ratio_token)',
      'Is this a VQ tokenizer?', 'No codebook lookup is reported. “Token” here means a continuous C-dimensional vector. Flattening order is an implementation convention.',
      'The shared projection receives gradients from all tubelets. The input video is data; the optimizer updates the projection weights.', '#tokenizer', '#S4'],
    encoder: ['02 / THE ENCODER', 'Video tokens talk to video tokens.',
      'Self-attention gets Q, K and V from video features. A local layer mixes spatial evidence within a temporal group. A global layer connects the entire observed clip, including later frames.',
      'Q = X @ Wq; K = X @ Wk; V = X @ Wv\nA = softmax(Q @ K.T / sqrt(d_head) + mask)\ncontext = A @ V\nmerge heads → output projection → residual\ntoken-wise MLP → residual\n# repeat with local/global routing',
      'What does the MLP do?', 'Attention mixes token addresses. The MLP transforms channels at each address using shared weights. Both preserve the token grid; information can spread far beyond the original tubelet.',
      'A query loss reaches F, then propagates through encoder layers to the tubelet projection. This is end-to-end learning, not a permanently frozen video backbone.', '#encoder', '#S2.SS2'],
    memory: ['03 / THE SCENE FEATURES', 'F stores context, not explicit coordinates.',
      'Each feature has already mixed evidence across the video. The decoder reads this shared bank to answer different geometric questions.',
      'F = encoder(video, aspect_ratio)\nF: [N, C]\nN = 6144 video tokens + metadata\nK_l = F @ Wk_l; V_l = F @ Wv_l',
      'Is F a voxel map or a cached movie?', 'It is a learned latent representation. No individual feature channel is promised to mean “depth” or “velocity.” A trained decoder maps it to geometric answers.',
      'The gradients from independent queries accumulate into the shared F. Its numeric values are recomputed after an encoder weight update.', '#model-atlas', '#S2.SS2'],
    query: ['04 / THE SECOND INPUT PATH', 'Describe the point and the question.',
      'The source pixel identifies a point. Target time asks when to locate it. Camera time chooses the coordinate frame for the answer. The local RGB crop supplies fine appearance evidence.',
      'q = FC(Fourier(u,v))\n  + E_src[t_src] + E_tgt[t_tgt] + E_cam[t_cam]\n  + MLP(crop_rgb(source_frame,u,v,9,9))\nH = MLP(q)\nH: [M, D]',
      'Why two patch sizes?', 'The 2 × 16 × 16 tubelet builds shared scene context. The 9 × 9 source-frame crop describes a particular query at finer spatial resolution.',
      'Gradients also train the query MLP, RGB-crop embedding and time embeddings. Geometry targets do not enter this branch.', '#geometry-lab', '#A1'],
    decoder: ['05 / THE DECODER', 'Questions read the shared scene.',
      'Cross-attention uses the current query state for Q and the encoder features for K and V. A query-wise MLP updates the result before the next layer.',
      'Q: [M, d_head]  ← point queries\nK,V: [N, d_head] ← F\nA = softmax(Q @ K.T / sqrt(d_head))\nA: [M,N]; context = A @ V\n8 cross-attention decoder layers',
      'Why no query self-attention?', 'Each output can be computed independently. You can batch or split queries while reusing F; one question does not need the other questions to define its answer.',
      'A geometric error trains both the query stream and the scene features it reads. Independent forward queries still contribute to shared parameter gradients.', '#decoder', '#S2.SS2'],
    heads: ['06 / OUTPUTS', 'Convert features into geometry.',
      'Learned projections map each final query feature to a 3D answer and auxiliary predictions. Coordinates are expressed in the requested camera reference.',
      'XYZ          3\nimage UV     2\nvisibility   1\nmotion       3\nsurface normal 3\nconfidence   1\n──────────────\ntotal       13',
      'Where are depth and camera pose?', 'Depth uses the Z output with source, target and camera times equal. Camera pose is recovered from corresponding 3D predictions expressed in different camera frames.',
      'Each supervised head contributes gradients through the shared decoder. A missing normal label, for example, does not remove an available 3D label.', '#geometry-lab', '#S2.SS3'],
    loss: ['07 / TRAINING', 'Teach an answer, update the whole path.',
      'Sample queries from labeled trajectories. Compare predictions only where each target exists. Backpropagation updates the heads, decoder, query embeddings and encoder.',
      'F = encoder(clip)\npred = decode(queries, F)\nloss = available_task_losses(pred, labels, masks)\ngradients = backward(loss)\nclip_global_norm(gradients, 10)\nAdamW.update(parameters, gradients)',
      'Does the model receive labels as input?', 'Labels supervise the loss, not the RGB-to-prediction path. Training query locations may be sampled using annotations. At inference you choose the queries without those annotations.',
      'Weights carry learned knowledge across clips. F belongs to one forward pass. Updating encoder weights invalidates its cached features.', '#geometry-loss', '#A1']
  };
  let selected = 'encoder', mode = 'forward';
  function render() {
    const e = entries[selected];
    ['label','title','copy','code','question','answer','gradient'].forEach((key,i) => $('map-detail-'+key).textContent=e[i]);
    $('map-detail-lab').href='anatomy.html'+e[7];
    $('map-detail-source').href=paper+e[8];
    $('map-detail-gradient').hidden=mode!=='backward';
    $('map-flow').classList.toggle('map-training',mode==='backward');
    document.querySelectorAll('.map-arrow').forEach(e=>e.textContent=mode==='backward'?'↑ ∂L':'↓');
    document.querySelector('.map-bridge>span:first-child').textContent=mode==='backward'?'∂L/∂F ↙':'F ↗ K, V';
    document.querySelectorAll('.map-back').forEach(e=>e.hidden=mode!=='backward');
    document.querySelectorAll('[data-model-node]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.modelNode===selected));
    document.querySelectorAll('[data-map-mode]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mapMode===mode));
  }
  document.querySelectorAll('[data-model-node]').forEach(b=>b.addEventListener('click',()=>{
    selected=b.dataset.modelNode; render();
    if(matchMedia('(max-width: 800px)').matches) document.querySelector('.map-inspector').scrollIntoView({block:'start'});
  }));
  document.querySelectorAll('[data-map-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mapMode;render();}));
  render();
  window.D4RTModelMap={snapshot:()=>({selected,mode})};
})();
