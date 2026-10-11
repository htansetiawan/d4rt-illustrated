/* Paper-grounded architecture explorer. No model weights or inference. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  if (!$('model-atlas')) return;
  const source = 'https://arxiv.org/html/2512.08924v2';
  const state = {node: 'local', mode: 'train', frames: 48, queries: 2048};
  const nodes = {
    video: {
      title: 'A clip has two destinations.', shape: 'RGB: [B, 48, 256, 256, 3]',
      body: 'The main route resizes the observed clip for the encoder. The second route supplies a small RGB crop around each source point directly to the query network. Original aspect ratio is embedded separately, so square resizing does not discard that metadata.',
      operation: 'clip → square RGB frames → encoder\nsource frame + (u,v) → 9 × 9 crop → query\noriginal W/H → aspect-ratio embedding',
      weights: 'Input preprocessing has no learned weights. The downstream aspect-ratio embedding is learned.',
      activation: 'RGB samples and metadata are inputs, not a learned scene representation.',
      training: 'Sample and augment the clip, then sample queries from available labeled trajectories. Spatial augmentation must transform the associated labels too.',
      inference: 'Only the observed RGB clip and requested queries are needed. Geometry labels and camera-pose labels are not supplied to the network.',
      link: '#S2.SS2', detail: '#tour'
    },
    patch: {
      title: 'Tokenization is a learned continuous projection.', shape: '[B, 6144, 1536] → [B, 6144, C]',
      body: 'At one image address, gather 16 × 16 RGB pixels from two neighboring frames. Flatten that volume into 1,536 scalars. A shared learned projection maps each volume to C channels; positional information is added. There is no VQ codebook in this described D4RT route.',
      operation: 'G = 48 / 2 = 24 time groups\nS = (256 / 16)² = 256 spatial tokens\nN = G × S = 6144\nx = flatten(tubelet)           # 1536 values\nz = x @ Wpatch + b + position',
      weights: 'The patch projection is learned. The exact positional-encoding implementation is not specified in this explorer.',
      activation: '6,144 vectors retain separate time-and-space addresses. Metadata tokens are excluded from this count.',
      training: 'Gradients from point-level geometric errors reach this projection through the decoder and encoder.',
      inference: 'The projection weights stay fixed; changing the clip changes its token activations.',
      link: '#S4', detail: '#tokenizer'
    },
    local: {
      title: 'Encoder: exchange evidence within each time slice.', shape: '[B × 24, 256, C] → same shape',
      body: 'The video encoder is a Vision Transformer. In its local, frame-wise attention operation, queries, keys and values all come from video features in the same spatial slice. “Local” does not mean a 3 × 3 window. A token can read the whole slice. With temporal tubelets, a slice is a token-time group.',
      operation: 'Q = X @ Wq; K = X @ Wk; V = X @ Wv\nA = softmax(Q @ Kᵀ / √d)      # within slice\ncontext = A @ V\nattention update → token-wise MLP\n# Figure 7 then shows the global operation.',
      weights: 'Learned attention projections and MLPs. Normalization, residuals and multi-head details are omitted from this routing diagram; the numerical inspector below opens a standard block.',
      activation: 'Attention mixes spatial addresses. The MLP mixes channels at each address without changing the token count.',
      training: 'VideoMAE supplies a useful encoder initialization. The subsequent D4RT geometry objective trains the encoder end-to-end.',
      inference: 'A local operation cannot directly read another time group. Earlier global layers may already have written temporal context into its input.',
      link: '#A1', detail: '#encoder'
    },
    global: {
      title: 'Encoder: connect different places at different times.', shape: '[B, 6144, C] → same shape',
      body: 'Global self-attention treats the video-token lattice as one sequence. A point that moved to a different patch can now contribute evidence directly. Both earlier and later observed frames are available. Figure 7 shows an MLP after this operation too.',
      operation: 'X = merge_time_and_space(X)\nQ,K,V = learned_projections(X)\nA = softmax(Q @ Kᵀ / √d)      # whole clip\ncontext = A @ V\nattention update → token-wise MLP',
      weights: 'The reported ViT-g encoder has 40 layers and about 1B parameters. Do not interpret the schematic as 40 repetitions of a two-attention pair; a full indexed layer schedule is not provided here.',
      activation: 'The logical score matrix is N × N per head and global layer. Efficient attention implementations need not materialize that entire matrix.',
      training: 'One query loss can influence evidence across the video through attention. This does not guarantee meaningful geometry at every token.',
      inference: 'This observed-clip reconstruction is non-causal. It differs from the causal temporal sublayer of the original Genie ST-transformer.',
      link: '#S2.SS2', detail: '#encoder'
    },
    scene: {
      title: 'F is a shared bank of contextual features.', shape: 'F: [B, N, C]',
      body: 'The encoder writes the global scene representation F. It is an array of continuous features, not an explicit mesh, a single pooled vector, or a learned discrete code index. Decoder layers read this same representation for any requested points.',
      operation: 'F = encoder(video)\n# F is reused for independent query batches:\nP_a = decoder(queries_a, F)\nP_b = decoder(queries_b, F)',
      weights: 'F itself is an activation, not an independently learned parameter. Encoder weights determine how a clip produces F.',
      activation: 'N and C describe the feature bank. Examples below count video tokens and omit aspect-ratio metadata.',
      training: 'Do not detach F: dL/dF carries supervision back into the encoder. An encoder weight update makes a previously cached F stale.',
      inference: 'For unchanged video and fixed weights, reuse F when changing query positions, target times or camera times.',
      link: '#S2.SS1', detail: '#learning'
    },
    query: {
      title: 'A point query bypasses the encoder on its second path.', shape: '5 query coordinates + [9, 9, 3] RGB crop → [d]',
      body: 'The source time and image coordinates identify a physical point. Target time asks when to locate it; camera time asks in which coordinate frame to express it. These are three distinct choices. The local crop preserves appearance detail that coarse video tokens can miss.',
      operation: 'q = FC(Fourier(u,v))\n  + E_source[t_source]\n  + E_target[t_target]\n  + E_camera[t_camera]\n  + MLP(source_RGB_crop_9x9)\nq = query_MLP(q)',
      weights: 'Coordinate and crop projections, time embeddings and query MLP are learned. The source crop contains pixels, not target geometry.',
      activation: 'For M queries, build M query features. Changing camera time changes the requested coordinates, not the physical identity of the point.',
      training: 'The paper samples 2,048 queries per clip. Thirty percent are sampled near boundaries; a separate 40% branch sets target time equal to camera time.',
      inference: 'Queries are chosen for the task: vary target time for a track, or query an image grid for a depth map.',
      link: '#A1', detail: 'reference.html#model-lifetime'
    },
    decoder: {
      title: 'Eight cross-attention layers read the shared scene.', shape: 'queries [B,M,d] × scene [B,N,C] → [B,M,d]',
      body: 'Queries supply Q; scene features supply K and V. Each layer reads evidence from F, then transforms each query feature through its MLP. Queries do not read one another. The decoder has about 144M parameters and produces one output feature per query.',
      operation: 'for layer in eight_decoder_layers:\n    Q = project(query_features)\n    K,V = project(F)\n    A = softmax(Q @ Kᵀ / √d)  # M × N\n    read = A @ V\n    query_features = update_and_MLP(read, query_features)\n# No M × M query self-attention.',
      weights: 'Query, key, value and output projections plus per-query MLPs are learned. Widths remain symbolic rather than being guessed.',
      activation: 'M controls decoding work independently of the encoder input. Independent predictions still share F and all model weights.',
      training: 'Gradients flow both through the query branch and through K,V into F. Sparse queries can therefore train the whole video encoder.',
      inference: 'Split a large query set into batches while reusing F. This architecture has no query-to-query state to carry between batches.',
      link: '#S2.SS2', detail: 'reference.html#decoder'
    },
    heads: {
      title: 'One query produces 13 output channels.', shape: '[B,M,d] → [B,M,13]',
      body: 'The shared decoded feature feeds simple learned projections. XYZ is the main answer; auxiliary outputs improve supervision and support downstream use. A camera matrix is not one of these 13 channels: camera parameters can be derived from point queries.',
      operation: 'XYZ position          3\nimage UV position     2\nvisibility            1\npoint displacement    3\nsurface normal        3\nconfidence            1\n───────────────────────\ntotal                13',
      weights: 'Output projection weights are learned. Figure 7 gives the output dimensions; it does not specify every output activation here.',
      activation: 'A predicted visibility score is different from having a valid 3D label. A synthetic occluded point may still have a known location.',
      training: 'Each head contributes a loss wherever its ground truth is available. Missing supervision does not mean the head is missing from the forward pass.',
      inference: 'Use XYZ for geometry. Decode corresponding point sets in two camera frames and align them to recover relative camera pose.',
      link: '#A1', detail: '#training'
    },
    loss: {
      title: 'Geometry errors train both branches.', shape: 'available labels + predictions → scalar loss → gradients',
      body: 'Normalize predicted and target point sets by their own mean depths, apply signed log, then compare 3D positions with L1. Other heads use their respective objectives. Labels are supplied to the loss; they are not concatenated into the encoder or query.',
      operation: '3D: L1(signlog(P / mean_depth(P)),\n       signlog(P* / mean_depth(P*)))\n2D and motion: L1\nnormal: cosine similarity objective\nvisibility: binary cross-entropy\nconfidence: weights 3D error, penalized by −log(c)\nloss.backward() → clip_grad_norm(10) → AdamW.step()',
      weights: 'AdamW updates the encoder, query branch, decoder and heads together. Gradients are derivatives of the objective, not new RGB or geometry inputs.',
      activation: 'Training retains or recomputes intermediate activations for backpropagation. Forward-only inference does not need this gradient graph.',
      training: 'Reported recipe: 500k steps; 2,500-step warmup to 1e−4; cosine decay to 1e−6; weight decay 0.03. Appendix A repeats a confidence term, so this diagram does not invent a corrected full loss.',
      inference: 'No labels, loss, gradient computation or optimizer update.',
      link: '#S2.SS3', detail: '#learning'
    }
  };
  const order = Object.keys(nodes);
  function select(node) {
    if (!nodes[node] || (node === 'loss' && state.mode === 'infer')) return;
    state.node = node;
    const n = nodes[node];
    for (const button of document.querySelectorAll('[data-atlas]')) button.setAttribute('aria-pressed', button.dataset.atlas === node);
    $('atlas-title').textContent = n.title;
    $('atlas-shape').textContent = n.shape;
    $('atlas-body').textContent = n.body;
    $('atlas-operation').textContent = n.operation;
    $('atlas-weights').textContent = n.weights;
    $('atlas-activation').textContent = n.activation;
    $('atlas-lifetime').textContent = state.mode === 'train' ? n.training : n.inference;
    $('atlas-lifetime-label').textContent = state.mode === 'train' ? 'During training' : 'During inference';
    $('atlas-source').href = source + n.link;
    $('atlas-deeper').href = n.detail;
    $('atlas-status').textContent = `${state.mode === 'train' ? 'Training' : 'Inference'} · ${n.title}`;
  }
  function mode(value) {
    state.mode = value;
    $('atlas-loss').disabled = value === 'infer';
    $('atlas-model').dataset.mode = value;
    $('atlas-return').textContent = value === 'train'
      ? 'Backward: loss → heads → decoder → {scene F → encoder, query branch}. Then AdamW updates weights.'
      : 'Forward only: video → F once. Reuse F for new queries while video and weights remain unchanged.';
    select(value === 'infer' && state.node === 'loss' ? 'heads' : state.node);
  }
  function counts(frames, queries) {
    const groups = frames / 2, spatial = 256, tokens = groups * spatial;
    return {groups, spatial, tokens, scalars: frames * 256 * 256 * 3,
      local: groups * spatial * spatial, global: tokens * tokens,
      temporal: spatial * groups * (groups + 1) / 2, cross: queries * tokens};
  }
  const format = n => n.toLocaleString('en-US');
  function budget() {
    const c = counts(state.frames, state.queries);
    $('atlas-counts').innerHTML = `<div><span>Video tokens</span><b>${format(c.tokens)}</b><small>${c.groups} × 16 × 16</small></div><div><span>Encoder · one local layer</span><b>${format(c.local)}</b><small>G × S² score entries / head</small></div><div><span>Encoder · one global layer</span><b>${format(c.global)}</b><small>(G × S)² score entries / head</small></div><div><span>Decoder · one cross-attention layer</span><b>${format(c.cross)}</b><small>M × G × S score entries / head</small></div>`;
    $('atlas-genie-cost').textContent = `On the same teaching token lattice, one Genie-style spatial + causal temporal pair permits ${format(c.local)} + ${format(c.temporal)} attention entries. The temporal part uses S × G(G+1)/2; the spatial part uses G × S². This compares routing, not the real Genie model’s token count or measured runtime.`;
    $('atlas-budget-note').textContent = state.frames === 48 && state.queries === 2048
      ? '48 frames and 2,048 queries match the reported D4RT training configuration. Counts exclude metadata tokens.'
      : 'Counterfactual shape calculation. This setting is not a reported D4RT training run or a guarantee that a checkpoint accepts this clip length.';
  }
  document.querySelectorAll('[data-atlas]').forEach(button => button.addEventListener('click', () => {
    select(button.dataset.atlas);
    if (window.innerWidth <= 650) {
      const inspector = $('atlas-inspector');
      inspector.focus({preventScroll: true});
      inspector.scrollIntoView({block: 'start', behavior: 'instant'});
    }
  }));
  $('atlas-mode').addEventListener('change', event => mode(event.target.value));
  function next() {
    const available = order.filter(n => state.mode === 'train' || n !== 'loss');
    select(available[(available.indexOf(state.node) + 1) % available.length]);
  }
  $('atlas-next').addEventListener('click', next);
  $('atlas-next-detail').addEventListener('click', next);
  $('atlas-frames').addEventListener('change', event => {state.frames = +event.target.value; budget();});
  $('atlas-queries').addEventListener('change', event => {state.queries = +event.target.value; budget();});
  mode('train'); budget();
  window.D4RTAtlas = {counts, snapshot: () => ({...state, counts: counts(state.frames, state.queries)})};
})();
