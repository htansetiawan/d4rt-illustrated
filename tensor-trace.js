/* Shape arithmetic only. No hidden dimensions or trained weights are assumed. */
(() => {
 'use strict';
 const $=id=>document.getElementById(id), fmt=n=>n.toLocaleString('en-US');
 const keys=['video','tubelet','embed','local','global','query','decoder','heads','train'];
 let queries=2048;
 function trace(key,s) {
   const index=(s.pair*16+s.row)*16+s.col;
   const shapes={
    video:[['Clip','[48, 256, 256, 3]','Time × height × width × RGB'],['Scalar values',fmt(48*256*256*3),'Before grouping pixels'],['Original aspect ratio','1 value → embedded token','Retained when resizing to a square']],
    tubelet:[['Grouped pixels','[24, 16, 16, 2, 16, 16, 3]','Group × patch row × patch column × tubelet pixels'],['Flattened tubelets','[6,144, 1,536]','24 × 16 × 16 addresses; 2 × 16 × 16 × 3 values each'],['Selected tubelet',String(index),`Frames ${2*s.pair}–${2*s.pair+1}; x ${16*s.col}–${16*s.col+15}, y ${16*s.row}–${16*s.row+15}`]],
    embed:[['Shared patch weights','[1,536, C]','The same learned projection at every address'],['Video embeddings','[6,144, C]','6,144 continuous vectors; position information is added'],['Encoder input','Video vectors + metadata','Append the aspect-ratio embedding']],
    local:[['Grouped video features','[24, 256, C]','A reshape of the same 6,144 addresses'],['Q, K, V per group / head','[256, d_head] each','Learned projections of the video features'],['Scores per group / head','[256, 256]','Softmax normalizes each query row over 256 keys'],['Mixed output per group','[256, C]','Merge heads and project; the MLP changes channels, not addresses']],
    global:[['Video features','[6,144, C]','All groups are in the same sequence'],['Q, K, V per head','[6,144, d_head] each','Every observed time can provide evidence'],['Scores per head','[6,144, 6,144]','Softmax normalizes each query row over all video keys'],['Contextual features','[6,144, C] + metadata','Token addresses remain; their receptive fields have expanded']],
    query:[['Question fields',`[${fmt(queries)}, 5]`,'u, v, source time, target time, camera time'],['Local RGB per question','[9, 9, 3] → embedding','243 scalar values from the source image'],['Query features',`[${fmt(queries)}, D]`,'Spatial, time and RGB embeddings are combined']],
    decoder:[['Q per head',`[${fmt(queries)}, d_head]`,'Projected from point-query features'],['K, V per head','[6,144, d_head] each','Projected from the shared video features'],['Scores per head',`[${fmt(queries)}, 6,144]`,'Each row reads scene tokens, not other point queries'],['Updated questions',`[${fmt(queries)}, D]`,'Value mixing → head merge → output projection → query MLP']],
    heads:[['Decoded features',`[${fmt(queries)}, D]`,'One feature vector per question'],['Predictions',`[${fmt(queries)}, 13]`,'XYZ 3 + UV 2 + visibility 1 + motion 3 + normal 3 + confidence 1'],['Depth request','Keep the Z channel','Set source = target = camera time']],
    train:[['Forward predictions','[2,048, 13]','Reported sampled-query count per training clip'],['Task labels + masks','Available targets only','A missing label disables its loss, not the whole point'],['Loss','Scalar','Normalize point sets, compare predictions and labels, combine tasks'],['Backpropagation','∂L/∂F → encoder weights','Queries share scene features and train the same parameters']]
   };
   const m=key==='train'?2048:queries;
   return {key,index,queries:m,rows:shapes[key],localScores:24*256*256,globalScores:6144*6144,decoderScores:m*6144};
 }
 let latest;
 function render(){
   const s=window.D4RTPipeline.snapshot();
   latest=trace(keys[s.stage],s);
   $('trace-rows').replaceChildren(...latest.rows.map(values=>{
     const tr=document.createElement('tr');
     values.forEach((v,i)=>{const el=document.createElement(i===0?'th':'td');if(i===0)el.scope='row';el.textContent=v;tr.append(el);});return tr;
   }));
   $('trace-query-control').hidden=!['query','decoder','heads'].includes(latest.key);
   $('trace-stage-link').href='#walk-'+latest.key;
   $('trace-count').textContent=`Per head, per layer: local ${fmt(latest.localScores)} · global ${fmt(latest.globalScores)} · decoder ${fmt(latest.decoderScores)} score entries.`;
   $('trace-meaning').textContent=latest.key==='train'
    ? 'Independent forward queries still share a training objective. Mean-depth normalization can couple their 3D losses; all query gradients accumulate into the encoder. This is why caching F across weight updates would be wrong.'
    : ['query','decoder','heads'].includes(latest.key)
      ? 'Change the query count: decoder rows change, but the video encoder still processes the same clip once. Chunking queries limits temporary decoder tensors while reusing F.'
      : 'Follow the token count: grouping pixels reduces the number of addresses. Self-attention spreads context between those addresses; the channel MLP transforms each address independently.';
 }
 $('trace-queries').addEventListener('change',e=>{queries=Number(e.target.value);document.dispatchEvent(new CustomEvent('d4rt-query-count'));});
 document.addEventListener('d4rt-stage',render);
 render();
 window.D4RTTensorTrace={snapshot:()=>structuredClone(latest)};
})();
