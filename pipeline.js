/* Original diagrams. No pretrained weights or claimed D4RT predictions. */
(() => {
'use strict';
const $=id=>document.getElementById(id), paper='https://arxiv.org/html/2512.08924v2';
const stages=[
 ['video','Video','Start with an observed clip.','V: [48,256,256,3]','Both object motion and camera motion live in these RGB frames. The model must infer their geometry; it receives no ground-truth camera poses.','Resize frames to 256 × 256\nKeep original aspect ratio as metadata\nRetain source RGB for the query crop','The shaded square stays at the same image address while the object moves through it. It is not yet a tracked point.','Our synthetic clip is only an illustration of the input.','#tokenizer','#S2.SS2'],
 ['tubelet','Tubelets','Two frames. One pixel volume.','[2,16,16,3] → [1536]','Take a 16 × 16 region from two consecutive frames. One tubelet holds both patches, including all three color channels.','g = floor(frame / 2)\nindex = (g × 16 + row) × 16 + column\nx = flatten(video[2g:2g+2, patch, RGB])','Temporal grouping happens before attention. The same region can contain different objects in the two frames.','The illustrated flattening order is time → row → column → RGB; the paper does not require this memory layout.','#tokenizer','#S4'],
 ['embed','Embedding','Pixels become a continuous vector.','[6144,1536] → [6144,C]','The same learned projection processes every tubelet. Add position information, and supply an embedded aspect-ratio token to the encoder.','e_i = x_i W_patch + b_patch\nz_i = e_i + position_i\nX = video tokens + aspect-ratio metadata','Token count shrinks, but a token is a vector, not a discrete vocabulary index. Training learns which combinations of pixels are useful for geometry.','C, exact positional encoding and metadata routing stay symbolic. Projection values are not fabricated.','#tokenizer','#A1'],
 ['local','Local attention','Let each temporal group share context.','24 groups × [256,C]','The local operation mixes spatial tokens within each group. Select a different token to see the complete set of addresses it can read.','Q = X W_Q; K = X W_K; V = X W_V\nA = softmax(QKᵀ / √d_head + local_mask)\ncontext = A V → project → token update','Video features supply queries, keys and values. Context returns to the video tokens; a token-wise MLP transforms their channels.','The lattice omits the aspect-ratio token. Local attention is shown at the post-tubelet temporal resolution.','#encoder','#S2.SS2'],
 ['global','Global attention','Connect different places and times.','[6144,C] → F: [6144,C]','Global self-attention allows a token to read every video address. Evidence about a moving object can arrive from another position in a later frame.','A = softmax(QKᵀ / √d_head)\nNo temporal causal mask over the observed clip\nInterleave local/global layers → scene bank F','The encoder is a 40-layer, 1B-parameter ViT-g. F is contextual memory for the decoder, not 6,144 isolated summaries of patches.','The paper does not list the full layer schedule. This diagram illustrates connectivity, not learned attention strengths.','#encoder','#S4'],
 ['query','Point query','Ask about a physical point.','q: (u,v,t_src,t_tgt,t_cam)','Pick a pixel in a source frame, choose when to locate it, and choose the camera coordinate frame for the answer. These three times have different jobs.','h = Fourier(u,v) + E_src(t_src)\n  + E_tgt(t_tgt) + E_cam(t_cam)\n  + embed(RGB crop 9 × 9 at source pixel)\nh → query MLP','The 9 × 9 RGB crop is a second path from the source image. It supplies fine detail lost in coarse encoder tokenization. Changing a query reuses F.','The highlighted source pixel is selected by image address. Changing source time may select a different physical point.','#decoder','#S2.SS2'],
 ['decoder','Cross-attention','The point query reads the scene.','Q: [M,d] · K,V: [N,d]','An eight-layer, 144M-parameter decoder repeatedly reads F. Each query is independent; there is no query-to-query self-attention.','Q = H W_Q; K = F W_K; V = F W_V\nA = softmax(QKᵀ / √d_head)  # [M,N]\nH ← cross-attention read + query MLP\nRepeat through 8 layers','The encoder exchanges information between video tokens. The decoder answers questions about that information. Parallel queries need not refer to nearby pixels.','The complete residual and normalization wiring is not specified here. The diagram shows which stream supplies each tensor.','#decoder','#A1'],
 ['heads','Predictions','One decoder, several geometric outputs.','[M,d] → 13 channels per query','Learned projections read the final query features. XYZ locates the point at the target time in the requested camera coordinate system.','XYZ: 3 · image UV: 2 · visibility: 1\nmotion: 3 · normal: 3 · confidence: 1\nTotal = 13 channels per query','Vary the queries to recover depth maps, trajectories or point clouds. Camera parameters can be recovered from geometric predictions; they are not a separate matrix among these 13 channels.','Output vectors shown here are symbolic. For actual sample reconstructions, visit the authors’ project.','#decoder','#S2.SS3'],
 ['train','Training','Send geometric error back through the model.','2048 sampled queries → masked losses','Labels are used after prediction. Each available task target contributes a loss; gradients train the decoder and both input branches end-to-end.','prediction = decode(queries, encode(video))\nloss = available_geometric_losses(prediction, labels)\nloss.backward()\nAdamW.step()  # recompute F on next forward pass','A small sampled set of queries can supervise a large shared scene representation. Inference fixes weights and reuses F instead of running backward.','This is the supervised D4RT stage. VideoMAE initialization is separate; this page does not simulate a full D4RT optimizer step.','#training','#A1']
];
const state={stage:0,pair:8,row:7,col:6,src:16,tgt:30,cam:0,mode:'train',route:'d4rt'};
let timer=null;
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,s,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${esc(s)}</text>`;
const rect=(x,y,w,h,fill='#fff',stroke='#d5d8ce',r=8)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;
const line=(x,y,x2,y2,color='#969e8d',dash=false)=>`<path d="M${x},${y} L${x2},${y2}" fill="none" stroke="${color}" stroke-width="1.6" ${dash?'stroke-dasharray="5 5"':''} marker-end="url(#arr)"/>`;
const box=(x,y,w,h,label,sub,fill='#fff')=>rect(x,y,w,h,fill)+text(x+18,y+30,label)+text(x+18,y+53,sub,'small');
const defs='<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10" fill="#969e8d"/></marker></defs>';
const pixel=(t,x,y)=>{
 const cx=40+t*3.5,cy=125+18*Math.sin(t/11);
 if((x-cx)**2+(y-cy)**2<25**2)return [108,133,111];
 if(x>177&&x<219&&y>84&&y<173)return [155,147,131];
 const b=223+Math.floor(y/32)%2*5;
 return [b,b+1,b-3];
};
const frames=new Map();
function frame(t){if(frames.has(t))return frames.get(t);const c=document.createElement('canvas');c.width=c.height=256;const ctx=c.getContext('2d'),im=ctx.createImageData(256,256);for(let y=0;y<256;y++)for(let x=0;x<256;x++){let i=(y*256+x)*4;im.data.set([...pixel(t,x,y),255],i);}ctx.putImageData(im,0,0);frames.set(t,c);return c;}
function patch(t){const c=document.createElement('canvas');c.width=c.height=16;c.getContext('2d').drawImage(frame(t),state.col*16,state.row*16,16,16,0,0,16,16);return c.toDataURL();}
function raw(){const a=[];for(let t=2*state.pair;t<2*state.pair+2;t++)for(let y=state.row*16;y<state.row*16+16;y++)for(let x=state.col*16;x<state.col*16+16;x++)a.push(...pixel(t,x,y));return a;}
function imageFrame(x,y,w,t,mark=true){let s=`<image href="${frame(t).toDataURL()}" x="${x}" y="${y}" width="${w}" height="${w}"/>`;if(mark)s+=rect(x+state.col*w/16,y+state.row*w/16,w/16,w/16,'none','#253d2a',0);return s;}
function vector(x,y,count=12){let s='';for(let k=0;k<count;k++)s+=rect(x+k*14,y,10,56,k%3===0?'#859581':k%3===1?'#bdc6b6':'#dce1d6','#fff',1);return s;}
function readable(){const p=state.pair,addr=state.row*16+state.col;return Array.from({length:6144},(_,i)=>i).filter(i=>state.route==='genie'?Math.floor(i/256)<=p&&i%256===addr:state.route==='composed'?Math.floor(i/256)<=p:state.stage===3?Math.floor(i/256)===p:true);}
function drawAttention(){const c=$('pipe-attention'),ctx=c.getContext('2d');ctx.clearRect(0,0,1000,440);const allowed=new Set(readable()),selected=(state.pair*16+state.row)*16+state.col;ctx.fillStyle='#f9faf6';ctx.fillRect(0,0,1000,440);for(let g=0;g<24;g++){const x=40+(g%6)*162,y=30+Math.floor(g/6)*97;ctx.fillStyle='#75796e';ctx.font='11px monospace';ctx.fillText(`t=${2*g},${2*g+1}`,x,y-10);for(let r=0;r<16;r++)for(let col=0;col<16;col++){const i=g*256+r*16+col;ctx.fillStyle=i===selected?'#253d2a':allowed.has(i)?'#9aaa8e':'#e3e6df';ctx.fillRect(x+col*4.6,y+r*4.6,3.8,3.8);}if(g===state.pair){ctx.strokeStyle='#253d2a';ctx.lineWidth=1.5;ctx.strokeRect(x-4,y-4,81,81);}}ctx.font='12px monospace';ctx.fillStyle='#52614a';ctx.fillText('Each tile: 16 × 16 spatial addresses  /  dark square: selected token  /  green: readable evidence',40,430);}
function diagram(){const {stage,pair,row,col,src,tgt,cam}=state;const t=2*pair;let s=defs;
 if(stage===0){s+=text(40,40,'OBSERVED RGB · 48 FRAMES','label');for(let i=0;i<4;i++){let f=Math.min(47,t+i-1);f=Math.max(0,f);s+=imageFrame(45+i*238,83,205,f)+text(45+i*238,317,`frame ${f}`,'mono');}s+=text(45,366,'The same image address is highlighted across time. The green object moves through the grid.','small');}
 if(stage===1){s+=text(40,38,'A TUBELET HAS DEPTH IN TIME','label');s+=imageFrame(42,76,218,t)+imageFrame(294,76,218,t+1)+text(42,319,`frame ${t}`,'mono')+text(294,319,`frame ${t+1}`,'mono');s+=line(532,176,591,176);s+=`<image href="${patch(t+1)}" x="636" y="78" width="150" height="150" style="image-rendering:pixelated"/><image href="${patch(t)}" x="598" y="116" width="150" height="150" style="image-rendering:pixelated"/>`;s+=rect(598,116,150,150,'none','#60715a',0)+text(606,294,'2 × 16 × 16 × RGB','mono')+line(796,175,840,175)+text(847,158,'1,536','big')+text(847,183,'scalars','small');s+=text(42,366,`Crop x=${col*16}…${col*16+15}, y=${row*16}…${row*16+15}. The temporal slices share an address, not a point identity.`,'small');}
 if(stage===2){s+=text(40,38,'ONE SHARED PROJECTION FOR EVERY TUBELET','label')+box(40,107,211,90,'Flatten the RGB volume','1,536 input scalars');s+=text(57,220,raw().slice(0,9).join(' · '),'mono')+line(272,151,331,151)+box(350,107,225,90,'Learned W_patch + b','1,536 channels → C channels','#eef1e9')+line(594,151,648,151);s+=vector(672,121)+text(672,213,'one C-dimensional token','small');s+=box(350,268,225,68,'Position information','Where + when')+line(575,299,747,240)+text(40,365,'Repeat at 6,144 video addresses. Add original-aspect-ratio metadata to the encoder input.','small');}
 if(stage===5){s+=text(40,35,'A SECOND BRANCH STARTS AT THE SOURCE FRAME','label')+imageFrame(40,67,220,src,false)+rect(40+(col*16+3.5)*220/256,67+(row*16+3.5)*220/256,9*220/256,9*220/256,'none','#253d2a',0)+text(40,310,`source frame ${src}`,'mono');const u=(col*16+8)/256,v=(row*16+8)/256;s+=line(274,174,330,174)+box(349,57,260,66,'Spatial embedding',`Fourier(u=${u.toFixed(3)}, v=${v.toFixed(3)})`)+box(349,140,260,66,'Three time embeddings',`source ${src} · target ${tgt} · camera ${cam}`)+box(349,223,260,66,'Local RGB embedding','9 × 9 crop at the source pixel');s+=line(621,87,697,170)+line(621,174,697,174)+line(621,257,697,178)+box(715,130,238,88,'Sum → query MLP','One token per question','#eef1e9')+text(350,350,'Changing target or camera time changes the question; the encoded clip F stays reusable.','small');}
 if(stage===6){s+=text(40,38,'TWO INPUT STREAMS · DIFFERENT ROLES','label')+box(40,80,230,83,'Scene memory F','Shared across every query','#eef1e9')+box(40,236,230,83,'Point queries H','One row per question');s+=line(282,118,340,118)+line(282,275,340,275)+box(355,77,180,74,'Keys and values','K = FW_K · V = FW_V')+box(355,239,180,74,'Queries','Q = HW_Q');s+=line(550,115,614,191)+line(550,277,614,195)+box(630,153,326,89,'softmax(QKᵀ / √d) V','M questions read N scene features','#eef1e9');s+=text(635,285,'Query update → MLP → repeat (8 layers)','small')+text(40,367,'No H-to-H attention. Questions do not exchange information with one another.','small');}
 if(stage===7){s+=text(40,38,'ONE SET OF HEADS PER QUERY','label')+box(40,157,245,84,'Final query features','Each query is decoded independently','#eef1e9');const names=[['XYZ',3],['Image UV',2],['Visibility',1],['Motion',3],['Normal',3],['Confidence',1]];names.forEach(([name,n],i)=>{const x=440+(i%3)*178,y=74+Math.floor(i/3)*163;s+=line(300,199,x-10,y+43)+box(x,y,153,90,name,`${n} channel${n===1?'':'s'}`);});s+=text(42,367,'Change the query schedule to request a track, depth map, or point cloud from the same F.','small');}
 if(stage===8){s+=text(40,34,'FORWARD PREDICTION → SUPERVISION → PARAMETER UPDATE','label');
 s+=box(40,65,190,74,'RGB clip','48 × 256 × 256')+line(242,103,292,103)+box(305,65,190,74,'Encoder','Learned video features')+line(505,103,556,103)+box(569,65,190,74,'Query decoder','2,048 sampled queries')+line(770,103,800,103)+box(812,65,154,74,'Heads','Predictions');
 s+=box(305,167,190,64,'Query embeddings','Coordinates + source RGB')+line(505,195,662,195)+line(662,195,662,144);
 s+=box(40,266,260,70,'Known point trajectories','Labels + availability masks')+line(312,301,558,301)+box(569,266,397,70,'Geometric + auxiliary losses','Normalize sets → compare available labels','#eef1e9')+line(890,149,890,257);
 s+=line(565,282,535,282,'#637e63',true)+line(535,282,535,152,'#637e63',true)+line(535,152,403,152,'#637e63',true)+line(403,152,403,143,'#637e63',true)+line(535,222,503,222,'#637e63',true);
 s+=text(40,376,'Dashed path: gradients reach the encoder and query embeddings through the decoder.','small');}
 return s;
}
function stop(){if(timer)clearInterval(timer);timer=null;$('pipe-play').textContent='Play clip';$('pipe-play').setAttribute('aria-pressed','false');}
function select(n){stop();state.stage=Math.max(0,Math.min(state.mode==='infer'?7:8,n));state.route='d4rt';$('pipe-route').value='d4rt';render();}
function render(){const a=stages[state.stage],attention=state.stage===3||state.stage===4;
 $('pipe-kicker').textContent=`${String(state.stage+1).padStart(2,'0')} / ${state.mode==='train'?'TRAINING':'INFERENCE'} · ${a[1].toUpperCase()}`;
 ['title','shape','explanation','code','insight','boundary'].forEach((id,i)=>$("pipe-"+id).textContent=a[i+2]);
 $('pipe-lab').href='anatomy.html'+a[8];$('pipe-source').href=paper+a[9];
 $('pipe-diagram').toggleAttribute('hidden',attention);$('pipe-attention').hidden=!attention;
 $('pipe-address-controls').hidden=state.stage>5;$('pipe-play').hidden=state.stage>2;
 $('pipe-query-controls').hidden=state.stage!==5;$('pipe-routing-controls').hidden=!attention;
 for(const key of ['pair','row','col','src','tgt','cam']){$('pipe-'+key).value=state[key];$(key+'-value').textContent=key==='pair'?`${2*state.pair}–${2*state.pair+1}`:state[key];}
 if(attention)drawAttention();else{$('pipe-diagram').innerHTML=diagram();$('pipe-diagram').setAttribute('aria-label',a[2]+' '+a[4]);}
 const index=(state.pair*16+state.row)*16+state.col;
 let readout=`Selected token ${index.toLocaleString('en-US')} / 6,143 · frames ${2*state.pair}–${2*state.pair+1} · row ${state.row}, column ${state.col}.`;
 if(attention){const n=readable().length;readout+=` ${n.toLocaleString('en-US')} readable addresses. `+(state.route==='genie'?'Same spatial address, present and earlier groups only.':state.route==='composed'?'Two-hop receptive field: all spatial addresses in present and earlier groups.':state.stage===3?'Spatial mixing within this group.':'Full observed clip, including later groups.');}
 if(state.stage===2)readout+=' First RGB values: '+raw().slice(0,6).join(', ')+'. No VQ lookup.';
 if(state.stage===5)readout=`Source identifies the point at frame ${state.src}. Locate it at frame ${state.tgt}. Express XYZ in the camera frame ${state.cam}. Scene memory F is unchanged.`;
 const queryCount=Number($('trace-queries')?.value||2048);
 if(state.stage>=6)readout=state.stage===6?`Cross-attention scores per head: M × N. With ${queryCount.toLocaleString('en-US')} queries and 6,144 video tokens: ${(queryCount*6144).toLocaleString('en-US')} scores (excluding metadata).`:state.stage===7?'3 + 2 + 1 + 3 + 3 + 1 = 13 output channels. Weights are '+(state.mode==='train'?'learned by the next stage.':'fixed; F can be reused for more queries.'):'Ground-truth labels enter the loss, not the model input. Missing labels disable only their corresponding supervision.';
 $('pipe-readout').textContent=readout;
 document.querySelectorAll('[data-pipe-stage]').forEach((b,i)=>{b.disabled=state.mode==='infer'&&i===8;if(i===state.stage)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
 $('pipe-prev').disabled=state.stage===0;$('pipe-next').disabled=state.stage===(state.mode==='infer'?7:8);
 document.dispatchEvent(new CustomEvent('d4rt-stage',{detail:{...state,key:a[0]}}));
}
stages.forEach((a,i)=>{const b=document.createElement('button');b.type='button';b.dataset.pipeStage=a[0];b.innerHTML=`<small>${String(i+1).padStart(2,'0')}</small>${a[1]}`;b.addEventListener('click',()=>select(i));$('pipe-tabs').append(b);});
for(const key of ['pair','row','col','src','tgt','cam'])$('pipe-'+key).addEventListener('input',e=>{stop();state[key]=Number(e.target.value);render();});
$('pipe-mode').addEventListener('change',e=>{state.mode=e.target.value;select(state.stage);});
$('pipe-route').addEventListener('change',e=>{state.route=e.target.value;render();});
document.addEventListener('d4rt-query-count',render);
$('pipe-next').addEventListener('click',()=>select(state.stage+1));$('pipe-prev').addEventListener('click',()=>select(state.stage-1));
$('pipe-play').addEventListener('click',()=>{if(timer)return stop();$('pipe-play').textContent='Pause clip';$('pipe-play').setAttribute('aria-pressed','true');timer=setInterval(()=>{state.pair=(state.pair+1)%24;render();},450);});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
$('pipe-attention').addEventListener('click',e=>{const b=e.currentTarget.getBoundingClientRect(),x=(e.clientX-b.left)*1000/b.width,y=(e.clientY-b.top)*440/b.height;for(let g=0;g<24;g++){const dx=x-(40+g%6*162),dy=y-(30+Math.floor(g/6)*97);if(dx>=0&&dy>=0&&dx<16*4.6&&dy<16*4.6){state.pair=g;state.row=Math.floor(dy/4.6);state.col=Math.floor(dx/4.6);render();break;}}});
$('pipe-genie-jump').addEventListener('click',()=>{select(4);state.route='composed';$('pipe-route').value='composed';render();$('walkthrough').scrollIntoView();});
// Existing published deep links continue to open their original detailed experiments.
const oldHashes=['model-atlas','tokenizer','encoder','decoder','training','genie','lifetime','tour','learning','sources','overview','execution','workbench','anatomy','microscope','query','geometry','attention','training-lab','model-lifetime','architecture','query-lab','recipes','efficiency','understanding','gradient-lab','attention-math','tubelet-volume'];
function followLegacy(){if(oldHashes.includes(location.hash.slice(1)) || /^#stage-[0-5]$/.test(location.hash))location.replace('anatomy.html'+location.hash);}
function followStage(){
 const index=stages.findIndex(s=>location.hash==='#walk-'+s[0]);
 if(index<0)return;
 if(index===8){state.mode='train';$('pipe-mode').value='train';}
 select(index);
 $('walkthrough').scrollIntoView({block:'start'});
}
window.addEventListener('hashchange',()=>{followLegacy();followStage();});followLegacy();
document.addEventListener('click',e=>{
 const link=e.target.closest('a[href^="#walk-"]');
 // A repeated permalink has no hashchange event, but must still select its step.
 if(link&&link.hash===location.hash){e.preventDefault();followStage();}
});
render();
window.D4RTPipeline={snapshot:()=>({...state,key:stages[state.stage][0],index:(state.pair*16+state.row)*16+state.col,readable:readable(),raw:raw()}),pixel};
followStage();
})();
