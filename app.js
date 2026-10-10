(() => {
  'use strict';
  const $=id=>document.getElementById(id),S=window.D4RTScene;
  const state={src:0,tgt:6,cam:0,selected:S.presets.sphere,yaw:.65,path:true};
  let sourcePoints=[],timer=null,recipe='track',tracked=0;
  const pad=n=>String(n+1).padStart(2,'0');
  const fmt=n=>(Math.abs(n)<.005?0:n).toFixed(2).replace('-','−');
  function update(){
    ['src','tgt','cam'].forEach(k=>{$(`${k}-out`).value=pad(state[k]);$(`${k}-time`).value=state[k];});
    $('source-frame-label').textContent=`FRAME ${pad(state.src)}`;$('target-frame-label').textContent=`FRAME ${pad(state.tgt)}`;
    sourcePoints=S.render($('source-canvas'),{time:state.src,source:true,camTime:state.src,selected:state.selected});
    S.render($('world-canvas'),{time:state.tgt,camTime:state.cam,selected:state.selected,yaw:state.yaw,path:state.path});
    const spec=S.byId.get(state.selected),sourcePoint=S.project(S.position(spec,state.src),S.camera(state.src),$('source-canvas').clientWidth,$('source-canvas').clientHeight);
    const u=sourcePoint.x/$('source-canvas').clientWidth,v=sourcePoint.y/$('source-canvas').clientHeight;
    const p=S.cameraPoint(S.position(spec,state.tgt),S.camera(state.cam));
    $('query-value').textContent=`q = (${u.toFixed(2)}, ${v.toFixed(2)}, ${state.src+1}, ${state.tgt+1}, ${state.cam+1})`;
    $('point-value').textContent=`P = (${p.map(fmt).join(', ')})`;
    const same=state.src===state.tgt&&state.tgt===state.cam;
    $('query-insight').textContent=same?`All three times match. The point’s Z coordinate (${fmt(p[2])}) is its depth in this camera frame.`:spec.type==='floor'?`This floor point never moves in our scene. Camera ${pad(state.cam)} supplies the coordinate system; moving target time alone leaves P unchanged.`:`We identify the point in frame ${pad(state.src)}, locate it at frame ${pad(state.tgt)}, and express that position in camera ${pad(state.cam)} coordinates. Orbiting the illustration changes none of these choices.`;
  }
  function stop(){if(timer)clearInterval(timer);timer=null;$('play-time').textContent='▶ Play target time';$('play-time').setAttribute('aria-pressed','false');}
  ['src','tgt','cam'].forEach(k=>$(`${k}-time`).addEventListener('input',e=>{stop();state[k]=Number(e.target.value);update();}));
  $('point-select').addEventListener('change',e=>{state.selected=S.presets[e.target.value];update();});
  $('show-path').addEventListener('change',e=>{state.path=e.target.checked;update();});
  $('play-time').addEventListener('click',()=>{if(timer){stop();return;}$('play-time').textContent='Ⅱ Pause';$('play-time').setAttribute('aria-pressed','true');timer=setInterval(()=>{state.tgt=(state.tgt+1)%12;update();},650);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  $('source-canvas').addEventListener('click',e=>{const b=e.target.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top;let best=null,d=400;sourcePoints.forEach(p=>{const v=(x-p.x)**2+(y-p.y)**2;if(v<d){best=p;d=v;}});if(best){state.selected=best.spec.id;$('point-select').value=best.spec.type;update();}});
  let drag=null;
  $('world-canvas').addEventListener('pointerdown',e=>{drag={x:e.clientX,yaw:state.yaw};e.target.setPointerCapture(e.pointerId);});
  $('world-canvas').addEventListener('pointermove',e=>{if(drag){state.yaw=drag.yaw+(e.clientX-drag.x)*.009;update();}});
  ['pointerup','pointercancel','lostpointercapture'].forEach(type=>$('world-canvas').addEventListener(type,()=>{drag=null;}));
  $('world-canvas').addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();state.yaw+=e.key==='ArrowLeft'?-.15:.15;update();}});
  $('reset-orbit').addEventListener('click',()=>{state.yaw=.65;update();});
  const stages=[
    ['THE INPUT','A clip, not an isolated image.','The input is an RGB video. Across its frames, both the objects and the camera may move.','V : T × H × W × 3'],
    ['THE ENCODER','Connect evidence across frames.','A vision transformer mixes within-frame and across-frame attention. It turns the clip into shared features.','F = E(V)'],
    ['THE REPRESENTATION','A common notebook for the scene.','The global scene representation stays fixed while queries are decoded. The notebook metaphor is illustrative; these are learned continuous features.','F : N × C'],
    ['THE DECODER','Ask about one physical point.','A query combines position, three time indices and local appearance. Cross-attention reads the scene features; decoder queries do not attend to each other.','q + local RGB patch → cross-attend to F'],
    ['THE OUTPUT','A position in the chosen coordinates.','The main output is a 3D point. The model also learns auxiliary quantities such as visibility.','P = D(q, F) ∈ ℝ³']
  ];
  function graphic(stage){let parts='';if(stage===0){for(let i=0;i<4;i++)parts+=`<rect x="${35+i*22}" y="${70-i*14}" width="100" height="62" rx="4" fill="#fafaf8" stroke="#a8afa1"/><circle cx="${65+i*30}" cy="${100-i*14}" r="9" fill="#919d8c"/>`;}else if(stage===4){parts='<path d="M45 110H205M115 140V20M65 145 160 45" stroke="#d2d6ca"/><circle cx="145" cy="65" r="9" fill="#7f8e77"/><circle cx="145" cy="65" r="16" fill="none" stroke="#c5cebf"/><path d="M145 65V110H115" fill="none" stroke="#a9b69f" stroke-dasharray="3 4"/>';}else{for(let i=0;i<35;i++){const x=37+(i%7)*23,y=25+Math.floor(i/7)*23;const opacity=.25+((i*13+stage*7)%11)/16;parts+=`<rect x="${x}" y="${y}" width="15" height="15" rx="2" fill="#768a70" opacity="${opacity}"/>`;}if(stage===3)parts+='<circle cx="233" cy="75" r="10" fill="#4e6249"/><path d="M220 75 75 40M220 75 140 60M220 75 95 100" stroke="#b1bda8" fill="none" stroke-dasharray="3 3"/>';}
    return `<svg viewBox="0 0 270 160">${parts}</svg>`;
  }
  function setStage(i){document.querySelectorAll('[data-stage]').forEach(b=>{const on=Number(b.dataset.stage)===i;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;});$('stage-panel').setAttribute('aria-labelledby',`stage-tab-${i}`);const a=stages[i];['stage-kicker','stage-title','stage-description','stage-code'].forEach((id,j)=>$(id).textContent=a[j]);$('stage-graphic').innerHTML=graphic(i);}
  document.querySelectorAll('[data-stage]').forEach(b=>b.addEventListener('click',()=>setStage(Number(b.dataset.stage))));
  const recipes={
    track:{kicker:'ONE IDENTITY, MANY TIMES',title:'Follow the same point.',description:'Keep the source point fixed and sweep target time. Keeping the reference camera fixed makes a shared-coordinate trajectory.',settings:[['source','one (u, v, t_src)'],['target','sweep 1 … T'],['camera','hold one reference']]},
    cloud:{kicker:'MANY POINTS, ONE MOMENT',title:'Assemble a 3D snapshot.',description:'Sample source pixels and query a common target time and camera frame. Each returned position joins the point cloud.',settings:[['source','many source pixels'],['target','one selected time'],['camera','one reference']]},
    depth:{kicker:'ONE FRAME, ONE COORDINATE',title:'Read the depth channel.',description:'Set source, target and camera to the same frame. Query its pixels and keep the Z component.',settings:[['source','pixels in frame i'],['target','i'],['camera','i → keep P_z']]},
    pose:{kicker:'SAME POINTS, TWO RULERS',title:'Recover camera motion.',description:'Hold the points and their time fixed, query them in two camera frames, then align the two 3D point sets.',settings:[['source','same sampled pixels'],['target','fixed'],['camera','i and j → align sets']]}
  };
  function recipeSvg(key){let g='<path d="M30 190H300M50 215 155 50M160 210V25" stroke="#e2e4dc" fill="none"/>';if(key==='track'){g+='<path d="M35 175C70 70 120 180 155 100S230 20 290 70" fill="none" stroke="#89997e" stroke-width="2" stroke-dasharray="3 5"/>';[[35,175],[91,121],[157,97],[221,51],[290,70]].forEach(([x,y],i)=>{g+=`<circle cx="${x}" cy="${y}" r="${i===2?7:4}" fill="${i===2?'#475e40':'#a8b19f'}"/><text x="${x-5}" y="${y+22}" font-size="10" fill="#8a9183">t${i+1}</text>`;});}else if(key==='cloud'||key==='depth'){for(let i=0;i<160;i++){const a=i*2.399,r=Math.sqrt(i/160)*78,x=165+Math.cos(a)*r,y=119+Math.sin(a)*r*.78;g+=`<circle cx="${x}" cy="${y}" r="${key==='depth'?4:2.2}" fill="${key==='depth'?`hsl(90 5% ${35+((i*7)%50)}%)`:'#8d9c82'}"/>`;}}else{for(let i=0;i<15;i++){const x=48+(i%5)*22,y=77+Math.floor(i/5)*29;g+=`<circle cx="${x}" cy="${y}" r="3" fill="#7f9074"/><circle cx="${x+150}" cy="${y-15+(i%5)*3}" r="3" fill="#b1b8a9"/>`;}g+='<path d="M145 45Q180 20 205 43" fill="none" stroke="#969f8b"/><text x="145" y="18" font-size="10" fill="#89917e">R, translation</text>';}
    return g;
  }
  function setRecipe(key){recipe=key;const a=recipes[key];document.querySelectorAll('[data-recipe]').forEach(b=>{const on=b.dataset.recipe===key;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;});$('recipe-panel').setAttribute('aria-labelledby',`recipe-tab-${key}`);$('recipe-kicker').textContent=a.kicker;$('recipe-title').textContent=a.title;$('recipe-description').textContent=a.description;$('recipe-settings').innerHTML=a.settings.map(([k,v])=>`<span><b>${k}</b>${v}</span>`).join('');$('recipe-svg').innerHTML=recipeSvg(key);$('recipe-svg').setAttribute('aria-label',a.title+' '+a.description);}
  document.querySelectorAll('[data-recipe]').forEach(b=>b.addEventListener('click',()=>setRecipe(b.dataset.recipe)));
  document.querySelectorAll('[role=tablist]').forEach(list=>list.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;const tabs=[...list.querySelectorAll('[role=tab]')],i=tabs.indexOf(document.activeElement);if(i<0)return;e.preventDefault();const n=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[n].click();tabs[n].focus();}));
  $('try-recipe').addEventListener('click',()=>{stop();state.src=0;state.cam=0;state.tgt=recipe==='depth'?0:6;state.path=recipe==='track';$('show-path').checked=state.path;state.selected=S.presets[recipe==='pose'?'floor':'sphere'];$('point-select').value=recipe==='pose'?'floor':'sphere';update();$('query-lab').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});$('tgt-time').focus({preventScroll:true});});
  const cells=[];for(let r=0;r<8;r++)for(let t=0;t<12;t++){const el=document.createElement('div');el.dataset.track=String((r-t%8+8)%8);el.title=`Frame ${t+1}, sample ${r+1}`;$('occupancy-grid').append(el);cells.push(el);}
  function occupancy(){cells.forEach(c=>{const id=Number(c.dataset.track);c.classList.toggle('covered',id<tracked);c.classList.toggle('latest',id===tracked-1);});$('covered-count').innerHTML=`${tracked*12}<span> / 96</span>`;$('track-count').textContent=`${tracked} ${tracked===1?'track':'tracks'} decoded · ${tracked*12} queries`;$('next-track').disabled=tracked===8;$('next-track').textContent=tracked===8?'All visible samples covered':'Trace the next point →';$('occupancy-grid').setAttribute('aria-label',`${tracked*12} of 96 visible samples covered by ${tracked} tracks.`);}
  $('next-track').addEventListener('click',()=>{tracked=Math.min(8,tracked+1);occupancy();});$('reset-tracks').addEventListener('click',()=>{tracked=0;occupancy();});
  document.querySelectorAll('[data-answer]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-answer]').forEach(a=>a.setAttribute('aria-pressed',String(a===b)));$('quiz-feedback').textContent=b.dataset.answer==='no'?'Exactly. A position is expressed relative to a coordinate system. Changing the reference camera can change x, y and z without moving the statue.':'The statue stays put, but the ruler moved. A different reference camera can assign different coordinates to the same physical position.';}));
  const resize=new ResizeObserver(()=>{update();S.render($('hero-canvas'),{time:6,selected:S.presets.sphere,yaw:.65,hero:true});});['source-canvas','world-canvas','hero-canvas'].forEach(id=>resize.observe($(id)));
  const sections=[...document.querySelectorAll('.chapter')];let queued=false;
  function progress(){queued=false;const max=document.documentElement.scrollHeight-innerHeight;$('reading-progress').style.width=`${max?100*scrollY/max:0}%`;let active=sections[0];sections.forEach(s=>{if(s.getBoundingClientRect().top<innerHeight*.35)active=s;});document.querySelectorAll('.rail nav a').forEach(a=>{const on=a.hash===`#${active.id}`;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});}
  addEventListener('scroll',()=>{if(!queued){queued=true;requestAnimationFrame(progress);}},{passive:true});
  setStage(0);setRecipe('track');occupancy();update();progress();
})();
