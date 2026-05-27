class NeuralNetwork {
  constructor(sizes, weights=null, biases=null){
    this.sizes = sizes.slice();
    this.weights = weights || this.sizes.slice(1).map((n,i)=>{
      const r = this.sizes[i]; const c = this.sizes[i+1];
      return Array.from({length:c},()=>Array.from({length:r},()=>(Math.random()*2-1)));
    });
    this.biases = biases || this.sizes.slice(1).map(n=>Array.from({length:n},()=>Math.random()*2-1));
  }
  feed(inputs){
    let a = inputs.slice();
    const activations = [a.slice()];
    for(let i=0;i<this.weights.length;i++){
      const w = this.weights[i]; const b = this.biases[i];
      const next = Array.from({length:w.length},(_,j)=>{
        let sum = b[j];
        for(let k=0;k<w[j].length;k++) sum += w[j][k]*a[k];
        return Math.tanh(sum);
      });
      a = next; activations.push(a.slice());
    }
    return {output: a, activations};
  }
  clone(){
    const w = this.weights.map(layer=>layer.map(row=>row.slice()));
    const b = this.biases.map(layer=>layer.slice());
    return new NeuralNetwork(this.sizes, w, b);
  }
  mutate(rate){
    this.weights.forEach(layer=>layer.forEach(row=>row.forEach((v,i,arr)=>{ if(Math.random()<rate) arr[i]+= (Math.random()*2-1)*0.5 })));
    this.biases.forEach(layer=>layer.forEach((v,i,arr)=>{ if(Math.random()<rate) arr[i]+= (Math.random()*2-1)*0.5 }));
  }
}

class SnakeGame {
  constructor(size=12){ this.size=size; this.reset(); }
  reset(){ this.snake=[[Math.floor(this.size/2),Math.floor(this.size/2)]]; this.dir=[0,-1]; this.placeFood(); this.alive=true; this.steps=0; this.score=0; this.maxLen=3 }
  placeFood(){
    while(true){ const x=Math.floor(Math.random()*this.size), y=Math.floor(Math.random()*this.size);
      if(!this.snake.some(p=>p[0]===x && p[1]===y)){ this.food=[x,y]; break }
    }
  }
  step(action){
    if(!this.alive) return;
    const dirs = [[0,-1],[1,0],[0,1],[-1,0]];
    if(action!=null){ const idx = action; this.dir = dirs[idx]; }
    const head = [this.snake[0][0]+this.dir[0], this.snake[0][1]+this.dir[1]];
    this.steps++;
    if(head[0]<0||head[1]<0||head[0]>=this.size||head[1]>=this.size || this.snake.some(p=>p[0]===head[0]&&p[1]===head[1])){ this.alive=false; return }
    this.snake.unshift(head);
    if(head[0]===this.food[0] && head[1]===this.food[1]){ this.score++; this.maxLen+=1; this.placeFood(); }
    if(this.snake.length>this.maxLen) this.snake.pop();
  }
  getInputs(options){
    const ins = [];
    const dir = this.dir; const dirs = [[0,-1],[1,0],[0,1],[-1,0]];
    if(options.includes('dir')){
      const one = dirs.map(d=> (d[0]===dir[0] && d[1]===dir[1])?1:0);
      ins.push(...one);
    }
    if(options.includes('foodVec')){
      const dx = (this.food[0]-this.snake[0][0])/this.size; const dy=(this.food[1]-this.snake[0][1])/this.size; ins.push(dx,dy);
    }
    if(options.includes('distFood')){
      const d = Math.hypot(this.food[0]-this.snake[0][0], this.food[1]-this.snake[0][1])/Math.hypot(this.size,this.size);
      ins.push(d);
    }
    if(options.includes('vision')){
      const left = [-dir[1], dir[0]]; const right=[dir[1], -dir[0]];
      const check = v=>{ const p=[this.snake[0][0]+v[0], this.snake[0][1]+v[1]]; return (p[0]<0||p[1]<0||p[0]>=this.size||p[1]>=this.size|| this.snake.some(s=>s[0]===p[0]&&s[1]===p[1]))?1:0 }
      ins.push(check(dir), check(left), check(right));
    }
    return ins;
  }
}

class Agent {
  constructor(nn,options,gridSize=12){ this.nn=nn; this.options=options; this.game=new SnakeGame(gridSize); }
  act(){ const input = this.game.getInputs(this.options.inputs); const out = this.nn.feed(input).output; let idx = 0; let best = -Infinity; for(let i=0;i<out.length;i++){ if(out[i]>best){best=out[i]; idx=i} } this.game.step(idx); }
}

class Evolver {
  constructor(cfg){ this.cfg=cfg; this.population=[]; this.generation=0; }
  init(){
    this.population = [];
    const layerCount = Number(document.getElementById('layers').value);
    const neurons = Number(document.getElementById('neurons').value);
    const inputs = this.cfg.inputsCount();
    const sizes = [inputs];
    for(let i=0;i<layerCount;i++) sizes.push(neurons);
    sizes.push(4);
    for(let i=0;i<this.cfg.population;i++){
      const nn = new NeuralNetwork(sizes);
      this.population.push({nn,fitness:0,score:0});
    }
  }
  async runOneGenerationAnimated(ui){
    const maxSteps = this.cfg.maxSteps;
    const pop = this.population;
    const stepsPerFrame = Number(document.getElementById('speed')?.value || 4);
    pop.forEach(p=>{ p.agent = new Agent(p.nn, this.cfg, 12); p.agent.game.reset(); p._steps=0; p._alive=true; });
    let currStep = 0; let aliveCount = pop.length;
    const canvases = ui.ensureCanvases(pop.length);
    await new Promise(resolve=>{
      const frame = ()=>{
        for(let s=0;s<stepsPerFrame;s++){
          currStep++;
          for(const p of pop){ if(!p._alive) continue; p.agent.act(); p._steps++; if(!p.agent.game.alive) { p._alive=false; aliveCount--; } }
          if(currStep>=maxSteps || aliveCount===0) break;
        }
        pop.forEach((p,i)=>{ drawGameOnCanvas(canvases[i].ctx, p.agent.game, canvases[i].canvas, !p._alive); canvases[i].label.textContent = `#${i+1} s:${p._steps} sc:${p.agent.game.score}`; if(!p._alive) canvases[i].canvas.classList.add('dead'); else canvases[i].canvas.classList.remove('dead'); });
        const pct = Math.min(100, Math.round(currStep/maxSteps*100));
        document.getElementById('genProgress').value = pct; document.getElementById('progressPct').textContent = pct+"%";
        if(currStep>=maxSteps || aliveCount===0){ resolve(); return; }
        requestAnimationFrame(frame);
      };
      requestAnimationFrame(frame);
    });
    for(const p of pop){ p.fitness = p.agent.game.score*100 + (p._alive? p._steps:0); p.score = p.agent.game.score }
    pop.sort((a,b)=>b.fitness-a.fitness);
    this.generation++;
    const survivors = Math.max(2, Math.floor(pop.length*0.2));
    const newPop=[]; for(let i=0;i<survivors;i++) newPop.push({nn:pop[i].nn.clone(),fitness:0,score:0});
    while(newPop.length<pop.length){ const i = Math.floor(Math.random()*survivors); const child = newPop[i].nn.clone(); child.mutate(this.cfg.mutationRate); newPop.push({nn:child,fitness:0,score:0}); }
    this.population = newPop;
    return {best:pop[0], generation:this.generation};
  }
}

// UI wiring and main loop
const UI = {
  init(){ this.gamesEl = document.getElementById('games'); this.weightsView = document.getElementById('weightsView'); this.generationEl = document.getElementById('generation'); this.bestFitnessEl = document.getElementById('bestFitness'); this.bestScoreEl = document.getElementById('bestScore'); this.bind(); this.running=false; this.evolver=null; },
  bind(){ document.getElementById('start').addEventListener('click',()=>this.start()); document.getElementById('pause').addEventListener('click',()=>this.pause()); document.getElementById('reset').addEventListener('click',()=>this.reset()); },
  readConfig(){ const inp = Array.from(document.querySelectorAll('.input-var:checked')).map(el=>el.value); const population = Number(document.getElementById('population').value); const mutation = Number(document.getElementById('mutation').value); const maxSteps = Number(document.getElementById('maxSteps').value); return {inputs:inp, population, mutationRate:mutation, maxSteps, inputsCount:()=>{ let c=0; if(inp.includes('dir')) c+=4; if(inp.includes('foodVec')) c+=2; if(inp.includes('distFood')) c+=1; if(inp.includes('vision')) c+=3; return c }}; },
  async start(){ if(this.running) return; this.running=true; const cfg = this.readConfig(); cfg.population = cfg.population; cfg.mutationRate = cfg.mutationRate; cfg.maxSteps = cfg.maxSteps; cfg.inputsCount = cfg.inputsCount; cfg.inputs = cfg.inputs; document.getElementById('generation').textContent='0'; this.evolver = new Evolver(cfg); this.evolver.init(); this.createCanvasCards(this.evolver.population.length); while(this.running){ const res = await this.evolver.runOneGenerationAnimated(this); this.updateUI(res); this.createCanvasCards(this.evolver.population.length); await this._shortDelay(50); } },
  pause(){ this.running=false },
  reset(){ this.pause(); this.evolver=null; this.gamesEl.innerHTML=''; this.weightsView.textContent='No data yet'; document.getElementById('generation').textContent='0'; this.bestFitnessEl.textContent='0'; this.bestScoreEl.textContent='0' },
  updateUI(res){ this.generationEl.textContent = res.generation; this.bestFitnessEl.textContent = Math.round(res.best.fitness); this.bestScoreEl.textContent = res.best.score; const nn = res.best.nn; this.weightsView.textContent = 'Sizes: '+nn.sizes.join(',')+"\n\nWeights:\n"+JSON.stringify(nn.weights.map(l=>l.map(r=>r.map(v=>+v.toFixed(2)))),null,2)+"\n\nBiases:\n"+JSON.stringify(nn.biases.map(l=>l.map(v=>+v.toFixed(2))),null,2); if(this.canvasCards) this.canvasCards.forEach((c,i)=>{ c.canvas.classList.toggle('best', i===0); }); },
  createCanvasCards(n){ this.gamesEl.innerHTML=''; this.canvasCards = []; for(let i=0;i<n;i++){ const card = document.createElement('div'); card.className='game-card'; const c = document.createElement('canvas'); c.width=140; c.height=140; c.className='game-canvas'; const label = document.createElement('div'); label.className='game-label'; label.textContent = `#${i+1}`; card.appendChild(c); card.appendChild(label); this.gamesEl.appendChild(card); this.canvasCards.push({canvas:c,ctx:c.getContext('2d'),label}); } return this.canvasCards; },
  ensureCanvases(n){ if(!this.canvasCards || this.canvasCards.length!==n) return this.createCanvasCards(n); return this.canvasCards; },
  _shortDelay(ms){ return new Promise(r=>setTimeout(r,ms)); }
}

function drawGameOnCanvas(g, game, canvas, dead=false){
  const w = canvas.width, h=canvas.height; g.clearRect(0,0,w,h);
  const size = game.size; const cell = Math.floor(Math.min(w,h)/size);
  g.fillStyle = '#f8fafc'; g.fillRect(0,0,w,h);
  g.strokeStyle = '#eef2f6'; g.lineWidth=1;
  for(let x=0;x<=size;x++){ g.beginPath(); g.moveTo(x*cell,0); g.lineTo(x*cell,cell*size); g.stroke(); }
  for(let y=0;y<=size;y++){ g.beginPath(); g.moveTo(0,y*cell); g.lineTo(cell*size,y*cell); g.stroke(); }
  if(game.food){ g.fillStyle='#e11d48'; g.fillRect(game.food[0]*cell+1, game.food[1]*cell+1, cell-2, cell-2); }
  if(game.snake){ for(let i=game.snake.length-1;i>=0;i--){ const s=game.snake[i]; if(i===0){ g.fillStyle='#0366d6'; } else { g.fillStyle='#0ea5a4'; } g.fillRect(s[0]*cell+1, s[1]*cell+1, cell-2, cell-2); } }
  g.fillStyle='rgba(0,0,0,0.6)'; g.font='12px sans-serif'; g.fillText(`s:${game.score}`, 6, 12);
  if(dead){ g.fillStyle='rgba(255,255,255,0.6)'; g.fillRect(0, h-18, 60, 18); g.fillStyle='#000'; g.fillText('dead',6,h-4); }
}

UI.init();

// Defensive: remove or hide injected dev-server overlays that show 'lost connection' messages
function removeDevOverlays(){
  const texts = ['lost connection','lost connection to','connection to the server','live-reload'];
  const els = Array.from(document.querySelectorAll('div,span,pre'));
  for(const el of els){
    const t = (el.textContent||'').toLowerCase();
    for(const s of texts){ if(t.includes(s)){ el.style.display='none'; const p = el.parentElement; if(p) p.style.display='none'; } }
  }
}
setInterval(removeDevOverlays,1000);

// Show connection status in our UI; if the page goes offline, indicate it
function updateConnStatus(){
  const el = document.getElementById('connStatus'); if(!el) return;
  el.textContent = navigator.onLine ? 'online' : 'offline';
  el.style.color = navigator.onLine ? '#0b6623' : '#b91c1c';
}
window.addEventListener('online', updateConnStatus);
window.addEventListener('offline', updateConnStatus);
updateConnStatus();
