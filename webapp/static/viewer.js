// Minimal dependency-free crystal viewer: depth-sorted shaded spheres on a
// canvas, unit-cell edges, drag to rotate, wheel/pinch to zoom.
(function () {
  const cv = document.getElementById('cv'), box = document.getElementById('viewer');
  if (!cv || !box) return;
  const ctx = cv.getContext('2d');
  // CPK-style colours; Li shares the app's lithium violet
  const COL = {Li:'#8c5ae8',O:'#d8432f',S:'#d9a400',Se:'#c47a00',Te:'#a06a1e',F:'#56b04a',Cl:'#2f9e57',Br:'#8a3a1e',I:'#6b2f8f',
    N:'#3a64d0',P:'#e07b1a',B:'#e8a0a0',C:'#555',H:'#d8d8d8',Si:'#c9a27a',Ge:'#668f8f',Sn:'#6e7d8a',Al:'#a6a6c8',Ga:'#c28f8f',
    In:'#a67573',La:'#5ec7e0',Zr:'#7fb3b3',Mg:'#79c94c',Ca:'#4fa04f',Na:'#ab5cf2',K:'#8f40d4',Cs:'#57178f',Rb:'#702eb0',
    Ba:'#00c900',Sr:'#00b000',Y:'#7fd4d4',Er:'#00c97a',Sb:'#9e63b5',As:'#bd80e3',Bi:'#9e4fb5',Be:'#c2ff00',Zn:'#7d80b0'};
  const RAD = {Li:0.62,O:0.62,S:0.82,Se:0.88,Te:0.95,F:0.58,Cl:0.80,Br:0.88,I:0.98,N:0.62,H:0.32,B:0.48,C:0.52,P:0.70};
  const col = e => COL[e] || '#8b97a3', rad = e => RAD[e] || 0.72;
  let L, atoms = [], center = [0,0,0], corners = [], span = 1;
  const edges = []; for (let i=0;i<8;i++) for (let j=i+1;j<8;j++){ const d=(i^j); if(d===1||d===2||d===4) edges.push([i,j]); }
  let rx = -0.45, ry = 0.6, zoom = 1, drag = null, auto = !matchMedia('(prefers-reduced-motion: reduce)').matches;

  function setStruct(S){
    if (!S) return;
    L = S.lattice; atoms = S.atoms || [];
    const corner = (a,b,c)=>[0,1,2].map(i=>a*L[0][i]+b*L[1][i]+c*L[2][i]);
    center = corner(.5,.5,.5);
    corners = []; for (const a of [0,1]) for (const b of [0,1]) for (const c of [0,1]) corners.push(corner(a,b,c));
    span = Math.max(...corners.map(p=>Math.hypot(p[0]-center[0],p[1]-center[1],p[2]-center[2])));
    const els = [...new Set([...atoms.map(a=>a.el), ...(S.elements||[])])];
    const lg = document.getElementById('vlegend');
    if (lg) lg.innerHTML = els.map(e=>`<span><i style="background:${col(e)}"></i>${e}</span>`).join('');
  }
  window.CSViewer = { set: setStruct };
  setStruct(window.STRUCT);

  function rot(p){
    let x=p[0]-center[0], y=p[1]-center[1], z=p[2]-center[2];
    let c=Math.cos(ry), s=Math.sin(ry); [x,z]=[c*x+s*z,-s*x+c*z];
    c=Math.cos(rx); s=Math.sin(rx); [y,z]=[c*y-s*z,s*y+c*z];
    return [x,y,z];
  }
  function draw(){
    if (!L) return;
    const dpr = window.devicePixelRatio||1, w = box.clientWidth, h = box.clientHeight;
    if (cv.width !== w*dpr || cv.height !== h*dpr){ cv.width=w*dpr; cv.height=h*dpr; }
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
    const scale = Math.min(w,h)/(2.3*span)*zoom, persp = span*4;
    const P = p => { const r=rot(p), f=persp/(persp+r[2]); return [w/2+r[0]*scale*f, h/2-r[1]*scale*f, r[2], f]; };
    const line = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() || '#888';
    ctx.strokeStyle=line; ctx.globalAlpha=.55; ctx.lineWidth=1;
    const pc = corners.map(P);
    ctx.beginPath(); for (const [i,j] of edges){ ctx.moveTo(pc[i][0],pc[i][1]); ctx.lineTo(pc[j][0],pc[j][1]); } ctx.stroke();
    ctx.globalAlpha=1;
    const pts = atoms.map(a=>{ const p=P(a.xyz); return {a, x:p[0], y:p[1], z:p[2], f:p[3]}; }).sort((u,v)=>v.z-u.z);
    for (const p of pts){
      const r = rad(p.a.el)*scale*p.f*0.55; if (r<0.6) continue;
      const g = ctx.createRadialGradient(p.x-r*.35,p.y-r*.35,r*.1,p.x,p.y,r);
      g.addColorStop(0,'#ffffff'); g.addColorStop(.25,col(p.a.el)); g.addColorStop(1,shade(col(p.a.el),-.45));
      ctx.globalAlpha = Math.max(.25, Math.min(1, p.a.occ));
      ctx.fillStyle=g; ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fill();
    }
    ctx.globalAlpha=1;
  }
  function shade(hex,amt){ const n=parseInt(hex.slice(1),16); let r=n>>16,g=n>>8&255,b=n&255;
    r=Math.round(r*(1+amt)); g=Math.round(g*(1+amt)); b=Math.round(b*(1+amt));
    return `rgb(${Math.max(0,r)},${Math.max(0,g)},${Math.max(0,b)})`; }
  function loop(){ if (auto && !drag) { ry += 0.004; } draw(); requestAnimationFrame(loop); }
  box.addEventListener('pointerdown', e=>{ drag={x:e.clientX,y:e.clientY}; auto=false; box.setPointerCapture(e.pointerId); box.style.cursor='grabbing'; });
  box.addEventListener('pointermove', e=>{ if(!drag) return; ry += (e.clientX-drag.x)*0.01; rx += (e.clientY-drag.y)*0.01; drag={x:e.clientX,y:e.clientY}; });
  const end = e=>{ drag=null; box.style.cursor='grab'; };
  box.addEventListener('pointerup', end); box.addEventListener('pointercancel', end);
  box.addEventListener('wheel', e=>{ e.preventDefault(); zoom = Math.min(4, Math.max(.4, zoom*(e.deltaY<0?1.1:0.9))); }, {passive:false});
  document.addEventListener('themechange', draw);
  loop();
})();
