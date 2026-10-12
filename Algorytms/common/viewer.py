"""Interactive HTML viewer: 3D trajectory + predictions of every evaluated model.

For each test trajectory one page is written to Results/viewer/. Moving the
time slider (or clicking the error / speed charts) shows the mosquito's recent
path, the true future and every model's predicted future for the chosen
horizon. Pages load plotly from a local plotly.min.js, so they work offline.

Usage (from the repo root, after running at least one algorithm):
    .venv/Scripts/python -m Algorytms.common.viewer
"""
import json

import numpy as np

from Algorytms.common import data
from Algorytms.common.data import FRAME_S, RESULTS_DIR

VIEW_DIR = RESULTS_DIR / "viewer"


def _r(a, nd=2):
    a = np.asarray(a, float)
    return np.where(np.isnan(a), None, np.round(a, nd)).tolist()


def model_names():
    """Result folders (relative to Results/, e.g. 'kalman/cv_0.336mps') that contain viewer data."""
    return sorted({p.parent.relative_to(RESULTS_DIR).as_posix() for p in RESULTS_DIR.rglob("viewer_data")})


def model_params(name):
    f = RESULTS_DIR / name / "params.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def model_label(name):
    """Display name stored by evaluate() in params.json, e.g. 'Kalman CV (0.336 m/s)'."""
    return model_params(name).get("label", name)


def page_payload(traj, models):
    payload = {"tid": traj.tid, "dt": FRAME_S, "pos": _r(traj.pos), "speed": _r(traj.speed, 3),
               "acc": _r(traj.acc, 2), "measured": traj.measured.astype(int).tolist(), "models": {}}
    for name in models:
        f = RESULTS_DIR / name / "viewer_data" / f"traj_{traj.tid:05d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        payload["models"][name] = {
            "label": model_label(name),
            "origins": z["origins"].tolist(), "err_steps": z["err_steps"].tolist(), "err": _r(z["err"].T, 2),
            "view_origins": z["view_origins"].tolist(), "view_steps": z["view_steps"].tolist(),
            "view_preds": _r(z["view_preds"], 1),
        }
        if "cloud_states" in z:   # particle cloud: [x, y, z, vx, vy, vz] per particle (best first) + weights
            payload["models"][name]["cloud"] = _r(z["cloud_states"], 3)
            payload["models"][name]["cloud_w"] = _r(z["cloud_weights"], 5)
            payload["models"][name]["q"] = model_params(name).get("q", 0)
    return payload


def build(tids=None):
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    js = VIEW_DIR / "plotly.min.js"
    if not js.exists():
        from plotly.offline import get_plotlyjs
        js.write_text(get_plotlyjs(), encoding="utf-8")
    models = model_names()
    tids = tids or data.test_ids()
    for tid in tids:
        payload = page_payload(data.load_trajectory(tid), models)
        html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":"))) \
                       .replace("__TIDS__", json.dumps(tids))
        (VIEW_DIR / f"traj_{tid:05d}.html").write_text(html, encoding="utf-8")
    (VIEW_DIR / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Trajectory viewer</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:32px}"
        "a{color:#2a78d6}</style><h1>Test trajectories</h1><p>Algorithms: " + ", ".join(map(model_label, models)) + "</p><ul>"
        + "".join(f"<li><a href='traj_{t:05d}.html'>trajectory {t}</a></li>" for t in tids) + "</ul>",
        encoding="utf-8")
    print(f"viewer for {len(tids)} trajectories, models {models} -> {VIEW_DIR / 'index.html'}")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prediction viewer</title>
<script src="plotly.min.js"></script>
<style>
:root{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--border:rgba(11,11,11,.10)}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--plane);color:var(--ink)}
header{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:12px 16px;background:var(--surface);border-bottom:1px solid var(--border)}
header h1{font-size:16px;margin:0 8px 0 0}label{font-size:13px;color:var(--ink2)}select,button{font:inherit;font-size:13px;padding:4px 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface);color:var(--ink);cursor:pointer}
#controls{display:flex;gap:8px;align-items:center;padding:8px 16px;background:var(--surface);border-bottom:1px solid var(--border)}
#slider{flex:1}#tlabel{font-variant-numeric:tabular-nums;font-size:13px;min-width:90px}
main{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:12px;padding:12px 16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px}
#p3d{height:640px}#perr{height:330px}#pspd{height:200px}#pcloud{height:420px}
#info{font-size:13px;line-height:1.6;color:var(--ink2);padding:4px 8px}#info b{color:var(--ink);font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
@media (max-width:900px){main{grid-template-columns:1fr}#p3d{height:460px}}
</style></head><body>
<header><h1 id="title"></h1>
<label>Trajectory <select id="tsel"></select></label>
<label>Prediction horizon <select id="hsel"></select></label>
<label><input type="checkbox" id="showall" checked> full trajectory in background</label></header>
<div id="controls"><button id="play">▶</button><button id="prev">◀ 0.1 s</button><button id="next">0.1 s ▶</button>
<input type="range" id="slider" min="0" value="0"><span id="tlabel"></span></div>
<main><div class="card"><div id="p3d"></div></div>
<div><div class="card"><div id="perr"></div></div><div class="card" style="margin-top:12px"><div id="pspd"></div></div>
<div class="card" style="margin-top:12px"><div id="info"></div></div>
<div class="card" id="cloudcard" style="margin-top:12px;display:none"><div id="pcloud"></div></div></div></main>
<script>
const D=__DATA__, TIDS=__TIDS__;
const C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"];
const RED="#e34948";
function gauss(){let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
const INK="#0b0b0b",INK2="#52514e",MUTED="#898781",GRID="#e1e0d9",SURF="#fcfcfb";
const names=Object.keys(D.models), M0=D.models[names[0]];
const N=D.pos.length, dt=D.dt, viewO=M0.view_origins, steps=M0.err_steps, vsteps=M0.view_steps;
const col=(a,k)=>a.map(r=>r[k]);
// fixed 3D box = flight volume + 50 cm, so predictions that leave the chamber go off-screen instead of rescaling
const BOX=[0,1,2].map(a=>{const v=col(D.pos,a);return [Math.min(...v)-50,Math.max(...v)+50]});
const SPAN=BOX.map(b=>b[1]-b[0]), ASPECT={x:SPAN[0]/Math.max(...SPAN),y:SPAN[1]/Math.max(...SPAN),z:SPAN[2]/Math.max(...SPAN)};
const LBL=nm=>D.models[nm].label||nm;
document.getElementById("title").textContent="Trajectory "+D.tid+" · "+(N*dt).toFixed(1)+" s · "+names.map(LBL).join(" vs ");
const tsel=document.getElementById("tsel");
TIDS.forEach(t=>{const o=document.createElement("option");o.value=t;o.textContent=t;if(t===D.tid)o.selected=true;tsel.appendChild(o)});
tsel.onchange=()=>{location.href="traj_"+String(tsel.value).padStart(5,"0")+".html"};
const hsel=document.getElementById("hsel");
steps.forEach((s,k)=>{const o=document.createElement("option");o.value=k;o.textContent=(s*dt).toFixed(2).replace(/\.?0+$/,"")+" s";hsel.appendChild(o)});
hsel.value=String(steps.indexOf(100)>=0?steps.indexOf(100):0);
const slider=document.getElementById("slider");slider.max=viewO.length-1;
let vi=0, timer=null;
const axis=(t)=>({title:{text:t,font:{color:INK2,size:12}},gridcolor:GRID,zeroline:false,tickfont:{color:MUTED,size:11},linecolor:"#c3c2b7"});
const base={paper_bgcolor:SURF,plot_bgcolor:SURF,font:{family:"system-ui,sans-serif",color:INK},margin:{l:56,r:12,t:32,b:40},showlegend:true,legend:{orientation:"h",y:1.12,x:0}};

function scene3d(){
  const o=viewO[vi], k=+hsel.value, H=steps[k], tr=[];
  const P=D.pos, hist=Math.max(0,o-100);
  if(document.getElementById("showall").checked)
    tr.push({type:"scatter3d",mode:"lines",x:col(P,0),y:col(P,1),z:col(P,2),line:{color:"#c3c2b7",width:2},name:"full trajectory",hoverinfo:"skip"});
  const ps=P.slice(hist,o+1);
  tr.push({type:"scatter3d",mode:"lines",x:col(ps,0),y:col(ps,1),z:col(ps,2),line:{color:INK2,width:5},name:"last 1 s"});
  const fut=P.slice(o,Math.min(N,o+H+1));
  tr.push({type:"scatter3d",mode:"lines",x:col(fut,0),y:col(fut,1),z:col(fut,2),line:{color:INK,width:5,dash:"dash"},name:"true future"});
  names.forEach((nm,m)=>{const md=D.models[nm];const j=md.view_origins.indexOf(o);if(j<0)return;
    const nk=vsteps.filter(s=>s<=H).length, pr=[P[o]].concat(md.view_preds[j].slice(0,nk));
    tr.push({type:"scatter3d",mode:"lines",x:col(pr,0),y:col(pr,1),z:col(pr,2),line:{color:C[m],width:6},name:"prediction: "+LBL(nm)});
    const e=pr[pr.length-1];
    tr.push({type:"scatter3d",mode:"markers",x:[e[0]],y:[e[1]],z:[e[2]],marker:{color:C[m],size:5},showlegend:false,hoverinfo:"skip"});
    if(md.cloud){const jc=md.view_origins.indexOf(o), cl=md.cloud[jc], T=H*dt, sd=Math.sqrt(md.q*T*T*T/3);
      const fut=cl.map(p=>[0,1,2].map(a=>p[a]+p[a+3]*T+sd*gauss()));
      tr.push({type:"scatter3d",mode:"markers",x:col(fut,0),y:col(fut,1),z:col(fut,2),marker:{color:C[m],size:2,opacity:0.35},
               name:"predicted particle cloud ("+hsel.options[hsel.selectedIndex].text+")",hoverinfo:"skip"});
      tr.push({type:"scatter3d",mode:"markers",x:[fut[0][0]],y:[fut[0][1]],z:[fut[0][2]],marker:{color:RED,size:6,symbol:"diamond"},
               name:"best particle (prediction source)",hoverinfo:"skip"});}});
  tr.push({type:"scatter3d",mode:"markers",x:[P[o][0]],y:[P[o][1]],z:[P[o][2]],marker:{color:INK,size:6},name:"now"});
  Plotly.react("p3d",tr,{...base,uirevision:"keep",margin:{l:0,r:0,t:32,b:0},
    scene:{xaxis:{...axis("x [cm]"),range:BOX[0]},yaxis:{...axis("y [cm]"),range:BOX[1]},zaxis:{...axis("z [cm]"),range:BOX[2]},
           aspectmode:"manual",aspectratio:ASPECT}},{responsive:true});
}
function errRange(k){   // log10 range over all models at horizon k, capped at 10 m
  let lo=Infinity,hi=0;names.forEach(nm=>D.models[nm].err[k].forEach(v=>{if(v!=null&&v>0){lo=Math.min(lo,v);hi=Math.max(hi,v)}}));
  return [Math.log10(lo)-0.1,Math.log10(Math.min(hi,1000))+0.1];
}
function errChart(){
  const k=+hsel.value, tr=names.map((nm,m)=>{const md=D.models[nm];
    return {type:"scattergl",mode:"lines",x:md.origins.map(o=>o*dt),y:md.err[k],line:{color:C[m],width:1.2},name:LBL(nm),
            hovertemplate:"t=%{x:.2f} s<br>error %{y:.2f} cm<extra>"+LBL(nm)+"</extra>"}});
  Plotly.react("perr",tr,{...base,title:{text:names.map(LBL).join(", ")+": prediction error at "+hsel.options[hsel.selectedIndex].text+" horizon",font:{size:13},x:0},
    xaxis:axis("flight time [s]"),yaxis:{...axis("error [cm] (log, clipped at 10 m)"),type:"log",range:errRange(k)},shapes:[cursor()]},{responsive:true});
}
function spdChart(){
  Plotly.react("pspd",[{type:"scattergl",mode:"lines",x:D.speed.map((_,i)=>i*dt),y:D.speed,line:{color:INK2,width:1},name:"speed",
    hovertemplate:"t=%{x:.2f} s<br>%{y:.2f} m/s<extra></extra>"}],
    {...base,showlegend:false,title:{text:"Mosquito speed",font:{size:13},x:0},xaxis:axis("flight time [s]"),yaxis:axis("m/s"),shapes:[cursor()],uirevision:"keep"},{responsive:true});
}
const CLOUD=names.find(nm=>D.models[nm].cloud);
if(CLOUD)document.getElementById("cloudcard").style.display="block";
function cloudPanel(){
  if(!CLOUD)return;
  const md=D.models[CLOUD], o=viewO[vi], j=md.view_origins.indexOf(o); if(j<0)return;
  const c=md.cloud[j], w=md.cloud_w[j], z=D.pos[o], mm=v=>v*10;
  const rel=c.map(p=>[0,1,2].map(a=>mm(p[a]-z[a]))), wmax=Math.max(...w);
  const size=w.map(x=>3+9*Math.sqrt(x/wmax));
  const tr=[{type:"scatter3d",mode:"markers",x:col(rel,0).slice(1),y:col(rel,1).slice(1),z:col(rel,2).slice(1),
      marker:{color:"#898781",size:size.slice(1),opacity:0.6},name:"particles (size = weight)",
      text:w.slice(1).map(x=>"weight "+x.toExponential(2)),hovertemplate:"%{text}<extra></extra>"},
    {type:"scatter3d",mode:"markers",x:[rel[0][0]],y:[rel[0][1]],z:[rel[0][2]],marker:{color:RED,size:9},
      name:"most probable particle",text:["weight "+w[0].toExponential(2)+"<br>v = ("+c[0].slice(3).map(v=>(v/100).toFixed(2)).join(", ")+") m/s"],
      hovertemplate:"%{text}<extra></extra>"},
    {type:"scatter3d",mode:"markers",x:[0],y:[0],z:[0],marker:{color:INK,size:6,symbol:"x"},name:"measurement"}];
  Plotly.react("pcloud",tr,{...base,uirevision:"keepc",margin:{l:0,r:0,t:40,b:0},
    title:{text:"Particle cloud now ("+md.label+"), "+c.length+" of the particles shown",font:{size:12},x:0},
    scene:{xaxis:axis("Δx [mm]"),yaxis:axis("Δy [mm]"),zaxis:axis("Δz [mm]"),aspectmode:"cube"}},{responsive:true});
}
function cursor(){const x=viewO[vi]*dt;return {type:"line",xref:"x",yref:"paper",x0:x,x1:x,y0:0,y1:1,line:{color:INK,width:1}}}
function info(){
  const o=viewO[vi], k=+hsel.value;
  let h="<b>t = "+(o*dt).toFixed(2)+" s</b> · speed <b>"+(D.speed[o]??NaN).toFixed(2)+" m/s</b> · acceleration <b>"+(D.acc[o]??NaN).toFixed(1)+" m/s²</b> · height z <b>"+D.pos[o][2].toFixed(1)+" cm</b><br>";
  h+="Error at "+hsel.options[hsel.selectedIndex].text+": ";
  h+=names.map((nm,m)=>{const md=D.models[nm];const j=md.origins.indexOf(o);const e=j<0?null:md.err[k][j];
    return "<span class='sw' style='background:"+C[m]+"'></span>"+LBL(nm)+" <b>"+(e==null?"–":e.toFixed(2)+" cm")+"</b>"}).join(" · ");
  document.getElementById("info").innerHTML=h;
  document.getElementById("tlabel").textContent=(o*dt).toFixed(1)+" / "+(N*dt).toFixed(1)+" s";
}
function setVi(v){vi=Math.max(0,Math.min(viewO.length-1,v));slider.value=vi;scene3d();info();cloudPanel();
  const s={shapes:[cursor()]};Plotly.relayout("perr",s);Plotly.relayout("pspd",s);}
function jumpTo(t){const f=t/dt;let b=0;for(let i=0;i<viewO.length;i++)if(Math.abs(viewO[i]-f)<Math.abs(viewO[b]-f))b=i;setVi(b)}
slider.oninput=()=>setVi(+slider.value);
document.getElementById("prev").onclick=()=>setVi(vi-1);document.getElementById("next").onclick=()=>setVi(vi+1);
document.getElementById("play").onclick=(e)=>{if(timer){clearInterval(timer);timer=null;e.target.textContent="▶"}
  else{timer=setInterval(()=>{if(vi>=viewO.length-1){clearInterval(timer);timer=null;document.getElementById("play").textContent="▶";return}setVi(vi+1)},100);e.target.textContent="❚❚"}};
hsel.onchange=()=>{errChart();scene3d();info()};
document.getElementById("showall").onchange=scene3d;
errChart();spdChart();setVi(0);
document.getElementById("perr").on("plotly_click",ev=>jumpTo(ev.points[0].x));
document.getElementById("pspd").on("plotly_click",ev=>jumpTo(ev.points[0].x));
</script></body></html>
"""

if __name__ == "__main__":
    build()
