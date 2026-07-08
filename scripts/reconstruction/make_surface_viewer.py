"""
Build a self-contained interactive 3-D viewer (HTML) for a colored point-cloud
PLY, so the reconstruction can be rotated / zoomed / panned in any browser.

    python make_surface_viewer.py                 # combined stitched surface
    python make_surface_viewer.py depth_outputs/meshes/loc01_mesh.ply

Controls (baked into the page): left-drag = rotate, wheel = zoom,
right/shift-drag = pan, C = photo/height colour, Space = auto-spin, R = reset.

The output HTML embeds the geometry as base64 (no server, no external files,
works offline by double-clicking).
"""

import os
import sys
import base64
import numpy as np

MAX_POINTS = 220000     # downsample above this so the page stays light + smooth
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read_ply_points(path):
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr:
            hdr += f.readline()
        lines = hdr.decode().splitlines()
        nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
        has_face = any(l.startswith("element face") for l in lines)
        dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("r", "u1"), ("g", "u1"), ("b", "u1")]
        rec = np.fromfile(f, dtype=dt, count=nv)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float32)
    rgb = np.stack([rec["r"], rec["g"], rec["b"]], 1).astype(np.uint8)
    return xyz, rgb, has_face


def build(path):
    xyz, rgb, _ = read_ply_points(path)
    if len(xyz) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(xyz), MAX_POINTS, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    center = xyz.mean(0)
    xyz = xyz - center                       # center on origin
    radius = float(np.percentile(np.linalg.norm(xyz, axis=1), 99))
    zmin, zmax = float(xyz[:, 2].min()), float(xyz[:, 2].max())

    pos_b64 = base64.b64encode(xyz.tobytes()).decode("ascii")
    col_b64 = base64.b64encode(rgb.tobytes()).decode("ascii")

    html = _TEMPLATE
    html = html.replace("__TITLE__", os.path.basename(path))
    html = html.replace("__NPTS__", str(len(xyz)))
    html = html.replace("__RADIUS__", f"{radius:.4f}")
    html = html.replace("__ZMIN__", f"{zmin:.4f}")
    html = html.replace("__ZMAX__", f"{zmax:.4f}")
    html = html.replace("__POS__", pos_b64)
    html = html.replace("__COL__", col_b64)

    out = os.path.splitext(path)[0] + "_viewer.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    mb = os.path.getsize(out) / 1e6
    print(f"{len(xyz):,} points -> {out}  ({mb:.1f} MB)")
    print("Open it in any browser: left-drag rotate, wheel zoom, "
          "right-drag pan, C colour, Space spin, R reset.")
    return out


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>3-D surface — __TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#11131a;overflow:hidden;
    font-family:system-ui,Segoe UI,Arial,sans-serif;color:#dfe3ea}
  #c{display:block;width:100vw;height:100vh;cursor:grab}
  #c:active{cursor:grabbing}
  #hud{position:fixed;left:14px;top:12px;font-size:12.5px;line-height:1.5;
    background:rgba(20,23,32,.72);padding:10px 13px;border-radius:9px;
    border:1px solid #2a2f3d;pointer-events:none;max-width:270px}
  #hud b{color:#8fd0ff}
  kbd{background:#2a2f3d;border-radius:4px;padding:1px 5px;font-size:11px}
</style></head>
<body>
<canvas id="c"></canvas>
<div id="hud"><b>3-D surface reconstruction</b><br>__TITLE__ · __NPTS__ points<br>
  <kbd>drag</kbd> rotate · <kbd>wheel</kbd> zoom · <kbd>right-drag</kbd> pan<br>
  <kbd>C</kbd> photo/height colour · <kbd>Space</kbd> spin · <kbd>R</kbd> reset</div>
<script>
const NPTS=__NPTS__, RADIUS=__RADIUS__, ZMIN=__ZMIN__, ZMAX=__ZMAX__;
function b64f32(s){const b=atob(s),n=b.length,u=new Uint8Array(n);
  for(let i=0;i<n;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function b64u8(s){const b=atob(s),n=b.length,u=new Uint8Array(n);
  for(let i=0;i<n;i++)u[i]=b.charCodeAt(i);return u;}
const POS=b64f32("__POS__"), COL=b64u8("__COL__");

const cv=document.getElementById('c');
const gl=cv.getContext('webgl2',{antialias:true});
if(!gl){document.body.innerHTML='<p style="padding:2em">WebGL2 not available in this browser.</p>';}

const VS=`#version 300 es
layout(location=0) in vec3 pos; layout(location=1) in vec3 col;
uniform mat4 uMVP; uniform float uPS;
out vec3 vc; out float vz;
void main(){ gl_Position=uMVP*vec4(pos,1.0); gl_PointSize=uPS; vc=col; vz=pos.z; }`;
const FS=`#version 300 es
precision highp float;
in vec3 vc; in float vz; out vec4 o;
uniform int uMode; uniform vec2 uZ;
vec3 turbo(float t){ t=clamp(t,0.0,1.0);
  return clamp(vec3(
   34.61 + t*(1172.33 + t*(-10793.56 + t*(33300.12 + t*(-38394.49 + t*14825.05)))),
   23.31 + t*(557.33 + t*(1225.33 + t*(-3574.96 + t*(1073.77 + t*707.56)))),
   27.2 + t*(3211.1 + t*(-15327.97 + t*(27814.0 + t*(-22569.18 + t*6838.66)))))/255.0,0.0,1.0);}
void main(){
  vec2 d=gl_PointCoord-0.5; if(dot(d,d)>0.25) discard;
  vec3 c = uMode==0 ? vc : turbo((vz-uZ.x)/max(uZ.y-uZ.x,1e-6));
  float sh=0.65+0.35*(1.0-length(d)*2.0);           // fake round-point shading
  o=vec4(c*sh,1.0);
}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(o);return o;}
const prog=gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,VS));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(prog);gl.useProgram(prog);

const vao=gl.createVertexArray();gl.bindVertexArray(vao);
const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);
gl.bufferData(gl.ARRAY_BUFFER,POS,gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
const cb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,cb);
gl.bufferData(gl.ARRAY_BUFFER,COL,gl.STATIC_DRAW);
gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.UNSIGNED_BYTE,true,0,0);

const uMVP=gl.getUniformLocation(prog,'uMVP');
const uPS=gl.getUniformLocation(prog,'uPS');
const uMode=gl.getUniformLocation(prog,'uMode');
const uZ=gl.getUniformLocation(prog,'uZ');
gl.uniform2f(uZ,ZMIN,ZMAX);
gl.enable(gl.DEPTH_TEST);gl.clearColor(0.067,0.075,0.102,1);

// ---- tiny mat4 ----
function persp(f,a,n,fa){const t=1/Math.tan(f/2);return[t/a,0,0,0, 0,t,0,0,
  0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];}
function mul(a,b){const o=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){
  let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;}
function look(e,c,u){const z=nrm(sub(e,c)),x=nrm(crs(u,z)),y=crs(z,x);
  return[x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -dot(x,e),-dot(y,e),-dot(z,e),1];}
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const crs=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const nrm=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];};

// ---- camera state ----
// start facing the bone (down the depth axis) with a gentle tilt for 3-D feel
let yaw=0.12, pit=1.28, dist=RADIUS*2.4, tgt=[0,0,0], mode=0, spin=false;
function reset(){yaw=0.12;pit=1.28;dist=RADIUS*2.4;tgt=[0,0,0];}
function resize(){const dpr=Math.min(devicePixelRatio,2);
  cv.width=innerWidth*dpr;cv.height=innerHeight*dpr;gl.viewport(0,0,cv.width,cv.height);}
addEventListener('resize',resize);resize();

let drag=null,lx=0,ly=0;
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{drag=e.button===2||e.shiftKey?'pan':'rot';lx=e.clientX;ly=e.clientY;});
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
  if(drag==='rot'){yaw-=dx*0.008;pit=Math.max(0.05,Math.min(Math.PI-0.05,pit-dy*0.008));}
  else{const s=dist*0.0015;const r=[Math.cos(yaw),0,-Math.sin(yaw)];
    tgt[0]-=(-dx*s)*r[0];tgt[2]-=(-dx*s)*r[2];tgt[1]+=dy*s;}});
cv.addEventListener('wheel',e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.0011);
  dist=Math.max(RADIUS*0.25,Math.min(RADIUS*20,dist));},{passive:false});
addEventListener('keydown',e=>{if(e.code==='KeyC')mode^=1;
  else if(e.code==='KeyR')reset();else if(e.code==='Space'){spin=!spin;e.preventDefault();}});

function frame(){
  if(spin)yaw+=0.005;
  const ex=tgt[0]+dist*Math.sin(pit)*Math.sin(yaw);
  const ey=tgt[1]+dist*Math.cos(pit);
  const ez=tgt[2]+dist*Math.sin(pit)*Math.cos(yaw);
  const V=look([ex,ey,ez],tgt,[0,1,0]);
  const P=persp(1.05,cv.width/cv.height,RADIUS*0.05,RADIUS*40);
  gl.useProgram(prog);gl.uniformMatrix4fv(uMVP,false,mul(P,V));
  gl.uniform1f(uPS,Math.max(1.5,Math.min(5.0,cv.height/380)));
  gl.uniform1i(uMode,mode);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.bindVertexArray(vao);gl.drawArrays(gl.POINTS,0,NPTS);
  requestAnimationFrame(frame);
}
frame();
</script></body></html>"""


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(PROJECT_ROOT, "depth_outputs", "meshes", "combined_surface_points.ply")
    build(src)
