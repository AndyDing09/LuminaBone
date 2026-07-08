import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reconstruction")))
import bone_depth_batch as bd

# ---- Test 1: FC on an EXACT orthographic gradient field (hemisphere) ----
# True world slopes, sampled on a regular pixel grid. This is the ideal
# orthographic case the claim says still only recovers 0.5-0.8x.
H = W = 256
yy, xx = np.mgrid[0:H, 0:W].astype(float)
cx, cy = W/2, H/2
R = 90.0
r2 = (xx-cx)**2 + (yy-cy)**2
inside = r2 < R**2
# height toward camera h = sqrt(R^2 - r2); depth z = -h (bd convention: larger=farther)
h = np.zeros((H,W)); h[inside] = np.sqrt(np.clip(R**2 - r2[inside],0,None))
z_true = -h
# exact gradients of z_true
p = np.zeros((H,W)); q = np.zeros((H,W))
# dz/dx = x/h*(-)(-1)... compute analytically for depth z=-sqrt(R^2-r2)
# z = -sqrt(R^2-(x-cx)^2-(y-cy)^2); dz/dx = (x-cx)/sqrt(...) = (x-cx)/h
with np.errstate(divide='ignore', invalid='ignore'):
    p[inside] = (xx[inside]-cx)/h[inside]
    q[inside] = (yy[inside]-cy)/h[inside]
z_fc = bd.integrate_frankot_chellappa(p, q)
# compare relief within a slightly eroded interior to avoid the pole singularity
core = r2 < (R*0.85)**2
def relief(z,m): 
    v=z[m]; return np.percentile(v,98)-np.percentile(v,2)
rt = relief(z_true, core); rf = relief(z_fc, core)
print("TEST1 orthographic hemisphere exact gradients:")
print(f"  relief true={rt:.3f}  fc={rf:.3f}  ratio={rf/rt:.3f}")

# full-support relief
rt2=relief(z_true,inside); rf2=relief(z_fc,inside)
print(f"  full inside: true={rt2:.3f} fc={rf2:.3f} ratio={rf2/rt2:.3f}")

# ---- Test 2: smooth Gaussian bump, exact orthographic gradients ----
A=40.0; sig=50.0
g = A*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sig**2))
zt = -g
p2 = -(-(xx-cx)/sig**2)*g  # dz/dx = -dg/dx = -(-(x-cx)/sig^2)g = (x-cx)/sig^2 * g
p2 = ((xx-cx)/sig**2)*g
q2 = ((yy-cy)/sig**2)*g
zf = bd.integrate_frankot_chellappa(p2,q2)
m=np.ones((H,W),bool)
print("TEST2 gaussian bump exact orthographic gradients:")
print(f"  relief true={relief(zt,m):.3f} fc={relief(zf,m):.3f} ratio={relief(zf,m)/relief(zt,m):.3f}")
