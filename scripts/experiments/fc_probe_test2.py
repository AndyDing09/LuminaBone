import numpy as np, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reconstruction")))
import bone_depth_batch as bd

# --- Test A: full 1280x720 domain, bump fully inside, compare recovered depth ---
def bump_full(R_px, A=50.0, H=720, W=1280):
    j,i=np.meshgrid(np.arange(W),np.arange(H))
    x=j-W/2.0; y=-(i-H/2.0)   # +y up, matching normals_to_depth convention
    h=A*np.exp(-(x**2+y**2)/(2*R_px**2))     # height toward camera (px units)
    z_true=-h
    nx=-h*(-x/R_px**2); ny=-h*(-y/R_px**2); nz=np.ones_like(h)
    n=np.stack([nx,-ny,nz],axis=-1)  # note: normals_to_depth expects +y up; build consistent
    # actually build normals as (-h_x,-h_y,1) then feed; normals_to_depth does p=nx/nz,q=-ny/nz
    n=np.stack([-h*(-x/R_px**2), -h*(-y/R_px**2), np.ones_like(h)],axis=-1)
    n/=np.linalg.norm(n,axis=-1,keepdims=True)
    zr=bd.normals_to_depth(n,nz_floor=1e-6)
    zt=z_true-z_true.mean(); zr=zr-zr.mean()
    s=np.dot(zr.ravel(),zt.ravel())/np.dot(zt.ravel(),zt.ravel())
    resid=np.linalg.norm(zr-s*zt)/np.linalg.norm(zt)
    return s, resid
print("Test A: full-domain bump inside 1280x720")
for R in [20,50,100,200,300]:
    s,resid=bump_full(R)
    print(f"  R={R:4}px  lsq_scale={s:.4f}  rel_resid={resid:.4f}")

# --- Test B: direct FFT frequency response of _fc_solve on pure sinusoid ---
print("Test B: single-sinusoid reconstruction ratio vs wavelength")
H=W=256
for k in [1,2,4,8,32,64]:   # cycles across the field; k=1 is the broadest
    yy,xx=np.meshgrid(np.arange(W),np.arange(H),indexing='ij')
    z=np.cos(2*np.pi*k*xx/H)           # true surface, k cycles
    p=np.gradient(z,axis=1)            # dz/dcol
    q=np.gradient(z,axis=0)            # dz/drow
    zr=bd.integrate_frankot_chellappa(p,q)
    zc=z-z.mean(); zrc=zr-zr.mean()
    s=np.dot(zrc.ravel(),zc.ravel())/np.dot(zc.ravel(),zc.ravel())
    print(f"  k={k:3} cycles  recovered/true={s:.4f}")
