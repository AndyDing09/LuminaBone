import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reconstruction")))
import bone_depth_batch as bd

# How does the ASSUMED light elevation scale recovered relief?
# Render a Lambertian bump lit by 3 lights at TRUE elevation, then solve with an
# ASSUMED elevation and see the relief scale factor. This isolates the elevation
# lever the claim mentions (40 in bone_depth_batch vs ~75 in the slide/mesh path).
H=W=200
yy,xx=np.mgrid[0:H,0:W].astype(float)
cx,cy=W/2,H/2
A=15.0; sig=40.0
hgt=A*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sig**2))   # height toward camera
# true normals of surface z_toward = hgt: n ~ (-h_x,-h_y,1)
hx=(-(xx-cx)/sig**2)*hgt; hy=(-(yy-cy)/sig**2)*hgt
n=np.stack([-hx,-hy,np.ones_like(hgt)],-1)
n/=np.linalg.norm(n,axis=-1,keepdims=True)

az=[0.0,90.0,180.0]
for true_el in [75.0]:
    Ltrue=np.array([bd.light_vector(a,true_el) for a in az])
    I=[np.clip(n@Ltrue[k],0,None) for k in range(3)]   # lambertian, albedo=1
    for assumed_el in [75.0,55.0,40.0]:
        Lass=np.array([bd.light_vector(a,assumed_el) for a in az])
        nn,alb=bd.photometric_stereo(I,Lass)
        z=bd.normals_to_depth(nn)
        # true relief in same pixel units (integrate true slopes)
        ztrue=bd.normals_to_depth(n)
        m=np.ones((H,W),bool)
        rr=(np.percentile(z[m],98)-np.percentile(z[m],2))
        rt=(np.percentile(ztrue[m],98)-np.percentile(ztrue[m],2))
        print(f"true_el={true_el} assumed_el={assumed_el}: relief ratio (recovered/true)={rr/rt:.3f}")
