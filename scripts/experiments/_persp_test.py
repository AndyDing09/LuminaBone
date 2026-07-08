import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reconstruction")))
import bone_depth_batch as bd

# Perspective render of a sphere and reconstruct via bd's exact pipeline
# (true normals -> p=nx/nz, q=-ny/nz -> FC -> scale by lateral mm_per_px).
# Geometry roughly matching the rig.
W = H = 400
fov_mm = 37.0            # field width at the sphere-center plane
WD = 46.0               # working distance to sphere center plane (mm)
# focal length in px so that field width fov_mm maps across W px at distance WD:
# u = f * X/Z ; X spanning +-fov_mm/2 at Z=WD maps to +-W/2 px => f = (W/2)/(fov_mm/2/WD)
f = (W/2.0) / ((fov_mm/2.0)/WD)
cx=cy=W/2.0

def render_and_recon(Rs):
    # camera at origin looking +Z. sphere center at (0,0,WD) (in front).
    Cz = WD
    j,i = np.meshgrid(np.arange(W), np.arange(H))
    x = (j - cx)/f     # ray dir X/Z
    y = (i - cy)/f     # image y downward -> world y downward; we'll handle sign
    # ray: P = t*(x,y,1). Solve |P - C|^2 = Rs^2 , C=(0,0,Cz)
    dxv=x; dyv=y; dzv=np.ones_like(x)
    # (t dx)^2+(t dy)^2+(t dz - Cz)^2 = Rs^2
    A = dxv**2+dyv**2+dzv**2
    B = -2*Cz*dzv
    Cc = Cz**2 - Rs**2
    disc = B**2 - 4*A*Cc
    hit = disc>0
    t = np.zeros_like(x)
    t[hit] = (-B[hit]-np.sqrt(disc[hit]))/(2*A[hit])   # near intersection
    Px=t*dxv; Py=t*dyv; Pz=t*dzv
    Zdepth = Pz  # world depth (toward camera axis)
    # world normal (outward from center), camera coords +x right,+y up,+z toward cam
    Nx = Px-0; Ny=Py-0; Nz=Pz-Cz
    nrm=np.sqrt(Nx**2+Ny**2+Nz**2)+1e-12
    Nx/=nrm;Ny/=nrm;Nz/=nrm
    # outward normal points away from center; the camera-facing side has Nz<0
    # bd expects nz>=0 (toward camera +z). Our +z is toward scene (away from cam).
    # Flip to camera convention: cam +z toward camera = -world_z. And +y up = -image_y.
    n = np.zeros((H,W,3))
    n[...,0]=Nx
    n[...,1]=-Ny    # image y down -> +y up flips
    n[...,2]=-Nz    # toward camera
    # ensure nz>=0
    fl=n[...,2]<0; n[fl]*=-1
    # background: flat normal facing camera
    n[~hit]=np.array([0,0,1.0])
    z = bd.normals_to_depth(n)   # relative depth (larger=farther), pixel units
    mm_per_px = fov_mm / W
    z_mm = z*mm_per_px
    # true depth toward camera: nearer = smaller Z. bd depth larger=farther => compare z_mm to Zdepth
    m = hit.copy()
    # erode a bit to avoid limb where nz->0 (clamped)
    from scipy.ndimage import binary_erosion
    m = binary_erosion(m, iterations=8)
    def rel(a): 
        v=a[m]; return np.percentile(v,98)-np.percentile(v,2)
    true_relief = rel(Zdepth)
    rec_relief = rel(z_mm)
    # correlation of shape
    cc=np.corrcoef(Zdepth[m], z_mm[m])[0,1]
    print(f"Rs={Rs:5.1f}mm sphere-cap-in-view: true_relief={true_relief:.3f}mm  rec={rec_relief:.3f}mm  ratio={rec_relief/true_relief:.3f}  shapeCorr={cc:.3f}")

for Rs in [25.0, 40.0, 12.0, 8.0]:
    render_and_recon(Rs)
