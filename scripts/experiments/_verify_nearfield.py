import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reconstruction")))
import bone_depth_batch as bd

# Working image size as produced by the pipeline (1280x720 -> long edge 640)
W, H = 640, 360
FIELD_W_MM = 37.0
mmpp = FIELD_W_MM / W          # mm per pixel
WD = 46.0                       # working distance mm
OFF = 12.08                     # LED off-axis mm

# pixel -> surface coords (mm), center origin, flat plane z=0, camera/LEDs at z=WD
j = np.arange(W); i = np.arange(H)
xx, yy = np.meshgrid((j - W/2.0)*mmpp, (i - H/2.0)*mmpp)  # (H,W)

# TRUE rig LED positions (right az0, top az90, left az180), radius 12.08mm, at z=WD
leds_true = {
    2: np.array([ OFF, 0.0, WD]),   # right
    3: np.array([ 0.0, OFF, WD]),   # top
    4: np.array([-OFF, 0.0, WD]),   # left
}

n_flat = np.array([0.0, 0.0, 1.0])

def render(led_pos, model):
    dx = led_pos[0]-xx; dy = led_pos[1]-yy; dz = led_pos[2]-0.0
    r = np.sqrt(dx*dx+dy*dy+dz*dz)
    Lx, Ly, Lz = dx/r, dy/r, dz/r
    cos_inc = Lz   # n=(0,0,1)
    if model == 'point':
        return np.clip(cos_inc,0,None)/(r*r)          # 1/r^2 * cos
    if model == 'point_norm':                          # cos only, no 1/r^2
        return np.clip(cos_inc,0,None)
    if model == 'far':                                 # far-field: constant dir, no falloff
        # use direction to LED at image center
        c = led_pos/np.linalg.norm(led_pos)
        return np.clip(c[2],0,None)*np.ones_like(xx)
    raise ValueError

led_indices = sorted(bd.SINGLE_LED_AZIMUTH_DEG.keys())  # [2,3,4]
# Solve directions exactly as the code builds them
azimuths = [bd.SINGLE_LED_AZIMUTH_DEG[k] for k in led_indices]
light_dirs = np.array([bd.light_vector(a, bd.LIGHT_ELEVATION_DEG) for a in azimuths])

def run(model):
    lums = [render(leds_true[k], model) for k in led_indices]
    lums = bd.balance_exposure(lums)
    normals, albedo = bd.photometric_stereo(lums, light_dirs)
    z = bd.DEPTH_SIGN * bd.normals_to_depth(normals)
    # relief on interior (exclude 4% border like bone_mask does)
    b = max(1,int(0.04*min(H,W)))
    zin = z[b:-b, b:-b]
    lo = np.percentile(zin, 2.0); hi = np.percentile(zin, 98.0)
    p2p = hi-lo
    # also full peak-to-valley of the interior
    ptv = zin.max()-zin.min()
    return p2p, ptv, z

for model in ['far','point','point_norm']:
    p2p, ptv, z = run(model)
    print(f"{model:11s}  relief(p2-p98)={p2p:8.2f}   full PtV={ptv:8.2f}")

# reference falloff numbers
def falloff_range(led_pos):
    dx = led_pos[0]-xx; dy = led_pos[1]-yy; dz = led_pos[2]
    r = np.sqrt(dx*dx+dy*dy+dz*dz)
    val = (dz/r)/(r*r)   # cos/r^2
    val = val/val[H//2,W//2]
    row = val[H//2]
    return row.min(), row.max(), row.max()/row.min()
for k in led_indices:
    lo,hi,ratio = falloff_range(leds_true[k])
    print(f"LED {k} horizontal falloff across field: {lo:.3f}->{hi:.3f}  max/min={ratio:.3f}")
