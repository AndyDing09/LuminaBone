import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reconstruction")))
import bone_depth_batch as bd

def gauss_bump_test(H=512, W=512, mm_per_px=37.0/1280.0, R_mm=6.0, A_mm=2.0):
    # build a Gaussian bump surface (height toward camera). depth = -height.
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = W/2.0, H/2.0
    x = (j - cx) * mm_per_px
    y = -(i - cy) * mm_per_px
    s = R_mm  # gaussian sigma in mm
    h = A_mm * np.exp(-(x**2 + y**2)/(2*s**2))   # height toward camera
    # true DEPTH (away from camera) = -h  (+ const). Use z_true = -h
    z_true = -h
    # analytic gradients of height
    # dh/dx = h * (-x/s^2), dh/dy = h * (-y/s^2)
    dh_dx = h * (-x/s**2)
    dh_dy = h * (-y/s**2)
    # normals ~ (-h_x, -h_y, 1)
    nx = -dh_dx
    ny = -dh_dy
    nz = np.ones_like(h)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    z_rec = bd.normals_to_depth(n, nz_floor=1e-6)
    # remove DC from both
    z_true_c = z_true - z_true.mean()
    z_rec_c = z_rec - z_rec.mean()
    # amplitude ratio: peak-to-peak
    ptp_true = z_true_c.max() - z_true_c.min()
    ptp_rec = z_rec_c.max() - z_rec_c.min()
    # least squares scale
    scale = np.dot(z_rec_c.ravel(), z_true_c.ravel())/np.dot(z_true_c.ravel(), z_true_c.ravel())
    return ptp_rec/ptp_true, scale

for R in [3,6,15,30,60,120]:
    ratio, scale = gauss_bump_test(R_mm=R)
    print(f"R={R:4} mm  ptp_ratio={ratio:.3f}  lsq_scale={scale:.3f}")
