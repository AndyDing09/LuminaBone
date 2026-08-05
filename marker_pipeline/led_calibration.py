"""
LED photometric-stereo light calibration, after Quéau et al. (2017),
"LED-Based Photometric Stereo: Modeling, Calibration and Numerical Solution",
J Math Imaging Vis, Sect. 2.2.
================================================================================

WHY. bone_depth_batch.py uses the far-field model (one constant light direction
per LED, no distance falloff). On this rig the LED sits ~12 mm off the lens at a
~30 mm working distance, i.e. squarely in the NEAR field, so that model invents a
bowl and over-deepens the relief. nearfield_lambertian.py already fixes the
FORWARD model (inverse-square + cosine), but it *assumes* the light geometry:
LEDs symmetric at a measured offset, principal axis = +Z, equal brightness,
Lambertian emission. This module calibrates those assumptions away, exactly as
the paper does, producing for each LED i:

    x_s  its 3-D position          (mm, camera frame)
    n_s  its principal direction   (unit vector)
    mu   its anisotropy exponent   (from the datasheet half-angle)
    Psi  its intensity             (up to one factor common to all LEDs)

which is precisely the per-LED record nearfield_lambertian's forward model wants.

THE MODEL (paper Eq. 2.1), one term per LED i, at surface point x with normal n:

    I^i(p) = Psi_i * rho(p)
             * [ n_s^i . (x - x_s^i) / |x - x_s^i| ] ** mu_i     <- anisotropy cos^mu
             * {(x_s^i - x) . n(p)}_+ / |x_s^i - x| ** 3          <- 1/r^2 * Lambert

The bracket is cos^mu(theta), theta = angle off the LED axis. The last factor is
inverse-square attenuation (the extra power of |x_s - x| in the denominator turns
the un-normalised dot product in the numerator into a cosine, leaving 1/r^2).

Coordinates match nearfield_lambertian.py: X right, Y down, Z from the camera
into the scene, millimetres; the lens/optical centre is at the origin.

CALIBRATION (three independent steps, paper Sect. 2.2):

  0. mu   from the manufacturer half-angle theta_1/2   (Eq. 2.6).
  1. x_s  by TRIANGULATION off a spherical mirror (Sect. 2.2.1): each pose gives
          a 3-D ray through the LED (reflect the camera ray at the highlight);
          the least-squares meet of many rays is x_s.
  2. n_s, Psi by a linear least-squares fit on a white LAMBERTIAN PLANE with x_s
          and mu known (Sect. 2.2.2-2.2.3, Eqs. 2.12, 2.18-2.20). A change of
          variable m_s = Psi**(1/mu) n_s linearises the cos^mu nonlinearity.

Run:
    python led_calibration.py --selftest     validate every solver on synthetic
                                             ground truth (no captures needed)
"""

import argparse
import json
import numpy as np


# ===========================================================================
# Step 0 -- anisotropy from the datasheet (Eq. 2.5-2.6)
# ===========================================================================

def anisotropy_mu(theta_half_deg):
    """LED anisotropy exponent mu from the half-power angle (Eq. 2.6).

    The imperfect-Lambertian emitter model is Phi(theta) = Phi0 cos^mu(theta).
    theta_1/2 is where intensity halves (on the datasheet as the viewing angle
    at 50% relative intensity). Setting cos^mu(theta_1/2) = 1/2 gives

        mu = -log(2) / log(cos(theta_1/2)).

    theta_1/2 = 60 deg -> mu = 1 (a truly Lambertian source), as in the paper.
    """
    c = np.cos(np.radians(theta_half_deg))
    if not (0.0 < c < 1.0):
        raise ValueError("theta_half must be in (0, 90) deg")
    return float(-np.log(2.0) / np.log(c))


# ===========================================================================
# Step 1 -- LED position by specular-sphere triangulation (Sect. 2.2.1)
# ===========================================================================

def _pixel_ray(u, v, fx, fy, cx, cy):
    """Unit direction of the camera ray through pixel (u, v) (Z into scene)."""
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return d / np.linalg.norm(d)


def sphere_center_from_boundary(boundary_uv, fx, fy, cx, cy, radius_mm):
    """3-D centre of a mirror sphere of known radius from its silhouette.

    The camera rays grazing the sphere form a right circular cone whose axis
    points at the sphere centre; each grazing ray makes the same angle with the
    axis. So the unit boundary directions all lie on one small circle of the
    view sphere: they share a plane  a . d = cos(beta), whose normal a is the
    cone axis and whose offset cos(beta) is the half-angle. Fit that plane
    (a = least-variance eigenvector of the boundary directions), then the
    tangent-line geometry sin(beta) = R / |C| fixes the range:

        |C| = R / sin(beta),   C = |C| * a   (sign chosen so C is in front).
    """
    dirs = np.array([_pixel_ray(u, v, fx, fy, cx, cy) for u, v in boundary_uv])
    dbar = dirs.mean(axis=0)
    # plane best containing the boundary directions -> normal = least variance
    cov = np.cov((dirs - dbar).T)
    w, V = np.linalg.eigh(cov)
    a = V[:, 0]                                  # smallest-eigenvalue direction
    c = float(a @ dbar)
    if c < 0:                                    # orient the axis into the scene
        a, c = -a, -c
    beta = np.arccos(np.clip(c, -1.0, 1.0))
    dist = radius_mm / np.sin(beta)
    return a * dist


def reflect_ray_from_sphere(highlight_uv, sphere_center, radius_mm,
                            fx, fy, cx, cy):
    """3-D ray (origin, direction) from a mirror-sphere highlight to the LED.

    Trace the camera ray to the near sphere hit s, take the outward normal
    n = (s - C)/R, and reflect the surface->camera direction across n. By the
    law of reflection the reflected direction points from s toward the LED, so
    the LED lies on the ray {s + t * ell}.
    """
    d = _pixel_ray(*highlight_uv, fx, fy, cx, cy)          # camera -> point
    C = np.asarray(sphere_center, float)
    # near intersection of ray (t d) with sphere |t d - C|^2 = R^2
    b = -2.0 * d @ C
    cc = C @ C - radius_mm ** 2
    disc = b * b - 4.0 * cc
    if disc < 0:
        raise ValueError("highlight ray misses the sphere; check inputs")
    t = (-b - np.sqrt(disc)) / 2.0
    s = t * d
    n = (s - C) / radius_mm                                # outward normal
    to_cam = -s / np.linalg.norm(s)                        # surface -> camera
    ell = 2.0 * (n @ to_cam) * n - to_cam                  # -> toward LED
    return s, ell / np.linalg.norm(ell)


def triangulate_point(origins, directions):
    """Least-squares point closest to a bundle of 3-D rays o_k + t d_k.

    Minimises sum_k |(I - d_k d_k^T)(x - o_k)|^2 -- the squared perpendicular
    distance to each line. Two rays suffice in theory; more (the paper uses ten
    sphere poses) average out detection noise since the rays never meet exactly.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)            # projector onto d's normal plane
        A += P
        b += P @ np.asarray(o, float)
    return np.linalg.solve(A, b)


def calibrate_led_position(poses, fx, fy, cx, cy, radius_mm):
    """LED position x_s from several mirror-sphere poses (Sect. 2.2.1).

    poses: list of dicts, one per pose, each with
        'boundary_uv' : >=5 pixels on the sphere silhouette
        'highlight_uv': the LED reflection pixel on the sphere
    """
    origins, dirs = [], []
    for pz in poses:
        C = sphere_center_from_boundary(pz["boundary_uv"], fx, fy, cx, cy,
                                        radius_mm)
        s, ell = reflect_ray_from_sphere(pz["highlight_uv"], C, radius_mm,
                                         fx, fy, cx, cy)
        origins.append(s)
        dirs.append(ell)
    return triangulate_point(origins, dirs)


# ===========================================================================
# Step 2 -- principal direction + intensity on a Lambertian plane
#           (Sect. 2.2.2-2.2.3, Eqs. 2.12, 2.15, 2.18-2.20)
# ===========================================================================

def corrected_gray_level(J, u, v, fx, fy, cx, cy):
    """Undo the lens' cos^4 peripheral fall-off (Eq. 2.12): I = J / cos^4 alpha.

    alpha is the angle between the pixel's ray and the optical axis;
    cos alpha = 1 / |ray| with the ray in normalised camera coords, so
    cos^4 alpha = 1 / (xn^2 + yn^2 + 1)^2. This is a pure geometric vignette
    from the imaging law, present even with an ideal lens -- not sensor
    vignetting -- and must be removed before the photometric fit.
    """
    xn = (np.asarray(u, float) - cx) / fx
    yn = (np.asarray(v, float) - cy) / fy
    cos4 = 1.0 / (xn * xn + yn * yn + 1.0) ** 2
    return np.asarray(J, float) / cos4


def calibrate_led_direction_intensity(I, X, N, x_s, mu):
    """Principal direction n_s and intensity Psi of one LED (Eqs. 2.18-2.20).

    Inputs are stacked over every white-cell pixel of every plane pose:
        I  : (K,)    corrected gray levels
        X  : (K, 3)  3-D coordinates of those surface points (mm)
        N  : (K, 3)  surface (plane) normals at those points
        x_s: (3,)    LED position from Step 1
        mu : scalar  anisotropy from Step 0

    With x_s and mu known, the only nonlinearity left in Eq. 2.15 is the cos^mu
    term. The substitution m_s = Psi**(1/mu) n_s absorbs it: raising the model to
    the power 1/mu turns it into a LINEAR equation in m_s,

        m_s . (x - x_s) = [ I |x_s - x|^(3+mu) / {(x_s - x).n}_+ ]**(1/mu),

    one row per pixel. Least-squares solve, then read off n_s and Psi:

        n_s = m_s / |m_s|,   Psi = |m_s|**mu.

    (mu == 0, an isotropic source, degenerates: there is no direction to find and
    Psi follows from Eq. 2.17 in closed form.)
    """
    I = np.asarray(I, float)
    X = np.asarray(X, float)
    N = np.asarray(N, float)
    x_s = np.asarray(x_s, float)

    d = x_s[None, :] - X                                    # surface -> LED
    r = np.linalg.norm(d, axis=1)
    lambert = np.clip(np.einsum('kc,kc->k', d, N), 1e-9, None)   # {(x_s-x).n}_+

    if abs(mu) < 1e-6:                                      # isotropic (Eq. 2.17)
        model = lambert / r ** 3
        Psi = float((I @ model) / (model @ model))
        return np.array([0.0, 0.0, -1.0]), Psi

    rhs = (I * r ** (3.0 + mu) / lambert) ** (1.0 / mu)     # scalar per pixel
    A = X - x_s[None, :]                                    # rows (x - x_s)
    m_s, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    norm = np.linalg.norm(m_s)
    n_s = m_s / norm
    Psi = float(norm ** mu)
    return n_s, Psi


# ===========================================================================
# Persistence
# ===========================================================================

def led_record(x_s, n_s, mu, Psi):
    """One JSON-safe per-LED calibration entry (numpy arrays -> lists).

    This is exactly the schema nearfield_lambertian.load_led_calibration reads.
    """
    return dict(x_s=[float(v) for v in x_s],
                n_s=[float(v) for v in np.asarray(n_s) / np.linalg.norm(n_s)],
                mu=float(mu), Psi=float(Psi))


def save_calibration(path, leds):
    """leds: {index: {'x_s':[3], 'n_s':[3], 'mu':float, 'Psi':float}}.

    Values may be numpy arrays; they are coerced to plain floats/lists so the
    file round-trips through JSON and loads in nearfield_lambertian.py.
    """
    safe = {str(k): led_record(v["x_s"], v["n_s"], v["mu"], v["Psi"])
            for k, v in leds.items()}
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# Self-tests -- synthesise ground truth, run each solver, check recovery
# ===========================================================================

def _selftest_anisotropy():
    mu = anisotropy_mu(60.0)
    ok = abs(mu - 1.0) < 1e-9
    print(f"  anisotropy : theta_1/2=60deg -> mu={mu:.4f}  "
          f"(expect 1.000)  {'PASS' if ok else 'FAIL'}")
    return ok


def _selftest_position(seed=0):
    rng = np.random.default_rng(seed)
    fx = fy = 1000.0
    cx = cy = 320.0
    R = 20.0                                        # mirror-ball radius (mm)
    x_s_true = np.array([14.0, -9.0, 3.0])          # LED near the lens plane
    poses = []
    for _ in range(10):
        C = np.array([rng.uniform(-8, 8), rng.uniform(-8, 8),
                      rng.uniform(70, 110)])        # ball in front of camera
        # boundary: exact tangent cone directions about axis a = unit(C)
        a = C / np.linalg.norm(C)
        beta = np.arcsin(R / np.linalg.norm(C))
        tmp = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(a, tmp); e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        boundary_uv = []
        for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            d = (np.cos(beta) * a
                 + np.sin(beta) * (np.cos(phi) * e1 + np.sin(phi) * e2))
            boundary_uv.append((cx + fx * d[0] / d[2], cy + fy * d[1] / d[2]))
        # highlight: the sphere point whose outward normal is the half-vector
        # between the to-camera and to-LED directions (specular reflection).
        # Fixed-point iterate u <- unit(to_cam + to_led), s = C + R u.
        u = -C / np.linalg.norm(C)                   # start: point nearest camera
        for _ in range(200):
            s = C + R * u
            to_cam = -s / np.linalg.norm(s)
            to_led = (x_s_true - s) / np.linalg.norm(x_s_true - s)
            u = (to_cam + to_led); u /= np.linalg.norm(u)
        s = C + R * u
        uv = (cx + fx * s[0] / s[2], cy + fy * s[1] / s[2])
        poses.append(dict(boundary_uv=boundary_uv, highlight_uv=uv))

    x_s = calibrate_led_position(poses, fx, fy, cx, cy, R)
    err = float(np.linalg.norm(x_s - x_s_true))
    ok = err < 2.0
    print(f"  position   : x_s={np.round(x_s,2)}  true={x_s_true}  "
          f"err={err:.2f} mm  {'PASS' if ok else 'FAIL'}")
    return ok


def _selftest_direction_intensity(seed=1):
    rng = np.random.default_rng(seed)
    fx = fy = 1000.0
    cx = cy = 320.0
    mu = 1.0
    x_s_true = np.array([12.0, -4.0, 2.0])
    n_s_true = np.array([-0.15, 0.05, 1.0]); n_s_true /= np.linalg.norm(n_s_true)
    Psi_true = 7.3e4

    # A tilted white plane sampled over the field of view. Its normal faces the
    # camera (negative Z) so the LED, on the camera side, actually lights it.
    plane_n = np.array([0.08, -0.05, -1.0]); plane_n /= np.linalg.norm(plane_n)
    p0 = np.array([0.0, 0.0, 40.0])                 # a point on the plane
    I, X, N, U, V = [], [], [], [], []
    for _ in range(4000):
        u = rng.uniform(60, 580); v = rng.uniform(60, 580)
        d = _pixel_ray(u, v, fx, fy, cx, cy)
        t = (plane_n @ p0) / (plane_n @ d)          # ray-plane hit depth
        x = t * d
        r = np.linalg.norm(x_s_true - x)
        cos_emit = max(n_s_true @ (x - x_s_true) / np.linalg.norm(x - x_s_true), 0)
        lamb = max((x_s_true - x) @ plane_n, 0.0)
        Imod = Psi_true * cos_emit ** mu * lamb / r ** 3
        # what the sensor records: multiply the corrected level back by cos^4
        xn, yn = (u - cx) / fx, (v - cy) / fy
        J = Imod * (1.0 / (xn * xn + yn * yn + 1.0) ** 2)
        J *= 1.0 + 0.01 * rng.standard_normal()     # 1% sensor noise
        I.append(J); X.append(x); N.append(plane_n); U.append(u); V.append(v)

    Icorr = corrected_gray_level(I, U, V, fx, fy, cx, cy)   # undo cos^4
    n_s, Psi = calibrate_led_direction_intensity(Icorr, X, N, x_s_true, mu)
    ang = np.degrees(np.arccos(np.clip(n_s @ n_s_true, -1, 1)))
    perr = abs(Psi - Psi_true) / Psi_true * 100
    ok = ang < 1.0 and perr < 3.0
    print(f"  direction  : n_s err={ang:.3f} deg   intensity err={perr:.2f}%  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def selftest():
    print("LED CALIBRATION SELF-TEST (synthetic ground truth)")
    ok = True
    ok &= _selftest_anisotropy()
    ok &= _selftest_position()
    ok &= _selftest_direction_intensity()
    print(f"  => {'ALL PASS' if ok else 'FAILURE'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true",
                    help="validate every solver on synthetic ground truth")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    print("Nothing to do. Capture calibration images, then feed them to the "
          "solvers here (see module docstring). Run --selftest to validate.")


if __name__ == "__main__":
    main()
