"""Build a self-contained, theme-aware HTML version of the manuscript with the
figures embedded as data URIs. Run: python build_html.py  ->  LuminaBone_paper.html
"""
import base64, os
HERE = os.path.dirname(os.path.abspath(__file__))

def datauri(p):
    b = base64.b64encode(open(os.path.join(HERE, p), "rb").read()).decode()
    return f"data:image/png;base64,{b}"

FIG = {k: datauri(f"figures/{v}") for k, v in {
    "bowl": "fig2_bowl.png", "shot6": "fig3_shot6.png", "shot4": "fig4_shot4.png"}.items()}

CSS = """
:root{
  --bg:#fbfcfd; --panel:#f3f6f8; --abstract:#eef4f6; --text:#19222b;
  --muted:#5a6673; --accent:#0f6e8c; --accent-2:#0b5468; --rule:#dde5ea;
  --good:#1f7a4d; --bad:#b23b3b;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0e141a; --panel:#161f27; --abstract:#132029; --text:#e7eef3;
  --muted:#93a1ac; --accent:#3fb4d2; --accent-2:#68c8e0; --rule:#233240;
  --good:#4cc187; --bad:#e0736f;
}}
:root[data-theme="dark"]{
  --bg:#0e141a; --panel:#161f27; --abstract:#132029; --text:#e7eef3;
  --muted:#93a1ac; --accent:#3fb4d2; --accent-2:#68c8e0; --rule:#233240;
  --good:#4cc187; --bad:#e0736f;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);margin:0;
  font-family:Georgia,"Iowan Old Style","Times New Roman",serif;
  line-height:1.62;font-size:18px;-webkit-font-smoothing:antialiased}
.wrap{max-width:720px;margin:0 auto;padding:64px 24px 96px}
.eyebrow{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:600;
  color:var(--accent);margin:0 0 10px}
h1{font-size:33px;line-height:1.18;margin:0 0 18px;text-wrap:balance;
  letter-spacing:-.01em;font-weight:700}
h2{font-size:23px;margin:52px 0 6px;text-wrap:balance;letter-spacing:-.01em;
  padding-bottom:8px;border-bottom:1px solid var(--rule)}
h2 .n{color:var(--accent);font-family:system-ui,sans-serif;font-size:16px;
  font-weight:600;margin-right:.6em;vertical-align:.08em}
h3{font-size:18px;margin:26px 0 4px;color:var(--accent-2);
  font-family:system-ui,sans-serif;font-weight:650;letter-spacing:.005em}
p{margin:12px 0}
.byline{font-family:system-ui,sans-serif;color:var(--muted);font-size:15px;
  margin:0 0 4px}
.affil{font-family:system-ui,sans-serif;color:var(--muted);font-size:13px;
  margin:0 0 28px}
.abstract{background:var(--abstract);border:1px solid var(--rule);
  border-left:3px solid var(--accent);border-radius:6px;padding:22px 26px;
  font-size:16.5px;line-height:1.6}
.abstract p:first-child{margin-top:0}.abstract p:last-child{margin-bottom:0}
.kw{font-family:system-ui,sans-serif;font-size:13.5px;color:var(--muted);
  margin:14px 0 0}
.kw b{color:var(--text)}
figure{margin:30px 0;padding:0}
figure img{width:100%;display:block;border:1px solid var(--rule);border-radius:6px;
  background:#fff}
figcaption{font-family:system-ui,sans-serif;font-size:13.5px;color:var(--muted);
  margin-top:10px;line-height:1.5}
figcaption b{color:var(--text)}
.tablewrap{overflow-x:auto;margin:24px 0}
table{border-collapse:collapse;width:100%;font-family:system-ui,sans-serif;
  font-size:14.5px;font-variant-numeric:tabular-nums}
caption{caption-side:top;text-align:left;font-family:system-ui,sans-serif;
  font-size:13px;color:var(--muted);margin-bottom:8px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--rule)}
thead th{border-top:2px solid var(--text);border-bottom:1.5px solid var(--text);
  font-weight:650}
tbody tr:last-child td{border-bottom:2px solid var(--text)}
td.num,th.num{text-align:right}
.win{color:var(--good);font-weight:650}.fail{color:var(--bad);font-weight:650}
.hl{font-weight:700}
.refs{font-family:system-ui,sans-serif;font-size:13.5px;line-height:1.55;
  color:var(--muted);counter-reset:r}
.refs p{margin:7px 0;padding-left:30px;text-indent:-30px}
.appendix{margin-top:56px;background:var(--panel);border:1px solid var(--rule);
  border-radius:8px;padding:8px 30px 30px}
.appendix h2{border-color:var(--rule)}
strong{font-weight:700}
em{font-style:italic}
code{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.86em;
  background:var(--panel);padding:1px 5px;border-radius:4px}
hr{border:0;border-top:1px solid var(--rule);margin:40px 0}
"""

HTML = f"""<style>{CSS}</style>
<div class="wrap">
  <p class="eyebrow">Image-Guided Surgery · Photometric Stereo · Preprint</p>
  <h1>Near-Field Photometric Stereo for CT-Registered Surface Reconstruction with a
  Near-Coaxial Monocular Endoscope: Millimetre Accuracy on Spinal Bone</h1>
  <p class="byline">A. Ding <em>et al.</em></p>
  <p class="affil">Affiliation to be completed · Correspondence: andyding09@gmail.com</p>

  <div class="abstract">
    <p><b>Purpose.</b> Image-guided spine surgery depends on registering the exposed
    bony surface to a preoperative CT. We test whether a monocular endoscope with a
    ring of near-coaxial LEDs &mdash; the geometry forced by an 18&nbsp;mm scope &mdash;
    can reconstruct the exposed spinal surface accurately enough for per-vertebra CT
    registration, using only its own images and no added tracking hardware.</p>
    <p><b>Methods.</b> Four single-LED frames per pose drive a photometric-stereo
    solve. We compare far-field photometric stereo, a near-field point-source model
    (per-pixel light directions + inverse-square attenuation), and an inverse-square
    brightness-to-range baseline. Each reconstruction is anchored to CT through
    retroreflective fiducials visible in both modalities, via per-vertebra
    Perspective-<em>n</em>-Point pose and an affine depth calibration. Accuracy is the
    3-D fiducial registration error (FRE) against CT on five vertebral levels.</p>
    <p><b>Results.</b> With the near-field model, three of five levels reconstructed to
    <b>1.2&ndash;1.7&nbsp;mm</b> 3-D FRE with four fiducials each &mdash; comparable to
    clinical surface-based spine registration (&asymp;1&nbsp;mm, 20 points) and better
    than reported in-vivo endoscopic photometric stereo (&lt;3&nbsp;mm). Near-field
    modelling removed the systematic &ldquo;bowl&rdquo; of far-field integration (worst
    far-field level improved from 6.6 to 1.7&nbsp;mm FRE). The inverse-square baseline
    failed (7.8&ndash;9.6&nbsp;mm, physically inverted depth). A working-distance
    envelope of &asymp;55&nbsp;mm is characterised.</p>
    <p><b>Conclusion.</b> Near-coaxial monocular photometric stereo yields per-vertebra,
    CT-registered surfaces at millimetre accuracy within a defined envelope, without
    added radiation or tracking hardware; the dominant residual is the use of assumed
    rather than calibrated LED positions.</p>
  </div>
  <p class="kw"><b>Keywords</b> &nbsp;photometric stereo · near-field · endoscopy ·
  image-guided surgery · spine · CT registration · fiducial registration error</p>

  <h2><span class="n">1</span>Introduction</h2>
  <p>Spinal navigation registers each vertebra as an independent rigid body &mdash; the
  spine flexes at the intervertebral discs, so no single rigid transform fits multiple
  levels &mdash; using intraoperative imaging (cone-beam CT / O-arm, adding radiation)
  or a tracked surface digitiser. An optical method that reconstructs the exposed
  surface from the endoscope already in the field, with no added radiation and no
  external tracker, is attractive.</p>
  <p>Metric shape from a single moving endoscope is hard: structure-from-motion and
  SLAM need texture and parallax that wet bone provides poorly, and neural rendering
  needs many views and does not natively produce a CT-registered metric surface.
  <em>Photometric stereo</em> is attractive because it is <em>albedo-invariant</em>: it
  recovers the surface normal from <em>ratios</em> of brightness across illuminations,
  so a dark and a bright patch at the same tilt give the same normal &mdash; exactly the
  property brightness-based shape-from-shading lacks on textured bone.</p>
  <p>The obstacle is geometry. A ring of LEDs around an 18&nbsp;mm scope sits
  &asymp;6&nbsp;mm off the optical axis, so at 40&nbsp;mm working distance the lights are
  <em>near-coaxial</em> (&asymp;8.6&deg; off-axis; &asymp;17&deg; between opposing pairs)
  and <em>near-field</em> (direction and intensity vary across the field). Far-field
  photometric stereo mis-models this and integrates the residual brightness gradient
  into a spurious low-frequency curvature &mdash; the &ldquo;bowl&rdquo;.</p>
  <p><b>Contributions.</b> (1) A near-field photometric-stereo pipeline for a
  near-coaxial LED endoscope that removes the far-field bowl; (2) a per-vertebra CT
  registration protocol using dual-modality fiducials, PnP pose, and affine depth
  calibration, validated by 3-D FRE; (3) a controlled far-field / near-field /
  inverse-square comparison on CT-validated data; (4) a characterisation of the
  working-distance envelope set by the near-coaxial baseline; (5) a transparent failure
  analysis isolating assumed LED positions as the dominant residual error.</p>

  <h2><span class="n">2</span>Related work</h2>
  <p>Woodham's far-field photometric stereo [1] recovers normals from &ge;3 directional
  illuminations; Frankot&ndash;Chellappa integration [2] yields depth. Qu&eacute;au et
  al. [3] give a complete LED forward model with calibration; robust point-light
  <em>position</em> calibration reaches &lt;2.7&nbsp;mm [4], and multi-view near-field
  benchmarks now exist [5]. In-vivo endoscopic photometric stereo has reconstructed the
  colon at &lt;3&nbsp;mm mean depth error [6]; neural and Gaussian-splatting pipelines
  [7] excel at rendering but do not deliver a CT-registered metric surface.
  Surface-based spine registration reaches &asymp;0.96&nbsp;mm RMS with ~20 lamina
  points [8]; quality is reported as FRE/TRE [9]. We target metric, CT-validated,
  <em>per-vertebra</em> registration on an unusually near-coaxial rig.</p>

  <h2><span class="n">3</span>Materials and methods</h2>
  <h3>3.1 Imaging system</h3>
  <p>A monocular endoscope (native 640&times;480) carries four LEDs on an
  &asymp;6.05&nbsp;mm ring. At 40&nbsp;mm each LED is &asymp;8.6&deg; off-axis; opposing
  LEDs subtend &asymp;17&deg;, so the tilt-encoding lateral component is only
  &asymp;cos&nbsp;81&deg; &asymp; 15% of each light vector. Field of view
  &asymp;37&ndash;43&nbsp;mm. Retroreflective fiducials appear in CT as small surface
  craters, giving dual-modality correspondences. Each capture records a dark frame and
  four single-LED frames.</p>
  <h3>3.2 Calibration &amp; preprocessing</h3>
  <p>Intrinsics from a ChArUco target; working focal length (<code>fx&asymp;860&nbsp;px</code>)
  validated by 4-point PnP reprojection (0.42&nbsp;px residual). Frames are
  dark-subtracted, gamma-linearised, specular-inpainted, and per-frame median
  exposure-balanced.</p>
  <h3>3.3 LED index &rarr; azimuth mapping</h3>
  <p>The physical wiring is LED1&rarr;top (90&deg;), LED2&rarr;bottom (270&deg;),
  LED3&rarr;left (180&deg;), LED4&rarr;right (0&deg;). An incorrect assumption produced
  normals <em>anti-correlated</em> with CT (<em>r</em>&asymp;&minus;0.73); the correct
  mapping gives <em>r</em>&asymp;+0.97/+0.99 on two CT-validated levels &mdash; a common
  silent failure mode for multi-LED rigs.</p>
  <h3>3.4 Photometric-stereo models</h3>
  <p><em>Far-field:</em> constant per-LED direction, solve
  <em>I<sub>k</sub></em>=&rho;(<b>n</b>&middot;<b>L<sub>k</sub></b>) and integrate.
  <em>Near-field:</em> for a back-projected point at depth <em>D</em> and LED position
  <b>P<sub>s</sub></b>, the incident direction and 1/<em>r</em><sup>2</sup> attenuation
  are computed per pixel; writing
  <em>m<sub>s</sub></em>=<em>I<sub>s</sub></em>&middot;&#8741;<b>P<sub>s</sub></b>&minus;<b>X</b>&#8741;<sup>2</sup>
  linearises the point-source model to a per-pixel least-squares solve, iterated with a
  CT-anchored depth (4 iterations). LED positions are <em>assumed</em> from the ring
  geometry &mdash; the principal approximation.</p>
  <h3>3.5 Per-vertebra CT registration</h3>
  <p>Four non-coplanar fiducials give the CT&rarr;camera pose (SQPnP). Photometric depth
  is affine-calibrated, <em>z</em><sub>lens</sub>=<em>a&middot;z</em><sub>photo</sub>+<em>b</em>,
  on the fiducials; the surface is back-projected and transformed into CT coordinates.
  Each vertebra is registered independently (the spine flexes); a single rigid
  multi-level fit is used only as a rigidity check. The dense surface is masked
  (brightness, photometric-residual confidence, largest connected component, border
  crop) and mask-aware smoothed before meshing.</p>

  <h2><span class="n">4</span>Results</h2>
  <div class="tablewrap"><table>
    <caption>Table 1 &mdash; 3-D fiducial registration error (FRE), far-field vs
    near-field, over five vertebral levels. Levels 4&ndash;6 lie within the envelope.</caption>
    <thead><tr><th>Level</th><th>Working dist.</th><th class="num">Far-field FRE</th>
    <th class="num">Near-field FRE</th></tr></thead>
    <tbody>
      <tr><td>4</td><td>near</td><td class="num">1.54 mm</td><td class="num win">1.72 mm</td></tr>
      <tr><td>5</td><td>near</td><td class="num">1.22 mm</td><td class="num win">1.32 mm</td></tr>
      <tr><td>6</td><td>~55 mm</td><td class="num">6.57 mm</td><td class="num win">1.74 mm</td></tr>
      <tr><td>7</td><td>far (3 markers)</td><td class="num">11.14 mm</td><td class="num">9.72 mm</td></tr>
      <tr><td>8</td><td>~110 mm</td><td class="num">5.66 mm</td><td class="num fail">degenerate</td></tr>
    </tbody>
  </table></div>

  <h3>4.1 Far-field vs near-field</h3>
  <p>On levels 4&ndash;6 the near-field model gives <span class="hl">1.2&ndash;1.7&nbsp;mm</span>
  FRE with four fiducials each. The decisive gain is level&nbsp;6 (6.6&rarr;1.7&nbsp;mm).
  Fig.&nbsp;1 shows the same shot, same registration: the far-field surface bows while
  the near-field surface flattens. Even on level&nbsp;4 &mdash; registered to 1.5&nbsp;mm
  &mdash; the far-field surface arches between fiducials; near-field reduced the
  plane-fit deviation from 6.90 to 4.98&nbsp;mm (part of the residual is genuine bone
  relief). Figs.&nbsp;2&ndash;3 show near-field surfaces registered to CT with
  reconstructed fiducials on the ground truth.</p>

  <figure><img src="{FIG['bowl']}" alt="Far-field vs near-field surface, level 4">
    <figcaption><b>Figure 1.</b> Level 4, identical registration &mdash; only the light
    model differs. Far-field (left) integrates a bowl; near-field (right) flattens it.
    </figcaption></figure>

  <figure><img src="{FIG['shot6']}" alt="Level 6 near-field surface registered to CT">
    <figcaption><b>Figure 2.</b> Level 6 near-field surface registered into CT
    coordinates; reconstructed fiducials (open circles) on CT ground truth (crosses).
    3-D FRE 1.7&nbsp;mm, versus 6.6&nbsp;mm far-field.</figcaption></figure>

  <figure><img src="{FIG['shot4']}" alt="Level 4 near-field surface registered to CT">
    <figcaption><b>Figure 3.</b> Level 4 near-field surface registered to CT; 3-D FRE
    1.7&nbsp;mm.</figcaption></figure>

  <h3>4.2 Inverse-square baseline</h3>
  <p>A brightness-to-range baseline (<em>r</em>&prop;1/&radic;<em>I</em>) with identical
  CT anchoring failed: 7.8&ndash;9.6&nbsp;mm FRE, dense relief inflating to
  150&ndash;559&nbsp;mm (true CT range 17&ndash;26&nbsp;mm), and a physically inverted
  depth slope. The cause is intrinsic &mdash; inverse-square assumes constant albedo, so
  dark bone reads as &ldquo;far&rdquo; &mdash; quantifying why the normal-based solve is
  necessary.</p>
  <h3>4.3 Working-distance envelope</h3>
  <p>At 110&nbsp;mm the 6&nbsp;mm ring subtends only &asymp;3&deg;; tilt signal collapses
  and the near-field solve degenerates (level&nbsp;8). Level&nbsp;7 is limited separately
  by only three fiducials (under-constrained PnP). The method is reliable within
  &asymp;55&nbsp;mm &mdash; a property of the near-coaxial baseline, reported as an
  operating specification.</p>

  <h2><span class="n">5</span>Discussion</h2>
  <div class="tablewrap"><table>
    <caption>Table 2 &mdash; Accuracy in context.</caption>
    <thead><tr><th>Method / study</th><th class="num">Accuracy</th><th>Notes</th></tr></thead>
    <tbody>
      <tr><td class="hl">This work (near-field, 4&ndash;6)</td><td class="num hl">1.2&ndash;1.7 mm</td><td>4 fiducials/level</td></tr>
      <tr><td>Surface-based spine reg. [8]</td><td class="num">~0.96 mm</td><td>20 points, clinical</td></tr>
      <tr><td>In-vivo endoscopic PS [6]</td><td class="num">&lt;3 mm</td><td>colon, depth error</td></tr>
      <tr><td>Near-field light calib. [4]</td><td class="num">&lt;2.7 mm</td><td>light position</td></tr>
    </tbody>
  </table></div>
  <p>Our millimetre accuracy sits between clinical surface registration and reported
  in-vivo endoscopic photometric stereo, with fewer fiducials and a harder near-coaxial
  geometry, and &mdash; unlike appearance-oriented neural methods &mdash; yields a
  directly CT-registered metric surface. The contribution is that on a near-coaxial
  endoscopic rig the far-field approximation injects a systematic shape error a global
  scale/offset cannot remove, and a per-pixel point-source model removes it while
  preserving albedo-invariance.</p>
  <p><b>Limitations.</b> (1) <em>Assumed LED positions</em> bound the residual bowl; a
  mirror-sphere / matte-target calibration [3,4] is the next step, expected to move
  levels 6&ndash;8 toward 1&nbsp;mm. (2) Near-coaxial low SNR compresses recovered
  relief. (3) Reliable only to &asymp;55&nbsp;mm. (4) Three fiducials under-constrain
  pose (level&nbsp;7); specular markers are inpainted, leaving small holes. (5) Bench
  validation on one instrumented specimen; wet in-vivo fields are future work.</p>
  <p><b>Clinical relevance.</b> Per-vertebra registration matches spine navigation
  practice; the combined multi-level result is a bench consistency check. A calibrated
  CT-registered optical surface from the existing scope could reduce reliance on
  intraoperative radiation for re-registration.</p>

  <h2><span class="n">6</span>Conclusion</h2>
  <p>A near-coaxial monocular endoscope, driven by four-LED near-field photometric
  stereo and anchored to CT through dual-modality fiducials, reconstructs the exposed
  spinal surface to <b>1.2&ndash;1.7&nbsp;mm</b> per vertebra within a &asymp;55&nbsp;mm
  envelope &mdash; competitive with clinical surface registration and better than
  reported in-vivo endoscopic photometric stereo, without added radiation or tracking
  hardware. The far-field bowl is removed; the inverse-square shortcut is shown
  inapplicable to textured bone; and the dominant residual is localised to assumed LED
  positions, defining the next experiment.</p>

  <h2><span class="n">&nbsp;</span>References</h2>
  <div class="refs">
    <p>[1] R. J. Woodham, &ldquo;Photometric method for determining surface orientation from multiple images,&rdquo; <em>Opt. Eng.</em>, 19(1), 1980.</p>
    <p>[2] R. T. Frankot, R. Chellappa, &ldquo;A method for enforcing integrability in shape from shading,&rdquo; <em>IEEE TPAMI</em>, 10(4), 1988.</p>
    <p>[3] Y. Qu&eacute;au et al., &ldquo;LED-based photometric stereo: modeling, calibration and numerical solution,&rdquo; <em>JMIV</em>, 2018. arXiv:1707.01018.</p>
    <p>[4] &ldquo;Robust point light source calibration for near-field photometric stereo,&rdquo; <em>Applied Optics</em>, 62(36):9512, 2023.</p>
    <p>[5] &ldquo;LUCES-MV: multi-view near-field point-light photometric stereo,&rdquo; arXiv:2412.16737, 2024.</p>
    <p>[6] &ldquo;Photometric single-view dense 3D reconstruction in endoscopy,&rdquo; arXiv:2204.09083, 2022.</p>
    <p>[7] &ldquo;Diff2DGS: reliable reconstruction of occluded surgical scenes,&rdquo; arXiv:2602.18314, 2026.</p>
    <p>[8] &ldquo;Surface-based registration accuracy of CT-based image-guided spine surgery,&rdquo; PubMed 15526221.</p>
    <p>[9] J. M. Fitzpatrick, J. B. West, C. R. Maurer, &ldquo;Predicting error in rigid-body point-based registration,&rdquo; <em>IEEE TMI</em>, 17(5), 1998.</p>
  </div>

  <div class="appendix">
    <h2><span class="n">A</span>Publication strategy</h2>
    <p><b>Primary target &mdash; IJCARS</b> (Int. J. Computer Assisted Radiology &amp;
    Surgery), IF&nbsp;&asymp;2.8, Q1. Scope is exactly computer-assisted interventions,
    navigation, and registration. <em>Recommended first submission.</em></p>
    <p><b>Alternatives:</b> IEEE TBME (IF&nbsp;&asymp;4.4, Q1 &mdash; lead with the
    near-field modelling); Biomedical Optics Express (Q1 &mdash; lead with the optical /
    LED novelty); Medical Physics (Q1 &mdash; rigorous validation home).</p>
    <p><b>Reach (after LED calibration + a second specimen):</b> Medical Image Analysis
    and IEEE TMI (IF&nbsp;&asymp;10, top Q1).</p>
    <p><b>What Q1 reviewers will expect:</b> CT ground-truth metric (have it: 3-D FRE);
    a model ablation (have it: far / near / inverse-square); disclosed limits (have them:
    envelope, 3-marker level); reproducible code (release <code>marker_pipeline/</code>);
    and &mdash; before a top-tier attempt &mdash; a <em>calibrated</em> LED forward model
    plus &ge;2 specimens to convert &ldquo;millimetre on a bench&rdquo; into a clinical
    accuracy claim.</p>
  </div>
</div>
"""

open(os.path.join(HERE, "LuminaBone_paper.html"), "w", encoding="utf-8").write(HTML)
print("wrote LuminaBone_paper.html", len(HTML), "chars")
