# -*- coding: utf-8 -*-
# Elsevier graphical abstract for Paper 2 -- variant C: "SOLVER-FREE TRUST GATE".
# A clean modern ML-infographic pipeline (rounded nodes, soft tints, arrowheaded connectors):
#
#   [ differentiable surrogate ]      [ cross-seed sign-agreement GATE ]        [ two outcomes ]
#     PINN / neural operator    -->     re-train M seeds; do per-component   -->   signs AGREE  -> free autodiff grad (0 solves)
#     emits design Jacobian             gradient SIGNS agree?  split @ tau=0.9     signs DISAGREE-> one full-wave solve
#     dr/dg  for FREE (autodiff)
#
# CORE MESSAGE ("Direction, not magnitude"): which design gradients to TRUST is set by
# cross-seed SIGN AGREEMENT, not by gradient MAGNITUDE. Solver-free where seeds agree;
# spend ONE full-wave solve only where they disagree.
#
# HONEST NUMBERS ONLY (no fabrication):
#   hero  = AUC 0.91 predicting per-component gradient SIGN-correctness on an EXTERNAL
#           transfer-matrix (thin-film TMM) benchmark  [0.906, 95% CI [0.833,0.965],
#           880 pooled comps, 852 correct / 28 wrong  -- see AUC_CONFIDENCE_INTERVALS.md].
#   tag   = "solver-free".
#   line  = "trusts the impedance gradient, rejects the seed-unstable radiation gradient"
#           (reliability.npz: impedance/responsive comps sign-agreement = 1.0 -> trusted;
#            radiation comps sign-agreement 0.5-0.7 < tau=0.9 -> flagged. Verified.)
# We deliberately do NOT headline "63% fewer solves" (scoped to a controlled spectrum).
#
# Self-contained: needs only matplotlib (no .npz read at runtime; numbers are quoted constants
# whose provenance is documented above). Saves PDF + PNG next to this file.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- palette (cohesive, refined) ----------------
INK = '#2c3e50'   # slate ink -- all text
GRN = '#1e7a45'   # trust  (signs agree)
RED = '#d1495b'   # verify (signs disagree) -- soft rose-red
BLU = '#1f5fa6'   # surrogate / neutral nodes
MUT = '#8a97a3'   # muted grey -- secondary marks / connectors
PANEL = '#f5f7f9'  # soft card fill
EDGE = '#dce1e6'  # soft card edge


def tint(c, a):
    """Soft node tint: colour c at alpha a (returns rgba tuple)."""
    r = list(mcolors.to_rgba(c)); r[3] = a; return tuple(r)


plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.0,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})

# Single full-bleed axes in a 12 x 6 (in) data box -> total control over the infographic.
fig = plt.figure(figsize=(12.0, 6.0), facecolor='white')
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')


# ----------------------------------------------------------------------------- helpers
def rounded(x, y, w, h, fc, ec, lw=1.6, rad=0.10, z=2, ls='-'):
    # NOTE: never pass a patch-level alpha here -- it would clobber the alpha channel
    # carried by an rgba `fc` (the soft tints). Transparency lives in `fc` only.
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0.0,rounding_size={rad}',
                       facecolor=fc, edgecolor=ec, lw=lw, zorder=z, ls=ls)
    ax.add_patch(p); return p


def arrow(p0, p1, color=MUT, lw=2.4, z=5, rad=0.0, mut=16, shrinkA=2, shrinkB=2):
    a = FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}',
                        arrowstyle='-|>', mutation_scale=mut, lw=lw, color=color,
                        shrinkA=shrinkA, shrinkB=shrinkB, zorder=z,
                        joinstyle='round', capstyle='round')
    ax.add_patch(a); return a


def draw_check(cx, cy, s, color, lw=2.6, z=14):
    """Hand-drawn check mark (robust; no glyph-tofu risk)."""
    ax.plot([cx - 0.60 * s, cx - 0.14 * s, cx + 0.78 * s],
            [cy - 0.02 * s, cy - 0.46 * s, cy + 0.50 * s],
            color=color, lw=lw, solid_capstyle='round', solid_joinstyle='round',
            zorder=z, clip_on=False)


def draw_cross(cx, cy, s, color, lw=2.6, z=14):
    """Hand-drawn x mark (robust)."""
    ax.plot([cx - 0.50 * s, cx + 0.50 * s], [cy - 0.50 * s, cy + 0.50 * s],
            color=color, lw=lw, solid_capstyle='round', zorder=z, clip_on=False)
    ax.plot([cx - 0.50 * s, cx + 0.50 * s], [cy + 0.50 * s, cy - 0.50 * s],
            color=color, lw=lw, solid_capstyle='round', zorder=z, clip_on=False)


# ============================================================ TITLE BLOCK
fig.text(0.040, 0.930, 'Direction, not magnitude',
         ha='left', va='center', fontsize=30, fontweight='bold', color=INK)
# accent rule under the title (trust-green -> verify-red): a subtle "two streams" cue
ax.plot([0.50, 4.35], [5.34, 5.34], color=GRN, lw=3.4, solid_capstyle='round', zorder=3)
ax.plot([4.35, 5.95], [5.34, 5.34], color=RED, lw=3.4, solid_capstyle='round', zorder=3)
fig.text(0.040, 0.846,
         'A solver-free reliability gate for the design gradients of a differentiable surrogate',
         ha='left', va='center', fontsize=13.2, color=MUT, style='italic')

# Hero badge (top-right) -- the payoff an editor sees first.
bx, by, bw, bh = 8.42, 4.98, 3.30, 0.92
rounded(bx, by, bw, bh, tint(GRN, 0.13), GRN, lw=2.0, rad=0.16, z=4)
fig.text((bx + 0.24) / 12, (by + bh * 0.62) / 6, 'AUC',
         ha='left', va='center', fontsize=13, color=GRN, fontweight='bold')
fig.text((bx + 0.24) / 12, (by + bh * 0.26) / 6, '0.91',
         ha='left', va='center', fontsize=30, color=GRN, fontweight='bold')
fig.text((bx + 1.52) / 12, (by + bh * 0.70) / 6, 'predicts gradient',
         ha='left', va='center', fontsize=10.0, color=INK)
fig.text((bx + 1.52) / 12, (by + bh * 0.42) / 6, 'sign-correctness',
         ha='left', va='center', fontsize=10.0, color=INK)
fig.text((bx + 1.52) / 12, (by + bh * 0.12) / 6, 'external TMM benchmark',
         ha='left', va='center', fontsize=8.2, color=MUT, style='italic')

# pipeline lane centre-line
yc = 2.74

# ============================================================ LEFT: differentiable surrogate
lx, lw_ = 0.50, 3.02
lh = 2.46; ly = yc - lh / 2
rounded(lx, ly, lw_, lh, tint(BLU, 0.09), BLU, lw=2.0, rad=0.14, z=2)
ax.text(lx + lw_ / 2, ly + lh - 0.30, 'DIFFERENTIABLE SURROGATE',
        ha='center', va='center', fontsize=11.8, color=BLU, fontweight='bold')
ax.text(lx + lw_ / 2, ly + lh - 0.64, 'PINN  /  neural operator',
        ha='center', va='center', fontsize=10.4, color=INK)

# autodiff Jacobian chip
jx, jw = lx + 0.32, lw_ - 0.64
jh = 0.92; jy = ly + 0.52
rounded(jx, jy, jw, jh, 'white', BLU, lw=1.5, rad=0.12, z=3)
ax.text(jx + jw / 2, jy + jh * 0.68, 'design Jacobian',
        ha='center', va='center', fontsize=10.2, color=INK)
ax.text(jx + jw / 2, jy + jh * 0.27,
        r'$\partial \mathbf{r}/\partial \mathbf{g}$  for free  (autodiff)',
        ha='center', va='center', fontsize=13, color=BLU, fontweight='bold')
ax.text(lx + lw_ / 2, ly + 0.235, 'one forward pass  gives the full gradient',
        ha='center', va='center', fontsize=8.8, color=MUT, style='italic')

# ============================================================ CENTER: the GATE
gx, gw = 4.66, 2.84
gh = 3.30; gy = yc - gh / 2

# connector LEFT -> GATE (label centred in the gap, clear of both boxes)
arrow((lx + lw_ + 0.04, yc), (gx - 0.04, yc), color=INK, lw=2.8, mut=18)
ax.text((lx + lw_ + gx) / 2, yc + 0.32, r'$K$ comps', ha='center', va='center',
        fontsize=9.0, color=INK)
rounded(gx, gy, gw, gh, tint(MUT, 0.13), INK, lw=2.2, rad=0.15, z=2)
ax.text(gx + gw / 2, gy + gh - 0.30, 'TRUST GATE',
        ha='center', va='center', fontsize=13, color=INK, fontweight='bold')
ax.text(gx + gw / 2, gy + gh - 0.61, 'scored by cross-seed sign agreement',
        ha='center', va='center', fontsize=9.2, color=INK)

# mechanism box: re-train M seeds, read per-component gradient SIGNS
sx, sw = gx + 0.28, gw - 0.56
sh = 1.16; sy = gy + gh - 1.96
rounded(sx, sy, sw, sh, 'white', MUT, lw=1.4, rad=0.12, z=3)
ax.text(sx + sw / 2, sy + sh - 0.26, r're-train $M$ seeds', ha='center', va='center',
        fontsize=9.8, color=INK)
# two compact "sign" rows: + + + + + (agree, green) and + - + - + (disagree, red)
row_y1 = sy + sh - 0.62
row_y2 = sy + 0.26
gxc = sx + 0.30
for j, sgn in enumerate(['+', '+', '+', '+', '+']):
    ax.text(gxc + j * 0.305, row_y1, sgn, ha='center', va='center',
            fontsize=12.5, color=GRN, fontweight='bold')
ax.text(sx + sw - 0.30, row_y1, 'agree', ha='center', va='center',
        fontsize=8.4, color=GRN, style='italic')
for j, sgn in enumerate(['+', '-', '+', '-', '+']):
    ax.text(gxc + j * 0.305, row_y2, sgn, ha='center', va='center',
            fontsize=12.5, color=RED, fontweight='bold')
ax.text(sx + sw - 0.30, row_y2, 'disagree', ha='center', va='center',
        fontsize=8.4, color=RED, style='italic')

# SPLIT switch at tau=0.9 (two-way branch node)
nx, ny = gx + gw / 2, gy + 0.72
ax.add_patch(Circle((nx, ny), 0.235, facecolor='white', edgecolor=INK, lw=2.0, zorder=6))
ax.text(nx, ny, r'$\tau$', ha='center', va='center', fontsize=13, color=INK,
        fontweight='bold', zorder=7)
ax.text(nx, gy + 0.245, r'split at  $\tau = 0.9$', ha='center', va='center',
        fontsize=9.6, color=INK, fontweight='bold')
# internal connector: mechanism box -> split node
arrow((nx, sy - 0.02), (nx, ny + 0.255), color=INK, lw=1.9, mut=13, shrinkA=1, shrinkB=1)

# ============================================================ RIGHT: two outcomes
rx = 7.92
ow = 3.80
# --- TRUST outcome (top) ---
th = 1.40; ty = yc + 0.18
rounded(rx, ty, ow, th, tint(GRN, 0.12), GRN, lw=2.0, rad=0.16, z=2)
draw_check(rx + 0.50, ty + th / 2 + 0.06, 0.34, GRN, lw=3.2)
ax.text(rx + 1.02, ty + th - 0.34, 'signs agree', ha='left', va='center',
        fontsize=13, color=GRN, fontweight='bold')
ax.text(rx + 2.36, ty + th - 0.34, '->  TRUST', ha='left', va='center',
        fontsize=13, color=GRN, fontweight='bold')
ax.text(rx + 1.02, ty + th - 0.72, 'use the free autodiff gradient', ha='left', va='center',
        fontsize=10.2, color=INK)
ax.text(rx + 1.02, ty + 0.29, 'most components  -  0 solves', ha='left', va='center',
        fontsize=10.2, color=GRN, fontweight='bold')

# --- VERIFY outcome (bottom) ---
vh = 1.40; vy = yc - 0.18 - vh
rounded(rx, vy, ow, vh, tint(RED, 0.11), RED, lw=2.0, rad=0.16, z=2)
draw_cross(rx + 0.50, vy + vh / 2 + 0.03, 0.32, RED, lw=3.2)
ax.text(rx + 1.02, vy + vh - 0.34, 'signs disagree', ha='left', va='center',
        fontsize=13, color=RED, fontweight='bold')
ax.text(rx + 2.62, vy + vh - 0.34, '->  VERIFY', ha='left', va='center',
        fontsize=13, color=RED, fontweight='bold')
ax.text(rx + 1.02, vy + vh - 0.72, 'spend one full-wave solve', ha='left', va='center',
        fontsize=10.2, color=INK)
ax.text(rx + 1.02, vy + 0.29, 'few components  -  verified', ha='left', va='center',
        fontsize=10.2, color=RED, fontweight='bold')

# branch connectors GATE -> two outcomes (the two-way switch)
arrow((gx + gw + 0.04, yc + 0.10), (rx - 0.04, ty + th / 2),
      color=GRN, lw=2.8, rad=-0.20, mut=17)
arrow((gx + gw + 0.04, yc - 0.10), (rx - 0.04, vy + vh / 2),
      color=RED, lw=2.8, rad=0.20, mut=17)

# ============================================================ BOTTOM honest payoff strip
sy0 = 0.30
rounded(0.50, sy0, 11.22, 0.84, PANEL, EDGE, lw=1.2, rad=0.12, z=1)
# solver-free badge (left)
rounded(0.74, sy0 + 0.16, 2.30, 0.52, 'white', INK, lw=1.5, rad=0.20, z=2)
ax.text(0.74 + 1.15, sy0 + 0.42, 'SOLVER-FREE', ha='center', va='center',
        fontsize=12.5, color=INK, fontweight='bold')
# the honest, concrete verdict
draw_check(3.36, sy0 + 0.42, 0.17, GRN, lw=2.4)
ax.text(3.64, sy0 + 0.42, 'trusts the impedance gradient', ha='left', va='center',
        fontsize=11.4, color=INK)
draw_cross(7.30, sy0 + 0.42, 0.17, RED, lw=2.4)
ax.text(7.58, sy0 + 0.42, 'rejects the seed-unstable radiation gradient', ha='left', va='center',
        fontsize=11.4, color=INK)

# ---------------- save ----------------
out_pdf = os.path.join(DIR, 'fig_graphical_p2_C.pdf')
out_png = os.path.join(DIR, 'fig_graphical_p2_C.png')
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=150)
plt.close(fig)
try:
    from PIL import Image
    w, h = Image.open(out_png).size
    print(f'fig_graphical_p2_C (solver-free trust gate): PNG {w}x{h} px -> {DIR}')
except Exception:
    print('fig_graphical_p2_C (solver-free trust gate): written')
