# -*- coding: utf-8 -*-
# Elsevier graphical abstract for paper 2 -- VARIANT B: "the biggest gradient can be
# confidently wrong" (danger/reveal hook).
#
# Concept (foregrounds the CORE innovation -- DIRECTION, NOT MAGNITUDE):
#   * One BIG, bold, confident-looking design-gradient arrow = the hero ("largest gradient,
#     looks trustworthy"). This is the seed-UNSTABLE radiation/aperture gradient.
#   * Reveal: re-trained across M seeds its per-component SIGN FLIPS -> the single arrow
#     splits into several seed-arrows pointing opposite ways (real signs from
#     zones_multiseed.npz: rela comp0 = + - + - -). A subtle danger accent.
#   * Verdict: a MAGNITUDE rule TRUSTS it (wrong); our solver-free gate REJECTS it because
#     the cross-seed sign DISAGREES. Contrast: a second, smaller-but-sign-stable gradient
#     (the impedance/feed gradient, sign-agreement = 1.0) that IS trusted.
#
# Honest numbers only: headline = AUC 0.91 predicting gradient sign-correctness on an
# EXTERNAL transfer-matrix benchmark; "solver-free"; "free autodiff where seeds agree, one
# full-wave solve where they disagree". (No "63% fewer solves" headline -- scoped.)
#
# All arrows / ticks / dots are DRAWN (patches), never unicode glyphs, to avoid tofu boxes.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.lines import Line2D

DIR = os.path.dirname(os.path.abspath(__file__))

# ---- refined cohesive palette -------------------------------------------------
INK = '#2c3e50'   # slate ink -- all text
GRN = '#1e7a45'   # trust  (signs agree)
RED = '#d1495b'   # verify / danger (signs disagree)
MUT = '#8a97a3'   # muted slate-grey for secondary marks
PAPER = '#ffffff'

def tint(c, a):
    r = list(mcolors.to_rgba(c)); r[3] = a; return tuple(r)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'text.color': INK, 'axes.edgecolor': INK,
})

# ---- real per-seed signs of the seed-UNSTABLE (radiation/aperture) gradient ----
Z = np.load(os.path.join(DIR, 'zones_multiseed.npz'))
rela = Z['rela']                       # (10 seeds, 6 components) radiation gradient
unstable_signs = list(np.sign(rela[:5, 0]).astype(int))   # real comp0, first 5 seeds: [+,-,+,-,-]
stable_signs = [1, 1, 1, 1, 1]         # impedance/feed gradient: cross-seed signs agree (sa=1.0)

# ===============================================================================
fig = plt.figure(figsize=(12.0, 6.0))
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')

def A(x, y, **k):  # convenience: ax coords
    return ax

# ------------------------------------------------------------------ title block
ax.text(6.0, 5.66, 'Direction, not magnitude', ha='center', va='center',
        fontsize=30, fontweight='bold', color=INK)
ax.text(6.0, 5.16,
        "a differentiable surrogate's largest design gradient can be confidently sign-wrong",
        ha='center', va='center', fontsize=12.8, color=INK)
ax.text(6.0, 4.86,
        "— a solver-free gate catches it by cross-seed sign agreement, not magnitude",
        ha='center', va='center', fontsize=12.8, color=RED, style='italic')

# thin rule under the title
ax.add_line(Line2D([0.9, 11.1], [4.58, 4.58], color=tint(INK, 0.18), lw=1.1, zorder=1))

# ------------------------------------------------------------------ helper: drawn tick / cross
def draw_tick(cx, cy, s, col, lw=3.2):
    ax.add_line(Line2D([cx - 0.30*s, cx - 0.05*s], [cy - 0.02*s, cy - 0.26*s],
                       color=col, lw=lw, solid_capstyle='round', zorder=20))
    ax.add_line(Line2D([cx - 0.05*s, cx + 0.34*s], [cy - 0.26*s, cy + 0.30*s],
                       color=col, lw=lw, solid_capstyle='round', zorder=20))

def draw_cross(cx, cy, s, col, lw=3.2):
    ax.add_line(Line2D([cx - 0.26*s, cx + 0.26*s], [cy - 0.26*s, cy + 0.26*s],
                       color=col, lw=lw, solid_capstyle='round', zorder=20))
    ax.add_line(Line2D([cx - 0.26*s, cx + 0.26*s], [cy + 0.26*s, cy - 0.26*s],
                       color=col, lw=lw, solid_capstyle='round', zorder=20))

def big_arrow(x0, y0, x1, y1, col, lw, ms, z=10, alpha=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle='-|>', mutation_scale=ms, lw=lw, color=col,
                 shrinkA=0, shrinkB=0, zorder=z, alpha=alpha, capstyle='round'))

# =========================================================== LEFT: the HERO trap
# soft danger field behind the hero
ax.add_patch(FancyBboxPatch((0.45, 0.95), 5.65, 3.42,
             boxstyle='round,pad=0.02,rounding_size=0.14',
             facecolor=tint(RED, 0.045), edgecolor=tint(RED, 0.30), lw=1.2, zorder=0))
ax.text(0.72, 4.16, 'THE BIGGEST GRADIENT', ha='left', va='center',
        fontsize=12.5, fontweight='bold', color=RED)
ax.text(3.18, 4.16, 'can be confidently WRONG', ha='left', va='center',
        fontsize=12.5, color=INK, style='italic')

# --- (1) the single big confident arrow ---------------------------------------
hx = 1.35
big_arrow(hx, 3.62, hx, 2.05, RED, lw=11.0, ms=46, z=8)
ax.text(hx, 3.86, 'large gradient', ha='center', va='center',
        fontsize=11.0, color=INK, fontweight='bold')
ax.text(hx, 1.74, 'looks trustworthy', ha='center', va='center',
        fontsize=9.6, color=MUT, style='italic')
# a magnitude rule "trusts" it -> WRONG (dashed = the naive baseline)
ax.add_patch(FancyBboxPatch((hx - 1.02, 1.04), 2.04, 0.52,
             boxstyle='round,pad=0.02,rounding_size=0.08',
             facecolor='white', edgecolor=tint(RED, 0.55), lw=1.3, ls=(0, (4, 2)), zorder=9))
draw_cross(hx - 0.70, 1.30, 0.34, RED, lw=2.4)
ax.text(hx + 0.26, 1.30, 'magnitude rule\nTRUSTS it', ha='center', va='center',
        fontsize=8.6, color=RED, fontweight='bold', linespacing=0.95, zorder=21)

# --- reveal arrow: "re-train M seeds" -----------------------------------------
ax.add_patch(FancyArrowPatch((2.18, 2.85), (3.05, 2.85), arrowstyle='-|>',
             mutation_scale=22, lw=2.4, color=INK, zorder=8))
ax.text(2.61, 3.16, 're-train', ha='center', va='center', fontsize=9.5,
        color=INK, fontweight='bold')
ax.text(2.61, 2.92, r'$M$ seeds', ha='center', va='center', fontsize=9.5, color=INK)
ax.text(2.61, 2.58, 'same design', ha='center', va='center', fontsize=8.0,
        color=MUT, style='italic')

# --- (2) the split into seed-arrows pointing opposite ways --------------------
sx = np.linspace(3.62, 5.62, len(unstable_signs))
for xi, sgn in zip(sx, unstable_signs):
    up = sgn > 0
    col = GRN if up else RED
    y_lab = 2.05
    if up:
        big_arrow(xi, 2.30, xi, 3.50, col, lw=4.2, ms=20, z=11, alpha=0.92)
    else:
        big_arrow(xi, 3.50, xi, 2.30, col, lw=4.2, ms=20, z=11, alpha=0.95)
    ax.text(xi, 3.70 if up else 2.05, '+' if up else '−', ha='center',
            va='center', fontsize=12.5, color=col, fontweight='bold')
# baseline through the seed arrows
ax.add_line(Line2D([3.42, 5.82], [2.30, 2.30], color=tint(INK, 0.35), lw=1.2, zorder=9))
ax.text(4.62, 3.95, 'signs FLIP across seeds', ha='center', va='center',
        fontsize=10.6, color=RED, fontweight='bold')
ax.text(4.62, 1.74, 'cross-seed sign disagrees', ha='center', va='center',
        fontsize=9.0, color=INK, style='italic')

# our gate REJECTS
ax.add_patch(FancyBboxPatch((3.62, 1.04), 2.04, 0.52,
             boxstyle='round,pad=0.02,rounding_size=0.08',
             facecolor=tint(RED, 0.14), edgecolor=RED, lw=1.7, zorder=9))
draw_cross(3.95, 1.30, 0.34, RED, lw=2.8)
ax.text(4.90, 1.30, 'our gate\nREJECTS it', ha='center', va='center',
        fontsize=8.6, color=RED, fontweight='bold', linespacing=0.95, zorder=21)

# =========================================================== RIGHT: the contrast
ax.add_patch(FancyBboxPatch((6.30, 0.95), 5.25, 3.42,
             boxstyle='round,pad=0.02,rounding_size=0.14',
             facecolor=tint(GRN, 0.05), edgecolor=tint(GRN, 0.32), lw=1.2, zorder=0))
ax.text(6.56, 4.16, 'A SMALLER, SIGN-STABLE GRADIENT', ha='left', va='center',
        fontsize=11.5, fontweight='bold', color=GRN)
ax.text(6.56, 3.82, 'the impedance gradient: tiny, but every seed agrees', ha='left',
        va='center', fontsize=9.2, color=INK, style='italic')

# small stable arrow + its seed copies, all the same way
gx = np.linspace(6.95, 8.55, len(stable_signs))
for xi in gx:
    big_arrow(xi, 2.55, xi, 3.35, GRN, lw=3.6, ms=17, z=11, alpha=0.92)
    ax.text(xi, 3.52, '+', ha='center', va='center', fontsize=11.0,
            color=GRN, fontweight='bold')
ax.add_line(Line2D([6.78, 8.72], [2.55, 2.55], color=tint(INK, 0.35), lw=1.2, zorder=9))
ax.text(7.75, 2.26, 'signs AGREE across seeds', ha='center', va='center',
        fontsize=9.6, color=GRN, fontweight='bold')

# arrow to verdict
ax.add_patch(FancyArrowPatch((8.92, 2.95), (9.55, 2.95), arrowstyle='-|>',
             mutation_scale=20, lw=2.2, color=INK, zorder=8))

# trusted verdict card
ax.add_patch(FancyBboxPatch((9.62, 2.05), 1.78, 1.62,
             boxstyle='round,pad=0.02,rounding_size=0.10',
             facecolor=tint(GRN, 0.12), edgecolor=GRN, lw=1.8, zorder=9))
draw_tick(10.51, 3.18, 0.52, GRN, lw=3.4)
ax.text(10.51, 2.60, 'TRUSTED', ha='center', va='center', fontsize=11.5,
        color=GRN, fontweight='bold', zorder=21)
ax.text(10.51, 2.28, 'use free autodiff', ha='center', va='center',
        fontsize=8.4, color=INK, style='italic', zorder=21)

# the one-line takeaway bridging both halves
ax.text(8.92, 1.40, 'trusts impedance,  rejects the seed-unstable radiation gradient',
        ha='center', va='center', fontsize=9.6, color=INK)
ax.text(8.92, 1.12, 'direction (sign), not magnitude, decides trust',
        ha='center', va='center', fontsize=9.0, color=MUT, style='italic')

# =========================================================== BOTTOM payoff strip
ax.add_patch(FancyBboxPatch((0.45, 0.12), 11.10, 0.68,
             boxstyle='round,pad=0.01,rounding_size=0.10',
             facecolor=INK, edgecolor='none', zorder=5))

# headline AUC (signal green on ink) -- with its own external-benchmark provenance
ax.text(1.62, 0.575, 'AUC 0.91', ha='center', va='center', fontsize=24,
        fontweight='bold', color='#6fe6a0', zorder=6)
ax.text(1.62, 0.305, 'gradient sign-correctness', ha='center', va='center',
        fontsize=8.4, color='#e0e5ea', zorder=6)
ax.text(1.62, 0.165, 'external transfer-matrix benchmark', ha='center', va='center',
        fontsize=7.2, color='#9aa3ab', style='italic', zorder=6)

# divider
ax.add_line(Line2D([3.18, 3.18], [0.20, 0.72], color=tint('#ffffff', 0.28), lw=1.1, zorder=6))

# three payoff chips (evenly spaced, no right-edge collision)
def chip(xdot, dotcol, head, sub):
    ax.add_patch(Circle((xdot, 0.46), 0.060, color=dotcol, zorder=7))
    ax.text(xdot + 0.16, 0.555, head, ha='left', va='center', fontsize=10.0,
            color='white', fontweight='bold', zorder=7)
    ax.text(xdot + 0.16, 0.285, sub, ha='left', va='center', fontsize=8.0,
            color='#cdd4da', zorder=7)

chip(3.50, '#6fe6a0', 'Solver-free gate',
     'no full-wave solve to decide trust')
chip(6.55, GRN, 'Signs agree',
     'free autodiff gradient  (0 solves)')
chip(9.45, RED, 'Signs disagree',
     'one full-wave solve, only here')

# ------------------------------------------------------------------ save + report
out_pdf = os.path.join(DIR, 'fig_graphical_p2_B.pdf')
out_png = os.path.join(DIR, 'fig_graphical_p2_B.png')
fig.savefig(out_pdf, facecolor=PAPER)
fig.savefig(out_png, dpi=150, facecolor=PAPER)
plt.close(fig)
try:
    from PIL import Image
    w, h = Image.open(out_png).size
    print(f'fig_graphical_p2_B (biggest-gradient-wrong hook): PNG {w}x{h}px; '
          f'unstable seed signs={unstable_signs}; -> {DIR}')
except Exception:
    print('fig_graphical_p2_B: written')
