# -*- coding: utf-8 -*-
# Elsevier graphical abstract for paper 2 -- concept "SAME MAGNITUDE, OPPOSITE TRUST".
# Core message: for a differentiable surrogate's DESIGN gradients, which to trust is set by
# CROSS-SEED SIGN AGREEMENT (re-train M seeds; do per-component gradient signs agree?), NOT by
# gradient magnitude.  Two large identical-length design arrows sit side by side (same size).
# Behind each, a small fan of M seed-gradient arrows: LEFT all aligned (signs agree) -> TRUST the
# free autodiff gradient (0 solves); RIGHT fanned with flips (signs disagree) -> VERIFY with one
# full-wave solve.  Same size, opposite verdict.  Honest numbers only (AUC 0.91 on an external
# transfer-matrix benchmark; solver-free).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

DIR = os.path.dirname(os.path.abspath(__file__))
INK = '#2c3e50'    # slate ink (all text)
GRN = '#1e7a45'    # trust  (signs agree)
RED = '#d1495b'    # verify (signs disagree) -- soft rose-red
MUT = '#8a97a3'    # muted slate-grey for secondary marks

def tint(c, a):
    r = list(mcolors.to_rgba(c)); r[3] = a; return tuple(r)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'text.color': INK,
})

fig = plt.figure(figsize=(12.0, 6.0))
fig.patch.set_facecolor('white')
# single full-canvas axes in 0..1 coords so everything is placed by hand (no clipping)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# ----------------------------------------------------------------------------------------------
#  title block
# ----------------------------------------------------------------------------------------------
ax.text(0.5, 0.945, 'Direction, not magnitude', ha='center', va='center',
        fontsize=30, fontweight='bold', color=INK)
ax.text(0.5, 0.882,
        'which design gradients of a differentiable surrogate to trust is set by '
        'cross-seed sign agreement, not gradient size',
        ha='center', va='center', fontsize=12.5, color=MUT, style='italic')

# thin slate rule under the title for a clean editorial banner feel
ax.plot([0.085, 0.915], [0.842, 0.842], color=tint(INK, 0.22), lw=1.0, solid_capstyle='round')

# ----------------------------------------------------------------------------------------------
#  two contrast panels (left = TRUST / green, right = VERIFY / red)
# ----------------------------------------------------------------------------------------------
# panel rectangles (soft tinted cards)
PY0, PY1 = 0.350, 0.820          # panel vertical extent (leave a clear band for the punch line)
LPX0, LPX1 = 0.060, 0.487        # left card x-extent
RPX0, RPX1 = 0.513, 0.940        # right card x-extent

cards = {}
for key, (x0, x1, col, al) in {'L': (LPX0, LPX1, GRN, 0.085),
                               'R': (RPX0, RPX1, RED, 0.075)}.items():
    p = FancyBboxPatch((x0, PY0), x1 - x0, PY1 - PY0,
                       boxstyle='round,pad=0.006,rounding_size=0.022',
                       facecolor=tint(col, al), edgecolor=tint(col, 0.95), lw=1.8, zorder=1)
    ax.add_patch(p)
    cards[key] = p
left_card, right_card = cards['L'], cards['R']

# a true visual circle must be drawn as an ellipse: figure is 2:1, so width = height/ASP
ASP = (12.0 / 6.0)               # x-units per y-unit visual stretch

# --- geometry of the hero arrows: identical length on both sides (SAME MAGNITUDE) ---
LCx = 0.5 * (LPX0 + LPX1)         # left panel centre x
RCx = 0.5 * (RPX0 + RPX1)         # right panel centre x
ARR_Y = 0.520                     # baseline y the big (consensus) arrows point along
ARR_DX = 0.125                    # half-length (same for both => identical magnitude)
TAILx = lambda cx: cx - ARR_DX    # consensus-arrow tail x
# the seed fan radiates from a point INBOARD of the tail so backward (flipped) seeds stay
# inside the panel and never cross the centre divider
FANx = lambda cx: cx - ARR_DX + 0.040
SEED_LEN = 0.092                  # seed arrow length (shorter than the consensus arrow)

def big_arrow(cx, col):
    """the large consensus design gradient: a soft wide band + a crisp centre-line on top."""
    a = FancyArrowPatch((TAILx(cx), ARR_Y), (cx + ARR_DX, ARR_Y),
                        arrowstyle='-|>,head_width=1.0,head_length=0.95',
                        mutation_scale=30, lw=13.0, color=tint(col, 0.26), zorder=3,
                        capstyle='round', joinstyle='round')
    ax.add_patch(a)
    a2 = FancyArrowPatch((TAILx(cx), ARR_Y), (cx + ARR_DX, ARR_Y),
                         arrowstyle='-|>,head_width=0.7,head_length=0.85',
                         mutation_scale=22, lw=3.0, color=col, zorder=5,
                         capstyle='round', joinstyle='round')
    ax.add_patch(a2)

# --- fan of M seed-gradient arrows ON TOP of the consensus arrow ---
M = 7
def seed_fan(cx, angles_deg, col, clip):
    """M seed arrows sharing one tail, splayed by the given angles (deg, 0 = +x = 'positive sign')."""
    tail = np.array([FANx(cx), ARR_Y])
    for ang in angles_deg:
        th = np.deg2rad(ang)
        # multiply the y-component by ASP so the fan looks angular (not squashed) on a 2:1 canvas
        tip = tail + np.array([SEED_LEN * np.cos(th), SEED_LEN * np.sin(th) * ASP])
        a = FancyArrowPatch(tuple(tail), tuple(tip),
                            arrowstyle='-|>,head_width=0.55,head_length=0.8',
                            mutation_scale=11, lw=2.2, color=col, zorder=7,
                            capstyle='round', joinstyle='round', alpha=0.95)
        a.set_clip_path(clip)            # safety: never spill outside the panel
        ax.add_patch(a)
    # mark the shared tail (the design point)
    ax.add_patch(Ellipse((tail[0], tail[1]), width=0.012, height=0.012 * ASP,
                 facecolor=col, edgecolor='white', lw=1.0, zorder=8))

# draw consensus arrows first (soft band), then the seed fans on top
big_arrow(LCx, GRN)
big_arrow(RCx, RED)

# LEFT: all M seeds aligned -> tight bundle, all near 0 deg (same sign), no flips
left_ang = np.array([5.0, -4.0, 9.0, -8.0, 2.0, 12.0, -11.0])
seed_fan(LCx, left_ang, GRN, left_card)

# RIGHT: seeds disagree -> wide fan with 3 of 7 FLIPPED to point backwards (sign disagreement)
right_ang = np.array([9.0, -14.0, 150.0, 22.0, -158.0, -26.0, 142.0])
seed_fan(RCx, right_ang, RED, right_card)

# small explanatory tags under each fan
ax.text(LCx, ARR_Y - 0.118, r'$M$ re-trained seeds  $\rightarrow$  signs agree',
        ha='center', va='center', fontsize=10.0, color=GRN, style='italic')
ax.text(RCx, ARR_Y - 0.118, r'$M$ re-trained seeds  $\rightarrow$  some signs flip',
        ha='center', va='center', fontsize=10.0, color=RED, style='italic')

# --- big verdict badges (check / cross) inside coloured discs, top of each panel ---
def verdict(cx, cy, col, mark):
    # true circle via ellipse (width corrected for 2:1 figure)
    ax.add_patch(Ellipse((cx, cy), width=0.072 / ASP, height=0.072,
                 facecolor='white', edgecolor=col, lw=2.8, zorder=9))
    ax.text(cx, cy - 0.002, mark, ha='center', va='center', fontsize=29,
            color=col, zorder=10, fontweight='bold')

VY = 0.758
verdict(LCx, VY, GRN, r'$\checkmark$')
verdict(RCx, VY, RED, r'$\boldsymbol{\times}$')

# --- panel headline labels ---
ax.text(LCx, 0.690, 'TRUST', ha='center', va='center', fontsize=22,
        fontweight='bold', color=GRN, zorder=9)
ax.text(LCx, 0.646, 'use the free autodiff gradient', ha='center', va='center',
        fontsize=11.5, color=INK, zorder=9)
ax.text(LCx, 0.612, r'$\mathbf{0}$  full-wave solves', ha='center', va='center',
        fontsize=12.0, color=GRN, zorder=9, fontweight='bold')

ax.text(RCx, 0.690, 'VERIFY', ha='center', va='center', fontsize=22,
        fontweight='bold', color=RED, zorder=9)
ax.text(RCx, 0.646, 'the sign is unstable', ha='center', va='center',
        fontsize=11.5, color=INK, zorder=9)
ax.text(RCx, 0.612, r'$\mathbf{1}$  full-wave solve', ha='center', va='center',
        fontsize=12.0, color=RED, zorder=9, fontweight='bold')

# --- "SAME MAGNITUDE" dimension lines under each hero arrow (proof of identical length) ---
def measure(cx):
    y = ARR_Y - 0.060
    ax.annotate('', xy=(cx - ARR_DX, y), xytext=(cx + ARR_DX, y),
                arrowprops=dict(arrowstyle='|-|,widthA=0.4,widthB=0.4', color=MUT, lw=1.1),
                zorder=5)
measure(LCx); measure(RCx)

# central "=" on the equator -- the proof that the two design gradients have identical size
ax.text(0.5, ARR_Y, r'$=$', ha='center', va='center', fontsize=34, color=INK,
        fontweight='bold', zorder=7,
        bbox=dict(boxstyle='circle,pad=0.18', facecolor='white', edgecolor=tint(INK, 0.25), lw=1.0))

# ----------------------------------------------------------------------------------------------
#  punch line strip between the panels and the payoff bar
# ----------------------------------------------------------------------------------------------
ax.text(0.5, 0.298,
        'same size  —  opposite verdict',
        ha='center', va='center', fontsize=16.0, fontweight='bold', color=INK)
ax.text(0.5, 0.256,
        r'trust the $\mathbf{sign\ agreement\ across\ seeds}$, not the magnitude',
        ha='center', va='center', fontsize=12.0, color=INK)

# ----------------------------------------------------------------------------------------------
#  hero payoff strip (bottom)
# ----------------------------------------------------------------------------------------------
BY0, BY1 = 0.040, 0.190
ax.add_patch(FancyBboxPatch((0.060, BY0), 0.880, BY1 - BY0,
             boxstyle='round,pad=0.004,rounding_size=0.020',
             facecolor=INK, edgecolor='none', zorder=2))

byc = 0.5 * (BY0 + BY1)
# left: the headline number
ax.text(0.150, byc + 0.022, 'AUC 0.91', ha='center', va='center',
        fontsize=27, fontweight='bold', color='white', zorder=4)
ax.text(0.150, byc - 0.030, 'predicts gradient', ha='center', va='center',
        fontsize=9.3, color=tint('white', 0.92), zorder=4)
ax.text(0.150, byc - 0.052, 'sign-correctness', ha='center', va='center',
        fontsize=9.3, color=tint('white', 0.92), zorder=4)

# vertical separators
for xs in (0.262, 0.560):
    ax.plot([xs, xs], [BY0 + 0.022, BY1 - 0.022], color=tint('white', 0.28), lw=1.1, zorder=4)

# middle: what it trusts / rejects (the concrete physics payoff)
ax.text(0.412, byc + 0.028, 'trusts the impedance gradient', ha='center', va='center',
        fontsize=11.0, color='white', zorder=4)
ax.text(0.412, byc - 0.004, 'rejects the seed-unstable', ha='center', va='center',
        fontsize=11.0, color='white', zorder=4)
ax.text(0.412, byc - 0.030, 'radiation gradient', ha='center', va='center',
        fontsize=11.0, color='white', zorder=4)
# small green / red dots flanking this column
ax.add_patch(Ellipse((0.300, byc + 0.028), width=0.011 / ASP, height=0.011,
             facecolor=GRN, edgecolor='none', zorder=5))
ax.add_patch(Ellipse((0.300, byc - 0.017), width=0.011 / ASP, height=0.011,
             facecolor=RED, edgecolor='none', zorder=5))

# right: solver-free + external validation note
ax.text(0.748, byc + 0.030, 'SOLVER-FREE', ha='center', va='center',
        fontsize=15.5, fontweight='bold', color='white', zorder=4)
ax.text(0.748, byc - 0.018, 'the gate itself needs no full-wave solver', ha='center', va='center',
        fontsize=9.6, color=tint('white', 0.9), zorder=4)
ax.text(0.748, byc - 0.044, 'validated on an external transfer-matrix benchmark',
        ha='center', va='center', fontsize=8.6, color=tint('white', 0.72),
        style='italic', zorder=4)

# ----------------------------------------------------------------------------------------------
out_pdf = os.path.join(DIR, 'fig_graphical_p2_A.pdf')
out_png = os.path.join(DIR, 'fig_graphical_p2_A.png')
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=150)
plt.close(fig)
try:
    from PIL import Image
    w, h = Image.open(out_png).size
    print(f'fig_graphical_p2_A (same-magnitude-opposite-trust): PNG {w}x{h} px -> {DIR}')
except Exception:
    print('fig_graphical_p2_A: written')
