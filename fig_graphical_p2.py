# -*- coding: utf-8 -*-
# Elsevier graphical abstract for paper 2 -- "reliability map" (polished with plot-from-data
# scatter style: slate ink, soft bands, light dotted grid, white-edged markers, boxed annotation).
# Hero: real per-component data (reliability.npz, same as fig:refute):
#   x = gradient magnitude (log)  vs  y = cross-seed sign-agreement.
# At the SAME large magnitude there are both trusted (green) and flagged (red) components ->
# DIRECTION (vertical) decides trust, not MAGNITUDE (horizontal). Honest numbers only.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.transforms import offset_copy

DIR = os.path.dirname(os.path.abspath(__file__))
INK = '#2c3e50'    # slate ink (all text)        — the palette upgrade
GRN = '#1e7a45'    # trust (signs agree)
RED = '#d1495b'    # verify (signs disagree) — softer rose-red than the harsh brick
MUT = '#8a97a3'    # muted slate-grey for secondary marks
GRID = '#e3e6e9'

def tint(c, a):
    r = list(mcolors.to_rgba(c)); r[3] = a; return tuple(r)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK, 'axes.linewidth': 1.0,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})

R = np.load(os.path.join(DIR, 'reliability.npz'))
mag = R['magnitude']; sa = R['sign_agree']; thr = float(R['thr'])
resp, null, rad = slice(0, 6), slice(6, 12), slice(12, 18)

fig = plt.figure(figsize=(12.0, 6.0))

# ---------------- title block ----------------
fig.text(0.5, 0.945, 'Direction, not magnitude', ha='center', va='center',
         fontsize=26, fontweight='bold', color=INK)
fig.text(0.5, 0.872, 'which design gradients of a differentiable surrogate to trust — '
         'set by cross-seed sign agreement, not gradient size',
         ha='center', va='center', fontsize=12.5, color=MUT, style='italic')

# ---------------- hero: reliability map ----------------
ax = fig.add_axes([0.070, 0.150, 0.555, 0.600])
ax.set_xscale('log'); ax.set_xlim(5e-3, 45); ax.set_ylim(0.45, 1.07)
ax.grid(True, color=GRID, lw=0.7, ls=':', zorder=0); ax.set_axisbelow(True)

# soft trust / verify bands + threshold
ax.axhspan(thr, 1.07, color=tint(GRN, 0.10), zorder=0.5)
ax.axhspan(0.45, thr, color=tint(RED, 0.06), zorder=0.5)
ax.axhline(thr, color=INK, ls=(0, (5, 4)), lw=1.3, zorder=4)
_tr = offset_copy(ax.get_yaxis_transform(), fig=fig, x=-2, y=3, units='points')
ax.text(1.0, thr, r'trust threshold $\tau=0.9$', transform=_tr, ha='right', va='bottom',
        fontsize=9.0, color=INK, style='italic')
ax.text(6.0e-3, 1.022, 'TRUST', ha='left', va='center', fontsize=12, color=GRN, fontweight='bold')
ax.text(6.0e-3, 0.493, 'VERIFY', ha='left', va='center', fontsize=12, color=RED, fontweight='bold')

# real per-component points (white-edged, prominent)
ax.scatter(mag[resp], sa[resp], s=96, marker='o', facecolor=GRN, edgecolor='white',
           linewidths=1.3, zorder=6, label='reliable — signs agree')
ax.scatter(mag[null], sa[null], s=70, marker='X', facecolor=tint(RED, 0.55), edgecolor='white',
           linewidths=1.0, zorder=6)
ax.scatter(mag[rad], sa[rad], s=104, marker='X', facecolor=RED, edgecolor='white',
           linewidths=1.2, zorder=7, label='unreliable — signs disagree')

# vertical "same magnitude, opposite verdict" connector at the large-magnitude side
xm = float(np.median(mag[rad]))
ax.annotate('', xy=(xm, float(max(sa[resp])) - 0.012), xytext=(xm, float(np.median(sa[rad])) + 0.02),
            arrowprops=dict(arrowstyle='<->', color=MUT, lw=1.2), zorder=5)
ax.text(xm * 1.22, 0.815, 'same magnitude,\nopposite verdict', ha='left', va='center',
        fontsize=8.8, color=MUT)

# boxed annotation for the trap (large yet sign-unstable); a short straight arrow to the cluster
cx = float(np.median(mag[rad])); cy_lo = float(sa[rad].min())
ax.annotate('large gradients, yet sign-unstable —\nthe trap a magnitude rule falls for',
            xy=(cx, cy_lo + 0.01), xytext=(0.42, 0.555), textcoords='data',
            ha='center', va='center', fontsize=9.0, color=INK, zorder=9,
            bbox=dict(boxstyle='round,pad=0.34', facecolor=tint(RED, 0.12), edgecolor=INK, lw=0.9),
            arrowprops=dict(arrowstyle='->', color=INK, lw=1.1, shrinkB=6))

ax.set_xlabel(r'gradient magnitude  $|\partial \mathbf{r}/\partial g_k|$  (log scale)', fontsize=12.5)
ax.set_ylabel('cross-seed sign-agreement', fontsize=12.5)
ax.tick_params(direction='in', length=4, width=0.9, labelsize=10.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.015, 0.965), frameon=True, facecolor='white',
          edgecolor='#cccccc', framealpha=0.96, fontsize=9.0, handletextpad=0.4,
          borderpad=0.6, labelspacing=0.45)

# ---------------- right outcome panel ----------------
rp = fig.add_axes([0.660, 0.150, 0.320, 0.600]); rp.set_xlim(0, 1); rp.set_ylim(0, 1); rp.axis('off')

def pill(y, dotcol, head, sub, color):
    rp.add_patch(FancyBboxPatch((0.04, y - 0.082), 0.92, 0.164,
                 boxstyle='round,pad=0.018,rounding_size=0.045',
                 facecolor=tint(color, 0.10), edgecolor=color, lw=1.6, zorder=2,
                 transform=rp.transAxes))
    rp.plot([0.135], [y], 'o', ms=13, color=color, transform=rp.transAxes, zorder=3, clip_on=False)
    rp.text(0.235, y + 0.030, head, ha='left', va='center', fontsize=12.5, color=color,
            fontweight='bold', zorder=3)
    rp.text(0.235, y - 0.035, sub, ha='left', va='center', fontsize=9.6, color=INK, zorder=3)

pill(0.865, GRN, 'signs agree → TRUST', 'use the free autodiff gradient (0 solves)', GRN)
pill(0.640, RED, 'signs disagree → VERIFY', 'spend one full-wave solve', RED)

# headline result card
rp.add_patch(FancyBboxPatch((0.04, 0.070), 0.92, 0.395,
             boxstyle='round,pad=0.018,rounding_size=0.045',
             facecolor='#f5f7f9', edgecolor='#d6dce1', lw=1.1, zorder=1, transform=rp.transAxes))
rp.text(0.50, 0.395, 'solver-free reliability gate', ha='center', va='center', fontsize=11.5, color=INK)
rp.text(0.50, 0.285, 'AUC 0.91', ha='center', va='center', fontsize=30, fontweight='bold', color=GRN)
rp.text(0.50, 0.180, 'predicts gradient sign-correctness', ha='center', va='center', fontsize=10.2, color=INK)
rp.text(0.50, 0.122, 'on an external transfer-matrix benchmark', ha='center', va='center',
        fontsize=9.0, color=MUT, style='italic')

out_pdf = os.path.join(DIR, 'fig_graphical_p2.pdf')
out_png = os.path.join(DIR, 'fig_graphical_p2.png')
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=150)
plt.close(fig)
try:
    from PIL import Image
    w, h = Image.open(out_png).size
    print(f'fig_graphical_p2 (polished reliability-map): PNG {w}x{h} px; {len(mag)} pts; -> {DIR}')
except Exception:
    print('fig_graphical_p2 (polished reliability-map): written')
