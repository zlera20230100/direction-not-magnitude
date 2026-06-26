# -*- coding: utf-8 -*-
# fig_reliability (3 panels, 2-row layout to cut width) -- ensemble sign-agreement vs MC-dropout.
# (a) [top, full width] sign-agreement by group: DISTINCT per-group colours, colour-matched mean bars
#     (black-edged, labelled "mean"); MC-dropout points overlaid on the real-radiation group, annotated.
# (b) [bottom-left] reliability-classification AUC with bootstrap 95% CI; value labels sit above the
#     CI cap so they never touch the error bar.
# (c) [bottom-right] MC-dropout sign-agreement across dropout rates vs the deep ensemble.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'; ORG = '#e67e22'
d = np.load(os.path.join(DIR, 'reliability.npz'))
mc = np.load(os.path.join(DIR, 'mcdropout.npz'))
ci = np.load(os.path.join(DIR, 'reliability_ci.npz'))
thr = float(d['thr']); rng = np.random.default_rng(1)

fig = plt.figure(figsize=(9.6, 7.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.48, wspace=0.27)
a = fig.add_subplot(gs[0, :]); b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])

# ---- (a) sign-agreement by group: distinct colours + colour-matched, black-edged, labelled means ----
groups = [('responsive\n(reliable)', d['sa_resp'], GRN),
          ('null\n(unreliable)', d['sa_null'], SIG),
          ('real radiation\n(unreliable)', d['sa_rad'], ACC)]
for i, (lab, vals, col) in enumerate(groups):
    a.scatter(i + 0.09 * rng.standard_normal(len(vals)), vals, s=34, c=col, edgecolors='k',
              linewidths=0.4, zorder=3, alpha=0.85)
    # colour-matched group mean as a thick bar with a thin black outline so it reads against same-colour dots
    a.plot([i - 0.22, i + 0.22], [vals.mean(), vals.mean()], color=col, lw=3.2, solid_capstyle='round',
           zorder=6, path_effects=[pe.Stroke(linewidth=5.0, foreground='k'), pe.Normal()])
a.axhline(thr, color='0.45', ls='--', lw=1.0, zorder=1)
a.text(2.54, thr + 0.006, f'trust threshold {thr:.1f}', ha='right', va='bottom', fontsize=7.8, color='0.35')
xm = 2 + 0.09 * rng.standard_normal(len(mc['sa_mc']))
a.scatter(xm, mc['sa_mc'], s=44, marker='^', c=ORG, edgecolors='k', linewidths=0.4, zorder=5)
a.annotate('MC-dropout\n(1 model):\nfalse trust', xy=(1.93, 1.0), xytext=(1.5, 0.61),
           fontsize=8.0, color=ORG, va='center', ha='center', fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=ORG, lw=1.2, connectionstyle='arc3,rad=0.12'))
# state the group-mean convention ONCE (thick colour-matched, black-edged bars), not per-group
mean_proxy = Line2D([0], [0], color='0.6', lw=3.2, solid_capstyle='round',
                    path_effects=[pe.Stroke(linewidth=5.0, foreground='k'), pe.Normal()])
a.legend([mean_proxy], ['group mean'], loc='lower left', frameon=False, fontsize=8.4, handlelength=1.7)
a.set_xticks(range(3)); a.set_xticklabels([g[0] for g in groups], fontsize=8.8)
a.set_xlim(-0.5, 2.7); a.set_ylim(0.45, 1.06)
a.set_ylabel('ensemble sign-agreement (10 seeds)')
a.set_title('(a) cheap ensemble indicator separates reliable / unreliable', loc='left', fontsize=9.6, fontweight='bold')

# ---- (b) reliability-classification AUC with bootstrap CI; labels above the CI cap ----
names = ['ensemble\nsign-agree', 'ensemble\nSNR', 'naive\n|gradient|']
keys = ['sign-agreement', 'SNR', 'naive_gradient']
pts = [float(ci[k][0]) for k in keys]; los = [float(ci[k][1]) for k in keys]; his = [float(ci[k][2]) for k in keys]
cols = [GRN, GRN, SIG]
b.bar(range(3), pts, color=cols, edgecolor='k', lw=0.6, width=0.62, zorder=3)
b.errorbar(range(3), pts, yerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)],
           fmt='none', ecolor='k', elinewidth=1.0, capsize=4, zorder=5)
for i, v in enumerate(pts):
    b.text(i, his[i] + 0.03, f'{v:.3f}', ha='center', va='bottom', fontsize=8.4)   # above CI cap, clear of error bar
b.axhline(0.5, color=ACC, ls=':', lw=1.1, zorder=2)
b.text(2.46, 0.52, 'chance', ha='right', va='bottom', fontsize=7.8, color=ACC)
b.set_xticks(range(3)); b.set_xticklabels(names, fontsize=8.4)
b.set_ylabel('reliability AUC  (95% CI, $n{=}18$)'); b.set_ylim(0.0, 1.2)
b.set_title('(b) predicts the full-wave verdict, no solver', loc='left', fontsize=9.6, fontweight='bold')

# ---- (c) MC-dropout sign-agreement across dropout rates ----
ps = [0.05, 0.10, 0.20, 0.50]; sa_sweep = []
for p in ps:
    dd = np.load(os.path.join(DIR, f'mcdropout_p{p:.2f}.npz')); sa_sweep.append(float(dd['sa_mc'].mean()))
c.plot(ps, sa_sweep, 'o-', color=ORG, lw=1.9, ms=7, label='MC-dropout (1 model)')
c.axhline(float(d['sa_rad'].mean()), color=GRN, ls='-', lw=1.8, label='deep ensemble (10 seeds)')
c.axhline(thr, color='0.45', ls='--', lw=1.0); c.text(0.5, thr - 0.05, 'trust threshold', ha='right', fontsize=7.6, color='0.35')
c.set_xscale('log'); c.set_xticks(ps); c.set_xticklabels([f'{p:g}' for p in ps])
c.set_xlabel('MC-dropout rate $p$'); c.set_ylabel('radiation-gradient sign-agreement'); c.set_ylim(0.45, 1.05)
c.legend(loc='center left', frameon=False, fontsize=8.0)
c.set_title('(c) dropout is over-confident at every rate', loc='left', fontsize=9.6, fontweight='bold')

fig.savefig(os.path.join(DIR, 'fig_reliability.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(DIR, 'fig_reliability.pdf'), bbox_inches='tight')
print('saved fig_reliability (2-row); group means =', [round(float(g[1].mean()), 3) for g in groups],
      '; AUC =', [round(p, 3) for p in pts], '; sweep =', [round(s, 2) for s in sa_sweep])
