# -*- coding: utf-8 -*-
# fig_extbench (2 panels) for the thin-film transfer-matrix (TMM) benchmark.
# (a) ROC of the sign-agreement gate predicting whether each surrogate gradient component
#     is sign-correct against the TMM gradient, pooled over query points.
# (b) sign-agreement separated by sign-correct vs sign-wrong components, with the trust threshold.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'
d = np.load(os.path.join(DIR, 'extbench_tmm.npz'))
sa = d['all_sa']; corr = d['all_correct']; auc = float(d['auc']); tau = float(d['tau'])
ntr = float(d['n_trust_mean']); K = int(d['K']); cos_ad = float(d['cos_ad'])

fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={'width_ratios': [1.0, 1.05]})

# (a) ROC
fpr, tpr, _ = roc_curve(corr, sa)
a.plot(fpr, tpr, '-', color=SIG, lw=2.0, zorder=3, label=f'sign-agreement gate (AUC {auc:.2f})')
a.plot([0, 1], [0, 1], '--', color='0.6', lw=1.0, zorder=1, label='chance')
a.set_xlabel('false positive rate'); a.set_ylabel('true positive rate')
a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
a.legend(loc='lower right', frameon=False, fontsize=8.2)
a.set_title('(a) gate predicts gradient sign-reliability on an\nexternal photonics (TMM) benchmark',
            loc='left', fontsize=9.0, fontweight='bold')

# (b) separation of sign-agreement by outcome
rng = np.random.default_rng(0)
grp = [('sign-correct', sa[corr == 1], GRN), ('sign-wrong', sa[corr == 0], ACC)]
for i, (lab, vals, col) in enumerate(grp):
    if len(vals):
        b.scatter(i + 0.09 * rng.standard_normal(len(vals)), vals, s=20, c=col, edgecolors='k',
                  linewidths=0.25, alpha=0.55, zorder=3)
        b.scatter([i], [vals.mean()], marker='_', s=900, c='k', zorder=4)
b.axhline(tau, color='0.4', ls='--', lw=1.1, zorder=2)
b.text(1.45, tau + 0.006, f'trust threshold {tau}', ha='right', va='bottom', fontsize=7.8, color='0.3')
b.set_xticks([0, 1]); b.set_xticklabels([g[0] for g in grp]); b.set_xlim(-0.5, 1.5)
b.set_ylabel('ensemble sign-agreement (10 seeds)'); b.set_ylim(0.45, 1.03)
b.text(0.02, 0.40,
       f"at $\\tau={tau}$: {ntr:.1f}/{K} components certified trustworthy;\n"
       f"hybrid follows the free gradient, preserving the\nTMM descent direction (cosine {cos_ad:.3f}) at "
       f"$<\\!1$ solve/grad",
       transform=b.transAxes, fontsize=7.8, va='top', ha='left', color='0.2',
       bbox=dict(boxstyle='round,pad=0.4', fc='#eef3ee', ec='0.7', lw=0.7))
b.set_title('(b) the gate separates reliable from unreliable\ngradient components', loc='left',
            fontsize=9.0, fontweight='bold')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(DIR, f'fig_extbench.{ext}'), dpi=320, bbox_inches='tight')
print('wrote fig_extbench.pdf/.png ; AUC', round(auc, 3), 'trusted', round(ntr, 1), '/', K)
