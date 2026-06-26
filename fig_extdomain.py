# -*- coding: utf-8 -*-
# fig_extdomain (2 panels) -- the SECOND external benchmark: a 1-D heat/diffusion BVP, a structurally
# different (elliptic, non-resonant) physics family. Tests whether the sign-agreement reliability
# ranking generalises beyond the resonant cases (EM-PINN + thin-film TMM).
# (a) ROC of the gate on the diffusion bench (AUC 0.761 [0.720, 0.802]) overlaid on the resonance-domain
#     TMM ROC (AUC 0.906): the ranking generalises but is WEAKER in diffusion.
# (b) per-domain AUC with bootstrap 95% CI, plus the diffusion class balance (171 wrong / 1429 correct).
# Conclusion: the reliability ranking generalises to a diffusion physics, though weaker; the
# solver-saving payoff is physics-dependent (negligible here, where sign-fragile components are low-leverage).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'; ORG = '#e67e22'

dp = np.load(os.path.join(DIR, 'extbench_poisson.npz'))   # diffusion (heat) -- the new domain
dt = np.load(os.path.join(DIR, 'extbench_tmm.npz'))        # resonance (thin-film TMM) -- existing

sa_p = dp['all_sa']; cor_p = dp['all_correct'].astype(int)
sa_t = dt['all_sa']; cor_t = dt['all_correct'].astype(int)
auc_p = float(dp['auc']); auc_t = float(dt['auc']); tau = float(dp['tau'])
nw = int(dp['n_sign_wrong']); nc = int(dp['n_sign_correct'])

# Display the query-point-CLUSTERED 95% CIs reported throughout the manuscript: the K gate components
# within one query point share the ensemble draw, so an independence-assuming component bootstrap would
# understate the interval. These are the reported values (TMM via auc_ci_device_external.py; heat via the
# same query-point-clustered recipe), so the figure matches the tables/abstract/captions exactly.
lo_t, hi_t = 0.843, 0.963   # resonance (TMM), clustered over the 80 query points
lo_p, hi_p = 0.728, 0.793   # diffusion (heat), clustered over query points

fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={'width_ratios': [1.05, 0.95]})

# ---- (a) ROC: diffusion vs resonance --------------------------------------------------------
fpr_t, tpr_t, _ = roc_curve(cor_t, sa_t)
fpr_p, tpr_p, thr_p = roc_curve(cor_p, sa_p)
a.plot([0, 1], [0, 1], '--', color='0.65', lw=1.0, zorder=2)
a.text(0.78, 0.70, 'chance', rotation=45, rotation_mode='anchor', color='0.55', fontsize=8,
       va='bottom', ha='center')
a.plot(fpr_t, tpr_t, '-', color=SIG, lw=2.3, zorder=3, solid_capstyle='round',
       label=f'resonance (TMM): AUC {auc_t:.3f}')
a.fill_between(fpr_p, tpr_p, color=GRN, alpha=0.10, zorder=1)
a.plot(fpr_p, tpr_p, '-', color=GRN, lw=2.3, zorder=4, solid_capstyle='round',
       label=f'diffusion (heat): AUC {auc_p:.3f}')
# operating point on the diffusion ROC at the trust threshold tau
j = int(np.argmin(np.abs(thr_p - tau)))
a.plot(fpr_p[j], tpr_p[j], 'o', ms=7.5, mfc='white', mec=GRN, mew=1.8, zorder=5)
a.annotate(rf'$\tau={tau}$', xy=(fpr_p[j], tpr_p[j]), xytext=(fpr_p[j] + 0.17, tpr_p[j] - 0.16),
           fontsize=8.5, color=GRN, va='center', arrowprops=dict(arrowstyle='-', color=GRN, lw=0.8))
a.set_xlabel('false positive rate'); a.set_ylabel('true positive rate')
a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
a.legend(loc='lower right', frameon=False, fontsize=8.2)
a.set_title('(a) the ranking generalises to diffusion, weaker than resonance',
            loc='left', fontsize=9.0, fontweight='bold')

# ---- (b) per-domain AUC with bootstrap CI ---------------------------------------------------
labels = ['diffusion\n(heat)', 'resonance\n(TMM)']
aucs = [auc_p, auc_t]; los = [lo_p, lo_t]; his = [hi_p, hi_t]; cols = [GRN, SIG]
xb = np.arange(2)
b.bar(xb, aucs, color=cols, edgecolor='k', lw=0.6, width=0.58, zorder=3)
b.errorbar(xb, aucs, yerr=[np.array(aucs) - np.array(los), np.array(his) - np.array(aucs)],
           fmt='none', ecolor='k', elinewidth=1.1, capsize=5, zorder=5)
for i, (v, lo, hi) in enumerate(zip(aucs, los, his)):
    b.text(i, hi + 0.02, f'{v:.3f}\n[{lo:.3f}, {hi:.3f}]', ha='center', va='bottom', fontsize=8.2)
b.axhline(0.5, color=ACC, ls=':', lw=1.1, zorder=2)
b.text(1.46, 0.515, 'chance', ha='right', va='bottom', fontsize=7.8, color=ACC)
b.set_xticks(xb); b.set_xticklabels(labels, fontsize=8.6)
b.set_ylabel('reliability AUC  (sign-agreement $\\to$ sign-correct)')
b.set_ylim(0.0, 1.15); b.set_xlim(-0.6, 1.6)
b.set_title('(b) above chance in both, but diffusion is weaker',
            loc='left', fontsize=9.0, fontweight='bold')

# class-balance / scope note placed in the clear margin BELOW the axes (not over the bars)
fig.text(0.74, -0.02,
         f'diffusion class balance: {nw} sign-wrong / {nc} correct;  '
         f'both CIs exclude chance.\nThe ranking generalises in direction, not strength; '
         f'the solver-saving payoff is physics-dependent.',
         fontsize=7.6, va='top', ha='center', color='0.3')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(DIR, f'fig_extdomain.{ext}'), dpi=320, bbox_inches='tight')
print(f'wrote fig_extdomain.pdf/.png ; diffusion AUC {auc_p:.3f} [{lo_p:.3f}, {hi_p:.3f}] '
      f'({nw} wrong / {nc} correct) vs resonance TMM AUC {auc_t:.3f} [{lo_t:.3f}, {hi_t:.3f}]')
