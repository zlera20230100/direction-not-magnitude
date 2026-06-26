# -*- coding: utf-8 -*-
# fig_selfaware (2 panels) for the correlated-seed-aware ("self-aware") reliability gate.
# (a) FALSE-TRUST RATE  P(sign-wrong | certified)  vs shared-bias fraction rho:
#     the plain gate climbs (the "rho-wall"), the global-abstain variant is no better (fails),
#     the per-component spread detector stays low -- it restores safety where the plain gate is
#     silently defeated. The rho=0.5-0.7 exploit band is shaded.
# (b) the global ICC meter rho-hat tracks the true rho (Pearson 0.998), so the regime IS detectable;
#     it carries a positive offset (shared true signal), so it is a relative meter, not a calibrated rho.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'; ORG = '#e67e22'
d = np.load(os.path.join(DIR, 'selfaware_gate.npz'))
rho = d['rho']
ft_plain = d['ft_plain']; ft_global = d['ft_global']; ft_percomp = d['ft_percomp']
pearson = float(d['pearson'])
rhh = d['rhohat_mean']; rhh_sd = d['rhohat_sd']
slope = float(d['fit_slope']); icpt = float(d['fit_intercept'])
# headline reductions in the exploit band (per-comp vs plain)
band = (rho >= 0.5) & (rho <= 0.7)
red = (1.0 - ft_percomp[band] / ft_plain[band]) * 100.0
red_med = float(np.mean(red))

fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={'width_ratios': [1.18, 1.0]})

# ---- (a) false-trust rate vs rho ------------------------------------------------------------
a.axvspan(0.5, 0.7, color=ORG, alpha=0.10, zorder=0)
a.text(0.6, a.get_ylim()[1], 'exploit\nband', ha='center', va='top', color='#9a5a12',
       fontsize=7.8, transform=a.get_xaxis_transform())
a.plot(rho, ft_plain, '-o', color=ACC, ms=5, lw=1.8, zorder=4,
       label='plain gate (the $\\rho$-wall)')
a.plot(rho, ft_global, '-^', color=ORG, ms=5.5, lw=1.6, zorder=3,
       label='global abstain variant (fails)')
a.plot(rho, ft_percomp, '-o', color=GRN, ms=5, lw=2.0, zorder=5,
       label='per-component self-aware gate')
# annotate the worst-case rho-wall point and the restored point
# value labels for ALL three endpoints, parked just RIGHT of rho=0.8 (colour-matched, no overlap)
a.annotate(f'{ft_global[-1]:.3f}', xy=(rho[-1], ft_global[-1]), xytext=(rho[-1]+0.012, ft_global[-1]),
           ha='left', va='center', fontsize=8.0, color=ORG)
a.annotate(f'{ft_plain[-1]:.3f}', xy=(rho[-1], ft_plain[-1]), xytext=(rho[-1]+0.012, ft_plain[-1]),
           ha='left', va='center', fontsize=8.0, color=ACC)
a.annotate(f'{ft_percomp[-1]:.3f}', xy=(rho[-1], ft_percomp[-1]), xytext=(rho[-1]+0.012, ft_percomp[-1]),
           ha='left', va='center', fontsize=8.0, color=GRN)
a.set_xlabel('shared-bias fraction $\\rho$  (seed-correlation / ICC)')
a.set_ylabel('false-trust rate  $P(\\mathrm{sign\\,wrong}\\mid\\mathrm{certified})$')
a.set_xlim(-0.02, 0.88); a.set_ylim(-0.004, 0.118)
a.legend(loc='upper left', frameon=False, fontsize=8.0)
a.text(0.04, 0.62,
       f'per-component cuts false trust\n$\\sim${red_med:.0f}% across $\\rho=0.5$-$0.7$',
       transform=a.transAxes, fontsize=8.2, va='top', ha='left', color=GRN,
       bbox=dict(boxstyle='round,pad=0.35', fc='#eef6f0', ec=GRN, lw=0.7))
a.set_title('(a) the per-component detector restores safety past the $\\rho$-wall',
            loc='left', fontsize=9.2, fontweight='bold')

# ---- (b) rho-hat tracks rho -----------------------------------------------------------------
b.plot([0, 0.85], [0, 0.85], '--', color='0.6', lw=1.0, zorder=1)
b.text(0.74, 0.80, 'identity', rotation=42, rotation_mode='anchor', color='0.55',
       fontsize=8, va='bottom', ha='center')
b.fill_between(rho, rhh - rhh_sd, rhh + rhh_sd, color=SIG, alpha=0.13, zorder=2,
               label='$\\hat{\\rho}$ across-trial $\\pm$1 s.d.')
b.plot(rho, rhh, '-o', color=SIG, ms=5, lw=2.0, zorder=4, label='$\\hat{\\rho}$ (global ICC meter)')
b.set_xlabel('true shared-bias fraction $\\rho$')
b.set_ylabel('estimated $\\hat{\\rho}$  (one-way ICC of the ensemble)')
b.set_xlim(-0.02, 0.82); b.set_ylim(0.0, 1.0)
b.legend(loc='lower right', frameon=False, fontsize=8.0)
b.text(0.04, 0.95,
       f'Pearson $r={pearson:.3f}$, Spearman $=1.000$\n'
       f'(monotone $\\Rightarrow$ regime detectable);\n'
       f'offset $\\approx{icpt:.2f}$: a relative meter,\nnot a calibrated $\\rho$',
       transform=b.transAxes, fontsize=8.0, va='top', ha='left', color='0.2',
       bbox=dict(boxstyle='round,pad=0.4', fc='#eef2f7', ec='0.7', lw=0.7))
b.set_title('(b) the regime is detectable, but only as a relative meter',
            loc='left', fontsize=9.2, fontweight='bold')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(DIR, f'fig_selfaware.{ext}'), dpi=320, bbox_inches='tight')
print(f'wrote fig_selfaware.pdf/.png ; plain FT 0->0.8 = {ft_plain[0]:.4f}->{ft_plain[-1]:.4f}, '
      f'percomp = {ft_percomp[0]:.4f}->{ft_percomp[-1]:.4f}, band-reduction ~{red_med:.0f}%, Pearson {pearson:.3f}')
