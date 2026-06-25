# -*- coding: utf-8 -*-
# fig_gate_vs_mag: the gate orders FD-verification by epistemic disagreement, NOT by gradient
# magnitude. (a) fraction of the dangerous SIGN-WRONG components caught vs #FD checks -- the gate
# catches them almost immediately while the obvious "verify largest-|gradient| first" heuristic
# misses them entirely until late; (b) descent-direction (cosine-to-truth) recovery vs #FD. Source:
# extbench_tmm_baseline.npz (TMM photonics benchmark).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.6,
    'axes.linewidth': 0.9, 'font.size': 10.5})
DIR = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(DIR, 'extbench_tmm_baseline.npz'))
GRN = '#1e7a45'; ACC = '#c0392b'; GREY = '#7f8c8d'
K = int(d['K']); x = np.arange(K + 1)
bg, bm, brd = int(d['budget80_gate']), int(d['budget80_mag']), int(d['budget80_rand'])

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))

# (a) fraction of sign-wrong components caught vs #FD
a = ax[0]
a.plot(x, d['caught_gate'], 'o-', color=GRN, lw=2.0, ms=5, label='gate (least sign-agreement first)')
a.plot(x, d['caught_mag'], 's--', color=ACC, lw=1.8, ms=4.5, label=r'baseline: largest $|\nabla|$ first')
a.plot(x, d['caught_rand'], '^:', color=GREY, lw=1.6, ms=4.5, label='random order')
a.axhline(0.8, color='0.6', lw=0.8, ls=':')
a.annotate(f'gate: {bg} checks', xy=(bg, 0.8), xytext=(bg + 1.2, 0.55), fontsize=8.2, color=GRN,
           arrowprops=dict(arrowstyle='-|>', color=GRN, lw=1.0))
a.annotate(r'largest-$|\nabla|$: misses them' '\n' 'until the 8th check', xy=(8, d['caught_mag'][8]),
           xytext=(2.4, 0.92), fontsize=8.0, color=ACC,
           arrowprops=dict(arrowstyle='-|>', color=ACC, lw=1.0, connectionstyle='arc3,rad=0.2'))
a.set_xlabel('number of FD verifications'); a.set_ylabel('fraction of sign-wrong components caught')
a.set_ylim(-0.03, 1.03); a.set_xlim(0, K)
a.legend(frameon=True, framealpha=0.92, edgecolor='none', loc='lower right')
a.set_title('(a) the gate catches the dangerous sign-wrong gradients first',
            loc='left', fontsize=9.4, fontweight='bold')

# (b) descent-direction recovery (cosine to truth) vs #FD
b = ax[1]
b.plot(x, d['cos_gate'], 'o-', color=GRN, lw=2.0, ms=5, label='gate order')
b.plot(x, d['cos_mag'], 's--', color=ACC, lw=1.8, ms=4.5, label=r'largest $|\nabla|$ first')
b.plot(x, d['cos_rand'], '^:', color=GREY, lw=1.6, ms=4.5, label='random order')
b.set_xlabel('number of FD verifications'); b.set_ylabel('cosine to full-FD gradient')
b.set_xlim(0, K)
b.legend(frameon=True, framealpha=0.92, edgecolor='none', loc='lower right')
b.set_title('(b) and recovers the descent direction with fewer solves',
            loc='left', fontsize=9.4, fontweight='bold')

fig.suptitle('The gate ranks by epistemic disagreement, not gradient magnitude (TMM benchmark)',
             x=0.012, ha='left', fontsize=10.2, fontweight='bold', y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(DIR, 'fig_gate_vs_mag.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(DIR, 'fig_gate_vs_mag.pdf'), bbox_inches='tight')
print('saved fig_gate_vs_mag; budget80 gate=%d mag=%d rand=%d' % (bg, bm, brd))
