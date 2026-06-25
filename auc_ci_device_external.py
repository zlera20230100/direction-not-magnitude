# -*- coding: utf-8 -*-
# Bootstrap 95% CIs for the reliability AUCs from saved arrays:
#  (1) the n=18 labelled device set (reliability.npz), and
#  (2) the external thin-film TMM benchmark (extbench_tmm.npz: all_sa score vs all_correct label),
# plus the external-benchmark class balance. No retraining.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(7)
NB = 5000

def boot_ci(y, x, nb=NB):
    y = np.asarray(y).astype(int); x = np.asarray(x, dtype=float)
    pt = roc_auc_score(y, x)
    bs = []
    n = len(y)
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], x[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, lo, hi, n

print("=== (1) n=18 device-labelled set (reliability.npz) ===")
d = np.load('reliability.npz')
y = d['label']
print(f"  class balance: {int(y.sum())} reliable / {int((1-y).sum())} unreliable  (n={len(y)})")
for name, key in [('sign-agreement','sign_agree'), ('SNR','snr'), ('naive |gradient|','magnitude')]:
    pt, lo, hi, n = boot_ci(y, d[key])
    print(f"  {name:>16}: AUC={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\n=== (2) external thin-film TMM benchmark (extbench_tmm.npz) ===")
e = np.load('extbench_tmm.npz', allow_pickle=True)
sa = np.asarray(e['all_sa']).ravel().astype(float)
correct = np.asarray(e['all_correct']).ravel().astype(int)
K = int(e['K']); NQ = int(e['NQ'])               # K components per query point, NQ query points
npos, nneg = int(correct.sum()), int((1 - correct).sum())
print(f"  pooled components: {len(correct)} = {NQ} query points x {K} components")
print(f"  class balance: {npos} sign-correct / {nneg} sign-wrong  (only {nneg} negatives -> must cluster)")
pt = roc_auc_score(correct, sa)

# Component-level bootstrap (independence-assuming). The K components of one query point share the
# same ensemble draw, so this UNDERSTATES the CI; kept for reference only, NOT the value in the paper.
_, lo_c, hi_c, _ = boot_ci(correct, sa)
print(f"  component-level CI (independence-assuming, reference only): [{lo_c:.3f}, {hi_c:.3f}]")

# Clustered bootstrap by query point: resample the NQ query points as blocks (each point's K
# components stay together). This is the CI reported in the paper.
sa_q = sa.reshape(NQ, K); cor_q = correct.reshape(NQ, K)
bs = []
for _ in range(NB):
    qi = rng.integers(0, NQ, NQ)
    ys = cor_q[qi].ravel(); xs = sa_q[qi].ravel()
    if len(np.unique(ys)) < 2:
        continue
    bs.append(roc_auc_score(ys, xs))
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  CLUSTERED CI (by {NQ} query points, REPORTED): AUC={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

# Operating-point confusion matrix at tau=0.9 (positive = gate trusts, i.e. sign-agreement >= tau).
tau = 0.9
trust = sa >= tau
TP = int((trust & (correct == 1)).sum()); FP = int((trust & (correct == 0)).sum())
FN = int((~trust & (correct == 1)).sum()); TN = int((~trust & (correct == 0)).sum())
spec = TN / (TN + FP) if (TN + FP) else float('nan')
print(f"  operating point tau={tau:g}: TP={TP} FP={FP} FN={FN} TN={TN}  "
      f"specificity={TN}/{TN + FP}={spec:.2f}  (gate falsely trusts {FP}/{nneg} sign-wrong)")
print(f"  reported point AUC in npz: {float(e['auc']):.3f}")
