# -*- coding: utf-8 -*-
# Bootstrap 95% confidence intervals for the reliability-classification AUC: resample the
# n=18 labelled components with replacement and recompute AUC.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.metrics import roc_auc_score
d = np.load('reliability.npz')
y = d['label']; N = len(y)
inds = {'sign-agreement': d['sign_agree'], 'SNR': d['snr'], 'naive |gradient|': d['magnitude']}
rng = np.random.default_rng(7)
NB = 5000
print(f"n = {N} labelled components ({int(y.sum())} reliable / {int((1-y).sum())} unreliable)")
out = {}
for name, x in inds.items():
    point = roc_auc_score(y, x)
    boot = []
    for _ in range(NB):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:   # need both classes
            continue
        boot.append(roc_auc_score(y[idx], x[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out[name] = (point, lo, hi)
    print(f"  {name:>16}: AUC = {point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
np.savez('reliability_ci.npz', **{k.replace(' ', '_').replace('|', ''): np.array(v) for k, v in out.items()}, n=N)
print("saved reliability_ci.npz")
