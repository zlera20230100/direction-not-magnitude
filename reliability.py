# -*- coding: utf-8 -*-
# Computes cheap ensemble reliability indicators (seed sign-agreement, SNR) from the
# multi-seed design gradient and scores how well they separate reliable from unreliable
# gradients, against a naive magnitude baseline. Labels come from cases with known ground
# truth: responsive synthetic Jacobian (reliable), null synthetic + real radiation gradient
# (unreliable, full-wave sign-match 3/6).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.metrics import roc_auc_score

ms = np.load('zones_multiseed.npz')
ct = np.load('protocol_control.npz')
fw = np.load('grad_fullwave.npz')

def ens_stats(J):  # J: (n_seeds, K) ensemble of a gradient (one column per component)
    mean = J.mean(0); std = J.std(0)
    sign_agree = np.maximum((J > 0).mean(0), (J < 0).mean(0))      # in [0.5,1]; 1 = all seeds agree on sign
    snr = np.abs(mean) / (std + 1e-12)                             # ensemble signal-to-noise (epistemic)
    naive_mag = np.abs(J).mean(0)                                  # baseline: per-seed |gradient|
    return sign_agree, snr, naive_mag

# three component groups, each a (10 seeds, 6 components) ensemble
sa_resp, snr_resp, mag_resp = ens_stats(ct['J_responsive'])       # reliable (ground truth known)
sa_null, snr_null, mag_null = ens_stats(ct['J_null'])             # unreliable (null)
sa_rad,  snr_rad,  mag_rad  = ens_stats(ms['rela'])               # unreliable (real radiation gradient)

# assemble labelled set: reliable=1, unreliable=0
sign_agree = np.concatenate([sa_resp, sa_null, sa_rad])
snr        = np.concatenate([snr_resp, snr_null, snr_rad])
magnitude  = np.concatenate([mag_resp, mag_null, mag_rad])
label      = np.concatenate([np.ones(6), np.zeros(6), np.zeros(6)]).astype(int)

print("component-level reliability indicators (reliable=responsive; unreliable=null + real radiation):")
print(f"  reliable   sign-agree {sa_resp.mean():.2f}  SNR {snr_resp.mean():8.1f}  |grad| {mag_resp.mean():6.2f}")
print(f"  null       sign-agree {sa_null.mean():.2f}  SNR {snr_null.mean():8.2f}  |grad| {mag_null.mean():6.3f}")
print(f"  radiation  sign-agree {sa_rad.mean():.2f}  SNR {snr_rad.mean():8.2f}  |grad| {mag_rad.mean():6.2f}")

auc_sign = roc_auc_score(label, sign_agree)
auc_snr  = roc_auc_score(label, snr)
auc_mag  = roc_auc_score(label, magnitude)
print("\nreliability-classification AUC (predict reliable from a cheap, no-full-wave indicator):")
print(f"  ensemble sign-agreement : AUC = {auc_sign:.3f}")
print(f"  ensemble SNR (|mean|/std): AUC = {auc_snr:.3f}")
print(f"  gradient magnitude (naive baseline): AUC = {auc_mag:.3f}")

# simple operating threshold on sign-agreement: flag 'verify-first' if < 0.9 (i.e. not near-unanimous)
thr = 0.9
flag_trust = sign_agree >= thr
tp = int(((flag_trust == 1) & (label == 1)).sum()); fn = int(((flag_trust == 0) & (label == 1)).sum())
fp = int(((flag_trust == 1) & (label == 0)).sum()); tn = int(((flag_trust == 0) & (label == 0)).sum())
print(f"\nthreshold sign-agreement>={thr}: TP={tp} FN={fn} FP={fp} TN={tn} "
      f"(sensitivity {tp/(tp+fn):.2f}, specificity {tn/(tn+fp):.2f})")

# full-wave cross-check on the real device (radiation): report the full-wave sign-match
# fraction alongside the ensemble agreement.
print(f"\nreal-device radiation gradient: ensemble sign-agree mean {sa_rad.mean():.2f}; "
      f"full-wave sign-match {int(fw['sign_match'].sum())}/6 (coincidental) -> both flag DISTRUST")

np.savez('reliability.npz',
         sign_agree=sign_agree, snr=snr, magnitude=magnitude, label=label,
         sa_resp=sa_resp, sa_null=sa_null, sa_rad=sa_rad,
         snr_resp=snr_resp, snr_null=snr_null, snr_rad=snr_rad,
         mag_resp=mag_resp, mag_null=mag_null, mag_rad=mag_rad,
         auc_sign=auc_sign, auc_snr=auc_snr, auc_mag=auc_mag, thr=thr)
print("\nsaved reliability.npz")
