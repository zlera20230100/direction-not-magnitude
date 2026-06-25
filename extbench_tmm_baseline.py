# -*- coding: utf-8 -*-
# Cheap-baseline check for the TMM benchmark: does ordering FD-verification by the gate
# (least sign-agreement first) beat the obvious heuristic of verifying the LARGEST-|gradient|
# components first? The reviewer's concern is that the dangerous case is "large-|J| but wrong",
# so verify-largest-|J| is the natural cheap competitor. We add that baseline (order_mag) to the
# exact extbench_tmm pipeline (identical seeds, so the AUC/sign-correct numbers reproduce) and
# report (a) cosine-to-truth vs #FD frontier and (b) fraction of the genuinely SIGN-WRONG
# components caught vs #FD budget, for gate / magnitude / random ordering.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

LAM0 = 1550.0
nH, nL, nsub, n0 = 2.35, 1.45, 1.52, 1.0
IDX = np.array([nH, nL, nH, nL, nH, nL, nH, nL, nH, nL, nH])
QW = LAM0 / (4.0 * IDX); NOMINAL = QW.copy(); NOMINAL[5] *= 2.0
K = IDX.size

def tmm_R(thick, lam):
    M = np.eye(2, dtype=complex)
    for nj, dj in zip(IDX, thick):
        delta = 2.0 * np.pi * nj * dj / lam
        c, s = np.cos(delta), np.sin(delta)
        M = M @ np.array([[c, 1j * s / nj], [1j * nj * s, c]], dtype=complex)
    B = M[0, 0] + M[0, 1] * nsub; C = M[1, 0] + M[1, 1] * nsub
    r = (n0 * B - C) / (n0 * B + C)
    return float(np.abs(r) ** 2)

_scan = np.linspace(LAM0 - 60, LAM0 + 60, 481)
_Rs = np.array([tmm_R(NOMINAL, l) for l in _scan]); _up = _scan > LAM0
LAM_EVAL = float(_scan[_up][np.argmin(np.abs(_Rs[_up] - 0.4))])

def tmm_grad_fd(thick, h=0.5):
    g = np.zeros(K)
    for k in range(K):
        tp = thick.copy(); tp[k] += h; tm = thick.copy(); tm[k] -= h
        g[k] = (tmm_R(tp, LAM_EVAL) - tmm_R(tm, LAM_EVAL)) / (2 * h)
    return g

RNG = np.random.default_rng(7)
NS = int(os.environ.get('NS', '600')); span = 0.30
X = NOMINAL[None, :] * (1.0 + span * (2 * RNG.random((NS, K)) - 1))
Y = np.array([tmm_R(x, LAM_EVAL) for x in X])
xm, xs = X.mean(0), X.std(0); ym, ysd = Y.mean(), Y.std()
Xt = torch.tensor((X - xm) / xs, dtype=torch.float32)
Yt = torch.tensor((Y - ym) / ysd, dtype=torch.float32).view(-1, 1)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

def train_one(seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = MLP(); opt = torch.optim.Adam(net.parameters(), lr=2e-3); lossf = nn.MSELoss()
    n = Xt.size(0); idx = torch.randperm(n, generator=g); tr = idx[:int(0.9 * n)]
    for ep in range(400):
        perm = tr[torch.randperm(tr.numel(), generator=g)]
        for b in perm.split(256):
            opt.zero_grad(); lossf(net(Xt[b]), Yt[b]).backward(); opt.step()
    return net

M = 10
nets = [train_one(s) for s in range(M)]

def ad_grad(net, xphys):
    xn = torch.tensor(((xphys - xm) / xs), dtype=torch.float32).view(1, -1).requires_grad_(True)
    y = net(xn); y.backward()
    return (xn.grad.detach().numpy().ravel()) * (ysd / xs)

NQ = 80
QP = NOMINAL[None, :] * (1.0 + 0.12 * (2 * RNG.random((NQ, K)) - 1))
TAU = 0.9
def cossim(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
rng2 = np.random.default_rng(123)
cos_gate = np.zeros((NQ, K + 1)); cos_mag = np.zeros((NQ, K + 1)); cos_rand = np.zeros((NQ, K + 1))
# fraction of sign-wrong components caught within the first n FD-verifications
caught_gate = np.zeros((NQ, K + 1)); caught_mag = np.zeros((NQ, K + 1)); caught_rand = np.zeros((NQ, K + 1))
nwrong_q = np.zeros(NQ)
all_sa, all_correct = [], []
for q, xq in enumerate(QP):
    Gm = np.array([ad_grad(net, xq) for net in nets])
    ad = Gm.mean(0); sa = np.maximum((Gm > 0).mean(0), (Gm < 0).mean(0))
    fd = tmm_grad_fd(xq)
    wrong = (np.sign(ad) != np.sign(fd))                  # the dangerous components
    nwrong_q[q] = wrong.sum()
    all_sa.append(sa); all_correct.append((~wrong).astype(int))
    order_gate = np.argsort(sa)                            # gate: least sign-agreement first
    order_mag = np.argsort(-np.abs(ad))                    # baseline: largest |gradient| first
    for n in range(K + 1):
        g = ad.copy(); g[order_gate[:n]] = fd[order_gate[:n]]; cos_gate[q, n] = cossim(g, fd)
        g = ad.copy(); g[order_mag[:n]] = fd[order_mag[:n]]; cos_mag[q, n] = cossim(g, fd)
        if wrong.sum() > 0:
            caught_gate[q, n] = wrong[order_gate[:n]].sum() / wrong.sum()
            caught_mag[q, n] = wrong[order_mag[:n]].sum() / wrong.sum()
        acc_c = 0.0; acc_w = 0.0; R = 8
        for _ in range(R):
            perm = rng2.permutation(K)
            g = ad.copy(); g[perm[:n]] = fd[perm[:n]]; acc_c += cossim(g, fd)
            if wrong.sum() > 0: acc_w += wrong[perm[:n]].sum() / wrong.sum()
        cos_rand[q, n] = acc_c / R; caught_rand[q, n] = acc_w / R

all_sa = np.concatenate(all_sa); all_correct = np.concatenate(all_correct)
auc = roc_auc_score(all_correct, all_sa)
cg, cm, cr = cos_gate.mean(0), cos_mag.mean(0), cos_rand.mean(0)
# average caught-fraction only over query points that HAVE a sign-wrong component
hasw = nwrong_q > 0
qg, qm, qr = caught_gate[hasw].mean(0), caught_mag[hasw].mean(0), caught_rand[hasw].mean(0)
print(f"TMM baseline check: NQ={NQ} K={K}, AUC={auc:.3f}, total sign-wrong={int(nwrong_q.sum())}")
print(f"\ncosine-to-truth vs #FD:  {'#FD':>3} {'gate':>8} {'mag-|J|':>8} {'random':>8}")
for n in range(K + 1):
    print(f"                         {n:>3} {cg[n]:>8.3f} {cm[n]:>8.3f} {cr[n]:>8.3f}")
print(f"\nfraction of SIGN-WRONG components caught vs #FD (over {int(hasw.sum())} query pts w/ a wrong comp):")
print(f"  {'#FD':>3} {'gate':>8} {'mag-|J|':>8} {'random':>8}")
for n in range(K + 1):
    print(f"  {n:>3} {qg[n]:>8.3f} {qm[n]:>8.3f} {qr[n]:>8.3f}")
# headline: how many FD to catch >=80% of sign-wrong components
def budget_to(curve, tgt=0.8):
    idx = np.where(curve >= tgt)[0]; return int(idx[0]) if len(idx) else K
bg, bm, brd = budget_to(qg), budget_to(qm), budget_to(qr)
print(f"\nFD budget to catch >=80% of sign-wrong components: gate {bg} | mag-|J| {bm} | random {brd}  (of {K})")
np.savez('extbench_tmm_baseline.npz', auc=auc, K=K, NQ=NQ, tau=TAU,
         cos_gate=cg, cos_mag=cm, cos_rand=cr,
         caught_gate=qg, caught_mag=qm, caught_rand=qr,
         budget80_gate=bg, budget80_mag=bm, budget80_rand=brd,
         total_sign_wrong=int(nwrong_q.sum()))
print("saved extbench_tmm_baseline.npz")
