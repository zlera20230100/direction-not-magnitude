# Correlated-seed-aware ("self-aware") gate — results & honest verdict

**Scope.** Paper-2 (Neurocomputing) revision experiment. Goal: turn the reliability gate's
biggest weakness — it is silently defeated by correlated / shared-seed bias near rho ≈ 0.7
(the Devil's-Advocate exploit) — into a *detected, handled* case. Synthetic only; reuses the
shared-bias data model of `shared_bias_sweep.py` / `hybrid_robust.py`. No manuscript or git
changes.

- Code: `selfaware_gate.py` · Data: `selfaware_gate.npz`
- Run: `D:\ANACONDA\python.exe selfaware_gate.py` (CPU, `OMP_NUM_THREADS=6`)
- Reproducible: fixed seeds; re-runs are bit-identical (verified).

---

## 1. Setup (identical to the existing shared-bias sweep)

Seed gradients `J[m,k] = a_true[k] + common[k] + idio[m,k]`, with
`a_true = [2.0, −1.5, 0.8, −0.6, 1.3, −0.9]`, K = 6 components, M = 10 retrained seeds,
trust threshold τ = 0.9, 8000 Monte-Carlo trials per rho.

- `common[k] ~ N(0, rho·σ_k²)` — one draw, **shared by every seed** (the shared bias).
- `idio[m,k] ~ N(0, (1−rho)·σ_k²)` — independent per seed.
- `σ_k` log-uniform in [0.05, 8]. Total per-component noise variance is σ_k² for **every** rho
  (std-matched), so only the *sharing* changes, never the marginal difficulty. rho is literally
  the intraclass correlation (ICC) — the fraction of noise variance that is seed-shared.

**Plain gate.** Certify component k (use the cheap autodiff ensemble mean `ad_k` instead of a
solver call) iff sign-agreement `sa_k = max(frac>0, frac<0) ≥ τ`.

**The exploit.** A large `common[k]` pushes all M seeds the same way → high sign-agreement on a
sign that may be **wrong**. The gate then certifies a sign-flipped component ("false trust").

---

## 2. The new algorithm

**rho-hat (global, per trial).** One-way random-effects / ICC(1,1) on `J`, treating the K
components as targets and the M seeds as raters:

```
MSB = M·Σ_k (mean_m J[·,k] − grand_mean)² / (K−1)
MSW = Σ_{m,k} (J[m,k] − mean_m J[·,k])² / (K·(M−1))
rho_hat = (MSB − MSW) / (MSB + (M−1)·MSW),  clipped to [0,1]
```

**Self-aware gate — two variants tested.**
- **GLOBAL** (the literal spec): if `rho_hat ≥ rho_gate` the ensemble is declared correlated and
  the gate **certifies nothing** (defers all K to the solver); else behaves as plain.
- **PER-COMPONENT** (discovered fix): certify k iff `sa_k ≥ τ` **and** `sd_k ≤ s_gate`, where
  `sd_k` is the observed across-seed std. Motivation: among certified comps, `sd_k ≈ √(1−rho)·σ_k`
  is a proxy for the noise scale; a shared-bias flip needs a large `common[k] ~ N(0, rho·σ_k²)`,
  i.e. it concentrates on **large σ_k → large observed sd_k**. So a certified component with an
  anomalously large across-seed spread is exactly the one a shared bias is most likely to have
  flipped.

Both thresholds are **calibrated on a clean (rho = 0) reference cohort** (only `J` is ever
observed): `rho_gate = P95(rho_hat | rho=0) = 0.874`, `s_gate = P95(sd_k | certified, rho=0) = 1.41`.
Each therefore costs ~5 % clean-case false-alarm by construction.

---

## 3. Results

### (1) Does rho-hat track the true rho? — YES (as a tracker).

| metric | value |
|---|---|
| Pearson r (rho-hat-mean vs rho) | **0.998** |
| Spearman | **1.000** |
| linear fit | rho_hat ≈ 0.559·rho + **0.349** |
| offset-corrected MAE | 0.189 |

rho-hat rises monotonically and rank-perfectly with rho. **Honest caveat:** it carries a large
positive offset (≈0.35 at rho = 0) because the *true signal* `a_true` is itself seed-shared
structure; raw ICC cannot separate "shared signal" from "shared bias". So rho-hat is a good
*relative* meter, not an unbiased estimate of rho.

### (2) False-trust rate P(sign-WRONG | certified), and solver cost (deferrals/trial, of K = 6).

| rho | FT plain | FT **global** | FT **per-comp** | cost plain | cost global | cost per-comp | err plain | err per-comp | err all-AD |
|----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0007 | 0.0007 | **0.0000** | 2.19 | 2.46 | 2.39 | 0.141 | 0.076 | 0.499 |
| 0.5 | 0.0438 | 0.0496 | **0.0130** | 1.47 | 2.22 | 2.06 | 0.879 | 0.265 | 1.168 |
| 0.6 | 0.0591 | 0.0675 | **0.0224** | 1.30 | 2.31 | 1.93 | 1.042 | 0.336 | 1.270 |
| 0.7 | 0.0702 | 0.0825 | **0.0339** | 1.11 | 2.65 | 1.75 | 1.161 | 0.431 | 1.342 |
| 0.8 | 0.0879 | 0.1060 | **0.0521** | 0.89 | 3.38 | 1.48 | 1.310 | 0.593 | 1.433 |

- **Plain gate confirms the rho-wall:** false-trust climbs 0.0007 → 0.088, and — perversely —
  solver cost *drops* (2.19 → 0.89) because the shared bias inflates sign-agreement, so the gate
  trusts **more** while being **more** wrong. Silent defeat.
- **GLOBAL self-aware gate fails (honest negative result).** It does **not** lower the false-trust
  *rate* — it is flat-to-worse (0.088 → 0.106 at rho = 0.8, +21 %) while solver cost **explodes**
  (0.89 → 3.38). Reason: abstaining on a *whole flagged trial* removes correct- and wrong-
  certified components in equal proportion, so the *ratio* wrong/certified is unchanged. A single
  trial-level scalar cannot point at *which* component is the flipped one (verified: rho-hat's AUC
  for the per-component wrong label is ≈ 0.43, i.e. chance).
- **PER-COMPONENT self-aware gate works.** False-trust is cut **by ~70 % at rho=0.5, easing to ~62 % at
  0.6 and ~52 % at 0.7** (0.044→0.013, 0.059→0.022, 0.070→0.034) and **−41 % at rho = 0.8**, while the
  assembled-gradient error falls to near the global gate's at far lower cost (rho = 0.8:
  err 0.593 vs 0.628, cost 1.48 vs 3.38). At rho = 0 it costs almost nothing (cost 2.19 → 2.39,
  FT → 0).

### (3) Safety restored where the plain gate fails (worst case, rho = 0.8).

| | plain | global | **per-comp** | oracle | all-AD |
|---|---:|---:|---:|---:|---:|
| false-trust rate | 0.0879 | 0.1060 (+21 %) | **0.0521 (−41 %)** | — | — |
| solver cost / K | 0.89 | 3.38 | **1.48** | 2.36 | 0 |
| grad rel-error | 1.310 | 0.628 | **0.593** | 0.216 | 1.433 |

The per-component gate is the only variant that improves safety *and* accuracy at modest cost.

### (4) Operating-point frontier (per-component gate; spread ceiling = percentile of rho=0 cohort).

False-trust / cost at rho = 0.8 as the ceiling tightens: P99 → 0.081/1.0, P95 → 0.052/1.5,
P90 → 0.036/1.8, P80 → 0.020/2.2, **P70 → 0.012/2.5** (vs plain 0.088/0.9). A tunable
safety↔cost knob: at a strict P70 ceiling the false-trust rate is **7× below** the plain gate.

### Robustness (sanity checks, not in the main npz)

The per-component advantage holds across seeds and ensemble sizes (M = 5, 10, 20); the reduction
**widens with M** (M = 20, rho = 0.7: 0.059 → 0.018, a 3.3× cut), consistent with `sd_k` being a
sharper noise-scale estimate at larger M.

---

## 4. Honest verdict

- **Does rho-hat track rho?** Yes — Pearson 0.998 / Spearman 1.000, monotone — but only as a
  *relative* meter (≈0.35 offset from confounding the true signal with shared bias). Usable as a
  regime indicator, not as a calibrated rho estimate.
- **Does the gate know when its independence assumption is violated?** Yes — rho-hat clearly
  separates the correlated regime.
- **Does the literal design (abstain-all when rho-hat high) fix the rho-wall?** **No.** This is a
  genuine negative result: a *global* abstention cannot lower the per-component false-trust *rate*
  (it only buys very expensive deferrals). Detecting the regime is necessary but not sufficient.
- **What actually fixes it?** A **per-component** self-aware test that drops certified components
  with anomalously large across-seed spread `sd_k` — the observable signature of the high-σ
  components on which shared-bias sign-flips ride. This cuts false-trust by ~70 % at rho=0.5,
  ~62 % at 0.6, ~52 % at 0.7, and 41 % at rho = 0.8, lowers gradient error well below the plain gate, and stays cheap
  at rho = 0 — restoring safety exactly in the regime where the plain gate is silently defeated.
- **Cost.** The per-component gate adds ~0.2 solver calls/trial at rho = 0 and ~0.6 at rho = 0.8
  (out of K = 6) — far cheaper than the global gate's blanket deferral, and tunable via the
  spread-ceiling percentile.

**Take-away for the paper.** The shared-seed exploit is *detectable and handlable*, but the
honest lesson is that the right detector is per-component spread, not a global correlation scalar.
The global ICC meter is a good diagnostic ("is this ensemble correlated?") yet does not by itself
repair the gate; the per-component spread ceiling does. Both findings — the negative (global) and
the positive (per-component) — should be reported.

## 5. Reproducibility

| item | value |
|---|---|
| Python | 3 (Anaconda, `D:\ANACONDA\python.exe`), CPU, OMP_NUM_THREADS=6 |
| numpy / scipy / sklearn | 2.3.5 / 1.16.3 / 1.7.2 |
| a_true | [2.0, −1.5, 0.8, −0.6, 1.3, −0.9] |
| K, M, τ | 6, 10, 0.9 |
| σ range | log-uniform [0.05, 8.0] |
| rho grid | 0.0 … 0.8 (step 0.1) |
| trials / rho | 8000 (calibration cohort 20000) |
| calibration | rho_gate = P95(rho-hat\|rho=0) = 0.874; s_gate = P95(sd_k\|cert,rho=0) = 1.41 |
| seeds | calibration 13; sweep 1000+round(1000·rho); op-sweep 7000+round(1000·rho) |
| determinism | re-runs bit-identical (verified) |
