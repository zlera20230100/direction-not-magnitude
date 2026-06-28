# Reproducibility code and data

**Direction, Not Magnitude: A Portable, Solver-Free Reliability Gate for Differentiable-Surrogate Design Gradients**

Xuan Qin, Xuan Shi, Nimako Samuel Boateng, Bokai Huang, Shengjun Wu, Kai You, Long Zhang\* (corresponding: 20230100@huat.edu.cn)
Hubei University of Automotive Technology, Shiyan 442002, China

This repository contains the figure generators and the frozen result data (`.npz`) needed to
reproduce every figure in the paper, together with the analysis and experiment scripts that
produced that data. Every figure can be regenerated from the provided `.npz` without rerunning any
solver or training.

---

## 1. Environment

- Python 3.11
- NumPy, SciPy, scikit-learn, Matplotlib (Times New Roman / STIX for figures)
- PyTorch 2.6.0 (only for the surrogate/ensemble experiment scripts; not needed to redraw figures)
- h5py and the openEMS Python bindings (`CSXCAD` / `openEMS`) — only for the full-wave scripts

Set `KMP_DUPLICATE_LIB_OK=TRUE` on Windows if you hit an OpenMP duplicate-runtime error.
Run scripts from inside this folder (they load their `.npz` inputs by relative name).

---

## 2. Quick reproduction (figures from the provided data)

```bash
python fig_unified.py        # -> fig_method, fig_forward, fig_jacobian, fig_inert, fig_remedy
python fig_reliability.py        # -> fig_reliability
python fig_reliability_calib.py  # -> fig_reliability_calib
python fig_endtoend.py           # -> fig_endtoend
python fig_extbench.py           # -> fig_extbench
python fig_hybrid_gradient.py    # -> fig_hybrid_gradient
```

These read the frozen `.npz` files and write the PDF/PNG figures used in the manuscript.

---

## 3. Figure -> generator -> data

| Figure | Generator | Main data file(s) |
|---|---|---|
| Method / device | `fig_unified.py` | `fpc_result.npz`, `hq_pattern.npz`, `movable.npz` |
| Forward accuracy | `fig_unified.py` | `fpc_result.npz`, `hq_pattern.npz` |
| Gradient-reliability map | `fig_unified.py` | `zones_multiseed.npz`, `grad_fullwave.npz` |
| Reliability indicator | `fig_reliability.py` | `reliability.npz`, `mcdropout.npz`, `reliability_ci.npz` |
| Calibration / ensemble size | `fig_reliability_calib.py` | `reliability_calib.npz` |
| End-to-end safe step | `fig_endtoend.py` | `endtoend.npz` |
| External benchmark (TMM) | `fig_extbench.py` | `extbench_tmm.npz` |
| Cost-optimal hybrid gradient | `fig_hybrid_gradient.py` | `hybrid_gradient.npz` |
| Operating boundary (inert) | `fig_unified.py` | `movable.npz`, `closure_directive.npz` |
| Resonance-wall remedies | `fig_unified.py` | (values in script) |

---

## 4. Experiment / analysis scripts

Self-contained (run anywhere with NumPy/PyTorch):

```bash
python hybrid_gradient.py     # reliability-gated hybrid gradient: cost-accuracy frontier -> hybrid_gradient.npz
python hybrid_robust.py       # frontier under Gaussian/Student-t/Laplace/correlated-seed noise
python hybrid_decouple.py     # label-decoupled AUC + ensemble-variance baseline
python extbench_tmm.py         # thin-film transfer-matrix external benchmark -> extbench_tmm.npz
python reliability_calib.py    # controlled reliability-spectrum calibration -> reliability_calib.npz
python reliability.py          # ensemble sign-agreement / SNR indicator + AUC -> reliability.npz
python reliability_ci.py       # bootstrap CI for the AUC -> reliability_ci.npz
python grad_fullwave.py        # full-wave finite-difference gradient audit -> grad_fullwave.npz
python endtoend.py             # trust->step->verify loop on a synthetic testbed -> endtoend.npz
python movable.py              # responsive/null control study -> movable.npz
python closure_directive.py    # directive full-wave closure / inertness check -> closure_directive.npz
python openems_fpc.py          # full-wave FPC reference (openEMS)
```

Require the project PINN framework (`pinn_model.py`, `config.py`, `main.py`, `visualizer.py`,
**not bundled here**; the figures above do not need them):

```bash
python zones_multiseed.py      # per-zone design Jacobian over multiple PINN seeds -> zones_multiseed.npz
python mcdropout.py            # MC-dropout uncertainty baseline -> mcdropout.npz
python train_hq.py             # high-Q PINN pattern -> hq_pattern.npz
```

---

## 5. Notes

- All results are simulation-based; the full-wave reference (openEMS / a second solver) is the baseline.
- Seeds and settings are fixed inside each script and listed in the manuscript's reproducibility table.
- License: MIT (see `LICENSE`). Citation metadata: `CITATION.cff`; archival metadata: `.zenodo.json`.
