# Note: this script needs the project PINN framework (pinn_model.py, config.py, main.py, visualizer.py),
# which is not bundled here; the figures in this repo reproduce from the provided .npz without it.
# Higher-accuracy training run: collocation pool doubled (100k->200k interior, 40k->60k
# boundary) and final refinement tripled (500->1500 epochs), aiming to lower the interior
# Helmholtz residual below the ~1e-1 plateau while keeping the broadside beam.
import os, sys, time, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
CODE = r"D:\实践三号“延安”\实践三号“延安”\代码"
PAP = r"D:\实践三号“延安”\论文"
sys.path.insert(0, CODE); os.chdir(CODE)
import logging; logging.disable(logging.INFO)
import config
config.SAMPLING_CONFIG['num_total_points'] = 200000
config.SAMPLING_CONFIG['num_boundary_points'] = 60000
config.TRAINING_CONFIG['epochs_final'] = 1500
print("[hq] points 200k/60k, epochs_final=1500", flush=True)

from config import FREQUENCY, device
from main import (setup_environment, load_antenna_geometry, prepare_training_data,
                  setup_model_and_trainer, train_system)
from visualizer import GainVisualizer

t0 = time.time()
setup_environment()
mesh = load_antenna_geometry(substrate_path="../模型/1.1.STL", patch_path="../模型/1-1.stl", headless=True)
data = prepare_training_data(mesh)
model, eq, trainer = setup_model_and_trainer(data)
print(f"[setup] scale={data['scale_factor']:.3f} t={time.time()-t0:.0f}s", flush=True)
trained = train_system(trainer, data)
torch.save(trained.state_dict(), os.path.join(PAP, '_real_model_hq.pth'))
np.savez(os.path.join(PAP, '_train_history_hq.npz'),
         total=np.array(trainer.loss_history),
         pde=np.array(trainer.far_pde_history),
         div=np.array(trainer.div_history))
pde = np.array(trainer.far_pde_history)
print(f"[train done] t={time.time()-t0:.0f}s  PDE last50={pde[-50:].mean():.4f} min={pde.min():.4f}", flush=True)

sf = data['scale_factor']
vis = GainVisualizer(trained, FREQUENCY, coord_scale=sf, device=device)
res = vis.calculate_gain(theta_res=46, phi_res=73)
g = res['gain_pattern']
print(f"[eval] boresight={float(np.mean(g[:,0])):.2f} dBi  pattern max={res['max_gain']:.2f} dBi", flush=True)
np.savez(os.path.join(PAP, 'hq_pattern.npz'), theta=np.degrees(res['theta']),
         phi=np.degrees(res['phi']), g=g, max_gain=res['max_gain'])
try:
    s11o, _, Zo, _, _ = vis.calculate_s11_probe()
    print(f"[eval] probe Zin={Zo.real:.1f}{Zo.imag:+.1f}j S11={s11o:.2f} dB", flush=True)
except Exception as e: print("probe fail:", e)
print("HQ DONE", flush=True)
