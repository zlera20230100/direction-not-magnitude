# -*- coding: utf-8 -*-
# Full-wave finite-difference check of the surrogate's per-zone aperture Jacobian.
# The surrogate (zones.py) partitions the 96 PRS cells into K=6 radial zones and reports
# rela[k] = d ln Q_ap / d g_k, with Q_ap = mean |E_z|^2 over the aperture grid at the PRS plane.
# Here the same normalized gradient is recomputed with an openEMS FDTD finite difference for a
# few representative zones, and compared (sign and ratio) to the surrogate values.
#
# Reuses the directive FPC model from closure_directive.py (SUB=24, AIRGAP=5.4, PATCH=2.8,
# same materials/mesh/feed). Zones loaded from zones.npz; the 96 geom.json cells are
# co-indexed with the PINN pp cells (max NN distance 3e-15 mm). Baseline g=1 for all zones,
# then for each selected zone scale L,W by g_k = 1+-0.05, dump the frequency-domain E-field on
# the aperture plane (z=z_s2+ZLIFT), and form d ln Q_ap/d g_k = (lnQ(+5%)-lnQ(-5%))/0.10.
# Baseline + 3 zones x 2 = 7 runs.
#
# Usage: python grad_fullwave.py [zones e.g. 0,2,5]
import os, sys, json, time, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.add_dll_directory(r"D:\openEMS_pkg\openEMS")
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0
import h5py

def P(*a): print(*a, flush=True)
PAPER = os.path.dirname(os.path.abspath(__file__))

# zones (surrogate partition) + cells (geom.json, co-indexed with PINN pp)
Z = np.load(os.path.join(PAPER, "zones.npz"))
zone = Z["zone"].astype(int)           # (96,) zone id per cell, surrogate's radial partition
rela = Z["rela"]                       # surrogate d lnQ_ap/dg_k
K = int(zone.max()) + 1
G = json.load(open(os.path.join(PAPER, "_geom.json")))
cells = np.array([c[:4] for c in G["cells"]])            # (96,4) x0,y0,x1,y1 [mm]
feed = G["feed"][0]; fx = 0.5*(feed[0]+feed[2]); fy = 0.5*(feed[1]+feed[3])
cxc = 0.5*(cells[:,0]+cells[:,2]); cyc = 0.5*(cells[:,1]+cells[:,3])
P(f"{len(cells)} cells; K={K} zones; cells/zone={[int((zone==k).sum()) for k in range(K)]}")
P(f"surrogate rela = {np.round(rela,3)}  signs={['+' if v>0 else '-' for v in rela]}")

# model constants (same as closure_directive.py directive model)
unit = 1e-3; f0, fc = 24e9, 12e9
SUB = 24.0; PATCH = 2.8; AIRGAP = 5.4; margin = 7.0
z_g, z_s1 = 0.0, 1.0; z_air = z_s1 + AIRGAP; z_s2 = z_air + 1.0      # PRS cells on z_s2 top face
epsR, tand = 4.4, 0.02
APFINE = 0.06
k0 = 2*np.pi*f0/C0
ZLIFT = 0.30                          # aperture dump plane lift above PRS (mm), in radiating air
# aperture extent = cell-center bounding box (matches surrogate's [cxmin,cxmax]x[cymin,cymax])
axmin, axmax = cxc.min(), cxc.max(); aymin, aymax = cyc.min(), cyc.max()

def run_design(gzones, tag):
    """gzones: len-K per-zone scale. Build directive FPC, dump aperture E-field, return Q_ap, S11."""
    sim = rf"D:\openEMS_pkg\sim_gradA_{tag}"
    FDTD = openEMS(EndCriteria=1e-3, NrTS=40000); FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['PML_8']*6)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX); g = CSX.GetGrid(); g.SetDeltaUnit(unit)
    fr4 = CSX.AddMaterial('FR4', epsilon=epsR, kappa=2*np.pi*f0*EPS0*epsR*tand)
    gnd = CSX.AddMetal('gnd'); prs = CSX.AddMetal('prs'); via = CSX.AddMetal('via'); dp = CSX.AddMetal('dpatch')
    fr4.AddBox([-SUB/2,-SUB/2,z_g],  [SUB/2,SUB/2,z_s1], priority=1)
    fr4.AddBox([-SUB/2,-SUB/2,z_air],[SUB/2,SUB/2,z_s2], priority=1)
    gnd.AddBox([-SUB/2,-SUB/2,z_g],[SUB/2,SUB/2,z_g], priority=10)
    xe=set(); ye=set()
    for i,(x0,y0,x1,y1) in enumerate(cells):
        s = float(gzones[zone[i]]); cxi=0.5*(x0+x1); cyi=0.5*(y0+y1)
        Lx=(x1-x0)*s; Ly=(y1-y0)*s
        nx0,nx1,ny0,ny1 = cxi-Lx/2,cxi+Lx/2,cyi-Ly/2,cyi+Ly/2
        prs.AddBox([nx0,ny0,z_s2],[nx1,ny1,z_s2], priority=10)
        xe.update([nx0,nx1,cxi]); ye.update([ny0,ny1,cyi])
    hp=0.10; zgap=0.12
    port = FDTD.AddLumpedPort(1,50.0,[fx-hp,fy-hp,z_g],[fx+hp,fy+hp,zgap],'z',1.0,priority=5,edges2grid='xy')
    via.AddBox([fx-hp,fy-hp,zgap],[fx+hp,fy+hp,z_s1], priority=10)
    xe.update([fx-hp,fx+hp]); ye.update([fy-hp,fy+hp])
    Lpy=PATCH; Lpx=PATCH*0.85
    dp.AddBox([-Lpx/2,-Lpy/2,z_s1],[Lpx/2,Lpy/2,z_s1], priority=12)
    xe.update([-Lpx/2,Lpx/2,0.0]); ye.update([-Lpy/2,Lpy/2,0.0])
    # aperture freq-domain E-field dump plane just above PRS
    z_ap = z_s2 + ZLIFT
    dump = CSX.AddDump('ap_E', dump_type=10, dump_mode=2, file_type=1)   # 10=E-field freq-domain, HDF5
    dump.AddFrequency([f0])
    dump.AddBox([axmin, aymin, z_ap], [axmax, aymax, z_ap], priority=0)
    def build(crit, lo, hi, tol=0.02):
        L = sorted(set([lo,hi]+list(crit))); out=[L[0]]
        for v in L[1:]:
            if v-out[-1] >= tol: out.append(v)
        return out
    ap = list(np.arange(-3.6, 3.6001, APFINE))
    xcrit = build(xe | set(ap) | {fx-hp,fx+hp}, -SUB/2-margin, SUB/2+margin)
    ycrit = build(ye | set(ap) | {fy-hp,fy+hp}, -SUB/2-margin, SUB/2+margin)
    g.AddLine('x', xcrit); g.AddLine('y', ycrit)
    g.AddLine('z', [-margin*1.4, z_g, zgap, z_s1, z_air, z_s2, z_ap, z_s2+margin*2.0])
    g.SmoothMeshLines('x',0.5,ratio=1.45); g.SmoothMeshLines('y',0.5,ratio=1.45); g.SmoothMeshLines('z',0.45,ratio=1.45)
    def dedupe(dirn, floor=0.035):
        L=list(g.GetLines(dirn)); keep=[L[0]]
        for v in L[1:-1]:
            if v-keep[-1] >= floor: keep.append(v)
        if L[-1]-keep[-1] >= floor: keep.append(L[-1])
        else: keep[-1]=L[-1]
        g.SetLines(dirn, keep)
    for d in ('x','y','z'): dedupe(d)
    nx=len(g.GetLines('x')); ny=len(g.GetLines('y')); nz=len(g.GetLines('z'))
    P(f"[{tag}] mesh {nx}x{ny}x{nz}={nx*ny*nz/1e6:.2f}M  running FDTD...")
    t0=time.time(); FDTD.Run(sim, cleanup=True, verbose=0); dt=time.time()-t0
    f = np.linspace(18e9,40e9,441); port.CalcPort(sim, f)
    s11 = 20*np.log10(np.abs(port.uf_ref/port.uf_inc)); i0=np.argmin(np.abs(f-f0))
    # read aperture E-field HDF5: mean |E_z|^2 over the plane
    h5 = os.path.join(sim, 'ap_E.h5')
    with h5py.File(h5,'r') as hf:
        fd = hf['FieldData']['FD']
        keys = sorted(fd.keys())
        # frequency-domain E-field: real & imag arrays 'f0_real','f0_imag' (shape ... x 3)
        rk = [k for k in keys if k.endswith('_real')][0]
        ik = [k for k in keys if k.endswith('_imag')][0]
        Er = np.array(fd[rk]); Ei = np.array(fd[ik])
    # openEMS FD field dump layout is (component=3, nz, ny, nx); find the size-3 component axis
    comp_ax = [a for a,s in enumerate(Er.shape) if s==3]
    comp_ax = comp_ax[0] if comp_ax else 0
    Ezr = np.take(Er, 2, axis=comp_ax); Ezi = np.take(Ei, 2, axis=comp_ax)   # Ez component
    Q_ap = float(np.mean(Ezr**2 + Ezi**2))
    P(f"[{tag}] S11@24={s11[i0]:.2f}dB  Q_ap=mean|Ez|^2={Q_ap:.4e}  Efield shape={Er.shape}  ({dt:.0f}s)")
    return dict(Q_ap=Q_ap, s11_24=float(s11[i0]), g=np.asarray(gzones,float))

def main():
    zlist = [int(z) for z in (sys.argv[1].split(',') if len(sys.argv)>1 else ['0','2','5'])]
    DELTA = 0.05
    results = {}
    P("\n=== baseline g=1 (all zones) ===")
    base = run_design(np.ones(K), 'base'); results['base'] = base

    fd_grad = {}; Q_plus = {}; Q_minus = {}
    for k in zlist:
        gp = np.ones(K); gp[k] = 1.0+DELTA
        gm = np.ones(K); gm[k] = 1.0-DELTA
        P(f"\n=== zone {k}: +{DELTA*100:.0f}% ===")
        rp = run_design(gp, f'z{k}p'); results[f'z{k}p']=rp
        P(f"\n=== zone {k}: -{DELTA*100:.0f}% ===")
        rm = run_design(gm, f'z{k}m'); results[f'z{k}m']=rm
        Q_plus[k]=rp['Q_ap']; Q_minus[k]=rm['Q_ap']
        fd = (np.log(rp['Q_ap']) - np.log(rm['Q_ap'])) / (2*DELTA)
        fd_grad[k] = fd
        P(f"  zone {k}: lnQ(+)={np.log(rp['Q_ap']):.4f} lnQ(-)={np.log(rm['Q_ap']):.4f} "
          f"-> FW dlnQ/dg={fd:+.3f}  | surrogate rela={rela[k]:+.3f}")

    # summary table
    P("\n===== TASK A: FULL-WAVE vs SURROGATE aperture-Jacobian =====")
    P(f"{'zone':>4s} {'FW dlnQ/dg':>11s} {'surr rela':>10s} {'sign match':>11s} {'ratio FW/surr':>14s}")
    rows=[]
    for k in zlist:
        sm = (np.sign(fd_grad[k])==np.sign(rela[k]))
        ratio = fd_grad[k]/rela[k] if rela[k]!=0 else np.nan
        P(f"{k:>4d} {fd_grad[k]:>+11.3f} {rela[k]:>+10.3f} {str(bool(sm)):>11s} {ratio:>14.3f}")
        rows.append((k, fd_grad[k], rela[k], bool(sm), ratio))

    np.savez(os.path.join(PAPER, "grad_fullwave.npz"),
             zones=np.array(zlist),
             g_pert=DELTA,
             Q_base=base['Q_ap'],
             Q_plus=np.array([Q_plus[k] for k in zlist]),
             Q_minus=np.array([Q_minus[k] for k in zlist]),
             fd_grad=np.array([fd_grad[k] for k in zlist]),
             surrogate_rela=np.array([rela[k] for k in zlist]),
             sign_match=np.array([np.sign(fd_grad[k])==np.sign(rela[k]) for k in zlist]),
             s11_base=base['s11_24'],
             zone_partition=zone, ZLIFT=ZLIFT, z_ap=z_s2+ZLIFT)
    P("\nsaved grad_fullwave.npz")

if __name__ == '__main__':
    main()
