# Full-wave openEMS model of the antenna from WANZHENG.step (HFSS export):
#   Fabry-Perot cavity antenna. Ground (PEC) at z=0; lower FR-4 0-1mm; air cavity
#   1-5.95mm; upper FR-4 5.95-6.95mm; 96-cell metasurface PRS (PEC) at z=6.95mm.
#   Probe feed (0.2x0.2x1mm post) at (0,1.25): z-lumped port 0->1mm.
# HFSS reference values: gain ~10.3-11.2 dBi, S11 ~ -12 dB, Zin ~ 50-90j.
import os, sys, json, numpy as np
os.add_dll_directory(r"D:\openEMS_pkg\openEMS")
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0
def P(*a): print(*a, flush=True)

G = json.load(open(r"D:\实践三号“延安”\论文\_geom.json"))
cells = G["cells"]; feed = G["feed"][0]
fx = 0.5*(feed[0]+feed[2]); fy = 0.5*(feed[1]+feed[3])
P(f"{len(cells)} cells; feed center ({fx:.3f},{fy:.3f})")

sim = r"D:\openEMS_pkg\sim_fpc"
unit = 1e-3; f0, fc = 24e9, 12e9
AIRGAP = float(sys.argv[5]) if len(sys.argv)>5 else 4.95   # air-cavity height (mm): cad=4.95, pinn-config=6.25
z_g, z_s1 = 0.0, 1.0
z_air = z_s1 + AIRGAP; z_s2 = z_air + 1.0            # air-top / upper-fr4 top (cells on top)
epsR, tand = 4.4, 0.02
SUB = float(sys.argv[1]) if len(sys.argv)>1 else 16.0
margin = 7.0

FDTD = openEMS(EndCriteria=1e-3, NrTS=40000)
FDTD.SetGaussExcite(f0, fc); FDTD.SetBoundaryCond(['PML_8']*6)
CSX = ContinuousStructure(); FDTD.SetCSX(CSX); g = CSX.GetGrid(); g.SetDeltaUnit(unit)

PTOP = float(sys.argv[2]) if len(sys.argv)>2 else 1.0   # probe top z (mm): 1.0=real post, 6.95=connect to cells
fr4 = CSX.AddMaterial('FR4', epsilon=epsR, kappa=2*np.pi*f0*EPS0*epsR*tand)
gnd = CSX.AddMetal('gnd'); prs = CSX.AddMetal('prs'); via = CSX.AddMetal('via')
# dielectric slabs
fr4.AddBox([-SUB/2,-SUB/2,z_g],  [SUB/2,SUB/2,z_s1], priority=1)
fr4.AddBox([-SUB/2,-SUB/2,z_air],[SUB/2,SUB/2,z_s2], priority=1)
# ground sheet at z=0 (high priority so it is not swallowed by the coplanar fr-4 face)
gnd.AddBox([-SUB/2,-SUB/2,z_g],[SUB/2,SUB/2,z_g], priority=10)
# 96 prs cells at z=6.95 (high priority, coplanar with upper-fr-4 top face)
xe=set(); ye=set()
for (x0,y0,x1,y1,_,_) in cells:
    prs.AddBox([x0,y0,z_s2],[x1,y1,z_s2], priority=10)
    xe.update([x0,x1, 0.5*(x0+x1)]); ye.update([y0,y1, 0.5*(y0+y1)])   # edges + midline
# probe feed: lumped port across base gap, PEC via from gap-top up to PTOP
hp = 0.10; zgap = 0.12
port = FDTD.AddLumpedPort(1, 50.0, [fx-hp,fy-hp,z_g],[fx+hp,fy+hp,zgap],'z',1.0,priority=5,edges2grid='xy')
if PTOP > zgap:
    via.AddBox([fx-hp,fy-hp,zgap],[fx+hp,fy+hp,PTOP], priority=10)
xe.update([fx-hp,fx+hp]); ye.update([fy-hp,fy+hp])
P(f"probe top z={PTOP}mm (gap 0..{zgap}, via {zgap}..{PTOP})")
# optional driven patch at z=1mm (a 2d perfecte sheet in hfss would not export to step)
PATCH = float(sys.argv[4]) if len(sys.argv)>4 else 0.0   # resonant length Ly (mm); 0=off
if PATCH > 0:
    dp = CSX.AddMetal('dpatch'); Lpy=PATCH; Lpx=PATCH*0.85
    dp.AddBox([-Lpx/2,-Lpy/2,z_s1],[Lpx/2,Lpy/2,z_s1], priority=12)
    xe.update([-Lpx/2,Lpx/2,0.0]); ye.update([-Lpy/2,Lpy/2,0.0])
    P(f"driven patch {Lpx:.2f}x{Lpy:.2f}mm @z={z_s1}mm, probe offset y={fy:.2f}mm")

# mesh: cell edges (fine in aperture) + coarse margins; merge slivers
APFINE = float(sys.argv[3]) if len(sys.argv)>3 else 0.0   # aperture uniform fine step (mm); 0=off
def build(crit, lo, hi, tol=0.03):
    L = sorted(set([lo, hi] + list(crit)))
    out=[L[0]]
    for v in L[1:]:
        if v-out[-1] >= tol: out.append(v)     # drop near-duplicates (kills slivers)
    return out
ap = []
if APFINE>0:
    ap = list(np.arange(-3.6, 3.6001, APFINE))   # uniform fine grid over the cell aperture
xcrit = build(xe | set(ap) | {fx-hp,fx+hp}, -SUB/2-margin, SUB/2+margin, tol=0.02 if APFINE else 0.06)
ycrit = build(ye | set(ap) | {fy-hp,fy+hp}, -SUB/2-margin, SUB/2+margin, tol=0.02 if APFINE else 0.06)
g.AddLine('x', xcrit); g.AddLine('y', ycrit)
g.AddLine('z', [-margin*1.4, z_g, zgap, z_s1, z_air, z_s2, z_s2+margin*2.0])
g.SmoothMeshLines('x', 0.5, ratio=1.45); g.SmoothMeshLines('y', 0.5, ratio=1.45)
g.SmoothMeshLines('z', 0.45, ratio=1.45)
# enforce min cell size: kill any sliver < floor that would wreck the timestep
def dedupe(dirn, floor=0.035):
    L = list(g.GetLines(dirn)); keep=[L[0]]
    for v in L[1:-1]:
        if v-keep[-1] >= floor: keep.append(v)
    if L[-1]-keep[-1] >= floor: keep.append(L[-1])
    else: keep[-1]=L[-1]
    g.SetLines(dirn, keep)
for d in ('x','y','z'): dedupe(d)
nf2ff = FDTD.CreateNF2FFBox()

nx=len(g.GetLines('x')); ny=len(g.GetLines('y')); nz=len(g.GetLines('z'))
dxmin = min(np.diff(g.GetLines('x')).min(), np.diff(g.GetLines('y')).min(), np.diff(g.GetLines('z')).min())
P(f"mesh {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.2f} M ; min cell={dxmin*1000:.0f} um ; SUB={SUB}mm")
P("running FDTD ..."); FDTD.Run(sim, cleanup=True, verbose=0)

f = np.linspace(18e9,40e9,441); port.CalcPort(sim, f)
s11 = 20*np.log10(np.abs(port.uf_ref/port.uf_inc)); Zin = port.uf_tot/port.if_tot
i0 = np.argmin(np.abs(f-f0)); imin = np.argmin(s11)
P(f"S11@24={s11[i0]:.2f}dB  Zin@24={Zin[i0].real:.0f}{Zin[i0].imag:+.0f}j  bestS11={s11[imin]:.2f}dB@{f[imin]/1e9:.2f}GHz")
# scan broadside gain vs frequency to locate where this structure actually radiates
fg = np.arange(20e9, 30.01e9, 1e9)
th = np.linspace(-np.pi/2, np.pi/2, 91)
best = (-99, 0); gains=[]
for fk in fg:
    r = nf2ff.CalcNF2FF(sim, fk, th, np.array([0.0]), outfile=f'nf_{int(fk/1e9)}.h5')
    gmax = 10*np.log10(r.Dmax[0]); gains.append(gmax)
    ip = np.argmin(np.abs(f-fk))
    P(f"  f={fk/1e9:4.0f}GHz  S11={s11[ip]:6.2f}dB  Dmax={gmax:6.2f}dBi")
    if gmax>best[0]: best=(gmax, fk)
gains=np.array(gains)
P(f"==> PEAK broadside D = {best[0]:.2f} dBi @ {best[1]/1e9:.0f} GHz")
# clean principal-plane patterns at 24 GHz for the validation figure
thp = np.linspace(-np.pi/2, np.pi/2, 181)
rE = nf2ff.CalcNF2FF(sim, f0, thp, np.array([0.0]),     outfile='nf_E.h5')   # phi=0  (E-plane, xz)
rH = nf2ff.CalcNF2FF(sim, f0, thp, np.array([np.pi/2]), outfile='nf_H.h5')   # phi=90 (H-plane, yz)
gE = 10*np.log10(rE.E_norm[0][:,0]**2 / np.max(rE.E_norm[0][:,0]**2) * rE.Dmax[0])
gH = 10*np.log10(rH.E_norm[0][:,0]**2 / np.max(rH.E_norm[0][:,0]**2) * rH.Dmax[0])
P(f"broadside D@24 = {10*np.log10(rE.Dmax[0]):.2f} dBi")
np.savez(r"D:\实践三号“延安”\论文\fpc_result.npz", f=f, s11=s11, Zr=Zin.real, Zi=Zin.imag,
         peakD=float(best[0]), peakF=float(best[1]), SUB=SUB, fg=fg, gains=gains,
         theta=np.degrees(thp), gE=gE, gH=gH, D24=float(10*np.log10(rE.Dmax[0])),
         PATCH=PATCH, AIRGAP=AIRGAP)
P("saved fpc_result.npz")
