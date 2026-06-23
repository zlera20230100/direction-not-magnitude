# -*- coding: utf-8 -*-
# Note: this script needs the project PINN framework (pinn_model.py, config.py, main.py, visualizer.py),
# which is not bundled here; the figures in this repo reproduce from the provided .npz without it.
# Multi-seed stability of the ZonePINN per-zone Jacobian. Retrains the K=6-zone
# geometry-conditioned PINN (same architecture/loss as zones.py) for several seeds, then
# per seed recomputes relf, rela, ||J_feed||, ||J_ap||, ratio and the aperture sign pattern.
# Reports mean+-spread and whether the sign pattern is seed-stable. Saves zones_multiseed.npz.
# Usage: python zones_multiseed.py [n_iters=3000] [seeds=0,1,2]
import os, sys, time, numpy as np, torch
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch.nn as nn
CODE = r"D:\实践三号“延安”\实践三号“延安”\代码"
PAPER = r"D:\实践三号“延安”\论文"
sys.path.insert(0, CODE); os.chdir(CODE)
import logging; logging.disable(logging.INFO)
from config import FREQUENCY, EXCITATION_CONFIG, Z_INTERFACE, device
from main import setup_environment, load_antenna_geometry
from pinn_model import NearFieldNet, FarFieldNet, MaxwellEquationsMicrostrip

N_ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEEDS = [int(s) for s in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0','1','2'])]
K = 6
G_LO, G_HI = 0.80, 1.20
print(f"device={device}  N_ITERS={N_ITERS}  seeds={SEEDS}  K={K}", flush=True)

setup_environment()
mesh = load_antenna_geometry(substrate_path="../模型/1.1.STL", patch_path="../模型/1-1.stl", headless=True)
points, entities, scale = mesh.sample_points()
bres = mesh.sample_boundary_points(); b_points, b_ent, b_normals = bres[:3]
pp = mesh.patch_params
def T(a): return torch.tensor(a, dtype=torch.float32, device=device)

cx = np.array([c['cx'] for c in pp])*1e-3; cy = np.array([c['cy'] for c in pp])*1e-3
loc = np.array(EXCITATION_CONFIG['location'])
r_cell = np.sqrt((cx-loc[0])**2 + (cy-loc[1])**2)
order = np.argsort(r_cell); zone = np.zeros(len(pp), dtype=np.int64)
for k, idx in enumerate(np.array_split(order, K)): zone[idx] = k
print(f"scale={scale:.4f} N_cells={len(pp)} cells/zone={[int((zone==k).sum()) for k in range(K)]}", flush=True)

ALL_X=T(points[:,0:1]*scale); ALL_Y=T(points[:,1:2]*scale); ALL_Z=T(points[:,2:3]*scale)
mask_pec=(b_ent=='ground_plane')|(b_ent=='patch'); mask_abc=(b_ent=='air_box')|(b_ent=='substrate')
def pack(m):
    idx=np.where(m)[0]
    if len(idx)==0: return None
    return dict(x=T(b_points[idx,0:1]*scale),y=T(b_points[idx,1:2]*scale),z=T(b_points[idx,2:3]*scale),
                nx=T(b_normals[idx,0:1]),ny=T(b_normals[idx,1:2]),nz=T(b_normals[idx,2:3]))
PEC=pack(mask_pec); ABC=pack(mask_abc)

class ZonePINN(nn.Module):
    def __init__(self, scale, pp, zone, K):
        super().__init__()
        self.coord_scale_k0=scale; self.freq_base=24.0e9; self.output_scale=200000.0; self.K=K
        self.near_net=NearFieldNet(4+K); self.far_net=FarFieldNet(4+K, coord_scale_k0=scale)
        self.register_buffer('probe_loc', torch.tensor(EXCITATION_CONFIG['location']).float())
        self.blending_sigma_phys=0.003
        self.register_buffer('cx', torch.tensor([c['cx'] for c in pp]).float()*1e-3)
        self.register_buffer('cy', torch.tensor([c['cy'] for c in pp]).float()*1e-3)
        self.register_buffer('Lc', torch.tensor([c['L'] for c in pp]).float()*1e-3)
        self.register_buffer('Wc', torch.tensor([c['W'] for c in pp]).float()*1e-3)
        self.register_buffer('zone', torch.tensor(zone, dtype=torch.long))
        self.mask_sharp=2.0e6
        self._g=torch.ones(K, device=device)
    def set_g(self, g): self._g = g if torch.is_tensor(g) else torch.tensor(g, dtype=torch.float32, device=device)
    def forward(self, x, y, z, freq):
        f_norm=freq/self.freq_base
        gcols=self._g.view(1,-1).expand(x.shape[0], self.K)
        inputs=torch.cat([x,y,z,f_norm,gcols], dim=1)
        E_near=self.near_net(inputs); E_far=self.far_net(inputs)
        s=self.coord_scale_k0
        px,py,pz=self.probe_loc[0]*s,self.probe_loc[1]*s,self.probe_loc[2]*s; sig=self.blending_sigma_phys*s
        W=torch.exp(-((x-px)**2+(y-py)**2+(z-pz)**2)/(2*sig**2+1e-30))
        return (W*E_near+(1-W)*E_far)*self.output_scale
    def get_patch_mask(self, x_phys, y_phys):
        s_cell=self._g[self.zone]
        mx=torch.sigmoid(self.mask_sharp*((self.Lc*s_cell)/2.0 - torch.abs(x_phys-self.cx)))
        my=torch.sigmoid(self.mask_sharp*((self.Wc*s_cell)/2.0 - torch.abs(y_phys-self.cy)))
        return torch.clamp((mx*my).sum(1,keepdim=True),0.0,1.0)

rad=EXCITATION_CONFIG['probe_radius']; n_src=4000
zmin=points[entities=='substrate'][:,2].min(); zmax=points[entities=='substrate'][:,2].max()
z_prs=Z_INTERFACE['sub2_top']; BATCH=4096

def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    r=rad*np.sqrt(np.random.rand(n_src,1)); th=2*np.pi*np.random.rand(n_src,1)
    SRC_X=T((loc[0]+r*np.cos(th))*scale); SRC_Y=T((loc[1]+r*np.sin(th))*scale); SRC_Z=T(np.random.uniform(zmin,zmax,(n_src,1))*scale)
    model=ZonePINN(scale,pp,zone,K).to(device)
    eq=MaxwellEquationsMicrostrip(FREQUENCY); eq.set_coord_scale(scale)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    cxmin,cxmax=model.cx.min().item(),model.cx.max().item(); cymin,cymax=model.cy.min().item(),model.cy.max().item()
    t0=time.time(); last_mpec=None
    for it in range(1,N_ITERS+1):
        model.train(); opt.zero_grad()
        g=torch.tensor(np.random.uniform(G_LO,G_HI,K), dtype=torch.float32, device=device); model.set_g(g)
        ia=torch.randint(0,ALL_X.shape[0],(BATCH,),device=device)
        xa=ALL_X[ia].detach().requires_grad_(True); ya=ALL_Y[ia].detach().requires_grad_(True); za=ALL_Z[ia].detach().requires_grad_(True)
        fa=torch.full_like(xa,FREQUENCY)
        l_all,_=eq.compute_physics_residuals(model,xa,ya,za,fa); l_div=eq.compute_divergence_residual(model,xa,ya,za,fa)
        isrc=torch.randint(0,SRC_X.shape[0],(1024,),device=device)
        xs=SRC_X[isrc].detach().requires_grad_(True); ys=SRC_Y[isrc].detach().requires_grad_(True); zs=SRC_Z[isrc].detach().requires_grad_(True)
        l_src,_=eq.compute_physics_residuals(model,xs,ys,zs,torch.full_like(xs,FREQUENCY))
        ip=torch.randint(0,PEC['x'].shape[0],(2048,),device=device)
        l_pec=eq.compute_pec_residuals(model,PEC['x'][ip],PEC['y'][ip],PEC['z'][ip],torch.full_like(PEC['x'][ip],FREQUENCY),PEC['nx'][ip],PEC['ny'][ip],PEC['nz'][ip])
        ib=torch.randint(0,ABC['x'].shape[0],(2048,),device=device)
        xb=ABC['x'][ib].detach().requires_grad_(True); yb=ABC['y'][ib].detach().requires_grad_(True); zb=ABC['z'][ib].detach().requires_grad_(True)
        l_abc=eq.compute_boundary_residuals(model,xb,yb,zb,torch.full_like(xb,FREQUENCY),ABC['nx'][ib],ABC['ny'][ib],ABC['nz'][ib])
        Np=2048; pad=0.0005
        xp=(torch.rand(Np,1,device=device)*(cxmax-cxmin+2*pad)+cxmin-pad)*scale
        yp=(torch.rand(Np,1,device=device)*(cymax-cymin+2*pad)+cymin-pad)*scale
        zp=torch.full_like(xp,z_prs*scale)
        Ep=model(xp,yp,zp,torch.full_like(xp,FREQUENCY)); Er_p,Ei_p=Ep[:,0:3],Ep[:,3:6]
        mask=model.get_patch_mask(xp/scale,yp/scale)
        E_tan_sq=(Er_p[:,0:1]**2+Er_p[:,1:2]**2+Ei_p[:,0:1]**2+Ei_p[:,1:2]**2)/(model.output_scale**2)
        l_mpec=torch.mean(E_tan_sq*mask)
        total=l_all*1.0+l_src*20.0+l_pec*1.0+l_abc*1.0+l_mpec*10.0+l_div*1.0
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        last_mpec=float(l_mpec)
        if it%max(1,N_ITERS//5)==0 or it==1:
            print(f"  seed{seed} it {it:5d}/{N_ITERS} tot {float(total):.2e} all {float(l_all):.2e} "
                  f"src {float(l_src):.2e} mpec {float(l_mpec):.2e} {(time.time()-t0)/it*1000:.0f}ms/it", flush=True)
    # per-zone Jacobian
    model.eval()
    nft=4096
    rr=torch.sqrt(torch.tensor(0.00015**2,device=device)+(0.00025**2-0.00015**2)*torch.rand(nft,1,device=device))
    ang=2*np.pi*torch.rand(nft,1,device=device)
    feed_xy=torch.cat([(loc[0]+rr*torch.cos(ang))*scale,(loc[1]+rr*torch.sin(ang))*scale],1)
    ng=48; gx=torch.linspace(cxmin,cxmax,ng,device=device)*scale; gy=torch.linspace(cymin,cymax,ng,device=device)*scale
    GX,GY=torch.meshgrid(gx,gy,indexing='ij'); ap_xy=torch.stack([GX.reshape(-1),GY.reshape(-1)],1)
    def jac(points_xy, zval):
        gvec=torch.ones(K,device=device,requires_grad=True); model.set_g(gvec)
        X=points_xy[:,0:1]; Y=points_xy[:,1:2]; Zc=torch.full_like(X,zval*scale)
        E=model(X,Y,Zc,torch.full_like(X,FREQUENCY)); Q=(E[:,2]**2+E[:,5]**2).mean()
        gg=torch.autograd.grad(Q,gvec)[0].detach().cpu().numpy(); Qv=Q.item()
        return Qv, gg/(Qv+1e-30)
    qf,relf=jac(feed_xy, EXCITATION_CONFIG['location'][2]+0.0005)
    qa,rela=jac(ap_xy, z_prs)
    return dict(relf=relf, rela=rela, qf=qf, qa=qa, last_mpec=last_mpec, sec=time.time()-t0)

all_relf=[]; all_rela=[]; out={}
for s in SEEDS:
    print(f"\n===== SEED {s} =====", flush=True)
    r=train_one(s)
    all_relf.append(r['relf']); all_rela.append(r['rela'])
    nf=np.linalg.norm(r['relf']); na=np.linalg.norm(r['rela'])
    sp=''.join('+' if v>0 else '-' for v in r['rela'])
    print(f"  seed{s}: ||J_feed||={nf:.3f} ||J_ap||={na:.3f} ratio(ap/feed)={na/nf:.2f} "
          f"ap-sign={sp}  ({r['sec']:.0f}s)", flush=True)
    out[f'seed{s}_relf']=r['relf']; out[f'seed{s}_rela']=r['rela']
    out[f'seed{s}_qf']=r['qf']; out[f'seed{s}_qa']=r['qa']

all_relf=np.array(all_relf); all_rela=np.array(all_rela)
nfeed=np.linalg.norm(all_relf,axis=1); nap=np.linalg.norm(all_rela,axis=1); ratio=nap/nfeed
signs=np.array([['+' if v>0 else '-' for v in row] for row in all_rela])
sign_stable=all((signs[:,k]==signs[0,k]).all() for k in range(K))

print("\n===== TASK B: MULTI-SEED ZonePINN JACOBIAN STABILITY =====", flush=True)
print(f"seeds={SEEDS}  iters={N_ITERS}", flush=True)
print(f"||J_feed||  per seed = {np.round(nfeed,3)}  mean={nfeed.mean():.3f} +- {nfeed.std():.3f}", flush=True)
print(f"||J_ap||    per seed = {np.round(nap,3)}    mean={nap.mean():.3f} +- {nap.std():.3f}", flush=True)
print(f"ratio ap/feed       = {np.round(ratio,2)}   mean={ratio.mean():.2f} +- {ratio.std():.2f}", flush=True)
print("aperture sign patterns per seed:")
for i,s in enumerate(SEEDS): print(f"   seed{s}: {''.join(signs[i])}", flush=True)
print(f"aperture sign pattern seed-STABLE: {sign_stable}  (smoke ref was +,+,-,-,+,-)", flush=True)
print("per-zone rela mean+-spread:")
for k in range(K):
    print(f"   zone{k}: {all_rela[:,k].mean():+.3f} +- {all_rela[:,k].std():.3f}  "
          f"signs={''.join(signs[:,k])}", flush=True)

np.savez(os.path.join(PAPER,"zones_multiseed.npz"),
         seeds=np.array(SEEDS), n_iters=N_ITERS, zone=zone,
         relf=all_relf, rela=all_rela,
         norm_feed=nfeed, norm_ap=nap, ratio=ratio,
         sign_stable=sign_stable, **out)
print("\nsaved zones_multiseed.npz", flush=True)
