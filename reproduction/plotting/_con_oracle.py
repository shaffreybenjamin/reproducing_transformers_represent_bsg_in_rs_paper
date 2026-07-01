"""Is belief IN the controllability subspace (mis-ranked, rescuable) or absent (lossy)?
Oracle: project the TRUE (supervised) belief subspace onto the controllability-CCA / gram-mean
top-K dirs; report captured energy (vs random K/D) and decode with d=5,10,20,40."""
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10

t0 = time.time(); tic = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
MODEL_DIR = Path(__file__).parent.parent / "models_paper1_sgd_ctx10"
device, d, L, N = "cpu", 5, 10, 80000
K_GRAM = 4

def sample_rrxor(T, pi, N, L, seed=1):
    rng = np.random.default_rng(seed); V, ns = T.shape[0], T.shape[1]
    st = rng.choice(ns, N, p=pi); seqs = np.empty((N, L), np.int64)
    for t in range(L):
        Px = np.stack([T[x][st].sum(1) for x in range(V)], 1); Px = np.clip(Px,1e-12,None); Px/=Px.sum(1,keepdims=True)
        x = (rng.random(N)[:,None] < np.cumsum(Px,1)).argmax(1); seqs[:,t]=x
        tr = T[x,st]/T[x,st].sum(1,keepdims=True); st = (rng.random(N)[:,None]<np.cumsum(tr,1)).argmax(1)
    return seqs

model = F10.load_model(MODEL_DIR / "rrxor_transformer.pt", device)
T, pi = np.array(rrxor(0.5,0.5)), np.array([2,1,1,1,1])/6.0; V=T.shape[0]
hooks=[f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]; Dd=len(hooks)*model.cfg.d_model
seqs = sample_rrxor(T,pi,N,L)
acts=[]
for i in range(0,N,8192):
    with torch.no_grad():
        _,c=model.run_with_cache(torch.from_numpy(seqs[i:i+8192]),names_filter=lambda n:n in hooks)
    acts.append(np.concatenate([c[h].numpy() for h in hooks],-1))
acts=np.concatenate(acts,0)
tic("acts ready")

B36,index=F10.msp_states(); eseqs,beliefs,idx=F10.enumerate_inputs(F10.N_CTX,index)
eacts=F10.collect_activations(model,eseqs,device,hooks)
Xf=np.concatenate([eacts[h] for h in hooks],-1).reshape(-1,Dd)
Yb=beliefs.reshape(-1,B36.shape[1]); fidx=idx.reshape(-1)
counts=np.bincount(fidx,minlength=len(B36)); wgt=1.0/np.clip(counts[fidx],1,None); wgt/=wgt.mean()
def com_r2(S):
    com=np.array([S[fidx==k].mean(0) for k in range(len(B36)) if (fidx==k).any()])
    bel=np.array([Yb[fidx==k][0] for k in range(len(B36)) if (fidx==k).any()])
    return LinearRegression().fit(com,bel).score(com,bel)
def decode_K(B,Kd):
    S=Xf@B[:,:Kd]; return LinearRegression().fit(S,Yb,sample_weight=wgt).score(S,Yb,sample_weight=wgt)
Btrue=np.linalg.qr(LinearRegression().fit(Xf,Yb,sample_weight=wgt).coef_.T)[0][:,:d]

# build controllability-CCA full canonical basis + gram-mean basis (k=K_GRAM)
Xs,gids=[],[]
for t in range(K_GRAM-1,L):
    Xs.append(acts[:,t,:]);
    g=np.zeros(N,np.int64)
    for j in range(K_GRAM): g=g*V+seqs[:,t-K_GRAM+1+j]
    gids.append(g)
X=np.vstack(Xs); g=np.concatenate(gids); ng=V**K_GRAM
F=np.zeros((len(g),ng)); F[np.arange(len(g)),g]=1.0
Xc=X-X.mean(0); Fc=F-F.mean(0); n=len(X)
Cxx=Xc.T@Xc/n+1e-1*np.eye(Dd); Cff=Fc.T@Fc/n+1e-3*np.eye(ng)
ex,Ux=np.linalg.eigh(Cxx); Wx=Ux@np.diag(ex**-0.5)@Ux.T
ef,Uf=np.linalg.eigh(Cff); Wf=Uf@np.diag(np.clip(ef,1e-12,None)**-0.5)@Uf.T
U2,s2,_=np.linalg.svd(Wx@(Xc.T@Fc/n)@Wf,full_matrices=False)
Bcca=Wx@U2                                  # (Dd, ng) full canonical activation dirs
Bcca=Bcca/np.linalg.norm(Bcca,axis=0,keepdims=True)
# gram-mean basis
sums=np.zeros((ng,Dd)); cnt=np.zeros(ng); np.add.at(sums,g,X); np.add.at(cnt,g,1)
m=cnt>0; gm=sums[m]/cnt[m][:,None]; w=cnt[m]/cnt[m].sum()
Xcm=(gm-(w[:,None]*gm).sum(0))*np.sqrt(w)[:,None]
Bgm=np.linalg.svd(Xcm,full_matrices=False)[2].T   # (Dd, r)
tic("bases ready")

def energy(B,Kk):
    Q=np.linalg.qr(B[:,:Kk])[0]; return float(np.sum((Q.T@Btrue)**2)/d)
print(f"\nbelief-energy of TOP-K dirs (random baseline K/{Dd}):")
print(f"{'K':>4}{'  con-CCA':>11}{'  gram-mean':>12}{'  random':>10}")
for Kk in [5,10,20,40]:
    print(f"{Kk:>4}{energy(Bcca,Kk):>11.3f}{energy(Bgm,Kk):>12.3f}{Kk/Dd:>10.3f}")
print(f"\ndecode with top-K dirs (does belief emerge with more dims?):")
print(f"{'K':>4}{'  con-CCA':>11}{'  gram-mean':>12}")
for Kk in [5,10,20,40]:
    print(f"{Kk:>4}{decode_K(Bcca,Kk):>11.3f}{decode_K(Bgm,Kk):>12.3f}")
print("\nref: obs-only 0.661, supervised 0.911")
tic("done")
