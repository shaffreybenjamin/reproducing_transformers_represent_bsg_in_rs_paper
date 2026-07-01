"""Does controllability ADD what observability lacks? Union the operator-OOM observability subspace
with the controllability-CCA subspace (scale-matched orthonormal concat) and decode. If the weak mode
lives in controllability but not observability, the union exceeds both -> controllability helps."""
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10
import fig14_observable_oom as F14

t0=time.time(); tic=lambda m: print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
SCR=Path("/tmp/claude-1000/-home-ben-mathematics-thesis-project-reproducing-transformers-represent-bsg-in-rs-paper/1bd34b5c-84ca-4b50-b39b-bbdaf648abc2/scratchpad")
MODEL_DIR=Path(__file__).parent.parent/"models_paper1_sgd_ctx10"; device,d,L,N="cpu",5,10,80000; K_GRAM=4

def sample(T,pi,N,L,seed=2):
    rng=np.random.default_rng(seed); V,ns=T.shape[0],T.shape[1]; st=rng.choice(ns,N,p=pi); s=np.empty((N,L),np.int64)
    for t in range(L):
        Px=np.stack([T[x][st].sum(1) for x in range(V)],1); Px=np.clip(Px,1e-12,None); Px/=Px.sum(1,keepdims=True)
        x=(rng.random(N)[:,None]<np.cumsum(Px,1)).argmax(1); s[:,t]=x
        tr=T[x,st]/T[x,st].sum(1,keepdims=True); st=(rng.random(N)[:,None]<np.cumsum(tr,1)).argmax(1)
    return s

model=F10.load_model(MODEL_DIR/"rrxor_transformer.pt",device); T,pi=np.array(rrxor(0.5,0.5)),np.array([2,1,1,1,1])/6.0; V=T.shape[0]
hooks=[f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]; Dd=len(hooks)*model.cfg.d_model
z=np.load(SCR/"rrxor_oom_order3_ops.npz"); C,Gs=z["C"],[g for g in z["Gs"]]
cols,fr=[C],[C]
for _ in range(5):
    nx=[Gx@f for Gx in Gs for f in fr]; cols+=nx; fr=nx
U_O=np.linalg.svd(np.hstack(cols),full_matrices=False)[0]                      # observability OOM basis

seqs=sample(T,pi,N,L); acts=[]
for i in range(0,N,8192):
    with torch.no_grad():
        _,c=model.run_with_cache(torch.from_numpy(seqs[i:i+8192]),names_filter=lambda n:n in hooks)
    acts.append(np.concatenate([c[h].numpy() for h in hooks],-1))
acts=np.concatenate(acts,0)
# controllability-CCA basis (k=K_GRAM, long prefixes)
Xs,gids=[],[]
for t in range(K_GRAM-1,L):
    Xs.append(acts[:,t,:]); gg=np.zeros(N,np.int64)
    for j in range(K_GRAM): gg=gg*V+seqs[:,t-K_GRAM+1+j]
    gids.append(gg)
X=np.vstack(Xs); g=np.concatenate(gids); ng=V**K_GRAM
F=np.zeros((len(g),ng)); F[np.arange(len(g)),g]=1.0
Xc=X-X.mean(0); Fc=F-F.mean(0); n=len(X)
ex,Ux=np.linalg.eigh(Xc.T@Xc/n+1e-1*np.eye(Dd)); Wx=Ux@np.diag(ex**-0.5)@Ux.T
ef,Uf=np.linalg.eigh(Fc.T@Fc/n+1e-3*np.eye(ng)); Wf=Uf@np.diag(np.clip(ef,1e-12,None)**-0.5)@Uf.T
U2=np.linalg.svd(Wx@(Xc.T@Fc/n)@Wf,full_matrices=False)[0]; Bcca=Wx@U2; Bcca/=np.linalg.norm(Bcca,axis=0,keepdims=True)
tic("bases ready")

B36,index=F10.msp_states(); es,bel,idx=F10.enumerate_inputs(F10.N_CTX,index); ea=F10.collect_activations(model,es,device,hooks)
Xf=np.concatenate([ea[h] for h in hooks],-1).reshape(-1,Dd); Yb=bel.reshape(-1,B36.shape[1]); fidx=idx.reshape(-1)
cnt=np.bincount(fidx,minlength=len(B36)); wgt=1.0/np.clip(cnt[fidx],1,None); wgt/=wgt.mean()
def dec(B,Kd):
    S=Xf@B[:,:Kd]; r=LinearRegression().fit(S,Yb,sample_weight=wgt);
    com=np.array([(Xf@B[:,:Kd])[fidx==k].mean(0) for k in range(len(B36)) if (fidx==k).any()])
    belc=np.array([Yb[fidx==k][0] for k in range(len(B36)) if (fidx==k).any()])
    return r.score(S,Yb,sample_weight=wgt), LinearRegression().fit(com,belc).score(com,belc)

print(f"\n{'subspace':<34}{'per-pos':>9}{'state-COM':>11}")
print(f"{'observability OOM (top-5)':<34}{dec(U_O,5)[0]:>9.3f}{dec(U_O,5)[1]:>11.3f}")
for Ko in [5,8]:
    for Kc in [5,10,20]:
        Uh=np.linalg.svd(np.hstack([U_O[:,:Ko],Bcca[:,:Kc]]),full_matrices=False)[0]
        a,b=dec(Uh,5)
        print(f"{f'union(obs top-{Ko}, con-CCA top-{Kc})':<34}{a:>9.3f}{b:>11.3f}")
print("ref: supervised 0.911")
tic("done")
