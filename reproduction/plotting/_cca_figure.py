"""RRXOR figure via the sampling-CCA subspace, evaluated honestly.
Panels: ground truth | supervised | sampling-CCA. Reports per-pos & state-COM on the enumeration
(full 36-state diamond) AND the consistent held-out-sampled per-pos (the method's own-distribution
score), plus the canonical-correlation spectrum for the elbow/d discussion.
"""
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10

t0=time.time(); tic=lambda m: print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
MODEL_DIR=Path(__file__).parent.parent/"models_paper1_sgd_ctx10"; FIG_DIR=Path(__file__).parent.parent/"figures"
device,d,L,N,H="cpu",5,10,80000,6

def sample(T,pi,N,L,seed):
    rng=np.random.default_rng(seed); V,ns=T.shape[0],T.shape[1]; st=rng.choice(ns,N,p=pi); s=np.empty((N,L),np.int64)
    for t in range(L):
        Px=np.stack([T[x][st].sum(1) for x in range(V)],1); Px=np.clip(Px,1e-12,None); Px/=Px.sum(1,keepdims=True)
        x=(rng.random(N)[:,None]<np.cumsum(Px,1)).argmax(1); s[:,t]=x
        tr=T[x,st]/T[x,st].sum(1,keepdims=True); st=(rng.random(N)[:,None]<np.cumsum(tr,1)).argmax(1)
    return s
def beliefs_for(seqs,T,pi):
    Nq,Lq=seqs.shape; ns=T.shape[1]; B=np.empty((Nq,Lq,ns)); b=np.tile(pi,(Nq,1))
    for t in range(Lq):
        nb=np.einsum('ni,nij->nj',b,T[seqs[:,t]]); nb/=nb.sum(1,keepdims=True); B[:,t]=nb; b=nb
    return B
def ffeat(fut,H,V):
    n=len(fut); fe=[]
    for k in range(1,H+1):
        idx=np.zeros(n,np.int64)
        for j in range(k): idx=idx*V+fut[:,j]
        oh=np.zeros((n,V**k)); oh[np.arange(n),idx]=1.0; fe.append(oh)
    return np.hstack(fe)

model=F10.load_model(MODEL_DIR/"rrxor_transformer.pt",device); T,pi=np.array(rrxor(0.5,0.5)),np.array([2,1,1,1,1])/6.0; V=T.shape[0]
hooks=[f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]; Dd=len(hooks)*model.cfg.d_model
seqs=sample(T,pi,N,L,0); bel=beliefs_for(seqs,T,pi)
acts=[]
for i in range(0,N,8192):
    with torch.no_grad():
        _,c=model.run_with_cache(torch.from_numpy(seqs[i:i+8192]),names_filter=lambda n:n in hooks)
    acts.append(np.concatenate([c[h].numpy() for h in hooks],-1))
acts=np.concatenate(acts,0); tic("acts+beliefs")

# ---- CCA subspace (unsupervised) on short prefixes (H=6) ----
trm=np.arange(N)%2==0
Xtr,Ftr,Ytr,Xte,Yte=[],[],[],[],[]
for t in range(L-H):
    Xtr.append(acts[trm,t]); Ftr.append(ffeat(seqs[trm,t+1:t+1+H],H,V)); Ytr.append(bel[trm,t])
    Xte.append(acts[~trm,t]); Yte.append(bel[~trm,t])
Xtr=np.vstack(Xtr); Ftr=np.vstack(Ftr); Ytr=np.vstack(Ytr); Xte=np.vstack(Xte); Yte=np.vstack(Yte)
Xc=Xtr-Xtr.mean(0); Fc=Ftr-Ftr.mean(0); n=len(Xtr)
ex,Ux=np.linalg.eigh(Xc.T@Xc/n+1e-1*np.eye(Dd)); Wx=Ux@np.diag(ex**-0.5)@Ux.T
ef,Uf=np.linalg.eigh(Fc.T@Fc/n+1e-3*np.eye(Ftr.shape[1])); Wf=Uf@np.diag(np.clip(ef,1e-12,None)**-0.5)@Uf.T
U2,cc,_=np.linalg.svd(Wx@(Xc.T@Fc/n)@Wf,full_matrices=False); B=Wx@U2[:,:d]
cons=LinearRegression().fit(Xtr@B,Ytr).score(Xte@B,Yte)
tic(f"CCA fit. canonical corr = {np.round(cc[:6],3)};  consistent held-out per-pos R2 = {cons:.3f}")

# ---- enumeration eval (full 36-state diamond), decode re-fit on enumeration (consistent) ----
B36,index=F10.msp_states(); es,ebel,eidx=F10.enumerate_inputs(F10.N_CTX,index); ea=F10.collect_activations(model,es,device,hooks)
Xf=np.concatenate([ea[h] for h in hooks],-1).reshape(-1,Dd); Yb=ebel.reshape(-1,B36.shape[1]); fidx=eidx.reshape(-1)
cnt=np.bincount(fidx,minlength=len(B36)); wgt=1.0/np.clip(cnt[fidx],1,None); wgt/=wgt.mean()
def com_r2(S):
    com=np.array([S[fidx==k].mean(0) for k in range(len(B36)) if (fidx==k).any()])
    bl=np.array([Yb[fidx==k][0] for k in range(len(B36)) if (fidx==k).any()])
    return LinearRegression().fit(com,bl).score(com,bl)
reg=LinearRegression().fit(Xf@B,Yb,sample_weight=wgt); pp=reg.score(Xf@B,Yb,sample_weight=wgt); sc=com_r2(Xf@B)
sup=LinearRegression().fit(Xf,Yb,sample_weight=wgt)
print(f"\nFULL 36-state diamond (enumeration):")
print(f"  supervised      per-pos {sup.score(Xf,Yb,sample_weight=wgt):.3f}  state-COM {com_r2(Xf):.3f}")
print(f"  obs-only OOM    per-pos 0.661           state-COM 0.911 (reference)")
print(f"  sampling-CCA    per-pos {pp:.3f}  state-COM {sc:.3f}")

# ---- figure ----
pca=PCA(4).fit(B36); tgt=pca.transform(B36)[:,[0,2]]; colors=np.array(F10.distinct_colors(len(B36)))
def draw(ax,xy,title,scatter=True):
    if scatter:
        ax.scatter(xy[:,0],xy[:,1],s=2,c=colors[fidx],alpha=.25,edgecolors="none")
        for k in range(len(B36)):
            mk=fidx==k
            if mk.any(): ax.scatter(*xy[mk].mean(0),s=90,c=[colors[k]],edgecolors="black",linewidths=.6,zorder=3)
        ax.scatter(tgt[:,0],tgt[:,1],s=150,facecolors="none",edgecolors="0.6",linewidths=1.0)
    else: ax.scatter(xy[:,0],xy[:,1],c=colors,s=90,edgecolors="white",linewidths=.5)
    ax.set_title(title); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 3"); ax.set_aspect("equal"); ax.spines[["top","right"]].set_visible(False)
    for setl,v in ((ax.set_xlim,tgt[:,0]),(ax.set_ylim,tgt[:,1])): lo,hi=v.min(),v.max(); p=.05*(hi-lo); setl(lo-p,hi+p)
fig,ax=plt.subplots(1,3,figsize=(16.5,5.6))
draw(ax[0],tgt,"Ground truth belief geometry",scatter=False)
draw(ax[1],pca.transform(sup.predict(Xf))[:,[0,2]],f"Supervised: full residual stream\ndecode R$^2$={sup.score(Xf,Yb,sample_weight=wgt):.2f}")
draw(ax[2],pca.transform(reg.predict(Xf@B))[:,[0,2]],f"Sampling-CCA subspace (5-D, H={H})\nper-pos R$^2$={pp:.2f}  state-COM={sc:.2f}")
out=FIG_DIR/"rrxor_sampling_cca.png"; fig.tight_layout(); fig.savefig(out,dpi=200,bbox_inches="tight",facecolor="white")
print(f"saved -> {out}"); tic("done")
