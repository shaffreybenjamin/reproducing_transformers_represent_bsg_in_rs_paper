"""Test the two criticisms:
 (1) train/eval mismatch: fit obs-CCA on sampled acts, decode on HELD-OUT SAMPLED acts with their
     TRUE beliefs (consistent distribution) vs on the enumeration. If consistent >> enumeration, the
     mismatch was depressing CCA.
 (2) gram pooling conflates belief: compute exact WITHIN-gram belief variance (does last-k determine
     belief at the pooled positions?).
True beliefs computed by forward filtering the sampled sequences (no labels used in the estimator;
used only to score, exactly like supervised).
"""
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.transition_matrices import rrxor
import fig10_rrxor_representation as F10

t0=time.time(); tic=lambda m: print(f"[{time.time()-t0:6.1f}s] {m}",flush=True)
MODEL_DIR=Path(__file__).parent.parent/"models_paper1_sgd_ctx10"; device,d,L,N="cpu",5,10,80000

def sample(T,pi,N,L,seed):
    rng=np.random.default_rng(seed); V,ns=T.shape[0],T.shape[1]; st=rng.choice(ns,N,p=pi); s=np.empty((N,L),np.int64)
    for t in range(L):
        Px=np.stack([T[x][st].sum(1) for x in range(V)],1); Px=np.clip(Px,1e-12,None); Px/=Px.sum(1,keepdims=True)
        x=(rng.random(N)[:,None]<np.cumsum(Px,1)).argmax(1); s[:,t]=x
        tr=T[x,st]/T[x,st].sum(1,keepdims=True); st=(rng.random(N)[:,None]<np.cumsum(tr,1)).argmax(1)
    return s

def beliefs_for(seqs,T,pi):
    N,Lq=seqs.shape; ns=T.shape[1]; B=np.empty((N,Lq,ns)); b=np.tile(pi,(N,1))
    for t in range(Lq):
        nb=np.einsum('ni,nij->nj',b,T[seqs[:,t]]); nb/=nb.sum(1,keepdims=True); B[:,t]=nb; b=nb
    return B

model=F10.load_model(MODEL_DIR/"rrxor_transformer.pt",device); T,pi=np.array(rrxor(0.5,0.5)),np.array([2,1,1,1,1])/6.0; V=T.shape[0]
hooks=[f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]; Dd=len(hooks)*model.cfg.d_model
seqs=sample(T,pi,N,L,0); bel=beliefs_for(seqs,T,pi)
acts=[]
for i in range(0,N,8192):
    with torch.no_grad():
        _,c=model.run_with_cache(torch.from_numpy(seqs[i:i+8192]),names_filter=lambda n:n in hooks)
    acts.append(np.concatenate([c[h].numpy() for h in hooks],-1))
acts=np.concatenate(acts,0)
tic("acts+beliefs ready")
tr_mask=np.arange(N)%2==0  # train/test split by sequence

# enumeration eval (the original)
B36,index=F10.msp_states(); es,ebel,eidx=F10.enumerate_inputs(F10.N_CTX,index); ea=F10.collect_activations(model,es,device,hooks)
Xf=np.concatenate([ea[h] for h in hooks],-1).reshape(-1,Dd); Yb=ebel.reshape(-1,B36.shape[1])

def future_feat(fut,H,V):
    n=len(fut); fe=[]
    for k in range(1,H+1):
        idx=np.zeros(n,np.int64)
        for j in range(k): idx=idx*V+fut[:,j]
        oh=np.zeros((n,V**k)); oh[np.arange(n),idx]=1.0; fe.append(oh)
    return np.hstack(fe)

def cca_fit(Xtr,Ftr):
    Xc=Xtr-Xtr.mean(0); Fc=Ftr-Ftr.mean(0); n=len(Xtr)
    ex,Ux=np.linalg.eigh(Xc.T@Xc/n+1e-1*np.eye(Dd)); Wx=Ux@np.diag(ex**-0.5)@Ux.T
    ef,Uf=np.linalg.eigh(Fc.T@Fc/n+1e-3*np.eye(Ftr.shape[1])); Wf=Uf@np.diag(np.clip(ef,1e-12,None)**-0.5)@Uf.T
    U2,s2,_=np.linalg.svd(Wx@(Xc.T@Fc/n)@Wf,full_matrices=False)
    return Wx@U2[:,:d], s2

def r2(Xa,Ya,B,Xb,Yb_): # fit decode on (Xa,Ya) projected, eval on (Xb,Yb_)
    reg=LinearRegression().fit(Xa@B,Ya); return reg.score(Xb@B,Yb_)

print(f"\n{'H':>2} {'consistent (sampled test)':>26} {'enumeration':>13} {'supervised(consistent)':>22}")
for H in [4,5,6]:
    Xtr,Ftr,Ytr,Xte,Yte=[],[],[],[],[]
    for t in range(L-H):
        Xtr.append(acts[tr_mask,t]); Ftr.append(future_feat(seqs[tr_mask,t+1:t+1+H],H,V)); Ytr.append(bel[tr_mask,t])
        Xte.append(acts[~tr_mask,t]); Yte.append(bel[~tr_mask,t])
    Xtr=np.vstack(Xtr); Ftr=np.vstack(Ftr); Ytr=np.vstack(Ytr); Xte=np.vstack(Xte); Yte=np.vstack(Yte)
    B,s=cca_fit(Xtr,Ftr)
    cons=r2(Xtr,Ytr,B,Xte,Yte)                                   # consistent sampled eval
    enum=r2(Xtr,Ytr,B,Xf,Yb)                                     # enumeration eval (original style)
    supc=LinearRegression().fit(Xtr,Ytr).score(Xte,Yte)         # supervised on consistent sampled
    print(f"{H:>2} {cons:>26.3f} {enum:>13.3f} {supc:>22.3f}   cc={np.round(s[:5],3)}",flush=True)

# criticism 2: within-gram belief variance (exact, on sampled positions used for controllability)
print("\nwithin-gram belief dispersion (does last-k-gram determine belief? small = yes):")
for k in [4,6]:
    Xs,gid,Bs=[],[],[]
    for t in range(k-1,L):
        g=np.zeros(N,np.int64)
        for j in range(k): g=g*V+seqs[:,t-k+1+j]
        gid.append(g); Bs.append(bel[:,t])
    g=np.concatenate(gid); Bb=np.vstack(Bs); ng=V**k
    tot_var=Bb.var(0).sum()
    sums=np.zeros((ng,Bb.shape[1])); cnt=np.zeros(ng); np.add.at(sums,g,Bb); np.add.at(cnt,g,1)
    m=cnt>0; gm=sums[m]/cnt[m][:,None]
    between=((cnt[m][:,None]*(gm-Bb.mean(0))**2).sum(0)/cnt[m].sum()).sum()
    within=tot_var-between
    print(f"  k={k}: within-gram belief var fraction = {within/tot_var:.3f}  (0=gram fully determines belief)")
tic("done")
