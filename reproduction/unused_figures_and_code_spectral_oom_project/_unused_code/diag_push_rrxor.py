"""Push RRXOR raw belief-geometry recovery past the 0.635 ceiling (unsupervised).

Ceiling diagnosis: belief is PERFECTLY linearly present (supervised 35-COM -> belief = 1.0),
but weakly-observable modes need a DEEP future to surface, and deep futures overfit (few
prefixes keep full descendants -> coverage wall). Two attacks:

  M1  multi-horizon ridge RRR : stack futures from h=1..H (shallow robust + deep informative),
      input-whiten, reduced-rank, ridge+rank by held-out one-step. Stays continuous/general.
  M2  predictive-future clustering : two prefixes are the SAME belief state iff they have the
      same future morph phi(w)=[P(.|wp)]. Cluster prefixes by phi (agglomerative), build the
      subspace from the cluster centres-of-mass. Exploits RRXOR's discrete 35-state structure.
      DGP-free: uses only the model softmax tree (belief used to SCORE only, via ARI + COM-geom).

All scored by the SAME 35-state COM-geometry R^2 as the 0.635 baseline.
"""
import itertools
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score

from simplexity.generative_processes.transition_matrices import rrxor
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13

VOCAB, D = 2, 5


def load():
    T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
    model = F14._load("rrxor_transformer.pt", "cpu")
    resid, soft, belief, _ = F14._collect(model, T, pi, "cpu")
    reach = {}
    for w in resid:
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok
    return resid, soft, belief, reach


def whiten_inv_sqrt(C, ridge):
    C = C + ridge * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    val, vec = np.linalg.eigh(C)
    return (vec / np.sqrt(np.clip(val, 1e-12, None))) @ vec.T


def main():
    resid, soft, belief, reach = load()
    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Aall = np.stack([resid[w] for w in allw])
    Yb = np.stack([belief[w] for w in allw])
    _, index = F13.msp_index()
    idx = np.array([index[tuple(np.round(belief[w], 5))] for w in allw])
    groups = defaultdict(list)
    for i, s in enumerate(idx):
        groups[s].append(i)

    def com_geom(B):
        s = Aall @ B
        com = np.array([s[g].mean(0) for g in groups.values()])
        bel = np.array([Yb[g[0]] for g in groups.values()])
        return LinearRegression().fit(com, bel).score(com, bel)

    # ---- baseline: OBS-OOM (0.635) ----
    orows, A, P, Gs, Uobs, sv = F14.observable_subspace(resid, soft, reach, VOCAB)
    print(f"baseline OBS-OOM           COM-geom = {com_geom(Uobs[:, :D]):.3f}")

    # ---- M1: multi-horizon ridge RRR ----
    def phi(w, H):
        paths = [p for L in range(H + 1) for p in itertools.product(range(VOCAB), repeat=L)]
        if not all((w + p) in soft for p in paths):
            return None
        return np.concatenate([soft[w + p] for p in paths])

    def rrr_multi(H, ridge):
        fitw = [w for w in allw if len(w) <= F14.MAX_LEN - H and phi(w, H) is not None]
        Af = np.stack([resid[w] for w in fitw])
        Phi = np.stack([phi(w, H) for w in fitw])
        Winv = whiten_inv_sqrt(Af.T @ Af / len(Af), ridge)
        M = (Af @ Winv).T @ Phi / len(Af)
        U_ = np.linalg.svd(M, full_matrices=False)[0]
        return Winv @ U_[:, :D]

    best_m1 = (0, None)
    for H in [2, 3, 4]:
        for ridge in [0.03, 0.1, 0.3, 1.0]:
            B = rrr_multi(H, ridge)
            cg = com_geom(B)
            if cg > best_m1[0]:
                best_m1 = (cg, (H, ridge))
    print(f"M1 multi-horizon ridge RRR COM-geom = {best_m1[0]:.3f}   (best H,ridge={best_m1[1]})")

    # ---- M2: predictive-future clustering ----
    def cluster_subspace(H, K, use_lda):
        sigw = [w for w in allw if len(w) <= F14.MAX_LEN - H and phi(w, H) is not None]
        Sig = np.stack([phi(w, H) for w in sigw])
        Sig = (Sig - Sig.mean(0)) / (Sig.std(0) + 1e-9)
        lab = AgglomerativeClustering(n_clusters=K).fit_predict(Sig)
        true_sub = np.array([index[tuple(np.round(belief[w], 5))] for w in sigw])
        ari = adjusted_rand_score(true_sub, lab)
        Asig = np.stack([resid[w] for w in sigw])
        if use_lda and K > D:
            B = LinearDiscriminantAnalysis(n_components=D).fit(Asig, lab).scalings_[:, :D]
        else:
            coms = np.array([Asig[lab == k].mean(0) for k in range(K) if (lab == k).any()])
            B = PCA(n_components=D).fit(coms).components_.T
        return com_geom(B), ari, len(sigw)

    print("M2 predictive-future clustering:")
    for H in [2, 3, 4]:
        for K in [35, 40, 50]:
            cg_pca, ari, n = cluster_subspace(H, K, use_lda=False)
            cg_lda, _, _ = cluster_subspace(H, K, use_lda=True)
            print(f"   H={H} K={K:2d} (n={n:4d}): ARI={ari:.3f}  COM-geom PCA={cg_pca:.3f}  LDA={cg_lda:.3f}")

    # ---- ORACLE: if clustering were perfect, top-5 PCA of the TRUE state-COMs ----
    true_coms = np.array([Aall[g].mean(0) for g in groups.values()])
    Bora = PCA(n_components=D).fit(true_coms).components_.T
    print(f"\nORACLE (PCA-5 of TRUE state-COMs)        COM-geom = {com_geom(Bora):.3f}  "
          f"<- clustering ceiling if states recovered")

    # ---- M3: cluster ACTIVATIONS projected onto a generous future subspace ----
    # belief is linearly present in a(w); project onto top-m OBS directions to kill nuisance
    # variance, cluster there (all 1046 prefixes -> no coverage wall), COMs -> 5-D subspace.
    print("M3 activation clustering in OBS-projected space (all prefixes, no coverage wall):")
    best_m3 = (0, None)
    for m in [8, 12, 20, 30]:
        Bgen = Uobs[:, :m]
        Sproj = Aall @ Bgen
        Sproj = (Sproj - Sproj.mean(0)) / (Sproj.std(0) + 1e-9)
        for K in [35, 45, 60]:
            lab = AgglomerativeClustering(n_clusters=K).fit_predict(Sproj)
            ari = adjusted_rand_score(idx, lab)
            coms = np.array([Aall[lab == k].mean(0) for k in range(K) if (lab == k).any()])
            B = PCA(n_components=D).fit(coms).components_.T
            cg = com_geom(B)
            if cg > best_m3[0]:
                best_m3 = (cg, (m, K, round(ari, 3)))
            print(f"   m={m:2d} K={K:2d}: ARI={ari:.3f}  COM-geom={cg:.3f}")
    print(f"M3 best COM-geom = {best_m3[0]:.3f}  (m,K,ARI={best_m3[1]})")

    # ---- M4: EM refine -- alternate (cluster in current subspace) / (refit subspace from COMs) ----
    print("M4 EM (cluster<->subspace), init from M1 best:")
    B = rrr_multi(best_m1[1][0], best_m1[1][1])
    for it in range(6):
        Sproj = Aall @ B
        Sproj = (Sproj - Sproj.mean(0)) / (Sproj.std(0) + 1e-9)
        lab = AgglomerativeClustering(n_clusters=40).fit_predict(Sproj)
        ari = adjusted_rand_score(idx, lab)
        coms = np.array([Aall[lab == k].mean(0) for k in range(40) if (lab == k).any()])
        B = PCA(n_components=D).fit(coms).components_.T
        print(f"   iter {it}: ARI={ari:.3f}  COM-geom={com_geom(B):.3f}")


    # ---- M5: RAGGED-horizon RRR -- every prefix contributes at its MAX available depth ----
    # Build Sigma_{a,phi} where each future-path column p is averaged over ALL prefixes that
    # have that descendant (no coverage wall: deep paths use short prefixes, shallow paths use
    # all). Input whiten by the full-prefix activation covariance. Then top-5 singular subspace.
    print("\nM5 ragged-horizon RRR (all prefixes at max depth):")
    Wfull = whiten_inv_sqrt(Aall.T @ Aall / len(Aall), 0.03)
    Aw = Aall @ Wfull
    best_m5 = (0, None)
    for Hmax in [3, 4, 5, 6]:
        paths = [p for L in range(1, Hmax + 1) for p in itertools.product(range(VOCAB), repeat=L)]
        cols = []
        for p in paths:
            avail = [(i, w) for i, w in enumerate(allw) if (w + p) in soft]
            if len(avail) < 10:
                continue
            ii = [i for i, _ in avail]
            # P(p|w) = product of conditional softmaxes along the path
            pv = np.ones(len(ii))
            for i_local, (gi, w) in enumerate(avail):
                pr, node = 1.0, w
                for t in p:
                    pr *= soft[node][t]; node = node + (t,)
                pv[i_local] = pr
            col = np.zeros(Aw.shape[1])
            col = (Aw[ii] * pv[:, None]).sum(0) / len(ii)
            cols.append(col)
        M = np.stack(cols, axis=1)                                  # (5, n_paths)
        U_ = np.linalg.svd(M, full_matrices=False)[0]
        B = Wfull @ U_[:, :D]
        cg = com_geom(B)
        if cg > best_m5[0]:
            best_m5 = (cg, Hmax)
        print(f"   Hmax={Hmax}: n_paths={len(cols):3d}  COM-geom={cg:.3f}")
    print(f"M5 best COM-geom = {best_m5[0]:.3f}  (Hmax={best_m5[1]})")

    # ---- dimension sweep: is belief recoverable in a slightly larger PREDICTIVE subspace? ----
    # (d is estimated, not known to be 5; the predictive subspace may need a few more dims)
    print("\nDIM SWEEP  COM-geom vs subspace dim d (predictive subspaces):")
    def rrr_full(H, ridge, dd):
        fitw = [w for w in allw if len(w) <= F14.MAX_LEN - H and phi(w, H) is not None]
        Af = np.stack([resid[w] for w in fitw]); Phi = np.stack([phi(w, H) for w in fitw])
        Winv = whiten_inv_sqrt(Af.T @ Af / len(Af), ridge)
        U_ = np.linalg.svd((Af @ Winv).T @ Phi / len(Af), full_matrices=False)[0]
        return Winv @ U_[:, :dd]
    rng = np.random.default_rng(0)
    Dfull = Aall.shape[1]
    for dd in [5, 6, 8, 10, 12, 15, 20]:
        cg_obs = com_geom(Uobs[:, :dd])
        cg_rrr = com_geom(rrr_full(3, 0.03, dd))
        # random-subspace control: how much COM-geom is "free" from regression dim alone
        rnd = np.mean([com_geom(np.linalg.qr(rng.standard_normal((Dfull, dd)))[0]) for _ in range(5)])
        print(f"   d={dd:2d}:  OBS-OOM={cg_obs:.3f}   multi-horizon-RRR={cg_rrr:.3f}   random-ctrl={rnd:.3f}")

    print(f"\nSUMMARY  baseline 0.635 | M1 {best_m1[0]:.3f} | M3 {best_m3[0]:.3f} | "
          f"M5 {best_m5[0]:.3f} | oracle(PCA-COM) 0.345 | supervised 1.000")


if __name__ == "__main__":
    main()
