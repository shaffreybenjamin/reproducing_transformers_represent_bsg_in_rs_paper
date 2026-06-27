"""Clean apples-to-apples head-to-head of the candidate RAW-SUBSPACE methods on BOTH
processes: same activation representation (per-layer-concat), same latent dim d (true),
same evaluation (Mess3 = per-prefix belief decode R^2 [continuum]; RRXOR = 35-state
COM-geometry R^2). Tests whether ONE method is best-or-tied on both.

Methods (all DGP-free, scored against truth only):
  CCA-1     predictive CCA on the one-step rescaled-children future (shallow)
  OBS-OOM   observability matrix [C, G_xC, G_xG_yC] SVD (deep operator-closure future)
  RRR-h     input-whitened reduced-rank reg onto horizon-h softmax future, h held-out
"""
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor, mess3
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13
import unsupervised_belief_oom as U
import fig23_unified_best as F23


def evaluate(name):
    if name == "RRXOR":
        T, pi, ck, d = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0, "rrxor_transformer.pt", 5
    else:
        T, pi, ck, d = np.array(mess3(x=0.05, a=0.85)), np.array([1, 1, 1]) / 3.0, "mess3_transformer.pt", 3
    vocab = T.shape[0]
    model = F14._load(ck, "cpu")
    resid, soft, belief, _ = F14._collect(model, T, pi, "cpu")
    reach = {}
    for w in resid:
        ok, pre = True, ()
        for t in w:
            if pre in soft and soft[pre][t] <= F14.EPS:
                ok = False; break
            pre = pre + (t,)
        reach[w] = ok

    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Aall = np.stack([resid[w] for w in allw])
    Yb = np.stack([belief[w] for w in allw])
    groups = None
    if name == "RRXOR":
        _, index = F13.msp_index()
        idx = [index[tuple(np.round(belief[w], 5))] for w in allw]
        groups = defaultdict(list)
        for i, s in enumerate(idx):
            groups[s].append(i)

    def score(B):
        s = Aall @ B
        per = LinearRegression().fit(s, Yb).score(s, Yb)
        if groups is None:
            return per, per
        cm = np.array([s[g].mean(0) for g in groups.values()])
        bl = np.array([Yb[g[0]] for g in groups.values()])
        return per, LinearRegression().fit(cm, bl).score(cm, bl)

    rows, X, P, Yc = F23.transitions(resid, soft, reach, vocab, F14.MAX_LEN)
    Wt = np.ones(len(rows))

    # CCA-1 (shallow one-step future)
    dirs, _ = U.predictive_cca(X, P, Yc, Wt)
    per_cca, geo_cca = score(dirs[:, :d])

    # OBS-OOM (deep operator-closure future)
    _, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab)
    per_obs, geo_obs = score(Uobs[:, :d])

    # RRR-h (input-whitened, horizon-h held-out)
    h_star, curve = F23.select_h(resid, soft, reach, rows, X, P, Yc, vocab, d, F14.MAX_LEN)
    cand = [w for w in rows if len(w) <= F14.MAX_LEN - h_star]
    Brrr = F23.fit_subspace(resid, soft, cand, vocab, d, h_star)
    per_rrr, geo_rrr = score(Brrr)

    metric = "decode" if groups is None else "COM-geom"
    print(f"\n=== {name}  (d={d}, {len(allw)} prefixes"
          f"{', 35 states' if groups else ', continuum'}) ===   [report: {metric} R^2]")
    print(f"  CCA-1   (shallow one-step):  per-prefix={per_cca:.3f}  {metric}={geo_cca:.3f}")
    print(f"  OBS-OOM (deep closure)    :  per-prefix={per_obs:.3f}  {metric}={geo_obs:.3f}")
    print(f"  RRR-h={h_star} (deep horizon)   :  per-prefix={per_rrr:.3f}  {metric}={geo_rrr:.3f}  "
          f"(held-out curve {[(h, round(r,2)) for h,r in curve]})")


if __name__ == "__main__":
    evaluate("RRXOR")
    evaluate("Mess3")
