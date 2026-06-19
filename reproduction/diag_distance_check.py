"""Check the RRXOR distance reversal using fig26's EXACT representation (resid_post concat)."""
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.transition_matrices import rrxor
import fig14_observable_oom as F14


def r2(x, y):
    return float(linregress(x, y).rvalue ** 2)


T, pi = np.array(rrxor(0.5, 0.5)), np.array([2, 1, 1, 1, 1]) / 6.0
model = F14._load("rrxor_transformer.pt", "cpu")
resid, soft, belief, _ = F14._collect(model, T, pi, "cpu")     # resid_post concat (256), == fig26
reach = {}
for w in resid:
    ok, pre = True, ()
    for t in w:
        if pre in soft and soft[pre][t] <= F14.EPS:
            ok = False; break
        pre = pre + (t,)
    reach[w] = ok
pw = F14.analytic_prefix_probs(resid, T, pi)
allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
Yb = np.stack([belief[w] for w in allw]); X = np.stack([resid[w] for w in allw])
groups = defaultdict(list)
for i, w in enumerate(allw):
    groups[tuple(np.round(belief[w], 5))].append(i)
gl = list(groups.values())
tb = np.array([Yb[g[0]] for g in gl])
nt = np.array([[(t @ T[x]).sum() for x in range(T.shape[0])] for t in tb]); nt = nt / nt.sum(1, keepdims=True)
print(f"states={len(tb)}  intrinsic next-token-dist vs belief-dist R^2={r2(pdist(nt), pdist(tb)):.3f}\n")

B = F14.observable_subspace(resid, soft, reach, 2, wmap=pw)[4][:, :5]
XB = X @ B

for tag, feat in [("SUPERVISED (full concat)", X), ("UNSUPERVISED (OBS-OOM subspace)", XB)]:
    # (a) per-prefix decode -> COM   (what fig34 used)
    rep_pp = LinearRegression().fit(feat, Yb).predict(feat)
    rp_pp = np.array([rep_pp[g].mean(0) for g in gl])
    # (b) COM -> decode  (fig26-certified state level)
    com = np.array([feat[g].mean(0) for g in gl])
    rp_cd = LinearRegression().fit(com, tb).predict(com)
    print(f"{tag}:  per-prefix decode R^2={LinearRegression().fit(feat, Yb).score(feat, Yb):.3f}   "
          f"state-COM decode R^2={LinearRegression().fit(com, tb).score(com, tb):.3f}")
    print(f"   (a) per-prefix->COM : belief-dist R^2={r2(pdist(tb), pdist(rp_pp)):.3f}  "
          f"next-token R^2={r2(pdist(nt), pdist(rp_pp)):.3f}")
    print(f"   (b) COM->decode     : belief-dist R^2={r2(pdist(tb), pdist(rp_cd)):.3f}  "
          f"next-token R^2={r2(pdist(nt), pdist(rp_cd)):.3f}\n")
