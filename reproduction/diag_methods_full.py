"""Definitive method comparison, same data / appropriate metric, RAW + ROLLOUT.

Three methods x two processes:
  CCA+ALS  : predictive CCA (one-step future) + ALS operator-consistency refit   [fig04]
  OBS-OOM  : observability-matrix SVD (deep operator-closure future), no ALS       [fig14/16]
  OBS+ALS  : the deep OBS-OOM subspace, THEN ALS-refined  -> deep future + clean operators

Question: does OBS+ALS combine OBS-OOM's deep-future RRXOR-raw advantage with CCA+ALS's
clean Mess3 rollout -> a single method best on both?

Metrics: Mess3 = prob-weighted belief-decode R^2 (continuum); RRXOR = 35-state COM-geom R^2.
"""
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.transition_matrices import rrxor, mess3
import unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig13_rrxor_unsupervised_oom as F13


def transitions(resid, soft, reach, vocab, max_len):
    rows = [w for w in resid if reach[w] and len(w) < max_len
            and all((w + (x,)) in resid for x in range(vocab))]
    X = np.stack([resid[w] for w in rows])
    P = np.stack([soft[w] for w in rows])
    Yc = np.stack([[resid[w + (x,)] for x in range(vocab)] for w in rows])
    return rows, X, P, Yc


def run_process(name):
    if name == "Mess3":
        d = 3
        hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85})
        vocab = hmm.vocab_size
        model, ctx = U.load_model("cpu")
        resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
        reach = {w: True for w in resid}
        groups = None
    else:
        d, vocab, max_len = 5, 2, F14.MAX_LEN
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
        prefix_prob = {w: (1.0 if reach[w] else 0.0) for w in resid}

    rows, X, P, Yc = transitions(resid, soft, reach, vocab, max_len)
    Wt = np.array([prefix_prob[w] for w in rows]); Wt = Wt / max(Wt.sum(), 1e-9) * len(Wt)
    allw = [w for w in resid if reach[w] and np.isfinite(belief[w]).all()]
    Yb = np.stack([belief[w] for w in allw])
    wcol = np.array([prefix_prob[w] for w in allw])
    if name == "RRXOR":
        _, index = F13.msp_index()
        idx = [index[tuple(np.round(belief[w], 5))] for w in allw]
        groups = defaultdict(list)
        for i, s in enumerate(idx):
            groups[s].append(i)

    def metric(states_dict):
        ks = [w for w in allw if w in states_dict and np.isfinite(states_dict[w]).all()]
        S = np.stack([states_dict[w] for w in ks]); Y = np.stack([belief[w] for w in ks])
        if groups is None:
            return U.belief_decode_r2(S, Y, np.array([prefix_prob[w] for w in ks]))
        gi = defaultdict(list)
        for i, w in enumerate(ks):
            gi[index[tuple(np.round(belief[w], 5))]].append(i)
        com = np.array([S[g].mean(0) for g in gi.values()]); bl = np.array([Y[g[0]] for g in gi.values()])
        return LinearRegression().fit(com, bl).score(com, bl)

    def evalB(B, ops, e):
        raw = metric({w: resid[w] @ B for w in allw})
        proj = {w: resid[w] @ B for w in resid if reach[w]}
        roll = U.rollout_states(proj, ops, e, max_len, vocab)
        return raw, metric(roll)

    out = {}
    # CCA+ALS
    dirs, _ = U.predictive_cca(X, P, Yc, Wt)
    Bc = U.als_refine_basis(X, P, Yc, Wt, dirs[:, :d], d)
    opsc, _ = U.fit_operators(X @ Bc, P, Yc @ Bc, Wt); ec, _ = U.recover_eval_functional(X @ Bc)
    out["CCA+ALS"] = evalB(Bc, opsc, ec)
    # OBS-OOM
    orows, A, oP, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab)
    Bo = Uobs[:, :d]
    _, opso, eo = F14.fit_at_dim(orows, A, oP, resid, vocab, Bo)
    out["OBS-OOM"] = evalB(Bo, opso, eo)
    # OBS+ALS
    Boa = U.als_refine_basis(X, P, Yc, Wt, Bo, d)
    opsoa, _ = U.fit_operators(X @ Boa, P, Yc @ Boa, Wt); eoa, _ = U.recover_eval_functional(X @ Boa)
    out["OBS+ALS"] = evalB(Boa, opsoa, eoa)

    m = "belief-decode" if groups is None else "COM-geom"
    print(f"\n=== {name}  (metric: {m} R^2) ===")
    print(f"  {'method':9s}   raw     rollout")
    for k, (r, ro) in out.items():
        print(f"  {k:9s}  {r:.3f}   {ro:.3f}")


if __name__ == "__main__":
    run_process("RRXOR")
    run_process("Mess3")
