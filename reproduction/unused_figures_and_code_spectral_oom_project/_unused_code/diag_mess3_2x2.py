"""Mess3 operator-rollout R^2 over the 2x2: {uniform, prob} weighting x {no-ALS, ALS}.
Scoring is prob-weighted belief-decode in all cells (comparable). Subspace = OBS-OOM."""
import numpy as np
from sklearn.linear_model import LinearRegression
from simplexity.generative_processes.builder import build_hidden_markov_model
import reproduction.estimators.unsupervised_belief_oom as U
import fig14_observable_oom as F14
import fig23_unified_best as F23

hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85}); vocab = hmm.vocab_size
model, ctx = U.load_model("cpu")
resid, soft, belief, prefix_prob, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
reach = {w: True for w in resid}
rows, X, P, Yc = F23.transitions(resid, soft, reach, vocab, max_len)
_, _, _, _, Uobs, _ = F14.observable_subspace(resid, soft, reach, vocab)
B_dec = Uobs[:, :3]
allw = [w for w in resid if np.isfinite(belief[w]).all()]
true_b = np.stack([belief[w] for w in allw]); wcol = np.array([prefix_prob[w] for w in allw])

def rollout_r2(Wt, als):
    B_op = U.als_refine_basis(X, P, Yc, Wt, B_dec, 3) if als else B_dec
    ops, _ = U.fit_operators(X @ B_op, P, Yc @ B_op, Wt)
    proj = {w: resid[w] @ B_op for w in resid}
    e, _ = U.recover_eval_functional(np.stack([proj[w] for w in allw]))
    roll = U.rollout_states(proj, ops, e, max_len, vocab)
    rw = [w for w in allw if w in roll and np.isfinite(roll[w]).all()]
    S = np.stack([roll[w] for w in rw]); Y = np.stack([belief[w] for w in rw])
    w = np.array([prefix_prob[w] for w in rw])
    return U.belief_decode_r2(S, Y, w)

Wt_u = np.ones(len(rows))
Wt_p = np.array([prefix_prob[w] for w in rows])
print("Mess3 operator-rollout belief-decode R^2 (prob-scored):")
print(f"  uniform + no-ALS : {rollout_r2(Wt_u, False):.3f}")
print(f"  uniform + ALS    : {rollout_r2(Wt_u, True):.3f}")
print(f"  prob    + no-ALS : {rollout_r2(Wt_p, False):.3f}")
print(f"  prob    + ALS    : {rollout_r2(Wt_p, True):.3f}")
