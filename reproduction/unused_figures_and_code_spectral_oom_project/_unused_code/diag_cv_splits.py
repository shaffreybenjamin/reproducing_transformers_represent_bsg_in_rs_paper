"""Unsupervised (observability-OOM + P(w)) cross-validation at several train fractions, on the
final Mess3 model. For each f: split prefixes, recover the subspace from TRAIN transitions,
fit the decode on TRAIN, and measure belief-reconstruction MSE on the HELD-OUT prefixes.
Reports the surviving transition count (the f^4 bottleneck) and averages over a few splits.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

from simplexity.generative_processes.builder import build_hidden_markov_model
import reproduction.estimators.unsupervised_belief_oom as U
import fig14_observable_oom as F14

D = 3
N_REPEAT = 4


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    hmm = build_hidden_markov_model("mess3", {"x": 0.05, "a": 0.85}); vocab = hmm.vocab_size
    model, ctx = U.load_model("cpu")
    resid, soft, belief, pp, max_len = U.collect_prefix_features_enumerated(model, hmm, ctx, "cpu")
    seqs = [s for s in resid if s in belief]
    Y = np.array([belief[s] for s in seqs]); Xact = np.stack([resid[s] for s in seqs])
    idx = {s: i for i, s in enumerate(seqs)}
    print(f"prefixes={len(seqs)}\n")
    print(f"{'train frac':>10} | {'test':>6} | {'transitions':>11} | {'CV-MSE (held-out)':>17} | {'train-MSE':>9}")
    print("-" * 70)

    for f in [0.3, 0.4, 0.5, 0.8]:
        cvs, trs, ntrans, ntest = [], [], [], []
        for r in range(N_REPEAT):
            rng = np.random.default_rng(r)
            is_tr = rng.random(len(seqs)) < f
            trk = [seqs[i] for i in np.where(is_tr)[0]]
            te = np.where(~is_tr)[0]
            sub = {k: resid[k] for k in trk}
            ssoft = {k: soft[k] for k in trk}
            spp = {k: pp[k] for k in trk}
            reach = {k: True for k in trk}
            rows, A, P, Gs, Uobs, sv = F14.observable_subspace(sub, ssoft, reach, vocab, wmap=spp)
            B = Uobs[:, :D]
            tr_i = np.array([idx[k] for k in trk])
            reg = LinearRegression().fit(Xact[tr_i] @ B, Y[tr_i])
            cvs.append(mse(reg.predict(Xact[te] @ B), Y[te]))
            trs.append(mse(reg.predict(Xact[tr_i] @ B), Y[tr_i]))
            ntrans.append(len(rows)); ntest.append(len(te))
        print(f"{int(f*100):>3}/{int((1-f)*100):<3}    | {int(np.mean(ntest)):>6} | "
              f"{int(np.mean(ntrans)):>11} | {np.mean(cvs):>17.4f} | {np.mean(trs):>9.4f}")
    print("\n(reference: supervised 20/80 held-out MSE = 0.0004; unsupervised 80/20 = 0.0016)")


if __name__ == "__main__":
    main()
