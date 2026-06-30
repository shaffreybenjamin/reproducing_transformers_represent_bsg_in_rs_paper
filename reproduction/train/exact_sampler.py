"""Exact-distribution data sampling shared by the per-process training scripts.

This replaces the two older per-script data paths (per-step HMM simulation in
train_mess3, and the continuous-stream-chopped-into-windows path in the others)
with the single pipeline used by epsilon-transformers' `quantum-public` training
code (Riechers, Elliott & Shai, "Neural networks leverage nominally quantum and
post-quantum representations"):

  * enumerate EVERY length-(n_ctx+1) observation window together with its EXACT
    probability, via the process's mixed-state presentation (MSP);
  * each training batch is `batch_size` i.i.d. draws from that exact distribution
    (`torch.multinomial` over the precomputed probability table) -- statistically
    identical to sampling whole windows from the stationary start, but exact and
    cheap (one on-GPU gather per step, no simulation, no jax in the train loop);
  * a window w (length n_ctx+1) gives input X = w[:-1] and target Y = w[1:].

It also returns the per-context-position myopic-entropy loss lower bound (the
information-theoretic optimum the cross-entropy can reach). That bound is read
straight from the already-generated MSP tree (the same single traversal that gives
the window probabilities), rather than a second jax-compiled pass -- matching how
their `generate_all_seqs` reads both from one mixed-state tree.

Enumeration is tiny for these processes (vocab^(n_ctx+1): mess3 3^9 = 19683,
binary processes 2^9 = 512), so building the table is a one-off at startup.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.mixed_state_presentation import MixedStateTreeGenerator


@dataclass
class ProcessData:
    """Precomputed exact-distribution sampling table for one process + context length."""

    sequences: torch.Tensor        # (N, n_ctx+1) int64: every reachable length-(n_ctx+1) window
    probs: torch.Tensor            # (N,) float32: exact P(window), normalised to sum to 1
    loss_lower_bound: torch.Tensor  # (n_ctx,) float32: myopic entropy per context position (nats)
    optimal_loss: float            # mean of loss_lower_bound -- the plotted optimum
    vocab_size: int

    def sample_batch(self, batch_size: int, generator: torch.Generator | None = None):
        """Draw `batch_size` i.i.d. windows from the exact distribution -> (X, Y)."""
        idx = torch.multinomial(self.probs, batch_size, replacement=True, generator=generator)
        w = self.sequences[idx]
        return w[:, :-1], w[:, 1:]

    def validation_data(self):
        """The FULL enumerated table as (X, Y, probs) -- their `validate_epoch_all` input."""
        return self.sequences[:, :-1], self.sequences[:, 1:], self.probs


def _myopic_entropy_from_tree(node_prob: dict, vocab_size: int, n_ctx: int) -> np.ndarray:
    """Myopic observation entropy per context length 1..n_ctx, from the MSP node probabilities.

    H(next token | c preceding tokens) = sum over length-c prefixes w of
    P(w) * H({P(w.o)/P(w)}_o), where P(w) and the child probabilities P(w.o) are the
    exact sequence probabilities already stored in the generated tree.
    """
    by_len: dict[int, list[tuple[tuple[int, ...], float]]] = defaultdict(list)
    for seq, p in node_prob.items():
        by_len[len(seq)].append((seq, p))

    ent_per_ctx = np.zeros(n_ctx, dtype=np.float64)
    for c in range(1, n_ctx + 1):
        total = 0.0
        for seq, pw in by_len[c]:
            if pw <= 0.0:
                continue
            entropy = 0.0
            for o in range(vocab_size):
                pc = node_prob.get(seq + (o,), 0.0)
                if pc > 0.0:
                    po = pc / pw
                    entropy -= po * math.log(po)
            total += pw * entropy
        ent_per_ctx[c - 1] = total
    return ent_per_ctx


def build_process_data(process_name: str, process_params: dict, n_ctx: int, device: str) -> ProcessData:
    """Enumerate the exact length-(n_ctx+1) window distribution and myopic-entropy bound."""
    hmm = build_hidden_markov_model(process_name, process_params)
    vocab_size = int(hmm.vocab_size)
    window_len = n_ctx + 1

    tree = MixedStateTreeGenerator(hmm, max_sequence_length=window_len).generate()

    node_prob: dict[tuple[int, ...], float] = {}
    windows: list[tuple[int, ...]] = []
    window_probs: list[float] = []
    for seq, node in tree.nodes.items():
        p = float(node.probability)
        node_prob[seq] = p
        if len(seq) == window_len and p > 0.0 and np.isfinite(p):  # reachable full windows only
            windows.append(seq)
            window_probs.append(p)

    seq_arr = np.asarray(windows, dtype=np.int64)
    prob_arr = np.asarray(window_probs, dtype=np.float64)
    prob_arr = prob_arr / prob_arr.sum()

    loss_lower_bound = _myopic_entropy_from_tree(node_prob, vocab_size, n_ctx).astype(np.float32)
    optimal_loss = float(loss_lower_bound.mean())

    return ProcessData(
        sequences=torch.from_numpy(seq_arr).to(device),
        probs=torch.from_numpy(prob_arr).float().to(device),
        loss_lower_bound=torch.from_numpy(loss_lower_bound).to(device),
        optimal_loss=optimal_loss,
        vocab_size=vocab_size,
    )
