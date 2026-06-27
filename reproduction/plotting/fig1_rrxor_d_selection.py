"""D-selection analysis for rrxor process."""
from pathlib import Path

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from simplexity.generative_processes.builder import build_hidden_markov_model
import d_selection_pipeline as DSP

PROCESS_NAME = "rrxor"
PROCESS_PARAMS = {"p1": 0.4, "p2": 0.6}
TRUE_D = 5
MODEL_FILE = "rrxor_transformer.pt"
MODEL_DIR = Path(__file__).parent.parent / "models"


def load_model(device):
    ckpt = torch.load(MODEL_DIR / MODEL_FILE, map_location=device, weights_only=False)
    cfg = HookedTransformerConfig.from_dict(ckpt["cfg"])
    cfg.device = device
    model = HookedTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["context_len"], ckpt.get("step")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmm = build_hidden_markov_model(PROCESS_NAME, PROCESS_PARAMS)
    model, ctx, step = load_model(device)
    print(f"Loaded {PROCESS_NAME} checkpoint (step={step}, device={device})")

    results = DSP.run_d_selection_pipeline(
        model, hmm, ctx, device, PROCESS_NAME, TRUE_D, max_d_test=50, verbose=True
    )

    summary_file = Path(__file__).parent / f"d_selection_{PROCESS_NAME}_results.txt"
    with open(summary_file, "w") as f:
        f.write(f"D-Selection Results: {PROCESS_NAME}\n")
        f.write(f"{'='*70}\n")
        f.write(f"True d: {TRUE_D}\n\n")
        f.write("SPECTRAL-OOM:\n")
        f.write(f"  Effective Rank (95%): {results['spectral_oom']['effective_rank_95']}\n")
        f.write(f"  Effective Rank (99%): {results['spectral_oom']['effective_rank_99']}\n")
        f.write(f"  Elbow Method:         {results['spectral_oom']['elbow']}\n")
        f.write(f"  MI Saturation:        {results['spectral_oom']['mi_saturation']}\n")
        f.write(f"  Agreement:            {results['spectral_oom']['agreement']}\n\n")
        f.write("CCA/RRR:\n")
        f.write(f"  Effective Rank (95%): {results['cca_rrr']['effective_rank_95']}\n")
        f.write(f"  Effective Rank (99%): {results['cca_rrr']['effective_rank_99']}\n")
        f.write(f"  Elbow Method:         {results['cca_rrr']['elbow']}\n")
        f.write(f"  MI Saturation:        {results['cca_rrr']['mi_saturation']}\n")
        f.write(f"  Agreement:            {results['cca_rrr']['agreement']}\n")

    print(f"\nResults saved to {summary_file}")


if __name__ == "__main__":
    main()
