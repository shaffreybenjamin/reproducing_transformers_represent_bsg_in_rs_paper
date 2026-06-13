# Unsupervised Recovery of Belief-State Geometry from Transformer Activations via Spectral/OOM Estimators: A Feasibility Assessment

## TL;DR
- **Yes, the activation-side unsupervised estimator is feasible for the Mess3 toy case, and the user's key insight is the load-bearing trick: because the projective normalizer η·T^(x)·1 equals the conditional emission probability P(x|w) — which the transformer already outputs as its own softmax — one can rescale the projective belief update into an ordinary *linear* one and recover the map from activations to the simplex with linear least squares, never touching the analytic ground-truth targets.**
- **The math is mature and directly transferable**: HKZ spectral HMM learning, Balle et al. weighted-automata Hankel methods, Jaeger OOMs, and PSR two-stage regression all recover per-token operators up to a GL(d) gauge from a Hankel matrix whose rank equals the minimal model dimension; the activation-side variant is essentially Dynamic Mode Decomposition / Ho-Kalman realization applied to residual-stream snapshots, validated by gauge-invariant spectra and rescaled rollout.
- **It will NOT scale to natural language as a single global linear state** (the minimal GHMM dimension of language is astronomical); the realistic route is the *local/factored* one — a dictionary of small low-dimensional linear-dynamical factors in orthogonal subspaces — which connects directly to the Astera "Transformers learn factored representations" result.

## Key Findings

1. **No one has published exactly the activation-side spectral/OOM estimator the user describes.** The closest published works are (a) spectral *distillation of weighted automata from RNNs* (Eyraud & Ayache; Okudono et al.), which query the network as a string oracle and learn from token statistics — i.e. still process-side; (b) the Astera/Simplex line, which uses *supervised* linear regression onto analytic belief targets; and (c) Levinson's "Finding Belief Geometries with Sparse Autoencoders" (arXiv:2604.02685, April 2026), which is unsupervised but uses SAE + k-subspace clustering + simplex-fitting (AANet), NOT operator/dynamics estimation. The user's specific proposal — fit a low-dim residual subspace and per-token linear maps A^(x) directly to activation dynamics — appears to be a genuine gap.

2. **The gauge story is exact and favorable.** HKZ Lemma 3 proves the learned operators satisfy B_x = (U^T O) A_x (U^T O)^{-1}, i.e. they are similar to the true operators A_x = T·diag(O_x) via the invertible gauge S = U^T O, and observable quantities Pr[x_1:t] = b_∞^T B_{x_t}…B_{x_1} b_1 telescope to be gauge-invariant. This means an activation-side estimator can only ever recover the geometry up to a linear change of basis — exactly the same indeterminacy the supervised regression resolves with a final fit. Gauge-invariant validation (eigenvalues, traces of word-operators) needs no ground truth.

3. **The projective→linear reduction is the crux and it works.** The ground-truth update η ↦ ηT^(x)/(ηT^(x)1) is a homography on the affine belief subspace. Generic homography recovery would need DLT. But the normalizer ηT^(x)1 = P(x|w) is observable as the network's own next-token probability, so multiplying through gives the *linear* relation P(x|w)·z(wx) ≈ z(w)A^(x), solvable by ordinary/alternating least squares on (activation, network-output) pairs only.

## Details

### (1) Mathematics of spectral learning / OOMs / WFAs / PSRs
The unifying object (Thon & Jaeger, JMLR 2015) is the **sequential system**: a process is described by an initial state, per-symbol observable operators T^(x) (= A_x), and an evaluation functional, with P(w = x_1…x_ℓ) = σ T^(x_ℓ)…T^(x_1) ω. HMMs ⊂ OOMs ⊂ (stochastic) weighted finite automata; PSRs are the input-output extension (Littman-Sutton-Singh).

The **Hankel matrix** H has rows indexed by prefixes p, columns by suffixes s, with H[p,s] = P(ps). The **fundamental theorem**: rank(H) equals the number of states of the minimal WFA computing the function (Fliess/Carlyle-Paz; Balle et al. 2014). HKZ instantiate this with the bigram matrix [P_{2,1}]_{ij} = Pr[x_2=i, x_1=j], whose factorization P_{2,1} = O·T·diag(π)·O^T gives **rank(P_{2,1}) = m = number of hidden states** (HKZ Lemma 2). Taking U = top-m left singular vectors of P_{2,1}, the operators are recovered as
- b_1 = U^T P_1,  b_∞ = (P_{2,1}^T U)^+ P_1,
- B_x = (U^T P_{3,x,1})(U^T P_{2,1})^+, with [P_{3,x,1}]_{ij} = Pr[x_3=i, x_2=x, x_1=j].

**Recovery is up to a GL(m) similarity transform** (gauge), as in Finding 2. The internal state b_t = (U^T O)·h_t is the true belief state h_t in the gauge basis, updated recursively as b_{t+1} = B_x b_t / (b_∞^T B_x b_t) — structurally identical to the MSP belief update with an observable normalizer.

**Finite-sample / PAC guarantees** (HKZ Theorem 6): for L1 accuracy ε on length-t joint probabilities, N = Õ( t² m / (ε² σ_m(O)² σ_m(P_{2,1})^4) + t² m·n_0(ε) / (ε² σ_m(O)² σ_m(P_{2,1})²) ) samples suffice w.p. 1−η. Per Hsu, Kakade & Zhang (arXiv:0811.4413), verbatim: "The sample complexity of the algorithm does not explicitly depend on the number of distinct (discrete) observations—it implicitly depends on this quantity through spectral properties of the underlying HMM," holding "under a natural separation condition (bounds on the smallest singular value of the HMM parameters)." The smallest singular value σ_m(P_{2,1}) is that separation/conditioning parameter; small σ_m ⇒ hard. Balle et al.'s analysis likewise quantifies complexity by the smallest singular value of the Hankel matrix.

### (2) Negative-probability problem and model-order selection
The OOM **negative probability problem (NPP)**: a learned operator model can assign negative "probabilities" because the non-negativity constraint is not enforced; it is in fact *undecidable* whether a candidate OOM violates it (Wiewiora 2007, as discussed in the hidden-quantum-Markov-models literature). **This is largely a non-issue for the activation-side estimator**, because the goal is to recover the network's *representation geometry*, not to build a valid generator — we never need the operators to define a normalized probability law; we only need them to predict the next activation/belief coordinate. The transformer's own softmax supplies a valid probability law separately.

**Model-order selection** = where to truncate the Hankel/snapshot singular-value spectrum to pick the latent dimension d. Standard practice: look for a spectral gap; discard singular values below a noise threshold (e.g. bootstrap SNR < 10 as in Markov-state-model practice); or use nuclear-norm regularization which produces a cleaner singular-value gap (Sun & Oymak). For Mess3 the truthful latent rank is 2 (a 2-simplex), so the spectrum should show a clear 3→ smaller gap (3 hidden states, 2 free dims after normalization).

### (3) Process-side vs activation-side estimation
- **Process-side**: build the Hankel matrix from token n-gram statistics, run spectral learning, get a GHMM. This recovers the data-generating process, not the network's representation. (scikit-splearn does exactly this.)
- **Activation-side** (the user's proposal): collect residual-stream activations z(w) at each context position; fit a d-dim subspace (PCA/SVD of the activation snapshot matrix); fit per-token linear maps A^(x) such that, after the P(x|w) rescaling, P_z(wx)·A^(x) tracks the next projected activation. This is **bilinear** in (subspace projector, operators) and is solvable by alternating least squares, or — once the subspace is fixed — by a stack of per-token linear regressions (a DMD-with-control / switched-linear-system fit). It is, to the best of the available literature, **unpublished** as such.

The activation-side fit is mathematically the **same object as Dynamic Mode Decomposition / Koopman operator approximation and subspace system identification** (Ho-Kalman realization, N4SID, van Overschee & De Moor): given snapshot pairs (z_t, z_{t+1}) under a known "input" symbol x_t, estimate the least-squares operator A^(x) = Z'_x Z_x^+. PyDMD implements the relevant DMD/DMDc machinery. The PSR **two-stage regression** (Hefny et al. 2015) is the statistically-consistent version: regress future features on history, then operators by linear regression — directly portable with activations as the feature map.

### (4) The projective wrinkle and the unsupervised rescaling trick
Belief update is projective (a homography / linear-fractional map) on the affine subspace, so naive linear regression of z(wx) on z(w) is mis-specified. Two routes:
- **Computer-vision homography / DLT**: treat (z(w), z(wx)) as projective point correspondences and solve the homogeneous system A·h = 0 via SVD (the null-space / smallest-singular-vector solution), exactly as in 2D homography estimation (8 DOF for the planar case); RANSAC for robustness. This is the principled tool if no normalizer were available.
- **The rescaling shortcut (recommended)**: since ηT^(x)1 = P(x|w) and the network outputs P(x|w) at each position, multiply the target through: P(x|w)·z(wx) ≈ z(w)·A^(x). This converts the homography into a *linear* map recoverable by ordinary least squares, using ONLY activations and the network's own output probabilities — fully unsupervised w.r.t. the generator. This is the single most important enabling observation in the user's plan.

### (5) Validation without ground truth
- **Rescaled rollout**: from an initial projected activation, iterate z_{t+1} = z_t A^(x_t) (with the P(x|w) rescaling) along held-out sequences and compare to actual projected activations (R²/cosine).
- **Reconstructed word probabilities** from estimated operators vs empirical n-gram frequencies.
- **Gauge-invariant comparisons**: eigenvalue spectra of A^(x) vs true T^(x); traces and determinants of word-operators B_w; these are similarity-invariant and need no alignment. (Riechers & Crutchfield "Spectral Simplicity" shows the belief metadynamics are governed precisely by these operator spectra.)
- **Procrustes gauge-fixing only at the very end**: align the recovered d-dim cloud to the known MSP fractal for visual overlay — used as a *display* step, not for learning.

### (6) DMD / Koopman / subspace ID equivalence
Confirmed same mathematical object: activation-side operator estimation = switched/controlled DMD = subspace identification. For stochastic/noisy data, **subspace DMD** (Takeishi et al. 2017) projects future snapshots onto past snapshots to consistently estimate the stochastic Koopman spectrum — directly relevant since residual activations are noisy. Koopman-invariant-subspace learning (Takeishi/Lusch) is the nonlinear analogue if a single linear subspace is insufficient.

### (7) Riechers & Elliott identifiability (arXiv:2509.03004)
This paper maps any linear latent model (classical, quantum, post-quantum) to a unique **minimal canonical GHMM**, solving the identifiability problem and giving minimality (dimension) bounds. For the estimator this sharpens the story in two ways: (i) it guarantees a well-defined *target* — the minimal GHMM dimension is the rank the Hankel/snapshot spectrum should reveal, so model-order selection has a principled answer; (ii) it tells us the recovered operators are unique up to the gauge, legitimizing gauge-invariant validation. The companion empirical paper ("Neural networks leverage nominally quantum and post-quantum representations," arXiv:2507.07432) shows networks can adopt GHMM (non-simplex) geometries, meaning the estimator must allow embeddings off the probability simplex — exactly the GHMM generality the user flags.

### (8) Code resources — what each can and cannot do
- **github.com/adamimos/epsilon-transformers**: Mess3 process generation, transformer training, activation collection, analytic MSP/belief computation, fractal plotting. CAN provide the entire experimental substrate and the supervised baseline; does NOT contain an unsupervised activation-side operator estimator.
- **scikit-splearn** (`pip install scikit-splearn`): spectral learning of WFAs via Hankel SVD; classes for Hankel matrix, WFA, learning, numerically stable PA minimization. CAN do process-side spectral learning from strings and is a ready reference for the SVD/operator extraction; does NOT operate on activations (you'd feed it token strings, not residuals) and is built around the prefix/suffix string Hankel, so the activation-side fit must be coded separately.
- **PyDMD**: standard + robust/subspace/controlled DMD variants. CAN do the activation-side operator least-squares and rollout; does NOT know about the projective normalization or multi-symbol switching out of the box (needs the P(x|w) rescaling and per-symbol operator bookkeeping added).
- **github.com/Astera-org/factored-reps**: training configs and figure-generation for the factored-representations experiments (independent/chain/noise-sweep processes, RNN/LSTM/transformer variants); the substrate for testing the factored/local route.

### (9) Mess3 specifically
Mess3 (Marzen & Crutchfield 2017) has 3 hidden states, 3 tokens, parameters α and x with β=(1−α)/2, y=1−2x; the labeled transition matrices T^(0),T^(1),T^(2) are the symmetric forms given in the Simplex papers. The canonical published setting is confirmed in Shai et al. (arXiv:2405.15943): "This process is called the mess3 process, and was defined in a paper by Sarah Marzen and James Crutchfield. In the work presented we use x=0.05, alpha=0.85." Its belief geometry is a 2-dimensional Sierpinski-like fractal in the 2-simplex. An activation-side estimator's success criterion: recovered latent dimension = 2; recovered per-token operator eigenvalues match T^(x) eigenvalues; rescaled rollout reproduces held-out activation trajectories; and the Procrustes-aligned recovered cloud reproduces the fractal.

Two methodological cautions are well-documented. First, the **constrained-vs-full belief gap** (Piotrowski, Riechers, Filan & Shai, arXiv:2502.01954): transformers "implement constrained Bayesian belief updating—a parallelized version of partial Bayesian inference shaped by architectural constraints," where "attention carries out an algorithm with a natural interpretation in the probability simplex" (their primary analysis "focuses on single-layer transformers, revealing how the first attention layer implements these constrained updates"). Consequently post-attention/pre-MLP activations realize a *constrained* (different) belief geometry, while the full fractal appears in the final residual stream. Second, the supervised baseline's recipe pins down exactly where the full geometry lives: per Shai et al. (arXiv:2405.15943), "we analyzed the final layer of the transformer's residual stream, before the layer norm and unembedding. Using linear regression, we identified a 2D subspace of the 64-dimensional residual activations that best matched the ground-truth belief distributions." The estimator should therefore be applied to the final-layer residual (pre-LayerNorm, pre-unembedding), and one should expect different geometries at intermediate layers.

### (10) Prior unsupervised-discovery work
PSR/2SR learning recovers predictive state without state supervision but from observations, not activations. Predictive-State Decoders and PSRNNs *inject* predictive-state structure into RNN training. DFA/WFA extraction from RNNs clusters or queries hidden states. The SAE belief-geometry paper is the only published unsupervised attempt to find belief simplices in activations, and it deliberately avoids dynamics. So the activation-side operator estimator is novel and worth doing.

### Scaling concern: factored/local route
The minimal GHMM dimension of natural language is astronomically large, so a single global low-rank linear state does not exist at tractable rank. The viable path is **local and factored**: a dictionary of small linear-dynamical factors, each a small estimated GHMM living in an orthogonal subspace of the residual stream. This connects directly to the Astera factored-representations result (Shai, Amdahl-Culleton, Christensen, Bigelow, Rosas, Boyd, Alt, Ray & Riechers, arXiv:2602.02385), which shows transformers represent conditionally-independent factors in orthogonal subspaces with dimension growing linearly rather than exponentially: "The dimension of the joint representation is (∏_n d_n) − 1, while the factored representation requires Σ_n (d_n − 1) dimensions, offering exponential reduction." Empirically, for five independent factors, "effective dimensionality converged to approximately 11 dimensions (for 95% variance explained), closely matching the factored prediction (10 dimensions) and strongly rejecting the joint prediction (135 dimensions for 95% CEV)." **Caveat on cleanliness**: MLP nonlinearity, LayerNorm, and the constrained-vs-full belief gap mean activations do NOT decompose perfectly linearly; the final-residual full-belief geometry is the cleanest target, intermediate layers are not. Expect the linear estimator to work well on toy HMMs at the final layer, degrade at intermediate layers, and require the factored/local treatment (plus possibly Koopman-style nonlinear lifts) beyond toy settings.

## Recommendations

**Staged plan for the first experiment (Mess3):**

1. **Reproduce the supervised baseline** using epsilon-transformers: train the Mess3 transformer (x=0.05, α=0.85), collect final-layer residual activations z(w) (pre-LayerNorm, pre-unembedding) and the network's softmax P(·|w), and compute the analytic MSP belief targets (for *evaluation only*).
2. **Fix the subspace**: PCA/SVD of the activation snapshot matrix; inspect the singular-value spectrum for the expected gap (target d=2–3). Benchmark to change the plan: if no clear gap at d≈2–3, the linear-state assumption is already failing — switch to the factored/local approach.
3. **Fit operators unsupervised** with the rescaling trick: for each token x, solve the linear least squares P(x|w)·P_d z(wx) ≈ P_d z(w) A^(x) over all positions (use PyDMD/DMDc or a hand-rolled lstsq; optionally PSR two-stage regression for consistency). No ground-truth targets used.
4. **Validate gauge-invariantly**: compare eig(A^(x)) to eig(T^(x)); check rescaled rollout R² on held-out sequences; reconstruct n-gram probabilities. Benchmark: eigenvalue match within a few percent and rollout R² > ~0.9 indicate success.
5. **Gauge-fix and overlay**: Procrustes-align the recovered cloud to the analytic fractal for the figure. Success = visual reproduction of the Mess3 fractal without ever having regressed onto the analytic belief targets.
6. **Then stress-test**: intermediate layers (expect constrained-belief mismatch), a GHMM/non-simplex process (Bloch-walk) to confirm off-simplex embeddings, then a two-factor process from factored-reps to pilot the dictionary-of-factors approach.

**Thresholds that change the approach**: no singular-value gap ⇒ abandon single global linear state, go factored; rollout R² low but eigenvalues right ⇒ subspace mis-fit, iterate the bilinear ALS; operators only work post-rescaling ⇒ confirms projective structure (good); needing many heads / complex eigenvalues ⇒ negative-eigenvalue regime requiring more attention heads (per Piotrowski et al.), expect harder fits.

**Reading path (ordered):**
1. Shai, Marzen, Riechers et al., "Transformers represent belief state geometry…" (arXiv:2405.15943) — the target phenomenon and supervised method.
2. Piotrowski, Riechers, Filan, Shai, "Constrained belief updates…" (arXiv:2502.01954) — constrained-vs-full geometry, "attention implements a spectral algorithm," Mess3 matrices.
3. Hsu, Kakade, Zhang, "A Spectral Algorithm for Learning HMMs" (arXiv:0811.4413) — the core operator construction, gauge lemma, PAC bounds.
4. Jaeger, "Observable Operator Models…" (Neural Computation 2000) — OOM foundations and NPP.
5. Balle, Carreras, Luque, Quattoni, "Spectral learning of weighted automata" (Machine Learning 2014) + Thon & Jaeger (JMLR 2015) — Hankel theory, the unifying WFA/OOM/PSR picture.
6. Hefny, Downey, Gordon, "Supervised Learning for Dynamical System Learning" / two-stage regression (NeurIPS 2015) — consistent regression-based estimation.
7. Brunton/Kutz DMD + Takeishi et al. "Subspace DMD" (arXiv:1705.04908) + PyDMD paper (arXiv:2402.07463) — the activation-side computational engine.
8. Riechers & Elliott, "Identifiability and minimality bounds…" (arXiv:2509.03004) — uniqueness/minimality of the GHMM target.
9. Shai et al., "Transformers learn factored representations" (arXiv:2602.02385) — the scaling/factored route.
10. Levinson, "Finding Belief Geometries with Sparse Autoencoders" (arXiv:2604.02685) — the closest unsupervised prior art and its validation pitfalls (tiling artifacts, barycentric prediction tests).

**Algorithmic skeleton:**
```
# Unsupervised activation-side OOM/DMD estimator
1. Collect {z(w)} residuals and {P(x|w)} softmax outputs over many sequences.
2. Z = stack(z(w)); U_d, S_d, _ = SVD(Z); choose d by spectral gap; P_d = U_d^T.
3. For each token x:
     gather pairs (a, b, p) = (P_d z(w), P_d z(wx), P(x|w)) over positions where x follows w
     solve  min_{A_x}  Σ || p*b  -  a A_x ||^2     # rescaled => linear least squares
4. Validate: eig(A_x) vs eig(T_x); rollout z_{t+1}=z_t A_{x_t} rescaled; n-gram recon.
5. (Optional bilinear) alternate: refit P_d given {A_x}, refit {A_x} given P_d (ALS).
6. Procrustes-align recovered states to analytic MSP for visualization only.
```

## Caveats
- The activation-side operator estimator as specified is, to the best of the surveyed literature (June 2026), **unpublished**; this assessment infers feasibility from the tight mathematical correspondence with HKZ/WFA/DMD/PSR theory, not from a demonstrated result.
- The rescaling trick assumes the network's softmax P(x|w) is a faithful surrogate for the true normalizer ηT^(x)1; this holds to the extent the transformer is well-trained and near-optimal — quantify the gap empirically.
- Linearity is exact only in the right basis at the final residual stream; LayerNorm (a projective/normalizing nonlinearity itself), MLP nonlinearity, and the constrained-belief gap will degrade intermediate-layer fits.
- The negative-probability problem is benign here only because we want representation geometry, not a valid generator — do not reuse these operators as a sampling model.
- Scaling to LMs requires the factored/local decomposition and possibly nonlinear (Koopman) lifts; a single global linear state will not exist at tractable rank. The factored route's losslessness depends on conditional independence of factors, which natural language only approximately satisfies.
- Mess3 parameter values and exact transition matrices should be taken from the source repos; α=0.85, x=0.05 is the canonical published setting, but other parameterizations appear across the papers (e.g. the constrained-belief paper sweeps several).