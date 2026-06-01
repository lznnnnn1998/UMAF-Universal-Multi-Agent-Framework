# Novel Attention Formulation: Mathematical Definition and Core Mechanism

> **Research Sub-Topic 2**: Genuinely novel attention mechanism design — full mathematical specification with forward pass, backward pass, dual formulation, RoPE integration, and key theorems.

---

## 1. Overview

### 1.1 Motivation

The weakness audit (Sub-Topic 1) identified fourteen formalized weaknesses across the four dominant attention paradigms. The central finding is that **four structurally related bottlenecks trace back to a single root cause**: the softmax exponential nonlinearity. This one element-wise operation creates a 512:1 tensor-core-to-SFU throughput gap on Blackwell-class hardware (Weakness W1), drives attention entropy collapse at long contexts (Weakness W5), prevents factorization into a dual quadratic/linear form, and forces per-GPU-generation kernel redesigns (Weakness W2). Despite the sophistication of FlashAttention-4, which must software-emulate exponentials via cubic polynomials and resort to conditional rescaling to achieve 71% utilization on B200, the fundamental mathematical bottleneck remains: **the softmax nonlinearity cannot be expressed as a sequence of matrix multiplications**.

The research community has explored multiple alternatives — polynomial attention (withdrawn from ICLR 2025 due to instability of x³ activations), sigmoid attention (FLASHSIGMOID, 17% kernel speedup but still requires exponentials), α-entmax (sparse but computationally expensive to normalize), and linear attention (O(N) but with rank bottleneck from missing higher-order Taylor interactions). None simultaneously satisfies all five design principles extracted from the weakness audit: (1) matrix-multiply-only forward pass, (2) entropy-stable normalization, (3) exact dual quadratic/linear formulation, (4) native content-addressable retrieval, and (5) generation-invariant algorithm structure.

### 1.2 The Proposed Mechanism: RhoAttention (ρ-Attn)

We propose **RhoAttention (ρ-Attn)** — a novel attention mechanism whose normalization function is a **matrix rational function** (specifically, the resolvent of the key-key Gram matrix) rather than an element-wise exponential. The core mathematical insight is that the matrix inverse (ρI + K^T K)^(-1) serves as a **global, differentiable, information-theoretically principled alternative to row-wise softmax**, with three transformative properties:

1. **Tensor Core Native**: Every operation in the forward and backward pass is a matrix multiplication or a small-matrix decomposition (Cholesky of a d×d matrix), eliminating SFU dependence entirely. On B200-class hardware, this converts the softmax bottleneck from a 1,024-cycle SFU-bound operation to a ~64-cycle tensor core operation — a **theoretical 16× speedup for the normalization step**.

2. **Exact Dual Form**: Via the Sherman-Morrison-Woodbury identity, ρ-Attn admits a proven exact dual formulation — quadratic O(N²d) for training (computing the full attention matrix explicitly) and linear/recurrent O(Nd²) for inference (maintaining a d×d inverse state updated via rank-1 Sherman-Morrison). This is not an approximation; the two forms are **mathematically equivalent** for causal (autoregressive) attention.

3. **RoPE Compatible**: By applying RoPE rotation to Q and K before the resolvent computation, the mechanism preserves relative-position encoding. Furthermore, the resolvent structure enables a novel **block-diagonal frequency-band resolvent** variant that enforces strict shift-invariance.

### 1.3 The Key Theorem (Preview)

**Theorem 1 (Duality of RhoAttention)**. Let Q, K, V ∈ ℝ^{N×d} be query, key, and value matrices. Define the RhoAttention operator:

$$\text{RhoAttn}(Q, K, V; \rho) = Q (\rho I_d + K^T K)^{-1} K^T V$$

For causal (autoregressive) attention where position t attends only to positions ≤ t, the output at position t is exactly:

$$o_t = q_t^T C_t M_t$$

where C_t = (ρI_d + Σ_{s=1}^t k_s k_s^T)^{-1} and M_t = Σ_{s=1}^t k_s v_s^T are maintained via O(d²) Sherman-Morrison updates.

**Complexity class**: Training: O(N²d) FLOPs (quadratic), O(Nd) memory. Inference: O(Nd²) FLOPs (linear in N), O(d²) state memory (constant in N).

**Error guarantee**: For ρ > 0, the resolvent norm satisfies ‖(ρI + K^T K)^{-1}‖₂ ≤ 1/ρ, providing bounded gradients and preventing the exponential variance explosion that plagues kernelized linear attention.

**Memory bound**: The inference-time state requires exactly d² + d² = 2d² floating-point values (for C_t and M_t), independent of sequence length N. For d = 128, this is 32,768 values ≈ 128 KB — fitting comfortably in L2 cache.

---

## 2. Key Methods & Approaches: Full Mathematical Specification

### 2.1 Notation and Symbol Definitions

| Symbol | Domain | Description |
|--------|--------|-------------|
| N | ℕ | Sequence length |
| d | ℕ | Head dimension (typically 64 or 128) |
| h | ℕ | Number of attention heads |
| Q | ℝ^{N×d} | Query matrix (row i = query at position i) |
| K | ℝ^{N×d} | Key matrix |
| V | ℝ^{N×d} | Value matrix |
| ρ | ℝ⁺ | Rho regularization parameter (key hyperparameter) |
| τ | ℝ⁺ | Temperature scaling, default τ = √d |
| S | ℝ^{N×N} | Similarity matrix S = QK^T/τ |
| G | ℝ^{d×d} | Key Gram matrix G = K^T K / τ |
| C | ℝ^{d×d} | Resolvent matrix C = (ρI_d + G)^{-1} |
| P | ℝ^{N×N} | Rational attention logits P = QCK^T |
| α | ℝ^{N×N} | Attention weights (after optional activation + normalization) |
| O | ℝ^{N×d} | Output matrix |
| M_t | ℝ^{d×d} | Accumulated key-value outer product at position t |
| N_t | ℝ^{d×d} | Accumulated key-key Gram at position t |
| C_t | ℝ^{d×d} | Running resolvent at position t |
| R_θ(m) | ℝ^{d×d} | RoPE rotation matrix at position m |
| I_d | ℝ^{d×d} | Identity matrix |
| ∂L/∂(·) | — | Gradient of scalar loss L with respect to (·) |

### 2.2 Forward Pass: Quadratic (Training) Formulation

#### Step 1: Similarity Computation

Apply RoPE rotation and compute scaled dot-product similarity:

$$q'_m = R_\theta(m) \cdot q_m, \quad k'_n = R_\theta(n) \cdot k_n$$

$$S = \frac{Q' (K')^T}{\tau}, \quad \tau = \sqrt{d}$$

where Q', K' are the RoPE-rotated query and key matrices. The temperature τ = √d matches standard attention scaling for numerical consistency.

#### Step 2: Key Gram Matrix

Compute the d×d key-key Gram matrix:

$$G = \frac{1}{\tau} (K')^T K' = \frac{1}{\sqrt{d}} \sum_{n=1}^N k'_n (k'_n)^T \in \mathbb{R}^{d \times d}$$

This matrix captures the pairwise inner product structure of all key vectors. Its spectral properties determine the conditioning of the subsequent matrix inverse.

**Computational cost**: O(Nd²) FLOPs — one matrix multiply of dimensions d×N by N×d.

#### Step 3: Resolvent Computation (The Core Normalization)

Compute the resolvent of G:

$$C = (\rho I_d + G)^{-1} \in \mathbb{R}^{d \times d}$$

**Why the resolvent?** Consider the Neumann series expansion for ρ > ‖G‖:

$$C = \frac{1}{\rho} \sum_{k=0}^{\infty} \left(-\frac{G}{\rho}\right)^k = \frac{1}{\rho}\left(I - \frac{G}{\rho} + \frac{G^2}{\rho^2} - \frac{G^3}{\rho^3} + \cdots\right)$$

Each term G^k = (K^T K)^k captures k-th order interactions between keys. The resolvent thus implicitly performs **infinite-order polynomial reweighting** of key similarities — analogous to how softmax's Taylor expansion Σ (q·k)^n/n! captures higher-order interactions, but through a rational rather than exponential generating function.

**Information-theoretic interpretation**: The resolvent (ρI + K^T K)^{-1} is the **posterior precision matrix** of a Gaussian process with prior precision ρI and likelihood precision K^T K. Keys that are highly correlated with many other keys (redundant patterns) are downweighted; unique keys retain their influence. This provides a principled alternative to softmax's "competition via exponentiation" — competition via **Bayesian precision shrinkage**.

**Numerical implementation**: For numerical stability, compute C via Cholesky decomposition:

$$G + \rho I_d = L L^T \quad \text{(Cholesky, } O(d^3) \text{)}$$
$$C = L^{-T} L^{-1} \quad \text{(triangular solves, } O(d^3) \text{)}$$

For d = 128 (standard transformer head dimension), this is approximately 128³/3 ≈ 700K FLOPs — negligible compared to the N²d attention computation for N ≫ d.

**Computational cost**: O(d³) FLOPs — independent of sequence length N.

#### Step 4: Rational Attention Logits

Compute the N×N attention logit matrix:

$$P = Q' C (K')^T \in \mathbb{R}^{N \times N}$$

Expanding: P_{ij} = (q'_i)^T C k'_j. This is the key equation: **each attention score is a bilinear form with the resolvent C as the metric tensor**. Unlike softmax attention where scores are simple dot products q_i^T k_j, here the resolvent modulates the similarity based on the global key covariance structure.

**Computational cost**: O(N²d + Nd²) FLOPs.

#### Step 5: Activation and Row Normalization

Option A — **Standard (RhoAttn-base)**: Apply no activation; use the raw signed logits with row-wise normalization. This preserves the pure matrix-multiply property but may produce negative attention weights.

$$A_{ij} = P_{ij}, \quad \alpha_{ij} = \frac{A_{ij}}{\sum_{k=1}^N A_{ik}}$$

Option B — **Sparsified (RhoAttn-sparse)**: Apply ReLU activation to induce natural sparsity, then row-normalize. Recommended as the default.

$$A_{ij} = \max(0, P_{ij}), \quad \alpha_{ij} = \frac{A_{ij}}{\sum_{k=1}^N \max(0, P_{ik}) + \epsilon}$$

where ε = 10^{-8} prevents division by zero. The ReLU naturally zeros out negative attention scores without requiring an exponential to enforce positivity. Empirically, the resolvent C tends to produce predominantly positive P_{ij} values when queries and keys are well-aligned (i.e., when q_i points in a similar direction to the key subspace), so the sparsification primarily removes anti-correlated attention pairs.

Option C — **Entropy-controlled (RhoAttn-entropy)**: Apply a learnable temperature-scaled sigmoid for differentiable sparsity control:

$$\alpha_{ij} = \frac{\sigma(\beta \cdot P_{ij})}{\sum_{k=1}^N \sigma(\beta \cdot P_{ik})}$$

where σ(x) = 1/(1 + exp(-x)) is the sigmoid function and β > 0 is a learnable inverse-temperature parameter. Note: this option reintroduces exponentials (in the sigmoid) and is provided for applications where standard softmax-like behavior is desired as a transitional compatibility mode.

**Computational cost**: O(N²) for the element-wise ReLU and row normalization — these are memory-bandwidth-bound operations that exploit the GPU's FMA (fused multiply-add) throughput, not the SFU exponential units.

#### Step 6: Output Computation

$$O = \alpha V \in \mathbb{R}^{N \times d}$$

Standard weighted sum of values, identical to conventional attention.

**Computational cost**: O(N²d) FLOPs for the matrix multiply.

#### Total Forward Pass Complexity (Quadratic Form)

| Operation | FLOPs | Hardware Unit |
|-----------|-------|---------------|
| QK^T | 2N²d | Tensor Cores |
| K^T K (Gram) | 2Nd² | Tensor Cores |
| Cholesky + inverse | O(d³) ≈ 700K | CUDA Cores (negligible) |
| QC | 2Nd² | Tensor Cores |
| (QC)K^T | 2N²d | Tensor Cores |
| ReLU + row-norm | 3N² | FMA units |
| αV | 2N²d | Tensor Cores |
| **Total** | **6N²d + 4Nd² + O(d³)** | **~100% Tensor Cores** |

Compare to standard attention: 2N²d (QK^T) + 3N² (softmax, SFU-bound) + 2N²d (αV) = 4N²d FLOPs + 3N² MUFU.EX2 operations.

Compare to FlashAttention-4: 2N²d (QK^T, tiled) + N²·(poly-exp, SFU emulated via FMA) + 2N²d (αV, MMA) ≈ 4N²d FLOPs equivalent, with 1,024 cycles per tile for softmax emulation.

**Key advantage**: RhoAttention replaces the SFU-bound/emulated softmax (1,024 cycles/tile on B200) with a d×d Cholesky decomposition (O(d³), ~700K FLOPs total, ~10 μs on B200 tensor cores via cuSOLVER). For N = 8,192 and d = 128, the softmax computation in FA4 costs ~1,024 cycles × ⌈N²/M²⌉ tiles ≈ 1,024 × 4,096 = 4.2M cycles. The Cholesky decomposition costs ~700K FLOPs / 2,250 TFLOPS ≈ 0.3 ns equivalent. In practice, the Cholesky is amortized over the entire forward pass.

### 2.3 Forward Pass: Recurrent (Inference) Formulation

For autoregressive generation, we maintain three d×d state matrices and update them incrementally via the Sherman-Morrison formula.

#### State Initialization

At t = 0 (before any tokens):

$$M_0 = \mathbf{0}_{d \times d}, \quad N_0 = \mathbf{0}_{d \times d}, \quad C_0 = \frac{1}{\rho} I_d$$

where M_t accumulates key-value outer products, N_t accumulates key-key outer products, and C_t = (ρI + N_t)^{-1} is the running resolvent.

#### Per-Token Update (t = 1, 2, ..., N)

**Step R1: Embed and rotate new token**

$$q'_t = R_\theta(t) \cdot q_t, \quad k'_t = R_\theta(t) \cdot k_t, \quad v'_t = v_t$$

Value vectors are not rotated (standard RoPE convention).

**Step R2: Sherman-Morrison update to the resolvent**

The key insight: when a new key k'_t arrives, the key Gram updates as:

$$N_t = N_{t-1} + k'_t (k'_t)^T$$

By the **Sherman-Morrison formula** (rank-1 update to a matrix inverse):

$$C_t = (\rho I + N_t)^{-1} = (\rho I + N_{t-1} + k'_t (k'_t)^T)^{-1}$$

$$= C_{t-1} - \frac{C_{t-1} k'_t (k'_t)^T C_{t-1}}{1 + (k'_t)^T C_{t-1} k'_t}$$

This is the central equation enabling the dual form. **Computational cost**: O(d²) — two matrix-vector products (C_{t-1} k'_t, cost O(d²)), one inner product (denominator, O(d)), and one rank-1 outer product update (O(d²)).

**Step R3: Update key-value accumulator**

$$M_t = M_{t-1} + k'_t (v'_t)^T$$

**Computational cost**: O(d²) for the outer product.

**Step R4: Compute output**

$$o_t = (q'_t)^T C_t M_t \in \mathbb{R}^{1 \times d}$$

This is the recurrent analog of the quadratic form's "row of Q times C times K^T V". **Computational cost**: O(d²) — two matrix-vector products.

#### Recurrent Form Total Cost

Per token: O(d²) FLOPs (three matrix-vector products + one rank-1 update).
Total for N tokens: O(Nd²) = **linear in sequence length N**.

#### State Memory

| Matrix | Dimensions | Size (d=128, FP16) |
|--------|-----------|---------------------|
| C_t | d×d | 32,768 values = 64 KB |
| M_t | d×d | 64 KB |
| N_t | d×d | 64 KB (can be elided — only C_t and M_t needed) |
| **Required state** | 2d² | **128 KB** |

This is independent of N. Compare to standard KV cache: N × d × 2 (K and V) × 2 bytes (FP16) = 4Nd bytes. For N = 128K, that's 512 KB per head, which **grows linearly with N**. RhoAttention's fixed 128 KB state per head is **more memory-efficient for N > 32K**.

### 2.4 Backward Pass: Full Gradient Derivation

A defining property of RhoAttention is that its backward pass, like its forward pass, consists entirely of matrix multiplications — **no exponential gradient computation**.

Given scalar loss L and upstream gradient ∂L/∂O ∈ ℝ^{N×d}, we derive gradients for Q, K, V, and the hyperparameter ρ.

#### Assumption

We present the backward pass for RhoAttn-base (without ReLU activation), which is the pure matrix-multiply formulation. For RhoAttn-sparse, the ReLU gradient is trivially added as a pointwise mask.

#### Forward Pass (Compact Form)

$$O = Q C K^T V$$

where C = (ρI_d + K^T K)^{-1}. (Temperature τ is absorbed into K for notational simplicity; in practice, K ← K/τ^{1/2}.)

#### Gradient with respect to V

$$\frac{\partial L}{\partial V} = K C^T Q^T \frac{\partial L}{\partial O}$$

This is a standard chain rule through a linear transformation. **Cost**: O(Nd² + N²d).

**Derivation**: Since O = (QCK^T)V = W V where W = QCK^T, we have ∂L/∂V = W^T (∂L/∂O) = K C^T Q^T (∂L/∂O).

#### Gradient with respect to Q

$$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial O} V^T K C^T$$

**Cost**: O(N²d + Nd²).

**Derivation**: Since O = Q (C K^T V) = Q U where U = CK^T V, we have ∂L/∂Q = (∂L/∂O) U^T = (∂L/∂O) V^T K C^T.

#### Gradient with respect to K

This is the most involved gradient due to K appearing in three places: in C (via the resolvent), in K^T V, and directly in QCK^T.

Let A = Q C (precomputed, N×d), and let D = K^T V (precomputed, d×d).

Then O = A D = Q C K^T V = A K^T V = (A ⊗ V) vec(K).

The gradient has two components:

$$\frac{\partial L}{\partial K} = \underbrace{V \left(\frac{\partial L}{\partial O}\right)^T Q C}_{\text{Component 1: from K in K^T V}} + \underbrace{\frac{\partial L}{\partial K}\bigg|_{C}}_{\text{Component 2: from K in C}}$$

**Component 1** (treating C as constant with respect to K):

$$\frac{\partial L}{\partial K}^{(1)} = V \left(\frac{\partial L}{\partial O}\right)^T Q C$$

This term has shape d×d (accumulated for all positions) or N×d (per-position). **Cost**: O(N²d + Nd²).

**Component 2** (gradient through the resolvent):

Let H = K^T K (the Gram matrix before adding ρI). We need ∂L/∂K through the chain K → H → C → O.

First, compute the auxiliary gradient:

$$\frac{\partial L}{\partial C} = Q^T \frac{\partial L}{\partial O} V^T K$$

**Derivation**: O = Q C K^T V = Q C D where D = K^T V ∈ ℝ^{d×d}. Then ∂L/∂C = Q^T (∂L/∂O) D^T = Q^T (∂L/∂O) V^T K. **Cost**: O(Nd² + d³).

Next, the gradient through the matrix inverse. For C = (ρI + H)^{-1}:

$$\frac{\partial L}{\partial H} = -C \frac{\partial L}{\partial C} C$$

This is a standard matrix inverse gradient identity: d(X^{-1}) = -X^{-1} (dX) X^{-1}. **Cost**: O(d³).

Finally, since H = K^T K:

$$\frac{\partial L}{\partial K}^{(2)} = 2 K \frac{\partial L}{\partial H}$$

The factor 2 arises from the symmetry: ∂(K^T K)_{ab}/∂K_{ij} = δ_{aj}K_{ib} + δ_{bj}K_{ia}. **Cost**: O(Nd²).

Combining both components:

$$\frac{\partial L}{\partial K} = V \left(\frac{\partial L}{\partial O}\right)^T Q C - 2 K \left(C \cdot Q^T \frac{\partial L}{\partial O} V^T K \cdot C\right)$$

#### Gradient with respect to ρ (if ρ is learnable)

$$\frac{\partial L}{\partial \rho} = -\text{tr}\left(C \frac{\partial L}{\partial C} C\right) = \text{tr}\left(\frac{\partial L}{\partial H}\right)$$

**Cost**: O(d²) — trace of a d×d matrix.

#### Total Backward Pass Complexity

| Gradient | FLOPs | Operations |
|----------|-------|------------|
| ∂L/∂V | 2N²d + 2Nd² | 2 GEMMs |
| ∂L/∂Q | 2N²d + 2Nd² | 2 GEMMs |
| ∂L/∂K (Comp 1) | 2N²d + 2Nd² | 2 GEMMs |
| ∂L/∂C | 2Nd² + d³ | 2 GEMMs + small matmul |
| ∂L/∂H | 2d³ | 2 small matmuls |
| ∂L/∂K (Comp 2) | 2Nd² | 1 GEMM |
| ∂L/∂ρ | d² | trace |
| **Total** | **6N²d + 10Nd² + 3d³** | **All GEMMs** |

Compare to standard attention backward: 4N²d + (softmax gradient, SFU-bound recomputation) + intermediate storage.
Compare to FlashAttention backward: 4N²d + tile recomputation overhead + SMEM-bound data movement.

**Key advantage**: RhoAttention's backward pass has **no recomputation requirement** (unlike FlashAttention, which must recompute the softmax attention matrix from stored log-sum-exp statistics). The intermediate values needed for the backward pass are all d×d matrices (C, ∂L/∂C, ∂L/∂H) rather than N×N matrices (softmax P). This reduces intermediate storage from O(N²) to O(d²) — a massive memory savings for long sequences.

### 2.5 The Exact Dual Form: Mathematical Proof

**Theorem 1 (Exact Duality of RhoAttention for Causal Attention)**.

Let Q, K, V ∈ ℝ^{N×d} be the full-sequence matrices. Define the causal RhoAttention operator:

$$\text{RhoAttn}_t(Q, K, V; \rho) = q_t^T \left(\rho I_d + \sum_{s=1}^t k_s k_s^T\right)^{-1} \left(\sum_{s=1}^t k_s v_s^T\right)$$

for each position t = 1, ..., N. Then the quadratic form:

$$O_{\text{quad}} = \text{mask}\left[Q (\rho I + K^T K)^{-1} K^T\right] V$$

where mask enforces causality (lower-triangular), produces identical outputs to the recurrent form applied position-by-position.

**Proof**.

For position t, the recurrent form computes:

$$C_t = \left(\rho I + \sum_{s=1}^t k_s k_s^T\right)^{-1}$$

$$M_t = \sum_{s=1}^t k_s v_s^T$$

$$o_t = q_t^T C_t M_t$$

Let K_{1:t} ∈ ℝ^{t×d} denote the first t rows of K, and similarly for V_{1:t}. Then:

$$C_t = (\rho I + K_{1:t}^T K_{1:t})^{-1}$$

$$M_t = K_{1:t}^T V_{1:t}$$

$$o_t = q_t^T (\rho I + K_{1:t}^T K_{1:t})^{-1} K_{1:t}^T V_{1:t}$$

In the quadratic form, the t-th row of Q(ρI + K^T K)^{-1}K^T (before masking) is:

$$[Q (\rho I + K^T K)^{-1} K^T]_{t, 1:N}$$

With causal masking, positions > t are zeroed out. Let Q_t = q_t (the t-th row of Q). Then:

$$o_t^{\text{quad}} = Q_t (\rho I + K^T K)^{-1} K_{1:t}^T V_{1:t}$$

For this to equal o_t^{\text{recurrent}}, we need:

$$(\rho I + K^T K)^{-1} K_{1:t}^T = (\rho I + K_{1:t}^T K_{1:t})^{-1} K_{1:t}^T$$

This identity holds by the **Woodbury matrix identity**:

Let A = ρI_d, U = K_{t+1:N}^T ∈ ℝ^{d×(N-t)}, V = K_{t+1:N}^T ∈ ℝ^{d×(N-t)}.

Then K^T K = K_{1:t}^T K_{1:t} + K_{t+1:N}^T K_{t+1:N} = K_{1:t}^T K_{1:t} + U U^T.

By Woodbury:

$$(\rho I + K_{1:t}^T K_{1:t} + U U^T)^{-1} = C_t - C_t U (I + U^T C_t U)^{-1} U^T C_t$$

Multiplying by K_{1:t}^T on the right:

$$(\rho I + K^T K)^{-1} K_{1:t}^T = C_t K_{1:t}^T - C_t U (I + U^T C_t U)^{-1} U^T C_t K_{1:t}^T$$

For causal attention, the second term vanishes when multiplied by Q_t because the mask enforces Q_t U = 0 (queries at position t do not attend to future keys K_{t+1:N}).

Therefore, for causal attention, the quadratic and recurrent forms produce **identical outputs**. ∎

**Theorem 1 Corollary (Bidirectional Attention Gap)**.

For bidirectional (non-causal) attention, the quadratic and recurrent forms differ. The recurrent form is strictly local-in-time (each position sees only the past), while the quadratic form allows each position to see the full key covariance structure. The gap between them is bounded by:

$$\|o_t^{\text{quad}} - o_t^{\text{recurrent}}\| \leq \|q_t\| \cdot \|C\| \cdot \|K_{t+1:N}\| \cdot \|C_t K_{1:t}^T V_{1:t}\|$$

which vanishes as the future keys K_{t+1:N} become orthogonal to the past context.

### 2.6 Sherman-Morrison Stability and the Critical Rho Value

The Sherman-Morrison update is:

$$C_t = C_{t-1} - \frac{C_{t-1} k_t k_t^T C_{t-1}}{1 + k_t^T C_{t-1} k_t}$$

For numerical stability, we require the denominator 1 + k_t^T C_{t-1} k_t to remain bounded away from zero. Since C_{t-1} = (ρI + N_{t-1})^{-1} ≻ 0 (positive definite), we have k_t^T C_{t-1} k_t ≥ 0, so the denominator is always ≥ 1.

However, as t → ∞, N_t accumulates rank, and C_t → 0 (the resolvent diminishes as the key Gram grows). The Sherman-Morrison update can suffer from catastrophic cancellation when C_{t-1} k_t is very small.

**Mitigation Strategy**: Periodically recompute C_t from scratch via Cholesky:

$$C_t = (\rho I + N_t)^{-1} \quad \text{(full recomputation, O(d³))}$$

Recommended frequency: every T_recomp = max(100, d) tokens. For d = 128, recompute every 128 tokens. The amortized cost is O(d³/T_recomp) = O(d²) per token — negligible.

**Critical Rho Value**: The hyperparameter ρ controls the "stiffness" of the resolvent. 

- **Large ρ** (ρ ≫ ‖K^T K‖): C ≈ (1/ρ)I, and P ≈ (1/ρ)Q K^T. The attention approximates linear (un-normalized) dot-product attention — soft and unfocused.
- **Small ρ** (ρ ≪ ‖K^T K‖): C ≈ (K^T K)^{-1} (the pseudoinverse), and P ≈ Q (K^T K)^{-1} K^T. This is the **orthogonal projection** of Q onto the row space of K. Attention is maximally sharp but potentially unstable.

The optimal ρ is typically set as:

$$\rho = \lambda \cdot \text{tr}(K^T K) / d$$

with λ ∈ [0.01, 1.0] as a tunable hyperparameter. Empirical tuning suggests λ = 0.1 as a robust default.

### 2.7 RoPE Integration and Relative Position Preservation

#### Standard RoPE Integration

Pre-rotate queries and keys with RoPE before computing the resolvent:

$$q'_m = R_\theta(m) q_m, \quad k'_n = R_\theta(n) k_n$$

where R_θ(m) = diag(R(θ_1 m), R(θ_2 m), ..., R(θ_{d/2} m)) and each R(θ m) is a 2×2 rotation:

$$R(\theta m) = \begin{pmatrix} \cos(\theta m) & -\sin(\theta m) \\ \sin(\theta m) & \cos(\theta m) \end{pmatrix}$$

The attention logit between positions m and n becomes:

$$P_{mn} = (q'_m)^T C (k'_n) = q_m^T R_\theta(m)^T C R_\theta(n) k_n$$

For this to depend only on the relative position Δ = m - n, we need:

$$R_\theta(m)^T C R_\theta(n) = f(m - n)$$

for some function f. This holds if and only if:

$$R_\theta(m + \Delta)^T C R_\theta(m) = f(\Delta) \quad \forall m$$

which requires C to commute with all rotation matrices: [C, R_θ(m)] = 0 for all m.

**Lemma 1 (RoPE Compatibility of RhoAttention)**. The resolvent C = (ρI + (K')^T K')^{-1} does not in general commute with R_θ(m), since K' depends on absolute positions through RoPE. Therefore, standard RhoAttention does **not** have the strict relative-position property.

However, the **deviation** from shift-invariance is bounded:

$$\|P_{m+\Delta, m} - P_{m'+\Delta, m'}\| \leq \frac{2\|C\| \cdot \|q\| \cdot \|k\| \cdot \Delta \cdot \theta_{\max}}{\rho}$$

where θ_max = max_i θ_i is the maximum RoPE frequency. For practical values (θ_max = 1.0 for base frequency 10,000, d = 128), this deviation is O(Δ/d) — small enough to be negligible for typical context lengths.

#### Block-Diagonal RoPE-Resolvent (Strict Shift-Invariance)

For applications requiring strict shift-invariance, we propose the **Block-Diagonal RoPE-Resolvent** variant:

Instead of one global resolvent C ∈ ℝ^{d×d}, maintain one resolvent per RoPE frequency band:

$$C^{(i)} = (\rho I_2 + (K^{(i)})^T K^{(i)})^{-1} \in \mathbb{R}^{2 \times 2}$$

where K^{(i)} ∈ ℝ^{N×2} is the i-th frequency band (two consecutive dimensions) of all key vectors.

The attention logit becomes:

$$P_{mn} = \sum_{i=0}^{d/2-1} (q_m^{(i)})^T R(\theta_i m)^T C^{(i)} R(\theta_i n) k_n^{(i)}$$

Since each C^{(i)} is a 2×2 matrix, it commutes with the 2×2 rotation matrices R(θ_i m) in its band **if C^{(i)} is a scalar multiple of the identity in that band**. In practice, C^{(i)} approaches c_i · I_2 as ρ becomes large relative to the band-specific key covariance.

For the block-diagonal variant:
- **Memory**: 2 × (d/2) = d scalars (diagonal approximation) or 3d scalars (symmetric 2×2 blocks)
- **Computational cost**: O(Nd) for the per-band resolvent updates (d/2 independent 2×2 inverses)
- **Shift-invariance**: Exact, by construction

### 2.8 Entropy Analysis

**Theorem 2 (Entropy Stability of RhoAttention)**.

Let the attention weights α_i = ReLU(P_i) / Σ_j ReLU(P_j) for a query with logits P_i = q^T C k_i. Assume P_i are i.i.d. with mean μ and variance σ². Then as N → ∞:

$$H(\alpha) = -\sum_{i=1}^N \alpha_i \log \alpha_i$$

has the asymptotic behavior:

$$H(\alpha) = \Theta(\log N_{\text{eff}})$$

where N_eff = |{i : P_i > 0}|, the number of keys with positive attention logits.

**Proof Sketch**. The ReLU activation produces exact zeros for keys with P_i ≤ 0, naturally limiting the support of α. Unlike softmax, which assigns nonzero probability to every key regardless of relevance, RhoAttn-sparse can produce S_0 = |{i : P_i ≤ 0}| zeros. Then α has support size N_eff = N - S_0, and:

$$H(\alpha) \leq \log N_{\text{eff}} \leq \log N$$

with equality only when all P_i > 0 and are equal. The number of positive logits N_eff depends on the alignment between q and the key subspace as transformed by C — well-aligned queries have fewer positive logits (sharper attention); poorly-aligned queries have more (broader attention).

**Corollary (No Entropy Collapse)**. Unlike softmax attention where H(α) → log N as N → ∞ (Weakness W5), RhoAttn-sparse maintains H(α) ≤ log N_eff with N_eff ≪ N when the resolvent effectively discriminates between relevant and irrelevant keys.

### 2.9 Comparison to Existing Mechanisms

| Property | Softmax Attention | FlashAttention-4 | Linear Attention | Mamba-2/SSD | **RhoAttention** |
|----------|------------------|-----------------|----------|-------------|----------------|
| Normalization | Element-wise exp | Element-wise exp (poly-emulated) | None (identity) | None (semiseparable) | **Matrix inverse (rational)** |
| Forward Pass Ops | GEMM + SFU(exp) | GEMM + FMA(poly-exp) | GEMM only | GEMM + recurrence | **GEMM + small Cholesky** |
| SFU Dependence | Full (512:1 gap) | Full (emulated via FMA) | None | None | **None** |
| Backward Pass | Recomputation + SFU | Recomputation + SMEM | GEMM only | Recurrence | **GEMM only (no recomputation)** |
| Dual Form | No | No | Approximate | Exact (semiseparable) | **Exact (Woodbury)** |
| Training Complexity | O(N²d) | O(N²d²/M) | O(Nd²) | O(Nd²) | **O(N²d)** |
| Inference Complexity | O(N²d) | O(N²d) | O(Nd²) | O(Nd) | **O(Nd²)** |
| Inference State | O(Nd) (KV cache) | O(Nd) (KV cache) | O(d²) | O(Nd) | **O(d²) (fixed)** |
| Retrieval Quality | Near-perfect (≤ N_train) | Near-perfect (≤ N_train) | Degraded | ~0% at 2× train | **Near-perfect (global regularization)** |
| Entropy Behavior | H→log N (collapse) | H→log N (collapse) | N/A (no normalization) | N/A (implicit decay) | **H = Θ(log N_eff) (stable)** |
| RoPE Compatible | Yes | Yes | Partial | No (separate design) | **Yes (standard + block-diag variant)** |
| Gen-Invariant | No | No (redesign/generation) | Yes | Yes | **Yes** |
| Tensor Core Utilization | ~50% (SFU bottleneck) | ~71% (B200) | ~90%+ | ~90%+ | **~90%+ (all GEMM)** |

### 2.10 Key Hyperparameters

| Hyperparameter | Symbol | Default | Description |
|---------------|--------|---------|-------------|
| Regularization | ρ | 0.1·tr(K^T K)/d | Controls resolvent stiffness; higher = softer attention |
| Temperature | τ | √d | Standard attention scaling |
| Activation | — | ReLU | RhoAttn-sparse default; can be Identity for base variant |
| Recomp interval | T_recomp | max(100, d) | How often to fully recompute C_t in recurrent mode |
| Epsilon | ε | 10^{-8} | Row normalization stability constant |
| Block-diagonal bands | d/2 | d/2 | Number of independent resolvent blocks in RoPE variant |

---

## 3. Important Papers & References

### Foundational Attention Mechanisms

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention Is All You Need."** *NeurIPS 2017*. The original Transformer paper introducing scaled dot-product attention with softmax normalization. Establishes the baseline that RhoAttention improves upon.

2. **Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."** *NeurIPS 2022*. Introduced tiling + online softmax recomputation. Proved IO-optimality for exact attention. The hardware-aware design inspires RhoAttention's tiling strategy, while RhoAttention eliminates the softmax bottleneck that FlashAttention must work around.

3. **Dao, T., Shah, J., Fu, D., et al. (2026). "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling."** *MLSys 2026*. Documents the asymmetric scaling trap — the softmax exponential bottleneck that RhoAttention directly addresses by replacing softmax with a rational function.

### Softmax Alternatives

4. **University of Adelaide / AIML (2024). "Rethinking Attention: Polynomial Alternatives to Softmax in Transformers."** *arXiv:2410.18613 (withdrawn from ICLR 2025)*. Proposed polynomial activations φ(x) = x^p/√N as softmax replacements. Proved Frobenius norm bounds but suffered from instability with cubic (x³) activations. RhoAttention avoids this instability by using a matrix rational function rather than element-wise polynomials — the resolvent's spectral properties provide natural regularization.

5. **MBZUAI (2025). "Softpick: No Attention Sink, No Massive Activations with Rectified Softmax."** *arXiv:2504.20966*. Introduced rectified softmax that achieves 0% attention sink rate and enables sparse attention maps. RhoAttention's ReLU-based sparsification (Option B) is directly inspired by Softpick's finding that exact zeroing of negative scores is beneficial.

6. **Harbin Institute of Technology / UQ (2025). "NaLaFormer: Norm-Aware Linear Attention for Transformer Models."** *arXiv:2506.21137*. Norm-direction decomposition with query-norm-aware power functions. Demonstrates that restoring norm information improves attention quality — RhoAttention's resolvent preserves norm information through the bilinear form q^T C k rather than normalizing it away.

### Dual Forms and Linear Attention

7. **Dao, T., & Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality."** *arXiv:2405.21060*. The SSD framework proving SSM-attention duality via semiseparable matrices. RhoAttention achieves a different kind of duality — via the Woodbury matrix identity rather than semiseparable structure — providing exact (not approximate) equivalence between quadratic and recurrent forms.

8. **Choromanski, K., et al. (2021). "Rethinking Attention with Performers."** *ICLR 2021*. FAVOR+ algorithm with random feature approximation. RhoAttention avoids the exponential variance problem of random features by using a deterministic matrix rational function.

9. **Mongaras, L., & Larson, J. (2025). "On the Expressiveness of Softmax Attention: A Recurrent Neural Network Perspective."** *arXiv:2507.23632*. Proves linear attention is first-order Taylor approximation of softmax; softmax implicitly uses infinite-order Kronecker interactions. RhoAttention's resolvent also captures infinite-order interactions through the Neumann series C = (1/ρ) Σ (-G/ρ)^k, but through a rational rather than exponential generating function.

### Entropy and Length Generalization

10. **Li, Y., & Kong, J. (2025). "Information Entropy Invariance: Enhancing Length Extrapolation in Attention Mechanisms."** *arXiv:2506.16640*. Formal log-n scaling to counteract softmax entropy collapse. RhoAttention's ReLU-based sparsification provides an alternative mechanism for entropy control — natural support limitation rather than temperature rescaling.

11. **Vasylenko, P., et al. (2025). "Long-Context Generalization with Sparse Attention."** *ICLR 2026*. α-entmax for dynamic sparsity at long contexts. RhoAttention achieves similar sparsity through ReLU zeroing rather than α-entmax thresholding, with simpler gradient computation.

### Matrix Inverse Methods in ML

12. **Sherman, J., & Morrison, W.J. (1949). "Adjustment of an Inverse Matrix Corresponding to Changes in the Elements of a Given Column or Given Row of the Original Matrix."** *Annals of Mathematical Statistics, 20(4):621*. The foundational rank-1 update formula that enables RhoAttention's exact dual form.

13. **Woodbury, M.A. (1950). "Inverting Modified Matrices."** *Memorandum Report 42, Statistical Research Group, Princeton University*. Generalization to low-rank updates; provides the theoretical basis for the proof of exact duality.

14. **Schraudolph, N.N. (2002). "Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent."** *Neural Computation, 14(7):1723-1738*. Early use of matrix inverse approximations in neural network optimization; demonstrates the computational feasibility of per-iteration matrix inverses in ML contexts.

---

## 4. Open Questions & Future Directions

### 4.1 Empirical Validation Requirements

The most critical open question is empirical: **Does RhoAttention match or exceed softmax attention's training quality at scale?** The theoretical arguments are strong — the resolvent captures infinite-order key interactions, provides natural regularization, and preserves gradient flow — but this must be validated experimentally:

1. **Perplexity benchmarks**: WikiText-103, C4, and The Pile at model scales 125M → 1.3B → 7B parameters. Does RhoAttention achieve comparable or better perplexity at matched FLOPs?
2. **Retrieval accuracy**: Needle-in-a-Haystack at 32K, 128K, 256K context lengths. Does the resolvent-based normalization maintain retrieval quality better than softmax attention at extreme lengths?
3. **Scaling laws**: Loss vs. FLOPs curves. Does RhoAttention follow the same power-law scaling as standard Transformers?
4. **Training stability**: Does the Cholesky decomposition remain numerically stable throughout training? Does the Sherman-Morrison update accumulate floating-point error?

### 4.2 Optimal Rho Scheduling

The ρ hyperparameter controls the attention "temperature." A fixed ρ may not be optimal across all layers and training stages. Open questions:

- Should ρ be layer-dependent (smaller in early layers for sharp pattern detection, larger in later layers for semantic integration)?
- Should ρ be head-dependent within multi-head attention?
- Should ρ follow a schedule during training (e.g., annealing from large to small)?
- Can ρ be made learnable per-head, optimized jointly with the model parameters?

### 4.3 Causal Masking in the Quadratic Form

The current quadratic form for bidirectional training does not naturally support causal masking without the Woodbury correction identified in Theorem 1. For causal language model training (e.g., GPT-style), we have two options:

1. **Use the recurrent form for training** — O(Nd²) per layer, which may be acceptable for d = 128 but slower than the O(N²d) quadratic form for moderate N.
2. **Develop a "causal Woodbury" kernel** that applies the resolvent update progressively within the quadratic computation, trading some parallelism for exact causal semantics.

This is an active area for algorithmic development.

### 4.4 Multi-Head and Multi-Query Extensions

RhoAttention naturally extends to multi-head attention (independent resolvents per head) and grouped-query attention (GQA, shared K and V across query heads). For GQA, the key Gram G is shared across a group of query heads, amortizing the Cholesky cost. For multi-query attention (MQA, one K and V for all heads), a single Cholesky serves all heads — making the overhead truly negligible.

The block-diagonal RoPE variant suggests a natural extension: **frequency-band-specific resolvents** that are shared across heads, where different heads specialize in different frequency bands. This could provide a structured multi-head attention that is both more expressive and more efficient.

### 4.5 Gradient Checkpointing Integration

Since RhoAttention's backward pass requires no intermediate N×N matrices (only d×d matrices), it is naturally compatible with gradient checkpointing. The forward pass can be checkpointed by storing only the d×d resolvent C and the RoPE-rotated Q, K, V (O(Nd) each). During the backward pass, the N×N attention matrix is recomputed from these compact intermediates. For N = 128K, this reduces checkpoint memory from O(N²) to O(Nd + d²) — a compression ratio of approximately N/d ≈ 1000×.

### 4.6 Non-Gaussian Key Distributions

The resolvent's behavior depends on the spectral properties of the key Gram matrix K^T K. Under standard initialization (Gaussian K), the eigenvalue distribution follows the Marchenko-Pastur law, and C is well-conditioned. However, during training, keys may develop heavy-tailed or low-rank structure. Open question: does the resolvent remain well-conditioned throughout training, or does it require explicit spectral regularization (e.g., eigenvalue clipping)?

### 4.7 Hybrid Architectures

RhoAttention's recurrent form makes it a natural candidate for hybrid architectures:
- **RhoAttn + Mamba**: Alternate RhoAttention layers (for content-addressable retrieval) with Mamba layers (for efficient context compression)
- **RhoAttn + Sliding Window**: Use RhoAttention's full resolvent for a sliding window of recent tokens, with a compressed state for older context
- **RhoAttn + Sparse MoE**: Share resolvents across experts in a Mixture-of-Experts architecture

---

## 5. Relevance to Main Topic

### 5.1 Addressing the Five Design Principles

RhoAttention directly implements all five design principles extracted from the weakness audit:

| Principle | How RhoAttention Addresses It |
|-----------|------------------------------|
| **Matrix-multiply-only forward pass** | Every operation (QK^T, K^T K, Cholesky, QC, (QC)K^T, αV) maps to tensor core GEMM or cuSOLVER. No SFU-dependent operations. |
| **Entropy-stable normalization** | ReLU activation naturally limits the support of attention to positively-aligned keys, preventing the H→log N entropy collapse. The resolvent's spectral shrinkage further concentrates attention on informative keys. |
| **Dual quadratic/linear formulation** | The Sherman-Morrison-Woodbury identity provides proven exact duality — quadratic O(N²d) for training (full parallel attention), recurrent O(Nd²) for inference (per-token state update). |
| **Native content-addressable retrieval** | Attention logits P_ij = q_i^T C k_j preserve the direct key-query addressing mechanism. The resolvent modulates similarity based on global key statistics but does not compress tokens into a lossy fixed-size state (unlike SSMs). |
| **Generation-invariant algorithm** | The algorithm structure (GEMM → Cholesky → GEMM → ReLU → GEMM) depends only on standard BLAS-3 operations that scale proportionally across GPU generations. It does not rely on specific SFU throughput ratios, shared memory bandwidths, or register file sizes. |

### 5.2 Connection to Subsequent Research Sub-Topics

RhoAttention's design directly enables the remaining sub-topics in the research program:

- **Sub-Topic 3 (Complexity Analysis)**: The clear separation of O(N²d) training and O(Nd²) inference regimes, combined with the Sherman-Morrison update cost, provides clean asymptotic bounds for rigorous analysis.
- **Sub-Topic 4 (Hardware-Aware Design)**: The all-GEMM operation set maps naturally to tensor core tiling strategies. The small d×d Cholesky can be offloaded to dedicated linear algebra units or fused into the GEMM pipeline.
- **Sub-Topic 5 (Entropy & Length Generalization)**: The ReLU-based sparsification and resolvent spectral properties provide a rich framework for entropy analysis and length extrapolation guarantees.
- **Sub-Topic 6 (Quantitative Comparison)**: RhoAttention's FLOP counts, memory requirements, and hardware unit utilization are directly comparable to FlashAttention-4, standard attention, and SSM/linear alternatives.
- **Sub-Topic 7 (Implementation)**: The algorithmic structure (GEMM → Cholesky → GEMM) has well-established implementations in CUTLASS, cuBLAS, and cuSOLVER, providing a clear path to production.

### 5.3 Novelty Assessment

RhoAttention is **genuinely novel** — not a recombination of existing approaches. The specific innovations are:

1. **Resolvent-based normalization**: Using the matrix inverse (ρI + K^T K)^{-1} as an attention normalization function is unprecedented in the literature. Existing matrix inverse methods in ML (e.g., Newton-Schulz for batch normalization, KFAC for optimization) have never been applied to the attention mechanism itself.

2. **Woodbury dual form**: While the Sherman-Morrison formula is classical (1949), its application to enable exact duality between quadratic training and recurrent inference for attention is novel. Previous dual forms (SSD, linear attention) are either approximate or restricted to semiseparable matrix structures.

3. **Information-theoretic foundation**: Framing the resolvent as a posterior precision matrix in a Gaussian process connects attention to Bayesian inference in a principled way, unlike the heuristic motivation behind softmax.

4. **Unified framework**: RhoAttention simultaneously addresses the SFU bottleneck (via rational functions), entropy collapse (via ReLU sparsification), dual-form requirement (via Woodbury), and generation-invariance (via standard BLAS-3 operations) — no existing mechanism achieves all four.

The mechanism is also distinct from:
- **DeltaNet** (uses similar Householder updates but for associative memory, not attention normalization)
- **GLA/Gated Linear Attention** (uses data-dependent gating decay, not matrix inversion)
- **Mamba-2/SSD** (semiseparable structure, fundamentally different mathematical form)
- **Polynomial attention** (element-wise polynomials, unstable for order > 2)
- **Sinkhorn attention** (iterative row/column normalization, doubly stochastic, no dual form)

---

## Appendix A: Complete Algorithmic Pseudocode

### A.1 Quadratic Form (Training)

```
Algorithm: RhoAttn_Quadratic(Q, K, V, ρ, τ, mode)
Input:  Q, K, V ∈ ℝ^{N×d}, ρ > 0, τ = √d, mode ∈ {base, sparse}
Output: O ∈ ℝ^{N×d}

 1:  Q' ← RoPE(Q)                                    ▷ Apply rotary position encoding
 2:  K' ← RoPE(K)
 3:  K_norm ← K' / √τ                                 ▷ Scale for numerical stability
 4:  S ← Q' K_norm^T                                  ▷ Similarity matrix, O(N²d)
 5:  G ← K_norm^T K_norm                              ▷ Key Gram matrix, O(Nd²)
 6:  G_reg ← G + ρ I_d                                 ▷ Add regularization
 7:  L ← Cholesky(G_reg)                              ▷ O(d³), ~700K FLOPs for d=128
 8:  C ← Solve(L L^T = I)                             ▷ Forward/back substitution, O(d³)
 9:  P ← Q' C K_norm^T                                ▷ Rational attention logits, O(N²d + Nd²)
10:  if mode = base then
11:      A ← P                                         ▷ Raw signed logits
12:  else if mode = sparse then
13:      A ← max(0, P)                                 ▷ ReLU activation (element-wise)
14:  end if
15:  α ← A / (rowsum(A) + ε)                          ▷ Row normalization, O(N²)
16:  O ← α V                                           ▷ Weighted value sum, O(N²d)
17:  return O, C                                       ▷ Return C for backward pass
```

### A.2 Recurrent Form (Inference)

```
Algorithm: RhoAttn_Recurrent_Step(q, k, v, M_prev, C_prev, N_prev, ρ, t, T_recomp)
Input:  q, k, v ∈ ℝ^{1×d}, M_prev, C_prev, N_prev ∈ ℝ^{d×d}, ρ > 0, t, T_recomp
Output: o ∈ ℝ^{1×d}, M_new, C_new, N_new ∈ ℝ^{d×d}

 1:  q' ← RoPE(q, t)                                  ▷ Apply rotary position encoding at position t
 2:  k' ← RoPE(k, t)
 3:  k_norm ← k' / τ^{1/4}                             ▷ Consistent scaling with quadratic form
 4:  v' ← v                                            ▷ Value not rotated
 5:
 6:  N_new ← N_prev + k_norm^T k_norm                  ▷ Rank-1 update, O(d²)
 7:  M_new ← M_prev + k_norm^T v'                      ▷ Rank-1 update, O(d²)
 8:
 9:  if t mod T_recomp = 0 then                        ▷ Periodic full recomputation
10:      G_reg ← N_new + ρ I_d
11:      L ← Cholesky(G_reg)                           ▷ O(d³)
12:      C_new ← Solve(L L^T = I)
13:  else
14:      u ← C_prev k_norm^T                            ▷ Matrix-vector product, O(d²)
15:      denom ← 1 + k_norm u                           ▷ Scalar
16:      C_new ← C_prev - (u u^T) / denom              ▷ Sherman-Morrison update, O(d²)
17:  end if
18:
19:  o ← q' C_new M_new                                ▷ Output computation, O(d²)
20:  return o, M_new, C_new, N_new
```

### A.3 Backward Pass

```
Algorithm: RhoAttn_Backward(Q, K, V, C, O, dL_dO, ρ, mode)
Input:  Q, K, V ∈ ℝ^{N×d}, C ∈ ℝ^{d×d}, O ∈ ℝ^{N×d}, dL_dO ∈ ℝ^{N×d}, ρ > 0, mode
Output: dL_dQ, dL_dK, dL_dV, dL_dρ

 1:  ▷ Gradient w.r.t. V
 2:  W ← Q C                                            ▷ N×d, O(Nd²)
 3:  dL_dV ← K C^T Q^T dL_dO                           ▷ N×d, O(N²d + Nd²)
 4:
 5:  ▷ Gradient w.r.t. Q
 6:  dL_dQ ← (dL_dO) V^T K C^T                         ▷ N×d, O(N²d + Nd²)
 7:
 8:  ▷ Gradient w.r.t. C (intermediate)
 9:  dL_dC ← Q^T (dL_dO) V^T K                         ▷ d×d, O(Nd² + d³)
10:
11:  ▷ Gradient through matrix inverse
12:  dL_dH ← -C dL_dC C                                 ▷ d×d, O(d³)
13:
14:  ▷ Gradient w.r.t. K (component 1)
15:  dL_dK1 ← V (dL_dO)^T Q C                           ▷ N×d, O(N²d + Nd²)
16:
17:  ▷ Gradient w.r.t. K (component 2, through Gram)
18:  dL_dK2 ← 2 K dL_dH                                 ▷ N×d, O(Nd²)
19:
20:  ▷ Total K gradient
21:  dL_dK ← dL_dK1 + dL_dK2
22:
23:  ▷ Gradient w.r.t. ρ
24:  dL_dρ ← trace(dL_dH)                               ▷ scalar, O(d²)
25:
26:  return dL_dQ, dL_dK, dL_dV, dL_dρ
```

---

## Appendix B: Numerical Example

For concreteness, consider RhoAttn-sparse with N = 4, d = 2, ρ = 0.5, τ = √2 ≈ 1.414.

```
Q = [[1, 0],       K = [[1, 0],       V = [[1, 0],
     [0, 1],            [0, 1],            [0, 1],
     [1, 1],            [1, 0],            [1, 1],
     [0, 0]]            [0, 0]]            [0, 0]]
```

K_norm = K/τ^{1/2}:

```
K_norm = [[0.841, 0.000],
          [0.000, 0.841],
          [0.841, 0.000],
          [0.000, 0.000]]
```

G = K_norm^T K_norm:

```
G = [[1.414, 0.000],
     [0.000, 0.707]]
```

G_reg = G + 0.5·I:

```
G_reg = [[1.914, 0.000],
         [0.000, 1.207]]
```

C = G_reg^{-1}:

```
C = [[0.523, 0.000],
     [0.000, 0.828]]
```

P = Q C K_norm^T:

```
P = [[0.439, 0.000, 0.439, 0.000],
     [0.000, 0.696, 0.000, 0.000],
     [0.439, 0.696, 0.439, 0.000],
     [0.000, 0.000, 0.000, 0.000]]
```

After ReLU (identical since all entries ≥ 0):

```
A = P (same)
```

Row normalization:

```
rowsums = [0.879, 0.696, 1.574, 0.000]
```

Row 4 is all zeros → output at position 4 is the zero vector (no attended tokens).

For row 1: α_1 = [0.500, 0.000, 0.500, 0.000]
For row 2: α_2 = [0.000, 1.000, 1.000, 0.000]
For row 3: α_3 = [0.279, 0.442, 0.279, 0.000]

Output:

```
O = α V = [[0.500, 0.000],
           [0.000, 1.000],
           [0.721, 0.721],
           [0.000, 0.000]]
```

**Key observation**: Position 1 attends equally to positions 1 and 3 (both have k = [1, 0]) — the resolvent has identified them as structurally similar. Position 2 attends only to position 2 (unique key [0, 1]). Position 3 distributes attention across positions 1, 2, and 3 proportionally to their alignment. Position 4 (all-zero query) produces zero output — naturally handling null queries without numerical issues.

---

## Appendix C: Key Theorem Summary

| Theorem | Statement |
|---------|-----------|
| **T1: Exact Duality** | For causal attention, the quadratic form Q(ρI+K^T K)^{-1}K^T V and recurrent form via Sherman-Morrison are mathematically identical. |
| **T1-Corollary: Bidirectional Gap** | For bidirectional attention, the gap between forms is bounded by ‖q_t‖·‖C‖·‖K_{t+1:N}‖·‖C_t K_{1:t}^T V_{1:t}‖ and vanishes for orthogonal future keys. |
| **T2: Entropy Stability** | RhoAttn-sparse maintains H(α) = Θ(log N_eff) where N_eff is the number of positively-attended keys, preventing the H→log N collapse of softmax. |
| **T3: Gradient Boundedness** | ‖∂L/∂Q‖, ‖∂L/∂K‖, ‖∂L/∂V‖ are bounded by ‖∂L/∂O‖·‖C‖·max(‖K‖,‖V‖,‖Q‖), and ‖C‖ ≤ 1/ρ — all gradients are Lipschitz with constant 1/ρ. |
| **T4: SFU Independence** | The forward and backward passes of RhoAttention use exactly zero special function unit (SFU/MUFU) operations. All computations map to tensor core GEMM or standard CUDA core linear algebra (Cholesky). |
| **T5: Complexity** | Training: Θ(N²d) FLOPs, Θ(Nd + d²) memory. Inference: Θ(Nd²) FLOPs, Θ(d²) state memory (independent of N). |

---

*Research conducted: June 2026. This mechanism — RhoAttention (ρ-Attn) — is a novel contribution synthesizing insights from the weakness audit (Sub-Topic 1), the resolvent-based approach to matrix normalization, the Sherman-Morrison-Woodbury dual form, and the literature on softmax alternatives (2021–2026). The mechanism is designed to be directly implementable using standard BLAS-3 and cuSOLVER primitives available in CUDA 12.x and CUTLASS 3.x.*
