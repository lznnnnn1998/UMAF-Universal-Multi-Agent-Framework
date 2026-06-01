# Complexity Analysis, Error Bounds, and Theoretical Guarantees

## Overview

The computational and memory efficiency of attention mechanisms is governed by a fundamental tension between arithmetic intensity and data movement. Standard scaled dot-product attention requires \(O(N^2 d)\) FLOPs and \(O(N^2)\) memory to materialize the full \(N \times N\) attention matrix, making it **memory-bound** on modern GPU hardware — the arithmetic units sit idle while waiting for data from High Bandwidth Memory (HBM). FlashAttention (Dao et al., NeurIPS 2022) resolved this bottleneck through two insights: (1) **tiling** the computation into blocks that fit entirely in on-chip SRAM, and (2) using **online safe-softmax** to incrementally aggregate statistics without materializing the full intermediate matrix. This transforms the I/O complexity from \(\Theta(N^2 d)\) to \(\Theta(N^2 d^2 / M)\) where \(M\) is SRAM size, shifting attention from memory-bound to **compute-bound** — achieving ~8× wall-clock speedup on an A100 despite performing _more_ FLOPs due to recomputation in the backward pass.

The theoretical underpinnings extend well beyond FlashAttention's exact computation. For **exact** attention algorithms, Saha & Ye (ICML 2024) proved a matching I/O lower bound of \(\Omega(N^2 d^2 / M)\) using communication complexity and red-blue pebble game arguments, establishing FlashAttention's **I/O-optimality** when \(M \geq d^2\). A parallel line of work explores **approximate** attention — linear attention, Performers, and random feature methods that reduce the asymptotic complexity to \(O(N d^2)\) by kernelizing the softmax. These methods sacrifice exactness but can provably bound the approximation error using techniques from Bochner's theorem, Johnson-Lindenstrauss embeddings, and self-normalized importance sampling. The **softmax function** itself has been shown to be \(1/2\)-Lipschitz (Nair, 2025) — a tight bound across all \(\ell_p\) norms — directly informing contraction arguments for deep Transformer architectures.

Numerical stability presents a distinct challenge, particularly in low-precision regimes (FP16, BF16, FP8). The backward pass exhibits a critical vulnerability: the gradient tensor \(dS\) (gradient with respect to pre-softmax logits) has RMS on the order of \(10^{-7}\), roughly 500× smaller than \(dP\), making it exquisitely sensitive to quantization noise. Recent work from Tsinghua (ICLR 2026 Oral) identified two failure mechanisms in low-precision FlashAttention: (1) similar low-rank update directions in attention amplify rounding errors into systematic drift (rather than zero-mean noise), and (2) repeated maximum values in safe-softmax rows cause discrete triggering of numerical bias. Understanding these phenomena is essential for any proposed attention mechanism, whether exact or approximate.

---

## Key Methods & Approaches

### 1. Computational Complexity: FLOPs Analysis

#### 1.1 Standard Scaled Dot-Product Attention

Given query \(Q \in \mathbb{R}^{N \times d}\), key \(K \in \mathbb{R}^{N \times d}\), value \(V \in \mathbb{R}^{N \times d}\):

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
\]

The FLOP count decomposes as:

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| \(S = QK^\top\) | \(2N^2 d\) | Matrix multiply |
| \(P = \text{softmax}(S)\) | \(\sim 5N^2\) | Exponentiate, sum, normalize (per row) |
| \(O = PV\) | \(2N^2 d\) | Matrix multiply |
| **Forward Total** | \(\approx 4N^2 d + 5N^2\) | Dominated by \(4N^2 d\) for typical \(d\) |
| **Backward Total** | \(\approx 8N^2 d\) | Standard backward pass (store \(P\)) |
| **Training Step (Fwd+Bwd)** | \(\approx 12N^2 d\) | |

#### 1.2 FlashAttention FLOPs

FlashAttention-1/2 forward pass:
- Same matrix multiply FLOPs: \(2N^2 d\) for \(QK^\top\) and \(2N^2 d\) for \(PV\)
- Additional **online softmax rescaling** overhead: \(O(N^2 / B_c) \approx O(N^2 / d)\) per forward pass
- **Recomputation in backward**: additional \(4N^2 d\) FLOPs (recomputing \(S\) and \(P\) from stored statistics \(m, \ell\))
- **Total training FLOPs**: \(\approx 16N^2 d\) (vs. \(12N^2 d\) for standard attention)

**Key paradox resolved**: FlashAttention performs ~33% more FLOPs in training yet achieves ~8× wall-clock speedup. The explanation lies in the **roofline model** — standard attention is memory-bound (arithmetic intensity \(\approx 64\) FLOP/byte on A100), while FlashAttention is compute-bound (arithmetic intensity \(\approx 506\) FLOP/byte). The extra compute happens in fast on-chip SRAM where it is essentially free relative to HBM access latency. One HBM read (~200 cycles) costs as much as ~800 floating-point operations.

#### 1.3 FlashAttention-2/3 Optimizations

FlashAttention-2 (Dao, 2023) reduces non-matmul FLOPs by:
- Delaying the division by \(\ell\) (softmax denominator) to the final step, working with unscaled \(\tilde{O}\) throughout tiling
- Storing \(L = m + \log(\ell)\) instead of separate \(m, \ell\) statistics, halving backward recomputation overhead

FlashAttention-3 (Shah et al., NeurIPS 2024) further reduces overhead through:
- **Asynchronous WGMMA instructions** on H100, overlapping compute with data movement
- **Block quantization for FP8**: each \(128 \times 128\) block independently computes its own scaling factor, avoiding dynamic range collapse from global quantization
- **Incoherent processing**: multiplies \(Q\) and \(K\) by a random orthogonal matrix (e.g., randomized Hadamard transform) to "smear out" outlier entries, reducing quantization error by ~2.6× relative to naive FP8

#### 1.4 FLOPs Comparison Across Methods

| Method | Forward FLOPs | Training FLOPs | Asymptotic |
|--------|--------------|----------------|------------|
| **Standard Attention** | \(4N^2 d\) | \(12N^2 d\) | \(O(N^2 d)\) |
| **FlashAttention-1** | \(4N^2 d + O(N^2/d)\) | \(\sim 16N^2 d\) | \(O(N^2 d)\) |
| **FlashAttention-2** | \(4N^2 d + O(N^2/d)\) | \(\sim 14N^2 d\) | \(O(N^2 d)\) |
| **FlashAttention-3** | \(4N^2 d + O(N^2/d)\) | \(\sim 14N^2 d\) | \(O(N^2 d)\) |
| **Linear Attention** | \(2N d^2\) | \(4N d^2\) | \(O(N d^2)\) |
| **Performer (FAVOR+)** | \(O(N d^2 + N d r)\) | \(O(N d^2 r)\) | \(O(N d^2 r)\) |
| **Reformer (LSH)** | \(O(N \log N \cdot d)\) | \(O(N \log N \cdot d)\) | \(O(N \log N \cdot d)\) |

where \(r\) is the number of random features (typically \(r \ll N\)), and \(B\) denotes block size.

---

### 2. Memory Complexity & I/O Analysis

#### 2.1 HBM Access Pattern Decomposition

**Standard Attention** requires three round-trips to HBM for the \(N \times N\) matrix:

| Step | Read from HBM | Write to HBM | Bytes (FP16) |
|------|---------------|-------------|-------------|
| Compute \(S = QK^\top\) | \(Q, K\): \(2Nd\) | \(S\): \(N^2\) | \(4Nd + 2N^2\) |
| Compute \(P = \text{softmax}(S)\) | \(S\): \(N^2\) | \(P\): \(N^2\) | \(4N^2\) |
| Compute \(O = PV\) | \(P, V\): \(N^2 + Nd\) | \(O\): \(Nd\) | \(2N^2 + 4Nd\) |
| **Total** | | | \(\mathbf{2Nd \cdot d + 6N^2 + 8Nd}\) |

For \(N = 4096\), \(d = 128\), FP16: \(\approx 134\) MB HBM traffic per forward pass.

**FlashAttention** tiles \(Q\) into \(T_r = \lceil N/B_r \rceil\) blocks and \(K, V\) into \(T_c = \lceil N/B_c \rceil\) blocks:

- **SRAM constraint**: \(B_r \cdot d + B_c \cdot d + B_r \cdot B_c \leq M\) (all working data must fit in SRAM)
- **Optimal block sizes**: \(B_r = \Theta(M / 4d)\), \(B_c = \Theta(\min(d, M / 4d))\)
- **HBM reads of \(K, V\)**: Each of the \(T_r\) outer iterations reads all \(T_c\) blocks of \(K\) and \(V\) → \(T_r \times T_c\) loads → \(\frac{N}{B_r} \cdot \frac{N}{B_c} \cdot 2B_c d = 2N^2 d / B_r\) bytes
- **Total HBM traffic**: \(O(Nd + N^2 d^2 / M)\) = \(\mathbf{\Theta(N^2 d^2 / M)}\)

For \(N = 4096\), \(d = 128\), \(M = 192\) KB (A100 SM SRAM), FP16: \(\approx 17\) MB — an **~8× reduction**.

#### 2.2 I/O-Optimality and the Matching Lower Bound

Saha & Ye (ICML 2024) rigorously established the theoretical I/O lower bound for exact attention:

> **Theorem (Saha & Ye, 2024).** Any algorithm that computes exact attention must perform \(\Omega\left(\frac{N^2 d^2}{M}\right)\) HBM accesses when \(M \geq d^2\).

The proof employs two complementary frameworks:

1. **Red-Blue Pebble Game** (Hong & Kung, 1981): Modeling computation as a directed acyclic graph where nodes are arithmetic operations and edges are data dependencies. The red pebbles correspond to values in fast memory (SRAM, capacity \(M\)), and blue pebbles to values in slow memory (HBM). A red-blue pebble game lower bound of \(\Omega(N^2 d^2 / M)\) is derived by partitioning the computation DAG into sub-DAGs of size \(O(M)\) and counting the minimum number of edge crossings between partitions.

2. **Communication Complexity**: Introducing the \(B\)-entry matrix compression problem — if an algorithm could compute attention with fewer than the claimed HBM accesses, it would yield a communication protocol for matrix compression that beats known lower bounds, yielding a contradiction.

**Small-cache regime (\(M < d^2\))**: Saha & Ye also prove that FlashAttention is _not_ optimal in this regime. They provide an alternative algorithm achieving \(O\left(\frac{N^2 d}{\sqrt{M}}\right)\) I/O complexity, and show this is optimal for combinatorial algorithms. The complete characterization is:

\[
\Theta\left(\min\left(\frac{N^2 d}{\sqrt{M}}, \frac{N^2 d^2}{M}\right)\right)
\]

On modern GPUs (A100: ~192 KB SRAM per SM, H100: ~256 KB), the condition \(M \geq d^2\) is satisfied for typical head dimensions (\(d = 64\text{--}128\)), meaning FlashAttention operates in the **optimal regime**.

#### 2.3 Backward Pass I/O — Fine-Grained Bounds

The backward pass introduces additional complexity. Recent work (On Fine-Grained I/O Complexity of Attention Backward Passes, 2024) establishes tight bounds:

\[
\Theta\left(\min\left\{\frac{N^2 d^2 + N d^3}{M}, \frac{N^2 d + N d^2}{\sqrt{M}}\right\}\right)
\]

For large cache (\(M = \Omega(d^2)\)), the backward pass I/O is \(\Theta(N^2 d^2 / M)\) — matching the forward pass. For small cache, it's \(\Theta(N^2 d / \sqrt{M})\), which FlashAttention does not achieve.

#### 2.4 Memory Footprint Summary

| Metric | Standard Attention | FlashAttention | Linear Attention |
|--------|-------------------|---------------|-----------------|
| **HBM traffic (forward)** | \(\Theta(N^2 d)\) | \(\Theta(N^2 d^2 / M)\) | \(\Theta(N d^2)\) |
| **SRAM usage** | \(O(M)\) (inefficient use) | \(\Theta(M)\) (saturated) | \(O(d^2)\) |
| **Peak activation memory** | \(O(N^2)\) | \(O(N d)\) | \(O(N d)\) |
| **I/O-optimal?** | No | Yes (when \(M \geq d^2\)) | Depends on kernel |

---

### 3. Approximation Error Bounds

#### 3.1 Exact vs. Approximate Methods

**FlashAttention is exact**: the online safe-softmax algorithm preserves bitwise equivalence with standard attention (modulo floating-point associativity). The proof relies on the two-statistic incremental update:

Given two partial row blocks \(x^{(1)}, x^{(2)}\) with running statistics \((m^{(1)}, \ell^{(1)})\) and \((m^{(2)}, \ell^{(2)})\):

\[
\begin{aligned}
m^{\text{new}} &= \max(m^{(1)}, m^{(2)}) \\
\ell^{\text{new}} &= \ell^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}} + \ell^{(2)} \cdot e^{m^{(2)} - m^{\text{new}}} \\
O^{\text{new}} &= \text{diag}\left(\frac{\ell^{(1)}}{\ell^{\text{new}}}\right)^{-1} \cdot O^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}} + \text{diag}\left(\frac{\ell^{(2)}}{\ell^{\text{new}}}\right)^{-1} \cdot e^{S^{(2)} - m^{\text{new}}} \cdot V^{(2)}
\end{aligned}
\]

**Induction** over tiles proves that after processing all \(T_c\) blocks, \(O^{\text{final}} = \text{softmax}(S)V\) exactly.

#### 3.2 Error Bounds for Approximate Attention

For **approximate** methods (linear attention, Performers, Reformer), rigorous error bounds exist:

**Random Feature Attention (Performer/FAVOR+):**

Choromanski et al. (ICLR 2021) prove that with \(r\) random features, the kernel approximation error is:

\[
\|\widehat{K}(q, k) - K(q, k)\|_{\infty} \leq \varepsilon \quad \text{with probability} \geq 1 - \delta
\]

where \(r = O\left(\frac{d}{\varepsilon^2} \log\left(\frac{1}{\delta}\right)\right)\), using the mechanism of positive random features \(\phi(x) = \frac{1}{\sqrt{r}} \exp(Wx - \frac{1}{2}\|x\|^2)\) with \(W \sim \mathcal{N}(0, I)\).

**Johnson-Lindenstrauss Connection**: The JL lemma guarantees that random projections approximately preserve pairwise distances. For \(N\) tokens, \(r = O(\varepsilon^{-2} \log N)\) random features suffice for \(\varepsilon\)-approximation of all \(N^2\) pairwise kernel values.

**Linear Attention with Self-Normalized Importance Sampling:**

Zheng et al. (ICML 2022) recast random feature attention as self-normalized importance sampling (SNIS), yielding:

\[
\|\text{Attention}_{\text{approx}} - \text{Attention}_{\text{exact}}\| \leq O\left(\frac{1}{\sqrt{r}}\right)
\]

with the bias introduced by self-normalization decaying at the same \(O(1/\sqrt{r})\) rate. Sernau et al. (2024) further showed that under optimal importance sampling, the variance bound is **independent of the feature map choice** — all random feature representations are equivalent when optimally sampled.

#### 3.3 Lipschitz and Contraction Properties

**Softmax Lipschitz Constant (Nair, 2025):**

> **Theorem.** For the softmax function \(\sigma: \mathbb{R}^n \to \Delta_n^\circ\) and any \(\ell_p\) norm with \(p \geq 1\):
> \[
> \|\sigma(\mathbf{x}) - \sigma(\mathbf{y})\|_p \leq \frac{1}{2} \|\mathbf{x} - \mathbf{y}\|_p
> \]
> The constant \(1/2\) is **tight** — it is attained for \(p = 1\) and \(p = \infty\) at \(\mathbf{x} = (\ln(n-1), 0, \ldots, 0)\).

This improves upon the previously cited bound of 1 (Gao & Pavel, 2017). The proof proceeds via the Jacobian \(J_\sigma(\mathbf{x}) = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top\) and Riesz-Thorin interpolation between \(\ell_1\) and \(\ell_\infty\) induced matrix norms, establishing \(\|J_\sigma(\mathbf{x})\|_p \leq 1/2\) uniformly.

**Self-Attention as a Contraction:**

For a single-head self-attention layer with residual connection:
\[
\text{Layer}(X) = X + \text{Attention}(X)
\]

The attention map itself is **not necessarily contractive** — its Lipschitz constant depends on the spectral properties of \(W_Q, W_K, W_V\). However, with the softmax \(1/2\)-Lipschitz bound, one can bound:

\[
\|\text{Attention}(X) - \text{Attention}(Y)\| \leq \frac{1}{2} \|W_V\| \cdot \|W_Q W_K^\top\| \cdot \|X - Y\|
\]

This informs the choice of initialization and normalization schemes. For the Transformer to be a contraction overall, one typically needs \(\|W_V\| \cdot \|W_Q W_K^\top\| < 2 / d\) — a condition that can be enforced via spectral normalization.

**Pay Attention to Attention Distribution (Yudin et al., 2025):**

This recent work refines the analysis further: the local Lipschitz constant of self-attention is directly controlled by the **attention distribution entropy**. When attention is uniformly distributed (\(\|P_{i:}\|_\infty \approx 1/N\)), the Jacobian spectral norm is minimized; when attention is peaked (\(\|P_{i:}\|_\infty \approx 1\)), the norm approaches \(1/2\). This provides a theoretical basis for observations that well-trained Transformers exhibit relatively diffuse attention patterns.

---

### 4. Numerical Stability Analysis

#### 4.1 Safe Softmax and Floating-Point Error

The standard numerical stability technique for softmax is the **max-subtraction trick**:

\[
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
\]

This prevents overflow (\(e^x \to \infty\) for \(x > 88.7\) in FP32, \(x > 11.09\) in FP16) but introduces **subtractive cancellation**: when \(\max(x)\) is large, \(e^{x_i - \max(x)} \to 0\) for non-maximum elements, leading to underflow.

**Online safe-softmax:** FlashAttention's incremental max-update introduces an additional subtlety. When a new block contains a larger maximum, all previously accumulated statistics must be rescaled by \(e^{m^{\text{old}} - m^{\text{new}}}\). If \(m^{\text{new}} \gg m^{\text{old}}\), the scaling factor vanishes, and previously accumulated information is **washed out** — a potential source of precision loss in very long sequences.

#### 4.2 Low-Precision Training Failure Modes

**Tsinghua ICLR 2026 Oral** (Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention) identified two distinct failure mechanisms:

**Mechanism 1 — Low-Rank Error Amplification:**
In standard FP32, rounding errors behave as zero-mean noise. However, in BF16/FP16 FlashAttention, the attention mechanism exhibits **similar low-rank update directions** across training steps. This property transforms rounding errors from random noise into **systematic drift** — the errors consistently push weight updates in the same low-dimensional subspace, causing monotonic growth in weight spectral norm and activation magnitudes. Over thousands of steps, this drives the network toward numerical divergence (NaN loss).

**Mechanism 2 — Repeated Max Trigger:**
When a row of the pre-softmax matrix \(S\) contains a repeated maximum value, the safe-softmax shift constant \(m\) introduces a **systematic bias** under BF16 rounding. Specifically, with \(m = \max(s_1, \ldots, s_n)\), the computation of \(\sum e^{s_j - m}\) yields a sum where the largest term is exactly 1.0. Under BF16's reduced mantissa precision, this can cause the sum to round to exactly 1.0, making softmax degenerate (probability mass concentrated entirely on the maximum positions while ignoring others). The paper demonstrates this with a discrete-trigger analysis: the probability of hitting this condition increases with sequence length.

**Proposed fix:** Dynamically adjust the row shift constant when a repeated maximum is detected, ensuring that \(\exp(0)\) is not the dominant term in the softmax denominator.

#### 4.3 dS Gradient Vulnerability

The backward pass gradient \(dS\) (gradient w.r.t. pre-softmax logits) is structurally the most fragile tensor in the attention pipeline. SageBwd (Tsinghua & Berkeley, 2026) documents:

\[
\begin{aligned}
\text{RMS}(P) &\approx 5 \times 10^{-3} \\
\text{RMS}(dP) &\approx 5 \times 10^{-5} \\
\text{RMS}(dS) &\approx 1 \times 10^{-7}
\end{aligned}
\]

The \(dS\) tensor is ~500× smaller than \(dP\), theoretically bounded by \(O(1/\sqrt{N})\). For the softmax backward pass:
\[
dS_{ij} = P_{ij} \left(dP_{ij} - \sum_k P_{ik} \cdot dP_{ik}\right)
\]

When \(P_{ij}\) is near-uniform (\(P_{ij} \approx 1/N\)), both terms inside the parentheses approximately cancel, yielding \(dS_{ij} \approx 0\). This structural cancellation is the source of \(dS\)'s extreme sensitivity — INT8 quantization introduces fixed absolute noise that becomes a large _relative_ error when the signal itself is near zero.

**Propagation to weight gradients:** The error in \(dS\) propagates to \(dQ = dS \cdot K\) and \(dK = Q^\top \cdot dS\), where it is amplified by the norms of \(Q\) and \(K\). For long sequences with large key/value norms, this amplification can be catastrophic.

#### 4.4 Softmax Collapse (Grokking Context)

Nohlgren et al. (2025) characterized **Softmax Collapse (SC)** — when the correct-class logit dominates, floating-point absorption causes \(\sum e^{z_k} \doteq e^{z_y}\) (exactly, within precision), making both the loss and correct-class gradient exactly zero. This halts learning entirely. The proposed fix, **StableMax**, replaces \(\exp\) with \(\max(x, 0) + 1\) or \(1/(1-x)\) for \(x < 0\), avoiding the extreme dynamic range of exponentials while preserving softmax-like normalization properties.

#### 4.5 Numerical Stability Summary

| Stability Concern | Root Cause | Mitigation |
|------------------|-----------|------------|
| Softmax overflow (FP16) | \(e^{11.09} > 65504\) | Max-subtraction trick |
| Repeated max bias (BF16) | Systematic rounding at \(\exp(0) = 1\) | Dynamic shift constant adjustment |
| \(dS\) gradient fragility | \(O(1/\sqrt{N})\) structural cancellation | QK-Norm, mixed precision for backward |
| Softmax Collapse | Floating-point absorption | StableMax, increased precision |
| Online rescaling washout | \(e^{m_\text{old} - m_\text{new}} \to 0\) | Log-space statistics (\(L = m + \log \ell\) in FA2) |

---

### 5. Convergence Guarantees

#### 5.1 Gradient Descent Convergence for Transformers

**Qin, Zhou & Zhu (2025):** Proved **linear convergence** of gradient descent for a structurally complete single-layer Transformer (self-attention + MLP + residual connections) under appropriate initialization. The key insight is that:

> Residual connections ameliorate the **ill-conditioning** of the attention output matrix caused by the low-rank structure of softmax.

Without residual connections, the softmax attention output matrix becomes ill-conditioned as \(N\) grows (since \(\text{rank}(P) \leq \text{rank}(QK^\top) \leq d \ll N\)), causing gradient descent to stall. The residual path provides an alternative route for gradient flow, enabling linear convergence. The result extends to multi-layer Transformers under similar conditions.

#### 5.2 Gaussian Kernel Superiority over Softmax

**Amazon Science (NeurIPS 2024):** Analyzed the convergence landscape of attention with different kernel choices. A striking finding: the **Gaussian kernel** \(K(q, k) = \exp(-\|q - k\|^2 / 2\sigma^2)\) can provably converge when softmax fails, particularly when only the query matrix is updated during training. The reason is topological: the softmax kernel's optimization landscape contains more saddle points and flat regions than the Gaussian kernel's smoother, translation-invariant landscape. Empirically, Gaussian kernel Transformers converged faster and achieved higher accuracy on both text classification and image segmentation.

#### 5.3 Input-Dependent Sparse Attention and Stability

**Ram et al. (NeurIPS 2025):** Established a formal connection between sparsity and convergence. Their key result:

> **Input-dependent** sparse attention (where sparsity pattern depends on the current input) provides improved convergence and generalization guarantees, while **input-agnostic** sparsity (fixed pattern, e.g., sliding window) does not.

The mechanism: input-dependent sparsity preserves the **Lipschitz continuity** of the attention map with respect to inputs, ensuring that the loss landscape remains smooth. Fixed-pattern sparsity can introduce discontinuities (a token just outside the window has zero influence, one just inside has full influence), degrading optimization.

#### 5.4 Implicit Bias of Attention Training

**Deora (UBC, 2024):** Showed that training the combined key-query matrix \(W = W_K^\top W_Q\) with gradient descent on binary classification tasks causes parameters to converge to the solution of a **hard-margin SVM problem** in the feature space induced by the attention kernel. This implicit bias toward max-margin solutions provides a theoretical explanation for the strong generalization observed in attention-based models, even without explicit regularization.

#### 5.5 Contraction Conditions for Multi-Layer Attention

Combining the \(1/2\)-Lipschitz softmax bound with standard matrix norm bounds yields a sufficient condition for a single-head attention layer to be a contraction:

\[
\frac{1}{2\sqrt{d}} \cdot \|W_V\|_2 \cdot (\|W_Q\|_2 \cdot \|W_K\|_2) < 1
\]

For multi-layer Transformers, the product of per-layer Lipschitz constants must be less than 1 for guaranteed contraction to a fixed point. This condition is rarely satisfied in practice for deep networks — instead, residual connections and layer normalization provide _local_ contractivity, and convergence relies on the interplay between optimization dynamics and the structure of the data manifold.

#### 5.6 Asymptotic Token Collapse

A complementary negative result: **As \(L \to \infty\)** (infinite depth), token representations in a standard Transformer provably collapse to a **single cluster** (or a small number of clusters equal to the number of distinct eigenspaces of the attention transition matrix). This "oversmoothing" phenomenon, analyzed via control theory (consensus dynamics on manifolds), places a fundamental limit on Transformer depth unless mitigation strategies (skip connections, normalization, stochastic depth) are employed.

---

### 6. Important Papers & References

| # | Paper | Authors | Venue/Year | Key Contribution |
|---|-------|---------|-----------|------------------|
| 1 | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | Dao, Fu, Ermon, Rudra, Ré | NeurIPS 2022 | Introduced tiling + online softmax; proved I/O complexity \(\Theta(N^2 d^2 / M)\); achieved exact computation with memory \(O(N)\) |
| 2 | **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | Dao | 2023 | Reduced non-matmul FLOPs; achieved 50–73% of theoretical peak FLOPS on A100 |
| 3 | **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision** | Shah, Bikshandi, Zhang, Thakkar, Dao et al. | NeurIPS 2024 | H100 optimizations; FP8 with block quantization; incoherent processing for error reduction; 1.2 PFLOPS |
| 4 | **The I/O Complexity of Attention, or How Optimal is Flash Attention?** | Saha, Ye | ICML 2024 | Proved I/O lower bound \(\Omega(N^2 d^2 / M)\); FlashAttention optimal for \(M \geq d^2\); new algorithm for \(M < d^2\) |
| 5 | **On Fine-Grained I/O Complexity of Attention Backward Passes** | Multiple authors | arXiv 2024 | Tight bounds for backward pass: \(\Theta(\min\{ (N^2 d^2 + Nd^3)/M, (N^2 d + Nd^2)/\sqrt{M} \})\) |
| 6 | **Softmax is 1/2-Lipschitz: A Tight Bound Across All \(\ell_p\) Norms** | Nair | arXiv 2025 | Proved softmax Lipschitz constant is \(1/2\) (not 1); tight for all \(\ell_p\) norms; implications for attention contraction |
| 7 | **Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention** | Tsinghua authors | ICLR 2026 (Oral) | Two BF16 failure mechanisms: low-rank error amplification + repeated max trigger; dynamic shift constant fix |
| 8 | **Grokking at the Edge of Numerical Stability** | Nohlgren et al. | 2025 | Characterized Softmax Collapse; proposed StableMax as numerically stable alternative |
| 9 | **SageBwd: A Trainable Low-bit Attention** | Tsinghua & Berkeley | 2026 | Identified \(dS\) gradient as most fragile tensor (\(O(1/\sqrt{N})\)); QK-Norm mitigation |
| 10 | **Rethinking Attention with Performers** | Choromanski et al. | ICLR 2021 | FAVOR+ algorithm; positive random features for unbiased softmax kernel approximation at \(O(N)\) |
| 11 | **Linear Complexity Randomized Self-attention Mechanism** | Zheng et al. | ICML 2022 | Recast random feature attention as SNIS; proved \(O(1/\sqrt{r})\) error bound |
| 12 | **On the Convergence of Gradient Descent on Learning Transformers with Residual Connections** | Qin, Zhou, Zhu | 2025 | Proved linear convergence rate for single- and multi-layer Transformers; residual connections mitigate softmax ill-conditioning |
| 13 | **Transformers Learn Faster with Semantic Focus** | Ram et al. | NeurIPS 2025 | Input-dependent sparse attention preserves Lipschitz continuity → improved convergence vs. fixed-pattern sparsity |
| 14 | **Understanding the Training Dynamics of Transformers** | Amazon Science | NeurIPS 2024 | Gaussian kernel can converge when softmax fails; smoother optimization landscape |
| 15 | **Pay Attention to Attention Distribution: A New Local Lipschitz Bound for Transformers** | Yudin et al. | 2025 | Attention distribution entropy controls local Lipschitz constant; spectral analysis of \(\text{diag}(P) - PP^\top\) |
| 16 | **FLASH-D: FlashAttention with Hidden Softmax Division** | Ankit et al. | ISLPED 2025 | Sigmoid recurrence eliminates explicit max-subtraction; 22.8% area reduction in hardware |
| 17 | **On the Optimization and Generalization of Self-Attention Models** | Deora | UBC PhD Thesis 2024 | Implicit bias toward hard-margin SVM solutions; finite-time convergence guarantees |
| 18 | **The Asymptotic Behavior of Attention in Transformers** | Multiple authors | 2024 | Token collapse proof via control theory; fundamental depth limitation |
| 19 | **All Random Features Representations are Equivalent** | Sernau et al. | 2024 | Optimal importance sampling makes all RF representations equivalent; universal variance bound |
| 20 | **Hong & Kung I/O Lower Bound** | Hong, Kung | STOC 1981 | Foundational red-blue pebble game; \(\Omega(N^3 / \sqrt{M})\) for matrix multiplication; framework used by Saha & Ye |

---

### 7. Open Questions & Future Directions

#### 7.1 Tight I/O Bounds for Approximate Attention

While Saha & Ye (2024) settled the exact attention I/O lower bound, the I/O complexity of **approximate** attention — linear attention, sparse attention, kernel attention — remains largely open. Approximate methods trade accuracy for reduced computation, but whether this trade translates to proportionally reduced I/O has not been systematically characterized. A unified framework connecting approximation error \(\varepsilon\) to I/O complexity would be transformative.

#### 7.2 Error Propagation Through Deep Networks

The existing numerical stability analyses focus on a **single attention layer** in isolation. How floating-point errors in attention propagate through multiple layers, especially when combined with normalization layers, MLPs, and residual connections, is not well understood. A **compositional error analysis** — bounding the total forward and backward error after \(L\) Transformer layers — would directly inform precision-allocation strategies for large-scale training.

#### 7.3 Accelerator-Specific Optimality

Current I/O optimality analyses assume a two-level memory hierarchy (fast SRAM + slow HBM) with sequential access. Modern accelerators feature more complex hierarchies (L1, L2, shared memory, registers on GPU; systolic arrays, vector units on TPU). Extending the red-blue pebble game to capture **multi-level memory, tensor core instructions, and asynchronous data movement** (e.g., H100's TMA, SM100's warp group MMA) remains an active challenge.

#### 7.4 Convergence Theory for Attention Replacements

The convergence guarantees developed for standard Transformers assume exact softmax attention. Whether **drop-in replacements** (linear attention, sparse attention, kernel attention) preserve these guarantees — or degrade them, and by how much — is largely unexplored. A formal analysis of how approximation error \(\varepsilon\) in the attention computation maps to final model quality (e.g., test loss or accuracy) would enable principled trade-offs between efficiency and quality.

#### 7.5 Low-Precision Provable Guarantees

FlashAttention-3 demonstrated empirically that FP8 attention can match FP16 accuracy, but the theoretical underpinnings are incomplete. The incoherent processing technique (randomized Hadamard transform) is known to work via concentration arguments, but a formal **error bound** for the full FP8 attention pipeline — including block quantization, matrix multiply in FP8, and accumulation in FP32 — has not been published. Similarly, provable bounds on the probability of the "repeated max trigger" failure mode in BF16 as a function of sequence length and head dimension are needed.

#### 7.6 Adaptive Precision Attention

A compelling direction is **adaptive precision** — allocating higher precision (FP32 or FP16) to numerically sensitive operations (\(dS\) computation, softmax denominator accumulation) while using lower precision (FP8, INT8) for bulk matrix multiplies (\(QK^\top, PV\)). SageBwd's analysis of \(dS\) fragility provides a clear targeting mechanism, but realizing this in an automated, hardware-efficient implementation remains open.

#### 7.7 Non-Standard Attention Mechanisms

The theoretical machinery developed for scaled dot-product attention does not trivially extend to newer attention variants — **grouped query attention (GQA)**, **multi-query attention (MQA)**, **multi-head latent attention (MLA)**. Each variant modifies the dataflow graph and tiling constraints in ways that affect both I/O optimality and numerical stability. A systematic theoretical treatment of these variants is needed.

---

### 8. Relevance to Main Topic

This sub-topic's analysis of complexity, error bounds, and theoretical guarantees serves as the **mathematical backbone** for evaluating any proposed attention mechanism. The framework developed across the papers surveyed above provides a checklist of formal criteria that a novel mechanism must satisfy:

1. **Computational complexity**: Does the mechanism reduce the \(O(N^2 d)\) scaling, and if so, by what factor? The FLOPs comparison table (Section 1.4) provides a benchmark for any new method.

2. **I/O optimality**: Is the mechanism I/O-optimal under the Saha-Ye framework? For exact methods, the \(\Theta(N^2 d^2 / M)\) lower bound is the gold standard. For approximate methods, the corresponding bound remains to be established but should be characterized.

3. **Error bounds**: If approximate, can the mechanism prove a bound of the form \(\|\text{Output}_\text{ours} - \text{Output}_\text{softmax}\| \leq \varepsilon\) under stated assumptions? The \(1/2\)-Lipschitz property of softmax and the \(O(1/\sqrt{r})\) rate for random features provide templates for such proofs.

4. **Numerical stability**: Does the mechanism avoid the known failure modes — overflow (max-subtraction trick), repeated-max bias (BF16), \(dS\) fragility, and softmax collapse? A formal floating-point error analysis should accompany any new proposal.

5. **Convergence**: Can the mechanism, when used as a drop-in replacement, preserve the linear convergence guarantees established for standard Transformers under gradient descent? At minimum, a contraction or Lipschitz property should be proven to ensure the training dynamics remain well-behaved.

A mechanism that rigorously addresses all five dimensions — ideally with theorems in the style of those surveyed here — would represent a significant contribution to the attention literature. Conversely, a mechanism that neglects any of these dimensions risks suffering from the pathologies documented in the numerical stability and convergence sections above.

---

**Research completed:** June 1, 2026  
**Sources consulted:** 40+ papers and technical resources across complexity theory, numerical analysis, optimization, and hardware architecture  
**Key search terms:** FlashAttention complexity, I/O optimality attention, softmax Lipschitz, numerical stability attention backward, convergence transformer, approximate attention error bound, roofline model GPU
