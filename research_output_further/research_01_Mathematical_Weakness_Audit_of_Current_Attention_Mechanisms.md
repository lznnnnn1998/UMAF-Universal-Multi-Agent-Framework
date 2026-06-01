# Mathematical Weakness Audit of Current Attention Mechanisms

> **Research Sub-Topic 1**: Systematic catalog of mathematical weaknesses across FlashAttention, RoPE positional encoding, State Space Models (SSMs), and Linear (Kernelized) Attention, ranked by severity and tractability.

---

## 1. Overview

Modern large language models are built on the attention mechanism, but this architecture harbors four distinct families of mathematical weaknesses that fundamentally constrain efficiency, length generalization, and retrieval quality. This audit systematically characterizes the precision failures across each family, quantifies them with concrete numbers and equations, and ranks them by combined severity and tractability.

**FlashAttention** has solved the I/O bottleneck of standard attention through tiling and recomputation, but exposes a deeper structural problem: the asymmetric hardware scaling trap. As NVIDIA transitions from A100→H100→B200→GB300, tensor core throughput grows 2.25× per generation while special function units (SFUs) for exponential computation remain unchanged. This creates a **512:1 throughput gap** between the matrix engines that compute QK^T/V^T and the exponential units that compute softmax — the forward pass spends equal time on GEMM and on a single exp operation. Each GPU generation requires a fundamental kernel redesign because the bottleneck shifts between memory bandwidth, register pressure, SFU throughput, and shared memory bandwidth in unpredictable ways.

**Rotary Position Embedding (RoPE)** suffers from a mathematical decomposition pathology: the d/2 independent frequency components are unequally trained within the finite training context, with low-frequency components observing less than one full period. During length extrapolation, these "activated" (high-frequency) components produce out-of-distribution rotation angles, triggering cascading attention disruption through all layers. The softmax attention distribution undergoes **entropy collapse** — H(attention) → log(N) as N → ∞ — transforming focused attention into uniform dispersion, while the attention matrix's spectral structure collapses toward low-frequency dominant components.

**State Space Models (SSMs)** compress all historical context into a fixed-size state vector through multiplicative decay, creating a fundamental information-theoretic bottleneck. The recurrent update h_t = Ā_t h_{t-1} + B̄_t x_t with Ā_t < 1 causes exponential forgetting of early tokens. Retrieval accuracy collapses from ~100% at the training context length (e.g., 8K) to near 0% at 2× that length. This "fuzzy memory" arises not from inadequate architecture but from the impossibility of content-addressable lookup through a compressed state — there is no native key-value retrieval mechanism.

**Linear (Kernelized) Attention** replaces the softmax nonlinearity with a feature map φ such that Attention(Q,K,V) ≈ φ(Q)(φ(K)^T V)/φ(Q)φ(K)^T 1, reducing complexity from O(N²d) to O(Nd²). But this formulation is fundamentally bounded by the rank of the accumulated key-value outer-product matrix M_t = Σ φ(k_s) ⊗ v_s ∈ ℝ^{d×d}, which saturates at rank ≤ d. For tasks requiring fine-grained token distinction (copying, retrieval), the rank bottleneck manifests as a degradation in ability to distinguish tokens — a problem that softmax attention's nonlinear coupling inherently avoids by enabling the attention matrix to have rank up to N.

The remainder of this audit formalizes each weakness with precise mathematical statements, concrete quantitative measurements, and a unified ranking framework.

---

## 2. Key Methods & Approaches: Formal Weakness Characterization

### 2.1 FlashAttention: The Asymmetric Hardware Scaling Trap

#### 2.1.1 Softmax Exponential Unit Bottleneck (Severity: CRITICAL)

**Formal Statement (Weakness W1):** Let T_tc be the tensor core throughput (ops/cycle) and T_exp be the exponential unit (MUFU.EX2) throughput (ops/cycle). On NVIDIA B200 (Blackwell), per Streaming Multiprocessor:

$$\frac{T_{tc}}{T_{exp}} = \frac{8192}{16} = 512$$

The forward pass of FlashAttention requires, per tile of dimensions M=N=d=128:

- GEMM (2 matrix multiplies): 1,024 cycles (tensor cores operating at 8,192 ops/cycle)
- Softmax exponential: 1,024 cycles (MUFU.EX2 operating at 16 ops/cycle)
- Shared memory traffic: 768 cycles

**The forward pass is therefore co-bottlenecked by tensor cores and exponential units**, despite the 512:1 raw throughput ratio. The exponential operation — a single element-wise nonlinearity — consumes as many cycles as two full matrix-matrix multiplies. This is the core mathematical inefficiency: the softmax nonlinearity cannot be expressed as a sequence of matrix operations that leverage tensor core throughput.

**Quantification across generations:**

| GPU Generation | Tensor Core Peak | MUFU/SFU Count | Ratio (TC:MUFU) | Primary Bottleneck |
|---|---|---|---|---|
| A100 (Ampere) | 312 TFLOPS (FP16) | 4/SM | ~200:1 | HBM bandwidth |
| H100 (Hopper) | 989 TFLOPS (FP16) | 4/SM | ~400:1 | Register pressure |
| B200 (Blackwell) | 2,250 TFLOPS (BF16) | 4/SM | **512:1** | SFU throughput (FWD), SMEM bandwidth (BWD) |
| GB300 (Blackwell Ultra) | ~2,500 TFLOPS (BF16) | **8/SM** | ~250:1 | Improved but not solved |

On GB300, SFU doubling yields only ~35% improvement for DeepSeek-V3 FP8 forward pass — the bottleneck shifts but does not vanish.

#### 2.1.2 Per-Generation Kernel Redesign Requirement (Severity: HIGH)

**Formal Statement (Weakness W2):** Let H_g be the hardware configuration of GPU generation g. The optimal tiling parameters (B_r, B_c), warp scheduling strategy, and recomputation policy for attention on H_g depend on the tuple (BW_HBM, BW_SMEM, T_tc, T_exp, R_reg, size_TMEM), which is not a monotonic function of g. Therefore, no attention kernel K is optimal across two successive generations g and g+1 without modification.

Empirical evidence:
- FA1 (A100): tiling-based, memory-bound optimization, 25-40% utilization
- FA2 (A100): better parallelism, fewer non-matmul FLOPs, 50-73% utilization
- FA3 (H100): warp specialization, async TMA, FP8 support, ~75% utilization — complete rewrite
- FA4 (B200): software-emulated exp via cubic polynomial, conditional rescaling, 2-CTA MMA, TMEM utilization, 71% utilization — another complete rewrite

Each generation requires O(months) of engineering effort by domain experts. The expense is not amortized — it recurs with each hardware release.

#### 2.1.3 Recomputation-Based Backward Pass Cost (Severity: MODERATE)

**Formal Statement (Weakness W3):** The FlashAttention backward pass computes gradients dQ, dK, dV by recomputing the softmax attention matrix P = softmax(QK^T/√d) from stored statistics (running max m, log-sum-exp ℓ) in each tile. The recomputation cost is:

$$\text{FLOPs}_{bwd}^{FA} = 4N^2 d + \underbrace{O(N^2 d \cdot \frac{d}{B_r})}_{\text{recomputation overhead}}$$

compared to standard attention's 4N²d (which stores the full P matrix). While the recomputation overhead is asymptotically absorbed by the tiling factor, **the backward pass of FA4 on B200 is shared-memory bandwidth bound** (3,328 cycles vs 2,560 cycles for tensor core MMA in the 1-CTA backward pass), meaning the recomputation pattern creates a new bottleneck distinct from the forward pass bottleneck.

---

### 2.2 Rotary Position Embedding (RoPE): Frequency Extrapolation Failure

#### 2.2.1 High-Frequency OOD Rotation Collapse (Severity: CRITICAL)

**Formal Statement (Weakness W4):** RoPE decomposes the dot-product attention score into d/2 independent frequency components:

$$q_m^\top k_n = \sum_{i=0}^{d/2-1} q_i^\top R_{\theta_i, m-n} k_i$$

where θ_i = b^{-2i/d} with base frequency b (typically 10,000) and R_θ is a 2D rotation matrix. Each component C_i = q_i^⊤ R_{θ_i, m-n} k_i oscillates at frequency ω_i ∝ θ_i.

During training with context length N_train, a component with frequency ω_i observes approximately α_i = ⌊N_train · ω_i / 2π⌋ full periods. For high-frequency components (small i), α_i ≫ 1 — the component is well-trained. For low-frequency components (large i), α_i < 1 — **the component has never completed a full rotation period**.

During extrapolation to N_test > N_train, the high-frequency components encounter rotation angles (θ_i · (m-n) for |m-n| > N_train) that were **never observed during training**. The resulting out-of-distribution (OOD) attention logits in the first layer cascade through all subsequent layers.

**Quantification (HoPE, Chen et al., 2024):**
- The "activated component" threshold can be **pre-calculated** from N_train before any training begins
- These components dominate attention in early training epochs (VAF analysis)
- On extrapolation: OOD logits appear **specifically in layer 1**, then disrupt all subsequent layers

#### 2.2.2 Attention Entropy Collapse (Severity: HIGH)

**Formal Statement (Weakness W5):** For softmax attention over sequence length N, the attention distribution for each query is:

$$p_i = \frac{\exp(s_i)}{\sum_{j=1}^N \exp(s_j)}$$

where s_i are the attention logits. As N → ∞, assuming the logits are bounded (which holds for any fixed model), the denominator grows as Θ(N) while the numerator is O(1), yielding:

$$p_i = \Theta\left(\frac{1}{N}\right), \quad H(p) = -\sum_i p_i \log p_i \to \log N$$

This is the **entropy collapse**: the attention distribution approaches maximum entropy (uniform distribution) regardless of the content. The model cannot maintain concentration on relevant tokens because softmax forces probability mass across all N positions.

**Quantification:**
- For softmax to maintain concentration c ∈ (0,1) on k high-value tokens, the required logit difference must grow as Ω(log N)
- α-entmax (α > 1) can maintain Θ(1/k) attention on top tokens **regardless of N**
- At N=128K (typical long-context inference), softmax entropy H(p)/H_max ≈ 0.95+ — near-complete dispersion

#### 2.2.3 Embedding Collapse Toward Low-Frequency Components (Severity: MODERATE)

**Formal Statement (Weakness W6):** For band f, the Gram matrix of RoPE-transformed keys is:

$$\Sigma_k = \sum_{j=1}^N \beta_j^2 R(\theta_j) k k^\top R(\theta_j)^\top$$

Under the low-frequency cone condition (γ_k < π/2), DoPE (Xiong et al., 2025) proves:

$$\lambda_{\max}(\Sigma_k) \geq N \beta_{\min}^2 \|k\|^2 \cos^2 \gamma_k$$

The top singular value scales as Θ(√N), producing a "bright band" outlier pattern in QK^T. Low-frequency components, having small ω_i and thus slow rotation, create near-constant attention patterns dominated by semantic rather than positional information — they form an **attention sink** that absorbs disproportionate probability mass.

**Quantification (DoPE, Xiong et al., 2025):**
- Band-wise matrix entropy H_{h,f} = -tr(Σ̃_{h,f} log Σ̃_{h,f}) ranges from near 0 (collapsed, attention-sink heads) to log 2 (healthy, balanced heads)
- Truncated effective rank ρ_h^r identifies heads where the dominant spectrum collapses
- Removing positional encoding from high-rank heads yields **up to 10-point accuracy improvements** without retraining
- Geometric analysis (Frayed RoPE, Wertheimer et al., 2025): RoPE shrinks the First Singular Value ratio by √2 to √d as n → ∞, dispersing key/query clusters and breaking the sink token mechanism

#### 2.2.4 Spectrum Damage via Linear Layers and Nonlinearities (Severity: MODERATE)

**Formal Statement (Weakness W7):** FoPE (Hua et al., 2024) proves that RoPE's clean frequency decomposition is corrupted by two mechanisms:

1. **Spectrum Leakage** (Linear layers): Y_m = Σ_{k=0}^{M-1} W_{km} X_k — weight matrices mix frequency components
2. **Spectrum Distortion** (Nonlinearities, Lemma 1): For any nonlinear g applied to multi-frequency input, g(x(n)) = Σ_{j,k} a_{j,k} cos(jω_1 + kω_2)n — producing harmonic intermodulation products that destroy clean frequency separation

The consequence: even if RoPE theoretically permits periodic extension (h̃_m(n + N_{ω_m}) = h̃_m(n)), the one-to-one frequency-to-coefficient correspondence is broken by spectrum damage, making true periodic extrapolation impossible without architectural intervention.

---

### 2.3 State Space Models (SSMs): The Fuzzy Memory Bottleneck

#### 2.3.1 Exponential Forgetting and Retrieval Collapse (Severity: CRITICAL)

**Formal Statement (Weakness W8):** The Mamba SSM recurrent update is:

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$

where Ā_t = exp(Δ_t · A) with A being a negative diagonal matrix (so Ā_t < 1 element-wise) and B̄_t = Δ_t · B_t. The contribution of token j to the output at position i is:

$$\alpha_{i,j} = C_i^\top \left(\prod_{k=j+1}^i \bar{A}_k\right) \bar{B}_j$$

Since ‖Ā_k‖ < 1 for all k, the product Π_{k=j+1}^i Ā_k → 0 exponentially in (i - j). This is **exponential forgetting**: the influence of early tokens decays multiplicatively, making exact token-level retrieval impossible beyond the effective memory horizon.

**Quantification (RULER benchmark, Mamba2-1.3B):**

| Context Length | Retrieval Accuracy |
|---|---|
| 8K (training length) | ~100% |
| 16K | **~0%** |
| 32K+ | **~0%** |

Vanilla Mamba2-1.3B at 16K: 0.27% average accuracy. With LAMB enhancement: 33.96%. But even enhanced, retrieval quality is far from attention's near-perfect performance within its training range.

#### 2.3.2 State Expansion vs. Quality Tradeoff (Severity: HIGH)

**Formal Statement (Weakness W9):** The SSM hidden state h_t ∈ ℝ^{N×d} has fixed dimension N (the state expansion factor). The effective memory capacity for exact token retrieval scales **exponentially** with N — because the state must encode O(exp(N)) distinguishable configurations to support content-addressable lookup of arbitrary tokens.

However, the computational cost of the state update scales as O(N²d) in the general case (Mamba-1) or O(Nd) with structured matrices (Mamba-2). Empirical scaling:

| State Dimension N | Relative Quality | Memory Cost | Retrieval at 16K |
|---|---|---|---|
| 16 (Mamba-1) | Baseline | Low | N/A (architecture limited to ~2K effective) |
| 64 (Mamba-2) | +2-3% perplexity | 4× | ~0% without intervention |
| 128 (Mamba-2) | +1-2% further | 8× | ~0% without intervention |
| 256+ | Diminishing returns | 16×+ | Still requires training-length matching |

The key insight: **expanding N yields quality improvements that saturate well before reaching retrieval parity with attention**. State caches account for up to 79.6% of memory in Mamba2-2.7B after weight quantization — this is a fundamental information-theoretic limit, not merely an engineering constraint.

#### 2.3.3 Absence of Content-Addressable Lookup (Severity: CRITICAL)

**Formal Statement (Weakness W10):** In standard attention, retrieval of value v_j given query q_i is a **direct, O(1) content-addressable lookup**:

$$o_i = \sum_{j=1}^N \underbrace{\text{softmax}(q_i^\top k_j / \sqrt{d})}_{\text{addressing weight}} \cdot v_j$$

In SSMs, retrieval of past information requires **decoding through the compressed state**:

$$o_i = C_i h_i = C_i \left(\sum_{j=1}^i \left(\prod_{k=j+1}^i \bar{A}_k\right) \bar{B}_j x_j\right)$$

There is no mechanism to directly address token j by its content — the state h_i is an exponentially weighted average, not an addressable memory. Tasks requiring associative recall (copying, needle-in-haystack, multi-hop QA) are **provably harder** for SSMs because they must route all retrieval through the fixed-dimensional state bottleneck.

**Empirical evidence:**
- Fully SSM models "reliably underperform on copying and associative recall tasks" (BabyLM 2025)
- Microsoft SAMBA: SSMs "struggle with memory recall due to their Markovian nature" — hybrid architectures (Mamba + sliding window attention) achieve 18.1% higher GSM8K accuracy
- The Mamba-2 "State Space Duality" reveals that SSMs are equivalent to masked linear attention with a **semiseparable mask** — a strictly less expressive form than full self-attention

#### 2.3.4 State Collapse Beyond Training Length (Severity: HIGH)

**Formal Statement (Weakness W11):** When inference context length exceeds training context length, some SSM hidden state channels exhibit catastrophic variance explosion. The Tsinghua University study (arXiv:2410.07145) identifies that specific attention heads (e.g., layer 38, heads 2, 4, 7 in Mamba-2 370M) retain >80% memory strength even at t = 8K, **refusing to release old information**. This is because the model overfits its state parameters to the training length — it never learned to forget, since the state capacity was never saturated during training.

The critical length K (context length at which state capacity saturates) scales **linearly** with state dimension N. Training on sequences ≥ K forces the model to learn proper forgetting dynamics. Training on sequences < K creates models that collapse catastrophically beyond their training horizon.

---

### 2.4 Linear (Kernelized) Attention: Rank and Approximation Bottlenecks

#### 2.4.1 Rank Saturation in the Recurrent State (Severity: CRITICAL)

**Formal Statement (Weakness W12):** Linear attention with feature map φ: ℝ^d → ℝ^m maintains a recurrent state:

$$M_t = M_{t-1} + \phi(k_t) \phi(v_t)^\top \in \mathbb{R}^{m \times m}$$

Since M_t is a sum of t rank-1 outer products:

$$\text{rank}(M_t) \leq \min(t, m)$$

For t > m (which occurs when sequence length exceeds the feature dimension), the state matrix saturates at rank m. Subsequent tokens must be encoded into a state that has exhausted its degrees of freedom — new information overwrites old information through superposition.

In contrast, softmax attention's attention matrix can have rank up to N (full rank) because the exponential nonlinearity introduces cross-dimensional coupling that linear attention's factorization cannot replicate. As proven by Mongaras & Larson (2025), linear attention corresponds to the **first-order (n=1) Taylor term** of the softmax exponential:

$$e^{q^\top k} = \sum_{n=0}^\infty \frac{1}{n!}(q^{\otimes n})^\top (k^{\otimes n})$$

Softmax attention implicitly uses all higher-order Kronecker product interactions (n=2, 3, ...), each with state dimension d^n. Linear attention discards all n ≥ 2 terms.

**Quantification:**

| Task | Standard Attention | Linear Attention (m=d=64) |
|---|---|---|
| WikiText-103 PPL | 18.3 | 19.1–19.5 (+4–7%) |
| Copying (length L) | ~100% for L ≤ N_train | Degrades for L > m |
| Needle-in-Haystack | ~100% within training range | Degraded; depends on m |
| Associative recall | Strong | Weak (non-injective mapping) |

#### 2.4.2 Approximation Error Bounds (Severity: HIGH)

**Formal Statement (Weakness W13):** For Performer (Choromanski et al., 2021), the softmax kernel is approximated as:

$$K_{sm}(q, k) = e^{q^\top k} = \mathbb{E}_{w \sim \mathcal{N}(0,I)}[e^{w^\top q - \|q\|^2/2} \cdot e^{w^\top k - \|k\|^2/2}]$$

The random feature estimator with m features has variance:

$$\text{Var}[\hat{K}_{sm}(q,k)] = \frac{1}{m}\left(e^{2q^\top k + \|q\|^2 + \|k\|^2} - e^{2q^\top k}\right)$$

which grows **exponentially** in ‖q‖² + ‖k‖². Without query/key normalization, this variance is **unbounded** — a single outlier query can catastrophically destabilize the approximation.

For Linformer (Wang et al., 2020), the projection dimension k must satisfy:

$$k = O\left(\frac{r \log N}{\epsilon^2}\right)$$

to achieve ‖Attention - LinAttention‖_F ≤ ε · ‖Attention‖_F, where r is the rank of the attention matrix. While k doesn't need to grow with N (only log N), **r grows with task complexity** — for tasks requiring fine-grained token distinction, r can approach N, making the bound vacuous.

**Quantification of the approximation gap:**

| Method | Error Source | Magnitude |
|---|---|---|
| Performer (m=256) | Variance of random features | ~2-5% perplexity degradation |
| Linformer (k=256) | Low-rank projection error | ~3-8% degradation on retrieval tasks |
| Linear attention (general) | Missing higher-order Taylor terms | Representational gap of Ω(log n) for certain functions |

#### 2.4.3 The Expressiveness-Rank Hierarchy (Severity: MODERATE)

**Formal Statement (Weakness W14):** The expressiveness of recurrent models is governed by the structure of their state transition matrices:

| Matrix Structure | Circuit Class | Example Models | Retrieval Capability |
|---|---|---|---|
| Diagonal | TC⁰ | Mamba, GLA, S4 | Cannot solve parity or mod-3 |
| Diagonal + rank-1 (Householder) | TC⁰+ | DeltaNet, RWKV-7 | Can solve parity |
| Diagonal + rank-r (r Householders) | Approaches NC¹ | DeltaProduct | Any permutation group word problem |
| Full matrix (softmax attention) | TC⁰ (but infinite-dimensional state) | Transformer | Near-perfect retrieval within training |

The fundamental result (Grazzi et al., 2024; DeltaProduct, 2025): **diagonal linear RNNs with eigenvalues in [0,1] are provably incapable of state-tracking** for addition modulo 3. Expanding to diagonal + rank-r progressively broadens the circuit class, but each increment in rank adds computational cost — a strict expressiveness-efficiency tradeoff with no free lunch.

---

## 3. Important Papers & References

### FlashAttention & Hardware Co-Design

1. **Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."** *NeurIPS 2022*. Introduced the tiling + online softmax recomputation paradigm. Proved that exact attention can be IO-optimal through algorithmic redesign rather than approximation.

2. **Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."** *ArXiv:2307.08691*. Improved parallelism, reduced non-matmul FLOPs. Achieved 50–73% A100 utilization.

3. **Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision."** *ArXiv:2407.08608*. Introduced warp specialization, TMA-based async data movement, FP8 support for H100. ~75% utilization.

4. **Dao, T., Shah, J., Fu, D., et al. (2026). "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling."** *MLSys 2026*. Software-emulated exponential, conditional rescaling, 2-CTA MMA, TMEM utilization for B200. 1,605 TFLOPS/s (71% utilization). The definitive treatment of the asymmetric scaling problem.

5. **NVIDIA (2026). "Making Softmax More Efficient with NVIDIA Blackwell Ultra."** *NVIDIA Technical Blog*. Documents the SFU doubling in GB300 and the 35% improvement in DeepSeek-V3 FP8 forward pass throughput.

### RoPE Formal Analysis

6. **Chen, Y., et al. (2024). "HoPE: A Novel Positional Encoding Without Long-Term Decay for Enhanced Context Awareness and Extrapolation."** *ACL 2025*. Component-level decomposition of RoPE; identifies "activated" frequency components that cause OOD logits in layer 1 during extrapolation. Proposes VAF-based analysis.

7. **Hua, Y., et al. (2024). "Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization."** Formalizes RoPE as Non-Uniform Discrete Fourier Transform; identifies spectrum damage (leakage + distortion) as the root cause of extrapolation failure.

8. **Xiong, Y., et al. (2025). "DoPE: Denoising Rotary Position Embedding."** *ArXiv:2511.09146*. Introduces truncated matrix entropy and spectral lower bounds; proves that low-frequency cone condition → attention sink formation. Shows 10-point improvements by removing PE from high-rank heads.

9. **Wertheimer, N., et al. (2025/2026). "Frayed RoPE and Long Inputs: A Geometric Perspective."** *ArXiv:2603.18017*. Geometric PCA analysis: RoPE disperses key/query clusters by shrinking the First Singular Value ratio, causing sink token failure.

### State Space Models

10. **Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."** *ArXiv:2312.00752*. Introduced selective SSMs with input-dependent parameters. Identified the inherent content-based reasoning weakness.

11. **Dao, T., & Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality."** *ArXiv:2405.21060*. Proves that SSMs with scalar-diagonal structure are equivalent to masked linear attention with a semiseparable matrix. The SSD framework enables N=64/128 states.

12. **Tsinghua University Study (2024). "State Collapse in Long-Context SSMs."** *ArXiv:2410.07145*. Identifies catastrophic variance explosion beyond training length. Shows Mamba-2 key retrieval drops from ~100% at 8K to ~0% at 16K.

13. **Li, L., et al. (2025). "LAMB: A Training-Free Method to Enhance the Long-Context Understanding of SSMs via Attention-Guided Token Filtering."** *ACL 2025 (short)*. Debiased attention and contrastive filtering: improves Mamba2-1.3B retrieval from 0.27% to 33.96% at 16K without retraining.

14. **Lee, H., et al. (2025). "Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling."** *BabyLM 2025*. Documents that fully SSM models "reliably underperform on copying and associative recall tasks."

### Linear Attention & Expressiveness Theory

15. **Choromanski, K., et al. (2021). "Rethinking Attention with Performers."** *ICLR 2021*. Introduced FAVOR+ algorithm: positive orthogonal random features with exponential tail bounds for unbiased softmax kernel approximation.

16. **Wang, S., Li, B., Khabsa, M., Fang, H., & Ma, H. (2020). "Linformer: Self-Attention with Linear Complexity."** *ArXiv:2006.04768*. Proved that attention matrices are low-rank; projection dimension k = O(r log n / ε²) suffices for ε-approximation.

17. **Mongaras, L., & Larson, J. (2025). "On the Expressiveness of Softmax Attention: A Recurrent Neural Network Perspective."** *ArXiv:2507.23632*. Proves linear attention is first-order Taylor approximation of softmax; softmax implicitly uses infinite-order Kronecker product interactions.

18. **Grazzi, R., et al. (2024). "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues."** Establishes circuit complexity boundaries: diagonal RNNs in TC⁰; Householder-based models approach NC¹.

19. **DeltaProduct (NeurIPS 2025). "Improving State-Tracking in Linear RNNs via Householder Products."** Proves r Householder reflections enable general orthogonal transformations, escaping the diagonal rank bottleneck.

### Entropy Analysis & Length Generalization

20. **Li, Y., & Kong, J. (2025). "Information Entropy Invariance: Enhancing Length Extrapolation in Attention Mechanisms."** *ArXiv:2506.16640*. Formal log-n scaling to counteract entropy collapse; proves H(p) → log N under softmax.

21. **Vasylenko, P., et al. (2025). "Long-Context Generalization with Sparse Attention."** Proposes α-entmax to maintain Θ(1/k) attention regardless of N; ASEntmax with learnable head-specific scaling.

22. **Su, J. (2023). "Entropy Invariance in Attention's Scale Operation."** *Blog series*. Pioneered the log-n scaling perspective and the Softmax₁ modification (add 1 to denominator).

---

## 4. Open Questions & Future Directions

### 4.1 Beyond Softmax: New Normalization Functions

The single most impactful research direction is replacing softmax with a normalization function that:
1. Is expressible as a sequence of matrix multiplies (no element-wise exponentials), leveraging tensor cores for the entire forward pass
2. Maintains stable entropy (not → log N) as N → ∞
3. Preserves the gradient properties needed for training stability
4. Admits a dual formulation (quadratic for training, linear/recurrent for inference)

Promising candidates: polynomial normalization, rational function normalization, sparse entmax variants, and the Softmax₁ "supermod" family. The key mathematical challenge is proving that such a function can match or exceed softmax's representational capacity while eliminating the SFU bottleneck.

### 4.2 Unifying RoPE Insights

The four independent analyses (HoPE, FoPE, DoPE, Frayed RoPE) converge on the same root cause — low-frequency components trained on less than one period within the training context — but propose different fixes. Open question: can these insights be unified into a single positional encoding that mathematically guarantees length generalization without retraining? The NUDFT formulation (FoPE) suggests that a Fourier-series approach with explicit frequency band management could achieve this.

### 4.3 SSM Retrieval Without Hybrid Architectures

The current "solution" to SSM retrieval weakness — hybrid Mamba+Attention architectures — is a practical workaround, not a mathematical resolution. Is there a principled way to add content-addressable lookup to the SSM state update without sacrificing the O(N) complexity? Candidates include: learned sparse attention patterns over the compressed state, differentiable memory addressing (like Neural Turing Machines but efficient), and state expansion with structured addressing.

### 4.4 The Rank Gap: Closing the Expressiveness Chasm

Linear attention's rank bottleneck (rank ≤ d for the recurrent state) versus softmax attention's implicit rank ≤ N is a fundamental gap. Can higher-order feature maps (φ_2(x) = x ⊗ x, φ_3(x) = x ⊗ x ⊗ x, ...) close this gap efficiently? The Mongaras-Larson Taylor expansion suggests that including Kronecker product terms of order 2 and 3 might capture most of the benefit while keeping the state dimension tractable (d² + d³ vs. infinite series). This is an open theoretical and empirical question.

### 4.5 Hardware-Software Co-Design for the Post-Blackwell Era

As NVIDIA adds SFU capacity (GB300 doubles MUFU/SFU) and new memory hierarchies (TMEM in Blackwell), the optimal attention algorithm is a moving target. The field needs:
- **Auto-tuning frameworks** that can discover optimal tiling and scheduling for each new GPU generation without months of manual engineering
- **Hardware-agnostic DSLs** (like CuTe) that compile attention algorithms to efficient kernels across generations
- **Theoretical models** of the hardware-computation bottleneck that predict, rather than react to, the next generation's constraints

### 4.6 Entropy-Controlled Attention

The "attention dilution" problem (H(p) → log N) and the "attention sink" problem (low-entropy bands absorbing all mass) are two sides of the same coin. Can a unified entropy regularization framework prevent both failure modes while remaining computationally efficient? This connects to information-theoretic approaches like α-entmax, Softmax₁, and log-n scaled attention — but the open question is whether these can be made hardware-efficient (avoiding the SFU bottleneck) while maintaining gradient stability.

---

## 5. Relevance to Main Topic

This weakness audit provides the foundational analysis for the broader research program on designing a next-generation attention mechanism. The catalog of 14 formalized weaknesses reveals a consistent pattern: **four independent research communities have each identified fundamental bottlenecks in their respective paradigms, and these bottlenecks are structurally related**.

### The Unifying Pattern

1. **The Nonlinearity Problem**: FlashAttention's SFU bottleneck, RoPE's softmax entropy collapse, and linear attention's rank limitation all trace back to the softmax exponential. A replacement normalization function is the single highest-leverage intervention.

2. **The Compression-Resolution Tradeoff**: SSMs compress context into a fixed state (losing retrieval resolution); linear attention compresses into a fixed-rank matrix (losing token distinction); softmax attention avoids compression (at O(N²) cost). A mechanism that dynamically allocates representational capacity — using higher resolution for important tokens and compression for context — could navigate this tradeoff.

3. **The Position Encoding Problem**: RoPE's frequency extrapolation failure and the broader "NoPE" challenge (can we eliminate explicit position encoding?) suggest that positional information should be integrated differently — perhaps as a property of the attention normalization function itself rather than as a rotation applied to Q and K.

4. **The Hardware Scaling Trap**: Each GPU generation shifts the bottleneck to whichever resource scaled worst. A truly future-proof attention mechanism must not depend on any single hardware unit — it must be expressible entirely in terms of operations that all scale together (primarily matrix multiplies).

### Design Principles for Next-Generation Attention

From this audit, we extract five design principles:

| Principle | Justification | Source Weakness |
|---|---|---|
| **Matrix-multiply-only forward pass** | Eliminates SFU bottleneck (512:1 gap) | W1, W2 |
| **Entropy-stable normalization** | Prevents attention dilution at long contexts | W5, W6 |
| **Dual quadratic/linear formulation** | Enables exact training + efficient inference | W3, W12, W13 |
| **Native content-addressable retrieval** | Avoids SSM-style compressed state bottleneck | W8, W10, W11 |
| **Generation-invariant algorithm** | Avoids per-GPU kernel redesign cycle | W2 |

These principles directly inform the novel mechanism design (sub-topic 2), the complexity analysis (sub-topic 3), and the hardware-aware implementation (sub-topic 4). The weaknesses ranked CRITICAL in this audit — particularly the softmax SFU bottleneck (W1), RoPE frequency extrapolation (W4), SSM fuzzy memory (W8, W10), and linear attention rank saturation (W12) — serve as the primary design constraints that any successor mechanism must address.

---

## Appendix A: Ranked Catalog of Weaknesses

### Ranking Methodology

Each weakness is scored on two axes (each 1–10):
- **Severity (S)**: Impact on model quality, throughput, or scaling. 10 = makes the mechanism unusable or fundamentally broken beyond a certain regime.
- **Tractability (T)**: How feasible it is to fix with known techniques. 10 = has a clear, implementable solution; 1 = requires fundamental theoretical breakthrough.

The **Priority** ranking balances severity against tractability: weaknesses that are both severe and tractable are highest priority; severe but intractable weaknesses require long-term research investment.

### Ranked Catalog

| Rank | ID | Weakness | Family | S | T | Priority Rationale |
|---|---|---|---|---|---|---|
| **1** | **W1** | **Softmax SFU bottleneck: 512:1 TC:MUFU throughput gap; forward pass co-bottlenecked by exp and GEMM** | FlashAttention | 10 | 7 | Fixable via alternative normalization function; highest immediate throughput impact |
| **2** | **W8** | **SSM exponential forgetting: retrieval collapses from ~100% (at train length) to ~0% (at 2×)** | SSM | 10 | 4 | Severe limitation but requires architectural change; training-free mitigations (LAMB) provide partial relief |
| **3** | **W12** | **Linear attention rank saturation: state matrix rank capped at d; missing higher-order Taylor interactions** | Linear Attention | 10 | 5 | Inherent to the factorization; higher-order feature maps possible but increase computational cost |
| **4** | **W10** | **SSM absence of content-addressable lookup: no native key-value retrieval mechanism** | SSM | 10 | 4 | Architectural limitation; hybrid architectures are a workaround, not a fix |
| **5** | **W4** | **RoPE high-frequency OOD rotation: unseen angles during extrapolation cause cascading layer disruption** | RoPE | 9 | 6 | Several fixes proposed (HoPE, FoPE, DoPE); unifying them is the challenge |
| **6** | **W5** | **Softmax entropy collapse: H(p) → log N as N→∞; attention becomes uniform regardless of content** | All (Softmax) | 9 | 7 | α-entmax, log-n scaling, and Softmax₁ provide tractable fixes; hardware efficiency remains a concern |
| **7** | **W2** | **Per-generation kernel redesign: bottlenecks shift between HBM BW → registers → SFU → SMEM BW** | FlashAttention | 8 | 5 | Auto-tuning and CuTe-DSL reduce engineering cost; fundamentally unsolvable without hardware-agnostic algorithms |
| **8** | **W11** | **SSM state collapse beyond training: overfitted state parameters refuse to forget; variance explosion** | SSM | 9 | 6 | Training on sequences ≥ critical length K prevents it; but K scales with state size, increasing training cost |
| **9** | **W9** | **SSM state expansion vs. quality: larger N improves quality but retrieval never reaches attention parity** | SSM | 8 | 5 | Diminishing returns after N=128; quantization (Q-Mamba) and pruning (PerfMamba) help |
| **10** | **W13** | **Linear attention approximation error: variance grows exponentially in ‖q‖²+‖k‖² without normalization** | Linear Attention | 8 | 6 | Query/key normalization mitigates variance; but introduces additional constraints on model design |
| **11** | **W6** | **RoPE embedding collapse: low-frequency components form attention sinks; Θ(√N) singular value growth** | RoPE | 7 | 7 | DoPE's Gaussian reparameterization provides a training-free fix with 10-point improvements |
| **12** | **W7** | **RoPE spectrum damage: linear layers + nonlinearities destroy clean frequency separation** | RoPE | 7 | 5 | Inherent to deep network structure; architectural changes needed (DSP-aware layers) |
| **13** | **W3** | **Recomputation-based backward: backward pass is SMEM-bound on B200; recomputation overhead scales with tiling** | FlashAttention | 6 | 6 | 2-CTA MMA in FA4 partially addresses; fundamental tension between memory savings and recomputation cost |
| **14** | **W14** | **Expressiveness-rank hierarchy: diagonal RNNs cannot solve mod-3; each rank increment adds cost** | Linear/SSM | 7 | 5 | Theoretical lower bound; r Householder products progressively expand circuit class |

### Summary of Formal Mathematical Statements

**W1** (SFU Bottleneck):
$$\frac{T_{tc}}{T_{exp}} = 512 \text{ on B200}; \quad \text{Cycles}_{GEMM} = \text{Cycles}_{exp} = 1024 \text{ per tile}$$

**W4** (OOD Rotation):
$$\exists i \text{ s.t. } \alpha_i = \lfloor N_{train} \cdot \theta_i / 2\pi \rfloor < 1 \implies \text{OOD}(C_i) \text{ at } N_{test} > N_{train}$$

**W5** (Entropy Collapse):
$$\lim_{N \to \infty} H(p) = \log N; \quad p_i = \Theta(1/N) \text{ for bounded logits}$$

**W8** (Exponential Forgetting):
$$\|\alpha_{i,j}\| = \|C_i^\top (\prod_{k=j+1}^i \bar{A}_k) \bar{B}_j\| \leq \|\bar{A}\|^{i-j} \cdot \text{const} \to 0 \text{ as } (i-j) \to \infty$$

**W10** (No Content Addressing):
$$\text{SSM retrieval: } o_i = C_i \cdot \text{compress}(x_1, ..., x_i); \quad \text{Attention retrieval: } o_i = \sum_j \text{lookup}(q_i, k_j) \cdot v_j$$

**W12** (Rank Saturation):
$$\text{rank}(M_t) \leq \min(t, m); \quad M_t = \sum_{s=1}^t \phi(k_s)\phi(v_s)^\top \text{ saturates at } t = m$$

---

## Appendix B: Quantitative Comparison Table

| Property | Standard Attention | FlashAttention-4 (B200) | Mamba-2 (N=128) | Linear Attention (d=64) |
|---|---|---|---|---|
| **FLOPs (FWD)** | 2N²d | 2N²d (exact) | O(Nd²) | 2Nmd (=128Nd) |
| **FLOPs (BWD)** | 4N²d | 4N²d + tile overhead | O(Nd²) | 4Nmd |
| **HBM Reads/Writes** | O(N²) | O(N²d²/M) | O(Nd) | O(Nd²) |
| **Memory (KV Cache)** | O(Nd) | O(Nd) | O(Nd) (state) | O(d²) (state only) |
| **Peak Throughput (B200)** | ~200 TFLOPS (memory-bound) | **1,605 TFLOPS** | N/A (not TOPS-limited) | ~800 TFLOPS (est.) |
| **Arithmetic Intensity** | ~64 FLOP/byte | ~506 FLOP/byte (compute-bound) | ~200+ FLOP/byte | ~500+ FLOP/byte |
| **Retrieval Quality** | Near-perfect (≤ N_train) | Near-perfect (≤ N_train) | ~0% at 2× N_train | Degraded for fine-grained tasks |
| **Length Generalization** | Poor (RoPE-dependent) | Poor (RoPE-dependent) | Catastrophic collapse | Moderate (rank-limited) |
| **SFU Dependence** | Full (softmax) | Full (softmax) | None (no softmax) | Low (approximation) |
| **Generation-Invariant?** | N/A | No (redesign per gen) | Yes (general algorithm) | Yes (general algorithm) |

---

*Research conducted: June 2026. Sources include FlashAttention-4 (MLSys 2026), HoPE (ACL 2025), FoPE, DoPE (2025), Frayed RoPE (2026), Mamba/Mamba-2, LAMB (ACL 2025), Performer (ICLR 2021), Linformer (2020), and associated theoretical analyses.*
