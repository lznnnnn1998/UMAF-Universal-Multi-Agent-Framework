# Attention Alternatives: State Space Models and Linear RNNs

## Overview

The self-attention mechanism has been the cornerstone of modern large language models since the Transformer architecture was introduced in 2017. However, its quadratic complexity O(N²) in sequence length creates fundamental bottlenecks: inference requires a growing key-value cache proportional to sequence length, training costs explode for long documents, and hardware utilization suffers from memory-bound attention operations. These limitations have motivated an intense research effort to find replacements or augmentations for self-attention that preserve its modeling power while achieving sub-quadratic complexity.

Two major families of alternatives have emerged as the most credible challengers. **State Space Models (SSMs)** — originating from control theory and signal processing — were adapted for deep learning through the Structured State Space (S4) framework, culminating in Mamba (S6) and Mamba-2, which achieve linear-time sequence processing via input-dependent selective parameters and hardware-aware parallel scans. **Linear Recurrent Architectures**, including RWKV (Receptance-Weight-Key-Value), xLSTM (Extended LSTM), and Google DeepMind's Griffin/Hawk family, reformulate sequence modeling as efficient recurrence with constant inference memory, leveraging modern parallelized training. A third paradigm, **Hybrid Architectures** (Jamba, Zamba, Griffin), strategically interleaves small amounts of attention with SSM or recurrent layers, achieving the best of both worlds: near-perfect retrieval from attention plus linear-time efficiency from SSMs.

As of mid-2025, the landscape has crystallized around a key insight formalized by Dao & Gu (2024): **Transformers and SSMs are mathematically dual formulations of the same underlying structured matrix multiplication** — differing only in whether the computation is expressed in "quadratic form" (attention) or "linear form" (recurrence). This duality unifies the field and suggests that the future lies not in choosing one approach over the other, but in principled hybridization that dynamically selects the optimal computation pattern per layer or per task.

---

## Key Methods & Approaches

### 1. State Space Models (SSMs): From S4 to Mamba-2

#### 1.1 Mathematical Foundation

A continuous-time state space model maps a 1-D input signal u(t) to output y(t) through an N-dimensional latent state h(t):

```
h'(t) = A·h(t) + B·u(t)
 y(t) = C·h(t) + D·u(t)
```

Where A ∈ ℝ^(N×N), B ∈ ℝ^(N×1), C ∈ ℝ^(1×N), D ∈ ℝ. For deep learning, this is discretized via a step size Δ into a recurrent formulation suitable for processing token sequences:

```
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t
```

where Ā = exp(ΔA) and B̄ = (ΔA)⁻¹(exp(ΔA) − I)·ΔB.

The crucial insight is that this recurrence can also be expressed as a convolution with an infinitely long kernel K̄ = (CB̄, CĀB̄, CĀ²B̄, ..., CĀ^(L-1)B̄), enabling efficient training via FFT while allowing recurrent inference.

#### 1.2 S4: Structured State Space (Gu et al., 2022)

**S4** introduced the key breakthrough of parameterizing the state matrix A using **HiPPO theory** (High-order Polynomial Projection Operators), which provides a mathematically optimal way to compress long sequences into bounded state representations. The HiPPO-LegS matrix ensures that the SSM state optimally memorizes input history:

```
A_nk = −{(2n+1)^(1/2)·(2k+1)^(1/2)  if n > k
       {n+1                            if n = k
       {0                              if n < k
```

S4 diagonalizes A into a normal-plus-low-rank (NPLR) form and uses the **Woodbury identity** to compute the convolution kernel efficiently, reducing the naive O(N²L) computation to O(NL). S4 achieved state-of-the-art on the Long Range Arena (LRA) benchmark with up to 60× faster inference than Transformers on sequences of length 16K.

#### 1.3 S4D: Diagonal State Space (Gu et al., 2022)

**S4D** simplified S4 by restricting A to a purely diagonal matrix A = diag(a₁, ..., a_N) with complex eigenvalues, eliminating the low-rank correction. This made the kernel computation a simple Vandermonde matrix multiplication while preserving performance. S4D demonstrated that the HiPPO-initialized diagonal SSM alone is sufficient for strong sequence modeling.

#### 1.4 H3: Hungry Hungry Hippos (Fu, Dao et al., 2023)

**H3** (ICLR 2023 Spotlight) was designed specifically for language modeling, addressing S4's failure on **associative recall** — the ability to attend to a token by its content rather than its position. H3 uses two SSM "hippos" working in tandem:

- **Memory hippo**: Remembers tokens from earlier in the sequence (via standard SSM recurrence)
- **Comparison hippo**: Compares tokens based on their relative position (via shifted SSM)
- Their outputs are combined through **multiplicative gating** (element-wise multiplication)

H3 at 2.7B parameters achieved lower perplexity than Transformers on OpenWebText, and a hybrid H3-Attention variant (125M with just 2 attention layers) outperformed pure Transformers by 1.0 PPL. The **FlashConv** kernel using fused block FFT achieved ~2× speedup on LRA.

#### 1.5 Mamba-1 / S6: Selective State Space (Gu & Dao, Dec 2023)

**Mamba** made SSMs truly competitive with Transformers through three innovations:

1. **Input-dependent selectivity**: Unlike S4's fixed A, B, C, Mamba makes parameters B, C, and the discretization step Δ data-dependent through learned linear projections:
   ```
   B_t = W_B·x_t,  C_t = W_C·x_t,  Δ_t = softplus(W_Δ·x_t + b_Δ)
   ```
   This allows selective information filtering — the model learns to ignore irrelevant tokens and retain important ones.

2. **Hardware-aware parallel scan**: Since selectivity breaks the fixed convolution (A_t is now also input-dependent via Δ_t), Mamba uses an associative scan algorithm that achieves O(log N) parallel depth while maintaining O(N) total work, optimized for GPU memory hierarchy through kernel fusion and recomputation strategies.

3. **Simplified architecture**: Replaces the SSM + MLP block of H3 with a single unified Mamba block, reducing parameters while increasing expressivity.

Mamba scales to 2.8B parameters with performance competitive against Transformers of similar size, including Llama and Pythia.

#### 1.6 Mamba-2 / SSD: State Space Duality (Dao & Gu, ICML 2024)

**Mamba-2** represents the most important theoretical advance in the field. The paper "Transformers are SSMs" proves that SSMs and attention are mathematically dual formulations through the concept of **semiseparable matrices**:

- The SSM transformation y = M·x produces a matrix M where M_ji = C_j·A_{j-1}···A_{i+1}·B_i — this is a **semiseparable matrix** (each submatrix below the diagonal has rank ≤ N)
- Linear attention produces y = (L ∘ QK^T)·V where L is a causal mask — when L has multiplicative structure (L_ij = ∏_{k=j+1}^i l_k), this is ALSO a semiseparable matrix
- **Theorem**: Structured Masked Attention (SMA) with a multiplicative causal mask is equivalent to an SSM with scalar-diagonal A

The **SSD (State Space Duality)** layer unifies both perspectives:

| Mode | Computation | Complexity | Best For |
|------|-------------|------------|----------|
| Attention form | Quadratic matrix multiply | O(N²) | Training (parallel) |
| Recurrence form | Sequential state update | O(N) | Inference (autoregressive) |
| SSD block decomposition | Hybrid block-matrix multiply | O(N²/B + BN) | Training on GPU Tensor Cores |

Key Mamba-2 improvements over Mamba-1:
- **State dimension**: N=16 → N=256 (8× expansion), enabled by scalar-A restriction
- **Training speed**: 2–8× faster via block decomposition using Tensor Core matrix multiplies instead of scan operations
- **Multi-head support**: Introduces Multi-Input SSM (analogous to multi-head attention)
- **Parallelism**: Supports tensor parallelism, sequence parallelism, and variable-length sequences
- A 2.7B Mamba-2 outperforms 6.9B Pythia; 4–6 attention layers added to Mamba-2 outperforms both pure Mamba-2 and Transformer++ at 8B scale

---

### 2. RWKV and Linear Recurrent Architectures

#### 2.1 RWKV: Receptance-Weight-Key-Value (Peng, 2021–2025)

**RWKV** (pronounced "RwaKuv") is a purely attention-free architecture that achieves Transformer-level language modeling through a channel-wise time-decay mechanism. The core formulation:

```
O_i = σ(R_i) · (∑_{j=1}^i e^{W_{i-j} + K_j}·V_j) / (∑_{j=1}^i e^{W_{i-j} + K_j})
```

Where:
- **R** = Receptance (sigmoid-gated "accept" gate, analogous to forget gate)
- **W** = Channel-wise time-decay (learnable, data-independent positional bias)
- **K, V** = Key and Value (similar role to Transformers, but no queries)

The critical insight is that W is **data-independent** — it's a learned per-channel decay rate. This means:
- **Training**: The summation can be computed in parallel via cumulative sum (like a Transformer)
- **Inference**: It's just a simple recurrence, no KV cache needed

**Version evolution:**
- **RWKV-4** (2021–2023): Original AFT-based formulation, scaled to 14B parameters
- **RWKV-5 "Eagle"** (2024): Matrix-valued states for richer representations; introduced multi-head WKV
- **RWKV-6 "Finch"** (2024): Data-dependent dynamic recurrence — W becomes partially input-dependent
- **RWKV-7 "Goose"** (March 2025): Generalized Delta Rule for state updates; described as a **meta-in-context learner** — performs implicit gradient descent on its internal state at each token; achieves state-of-the-art among open-source 3B models; supports 100+ languages

RWKV is deployed in production inside Microsoft Windows & Office, runs a 14B model in ~3GB VRAM in INT8 for arbitrary-length sequences, and is a Linux Foundation AI project under Apache 2.0.

#### 2.2 xLSTM: Extended Long Short-Term Memory (Beck et al., NeurIPS 2024)

**xLSTM**, from Sepp Hochreiter's group (the original LSTM inventors), modernizes LSTMs with two innovations:

1. **Exponential gating**: Standard sigmoid gates are replaced with exp(x), enabling **decisive memory updates** — the model can truly overwrite old memories rather than soft-blend. Stabilized via a normalizer state.

2. **Dual memory structures**:
   - **sLSTM** (scalar memory): Retains the memory-mixing recurrence of classic LSTMs — excels at state tracking and formal language tasks (where Transformers and SSMs fail)
   - **mLSTM** (matrix memory): Stores key-value associations via a covariance update rule C_t = f_t·C_{t-1} + i_t·v_t·k_t^T, fully parallelizable, with d×d storage capacity

At 300B token training scale (125M–1.3B parameters), xLSTM achieves lower perplexity than Llama, Mamba, and RWKV-4 on 568/571 domains of the PALOMA benchmark. Crucially, xLSTM **maintains stable perplexity when extrapolating from 2048 training context to 16,384 tokens** — a regime where Transformers fail completely.

**Limitations**: sLSTM is not parallelizable; mLSTM has d×d computational cost; CUDA kernels are less optimized than FlashAttention/Mamba scan.

#### 2.3 Griffin / Hawk / RecurrentGemma (Google DeepMind, 2024)

Google DeepMind introduced a family of efficient architectures based on the **Real-Gated Linear Recurrent Unit (RG-LRU)**:

**RG-LRU Core Design:**
- Diagonal recurrent weight a = σ(Λ), guaranteeing 0 ≤ a ≤ 1 (stable)
- Input gate i_t and recurrence gate r_t (no state-dependent gating for hardware efficiency)
- All-real arithmetic (complex numbers found unnecessary for language)
- No orthogonal polynomial initialization required (unlike many SSMs)

**Architecture variants:**
- **Hawk**: Pure RG-LRU recurrence (no attention). Exceeds Mamba-3B on downstream tasks with half the training tokens.
- **Griffin**: RG-LRU + local sliding-window attention (2048-token window). Matches Llama-2 with ~6–7× fewer training tokens; scales to 14B.
- **RecurrentGemma**: Open release of Griffin architecture, 2B parameters, comparable to Gemma-2B with better inference efficiency.

Key results: Griffin offers **higher throughput and lower latency** than Multi-Query Attention Transformers on long sequences, with **fixed-size state** (no growing KV cache). It extrapolates to sequences longer than those seen during training.

---

### 3. Hybrid Architectures

#### 3.1 Jamba (AI21 Labs, 2024)

**Jamba** is a triple-hybrid: Transformer attention + Mamba SSM + Mixture of Experts (MoE). At 52B total parameters (only 12B active during inference via MoE), it uses a striped 3:1:4 pattern (3× SSM, 1× Attention, 4× SSM) with global attention heads.

- **256K context window** with 4GB KV cache (vs. 32GB for Mixtral at equivalent context)
- **3× higher throughput** than comparable dense models
- Competitive with Llama-2 70B and Mixtral

#### 3.2 Zamba (Zyphra, 2024–2025)

**Zamba** uses a Mamba backbone with **shared attention blocks** interleaved (ABAB pattern in Zamba2). Notable for being the most **sample-efficient** hybrid — achieving competitive performance with far fewer training tokens than peers. Zamba2-2.7B upgrades to Mamba-2 blocks and adds a second shared attention block, substantially outperforming comparable models in inference latency and memory cost.

#### 3.3 Functional Segregation Discovery (2025)

A landmark mechanistic study of hybrid architectures revealed **complete functional segregation**:

- **Self-attention layers exclusively perform retrieval** — ablating attention causes catastrophic retrieval failure (0% accuracy)
- **SSM layers contribute nothing to retrieval**, even with prompting strategies like "Just Read Twice"
- Sparsifying attention to **15% of heads maintains near-perfect retrieval** while preserving 84% of MMLU performance
- This was validated across RecurrentGemma-2B, RecurrentGemma-9B, and Jamba-Mini

This finding explains why even small amounts of attention (7–10% of layers) in hybrids dramatically improve performance — SSMs have a fundamental "fuzzy memory" limitation for exact token-level lookup, which attention solves perfectly.

#### 3.4 Other Notable Hybrids

- **StripedHyena / StripedMamba**: Alternating Hyena convolutions with Mamba SSM layers
- **Samba**: Striped sliding-window attention + Mamba blocks
- **Falcon Mamba (TII)**: Pure Mamba 7B model from the Falcon team
- **Codestral Mamba (Mistral)**: Pure Mamba 7B specialized for code generation
- **MiniMax-Text-01**: Hybrid lightning attention + MoE at 100B+ scale (only non-transformer in LMSys top-20)

---

### 4. Comparative Benchmarks

#### 4.1 Where SSMs Outperform Attention

| Domain | SSM Advantage | Magnitude |
|--------|---------------|-----------|
| **Long-context training** | Linear complexity enables training on much longer sequences | 5–10× faster training at 16K+ tokens |
| **Genomics (DNA/RNA)** | Mamba-16 outperforms Transformers in RNA-Seq prediction | +2% R² (0.450 vs 0.437) |
| **Audio/speech** | Efficient long-duration modeling without quadratic blowup | Substantial |
| **Inference throughput** | Constant memory, no KV cache | 3–8× higher throughput at long contexts |
| **Length extrapolation** | xLSTM, Griffin maintain perplexity beyond training length | Transformers collapse |
| **Edge deployment** | RWKV 14B runs in 3GB VRAM, RWKV in MS Office | Infeasible for equivalent Transformers |

#### 4.2 Where SSMs Fall Short

| Task | Pure SSM Performance | Root Cause |
|------|---------------------|------------|
| **In-context learning (few-shot)** | Lags on 5-shot MMLU | Limited associative recall |
| **Exact copying / retrieval** | Fails on Phonebook, multi-hop QA | State capacity grows linearly with sequence |
| **Multi-query associative recall (MQAR)** | 70M Attention model beats 1.4B Hyena | Attention uses O(1) dimensions regardless of sequence |
| **State tracking / formal languages** | S4/H3/Hyena fail; Mamba partial | Expressiveness bounded by TC⁰ complexity class |
| **Context length generalization** | "State collapse" when input exceeds training length | Recurrent state overfitting |
| **Real-world training speed** | Sometimes slower than Transformers with Flash Attention | Ecosystem maturity gap |

#### 4.3 NVIDIA 8B Controlled Study (Waleffe et al., 2024)

The gold-standard comparison: same data (3.5T tokens), same scale (8B parameters):

| Architecture | Avg on 12 Standard Tasks | Long-Context (23 tasks) | Inference Speed |
|---|---|---|---|
| Pure Transformer | Baseline | Baseline | Baseline |
| Pure Mamba-2 | Matches/close | Lags on reasoning | Similar (scan vs FlashAttn) |
| **Mamba-2-Hybrid** (43% SSM + 7% Attn + 50% MLP) | **+2.65 pts above Transformer** | **Matches or exceeds** | **Up to 8× faster** |

#### 4.4 Architecture-by-Scale Distribution (2025)

- **0.7–1.5B**: Samba and RWKV7-World3 significantly outperform Llama 3.2/Qwen2.5 on multiple benchmarks
- **14–70B**: Only hybrids (Griffin, Jamba) remain competitive — no pure sub-quadratic models
- **100B+**: Only MiniMax-Text-01 (hybrid lightning attention + MoE) appears in LMSys top-20; no pure SSM cracks top-10

---

### 5. Theoretical Connections Between SSMs and Linear Attention

#### 5.1 The Unifying Semiseparable Matrix Framework

The fundamental theoretical contribution of Mamba-2 (Dao & Gu, ICML 2024) is proving that SSMs and linear attention are **the same mathematical object** computed in different ways:

**SSM form** (linear recurrence):
```
h_t = A_t·h_{t-1} + B_t·x_t
y_t = C_t^T·h_t
```
Unrolled: y = M_SSM·x where M_SSM_ji = C_j^T·A_{j-1}···A_{i+1}·B_i

**Linear attention form** (quadratic matrix multiply):
```
y = (L ∘ QK^T)·V = M_Attn·x
```
When L has multiplicative structure: M_Attn_ji = Q_j^T·K_i · ∏_{k=i+1}^j l_k

**Equivalence condition**: When the SSM's A matrix is scalar-diagonal (A = a·I), the SSM matrix M_SSM and attention matrix M_Attn are **both semiseparable matrices** — each submatrix below the diagonal has rank ≤ 1. They are algebraically equivalent, differing only in parameterization:
- SSM: parameterizes via (A, B, C, Δ)
- Attention: parameterizes via (Q, K, V, L)

#### 5.2 Dynamical Systems Framework (Sieber et al., NeurIPS 2024)

A unified control-theoretic framework:
```
h_i = Λ_i·h_{i-1} + B_i·u_i
y_i = C_i·h_i + D_i·u_i
```

| Model | Λ_i (State Transition) | B_i (Input) | Signature |
|-------|----------------------|-------------|-----------|
| Causal Linear Attention | η(q_{i-1},k_{i-1})/η(q_i,k_i)·I | ψ(k_i)^T·v_i/η(q_i,k_i) | Scalar eigenvalue, normalization-coupled |
| S6 (Mamba-1) | exp(-Δ_i·A) | Δ_i·W_B·u_i | Diagonal, input-gated discretization |
| SSD (Mamba-2) | a_i·I | b_i·x_i | Scalar-diagonal, multi-head |

The key structural finding: **both linear attention and Mamba couple the state transition and input projection through a single scalar parameter** — attention uses the kernel normalization factor, Mamba uses the discretization step Δ.

#### 5.3 Expressiveness Hierarchy

Formal results place these architectures in distinct complexity classes:

| Architecture | Complexity Class | Key Limitation |
|---|---|---|
| Transformers (standard) | Beyond TC⁰ (empirically) | Can learn induction heads, state tracking |
| RWKV-7 (latest) | Beyond TC⁰ | Meta-in-context learning, Delta Rule updates |
| Mamba (linear SSM) | TC⁰ | State tracking requires linear state growth |
| S4/H3/Hyena (linear SSM) | TC⁰ | Associative recall requires d_model ∝ seq_len |
| Classic RNN/LSTM | TC⁰ | Vanishing gradients limit long-range |

The recently published result that RWKV-7 can perform state tracking **beyond the TC⁰ bound** via its generalized Delta Rule is significant — it suggests that architectural innovations can push recurrent models past theoretical limits once thought inherent to the class.

#### 5.4 Computational Tradeoffs

The SSD framework reveals three ways to compute the same operation:

```
Layer Type     | Computation          | Parallelism       | Hardware
---------------+----------------------+-------------------+----------
Attention form | y = (L∘QK^T)·V       | Full parallel     | Tensor Cores (matmul)
SSM scan form  | Recursive state scan  | Associative scan  | Custom kernels
SSD blocks     | Block matrix decomp   | Hybrid            | Mix of matmul + scan
```

Optimal hardware mapping depends on sequence length: for short sequences, the attention form wins (Tensor Core utilization); for long sequences, the scan form wins (linear complexity); the SSD block decomposition provides a tunable tradeoff.

---

## Important Papers & References

### Foundational SSM Papers

1. **Gu, Dao et al. (2022)** — "Efficiently Modeling Long Sequences with Structured State Spaces" (S4), ICLR 2022. *Introduced the HiPPO-based structured state space for deep learning; achieved SOTA on Long Range Arena with 60× faster inference than Transformers.*

2. **Gu, Goel, Ré et al. (2022)** — "On the Parameterization and Initialization of Diagonal State Space Models" (S4D), NeurIPS 2022. *Simplified S4 to purely diagonal form; proved that HiPPO-initialized diagonal SSMs alone suffice for strong sequence modeling.*

3. **Fu, Dao, Saab, Thomas, Rudra, Ré (2023)** — "Hungry Hungry Hippos: Towards Language Modeling with State Space Models" (H3), ICLR 2023 Spotlight. *First SSM architecture explicitly designed for language modeling via dual-SSM gating for memory and comparison; scaled to 2.7B parameters.*

4. **Poli, Massaroli, et al. (2023)** — "Hyena Hierarchy: Towards Larger Convolutional Language Models." *Implicit long convolution with gating; achieved sub-quadratic complexity but later surpassed by Mamba on associative recall tasks.*

### Mamba & State Space Duality

5. **Gu & Dao (2023)** — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *Introduced input-dependent selectivity and hardware-aware parallel scan; the first SSM competitive with Transformers at 2.8B scale.*

6. **Dao & Gu (2024)** — "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Mamba-2), ICML 2024. **The theoretical breakthrough paper.** *Proved SSMs and linear attention are mathematically dual via semiseparable matrices; introduced SSD layer enabling 2–8× faster training than Mamba-1 and state dimension expansion to 256.*

### Linear Recurrent Architectures

7. **Peng, Alcaide, Anthony, et al. (2023–2025)** — "RWKV: Reinventing RNNs for the Transformer Era" (RWKV-4 through RWKV-7), Linux Foundation AI. *Purely attention-free architecture using channel-wise time-decay; RWKV-7 "Goose" uses generalized Delta Rule for state updates, achieving SoTA among 3B open models; deployed in Microsoft Office.*

8. **Beck, Pöppel, Spanring, et al. (2024)** — "xLSTM: Extended Long Short-Term Memory," NeurIPS 2024. *Modernized LSTMs with exponential gating and matrix memory; beats Llama, Mamba, and RWKV-4 on 568/571 PALOMA domains; extrapolates to 8× training context.*

9. **De, Smith, Scott, et al. (2024)** — "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models," Google DeepMind. *Introduced RG-LRU; Hawk (pure recurrence) and Griffin (recurrence + local attention); matches Llama-2 with 6–7× fewer tokens; RecurrentGemma open release.*

### Hybrid Architectures

10. **Lieber, Lenz, et al. (AI21 Labs, 2024)** — "Jamba: A Hybrid Transformer-Mamba Language Model." *Triple hybrid: Transformer + Mamba + MoE; 52B total, 12B active; 256K context with 4GB KV cache; 3× throughput improvement.*

11. **Zyphra (2024–2025)** — "Zamba: A Hybrid Mamba-Transformer Language Model." *Mamba backbone + shared attention blocks; most sample-efficient hybrid; Zamba2 upgrades to Mamba-2 with two shared attention blocks.*

12. **NVIDIA (Waleffe, Byeon, et al., 2024)** — "An Empirical Study of Mamba-based Language Models." *Definitive controlled comparison at 8B/3.5T tokens; Mamba-2-Hybrid with 7% attention outperforms pure Transformers on all 12 standard tasks; up to 8× faster inference.*

### Theoretical & Mechanistic Understanding

13. **Sieber, Lanza, et al. (2024)** — "Understanding the Differences in Foundation Models: Attention, State Space Models, and Recurrent Neural Networks," NeurIPS 2024. *Unified Dynamical Systems Framework (DSF); characterizes the structural relationship between all three model classes.*

14. **Anonymous (2025)** — "Functional Segregation in Hybrid Architectures: Some Attention is All You Need for Retrieval." *Landmark mechanistic analysis showing complete functional segregation between SSM and attention layers in hybrids; 15% of attention heads sufficient for retrieval.*

15. **Tsinghua Team (2024)** — "Stuffed Mamba: Oversized States Lead to the Inability to Forget." *Identified "state collapse" as the mechanism underlying Mamba-2's length generalization failure; proposed state normalization as mitigation.*

16. **Merrill, Sabharwal, et al. (2024)** — "The Illusion of State in State-Space Models." *Formal proof that linear SSMs are bounded by TC⁰ complexity class; explains fundamental limitations on state tracking.*

### Surveys

17. **Somvanshi et al. (2025)** — "From S4 to Mamba: A Comprehensive Survey on Structured State Space Models," arXiv:2503.18970. *30-page survey covering the full evolution S4→Mamba→S5→Jamba across NLP, speech, vision, and time-series.*

18. **Lv et al. (2025)** — "Technologies on Effectiveness and Efficiency: A Survey of State Spaces Models," Tsinghua University, arXiv:2503.11224. *Three-era taxonomy: Original SSM → Structured SSM (S4) → Selective SSM (Mamba), with mathematical foundations.*

---

## Open Questions & Future Directions

### 1. The Hybrid Sweet Spot

The strongest empirical signal from 2024–2025 research is that pure SSMs lag on tasks requiring precise retrieval, while even 7% attention largely closes the gap. Key open questions:
- What is the **minimum necessary attention** for given tasks? The 15%-of-heads result suggests there is substantial redundancy.
- Can **dynamic routing** route tokens to attention vs. SSM layers based on content? Current hybrids use static striped patterns.
- Is there an architecture where attention **emerges naturally** from SSM parameters when needed, rather than being hard-coded as separate layers?

### 2. Scaling Laws for SSMs

While Transformers have well-characterized scaling laws (Chinchilla, Kaplan), SSM and hybrid scaling is poorly understood:
- How do SSM training dynamics change as state dimension N grows? Mamba-2's jump from N=16 to N=256 was empirical.
- Is there a **critical state size** that depends on sequence length and task complexity?
- Do hybrid architectures follow the same power laws as Transformers, or do they have different optimal compute-to-data ratios?

### 3. State Collapse and Length Generalization

The "State Collapse" phenomenon (Tsinghua, 2024) reveals that SSM length generalization is not simply a capacity issue but a training pathology:
- Can **curriculum learning** (gradually increasing sequence length) prevent state collapse?
- Do **normalization schemes** (state normalization, channel-wise variance constraints) fully solve this?
- Understanding the theoretical relationship between state size N, training length K, and extrapolation ability remains incomplete.

### 4. Hardware-Aware Architecture Design

Current GPUs were designed for matrix multiplication — ideal for Transformers. SSMs require different primitives:
- Can **hardware-software co-design** produce SSM-optimized accelerators?
- The SSD block decomposition is one approach, but what other computational patterns become optimal at different scales?
- How do we reduce the **ecosystem maturity gap** between FlashAttention (years of optimization) and SSM scan kernels?

### 5. Mechanistic Interpretability of SSMs

- Transformers have well-understood circuits (induction heads, copy heads), but SSM mechanistic understanding is nascent
- How do Mamba's selective parameters implement specific computations?
- The finding that Mamba implements induction **not via the SSM state but via short convolutions** suggests unexpected computational strategies

### 6. In-Context Learning Beyond Attention

- RWKV-7's meta-in-context learning via Delta Rule updates represents a fundamentally different ICL mechanism
- Can recurrent models develop ICL capabilities that match attention without approximating attention?
- Is there a fundamental relationship between ICL ability and computational complexity class?

### 7. The Endgame Question

**Research Question: Are hybrid SSM-attention architectures the future of language model design, or will pure SSMs eventually match or surpass attention-based transformers across all tasks?**

The current evidence strongly favors hybrids for the near-to-medium term. The functional segregation finding — that SSM layers cannot perform retrieval regardless of scale — suggests this limitation may be architectural rather than merely a matter of scale. If so, hybridization is not a transitional phase but a **permanent architectural principle**, analogous to how biological brains use separate mechanisms for pattern recognition (cortical columns) and precise memory retrieval (hippocampus).

However, the rapid pace of innovation leaves room for disruption. RWKV-7's ability to perform state tracking beyond TC⁰ suggests that recurrent models can transcend previously assumed theoretical limits. A potential path to pure SSM dominance would require:
- A mechanism for exact, content-based retrieval that operates in linear time
- Resolution of the state collapse problem at scale
- Demonstration that SSMs can match Transformer ICL without architectural hybridization

---

## Relevance to Main Topic

This survey of attention alternatives directly addresses a core tension in modern language model design: **the trade-off between modeling power and computational efficiency**. The Transformer's self-attention provides unparalleled modeling flexibility but imposes a quadratic complexity burden that becomes prohibitive for long sequences, large-scale deployment, and resource-constrained settings.

The emergence of SSMs and linear recurrent architectures represents one of the most significant architectural innovations since the Transformer itself. Key connections to broader research include:

1. **Efficiency as a first-class design constraint**: As models scale to trillions of parameters and million-token contexts, sub-quadratic architectures transition from academic curiosity to practical necessity. The 8× inference speedup demonstrated by Mamba-2-Hybrid directly translates to cost and latency improvements in production.

2. **Theoretical unification**: Dao & Gu's duality framework reveals that the attention-vs-SSM debate is, in a precise mathematical sense, a false dichotomy. Both are instances of structured matrix multiplication, differing only in computational form. This suggests that future architectures should be designed in this unified space, selecting the optimal computation pattern per layer.

3. **Complementary strengths**: The functional segregation finding — SSMs for efficient sequence processing, attention for precise retrieval — mirrors the broader principle that heterogeneous architectures outperform homogeneous ones. This has implications beyond NLP, including multimodal models, retrieval-augmented generation, and agent architectures.

4. **Edge deployment**: RWKV's deployment inside Microsoft Office demonstrates that linear-time architectures enable capabilities (local, private, large-context LLMs) that are simply infeasible with pure Transformers. This expands the application space of language models significantly.

5. **The next architectural paradigm**: Just as Transformers displaced LSTMs in 2017, the hybrid and linear architectures of 2023–2025 may represent the early stages of the next dominant paradigm. The key question is whether the current hybridization trend stabilizes into a permanent architectural pattern or whether it is merely a bridge to pure SSMs that can match all of attention's capabilities through further innovation.
