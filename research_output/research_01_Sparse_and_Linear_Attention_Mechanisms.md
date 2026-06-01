# Research: Sparse and Linear Attention Mechanisms

## Overview

Self-attention is the core computational primitive of Transformer models, enabling each token to attend to every other token in a sequence. However, its O(n²) time and memory complexity with respect to sequence length n has long been recognized as a fundamental bottleneck for long-context modeling. As language models scale to process hundreds of thousands to millions of tokens, the quadratic cost of dense attention becomes prohibitive — a 1M-token sequence would require ~4 TB of memory for the attention matrix alone (fp16). This has motivated an intense research effort spanning 2020–2026 to develop efficient attention mechanisms that reduce this complexity while preserving modeling quality.

The landscape of efficient attention has evolved through three broad paradigms: (1) **sparse attention**, which selectively computes only a subset of token interactions based on predefined or learned patterns, reducing the number of pairwise computations while otherwise preserving the standard softmax attention formulation; (2) **linear (kernel-based) attention**, which reformulates the attention computation through the lens of kernel methods, using random feature maps to approximate the softmax kernel and rearranging matrix multiplications to achieve O(n) complexity without ever materializing the n×n attention matrix; and (3) **low-rank approximation**, which projects keys and values to a fixed, lower-dimensional space, exploiting the empirical observation that the attention matrix is inherently low-rank.

By 2024–2026, the field has converged on a powerful insight: **neither pure sparse attention nor pure linear attention alone is optimal**. The winning approach is **hybrid architectures** that combine both mechanisms — using sparse attention for high-fidelity token-level retrieval and linear attention for efficient global context compression. This report surveys the key methods in each paradigm, synthesizes empirical comparisons, and addresses the core research question: which paradigm offers the best accuracy-efficiency trade-off for long-context language modeling.

---

## Key Methods & Approaches

### 1. Sparse Attention Patterns

Sparse attention reduces the quadratic cost of self-attention by computing only a subset of the n×n attention matrix. The key challenge is designing sparsity patterns that maintain model quality while achieving meaningful computational savings.

#### 1.1 Fixed-Pattern Sparse Attention

**Longformer (Beltagy et al., 2020)** introduced three complementary sparse attention patterns that achieve O(n) complexity:

- **Sliding Window Attention**: Each token attends only to w tokens on either side (w=256). Complexity: O(n·w). Multiple layers of windowed attention build a large receptive field through information propagation across layers, analogous to CNNs.
- **Dilated Sliding Window**: Inspired by dilated CNNs, gaps of size d are introduced between attended tokens. Different dilation rates are used across attention heads, with dilation applied primarily in upper layers to expand the receptive field without additional computation.
- **Global Attention**: A small, fixed set of task-specific tokens (e.g., [CLS], question tokens) attend to all positions and are attended to by all positions. Uses separate linear projections for global vs. local attention.

Longformer achieved SOTA on character-level language modeling (text8, enwik8) and long-document tasks (WikiHop, TriviaQA), processing sequences up to 4,096 tokens — 8× BERT's limit. The Longformer-Encoder-Decoder (LED) variant extended this to seq2seq tasks.

**BigBird (Zaheer et al., NeurIPS 2020)** combined three attention patterns to form a block-sparse attention matrix:

- **Window (Local) Attention**: w neighboring tokens per position.
- **Global Attention**: g tokens (e.g., CLS) attend to all positions.
- **Random Attention**: r randomly selected tokens per position.

The key theoretical contribution was proving that this combination satisfies two critical properties: (1) **Universal approximation** — BigBird can approximate any continuous sequence-to-sequence function; (2) **Turing completeness** — BigBird is provably Turing-complete under standard precision assumptions, meaning no computational task solvable by full attention is theoretically impossible for BigBird. The O(1) global tokens play a crucial role in maintaining these theoretical guarantees while the random connections create a small-world graph with short path lengths between any two nodes. BigBird achieved SOTA on long-context NLP tasks and was extended to genomics (DNA sequence representation).

**Reformer (Kitaev et al., ICLR 2020)** took a fundamentally different approach using **Locality-Sensitive Hashing (LSH)**:

- Q and K are tied/shared, reducing the attention problem to finding nearest neighbors among queries.
- Angular LSH projects vectors onto a unit sphere using random projections, then partitions them into buckets.
- Attention is computed only within buckets, reducing complexity from O(L²) to O(L log L).
- Multi-round LSH provides more stable nearest-neighbor identification.
- Combined with **reversible residual layers** (RevNet) that reconstruct activations during backpropagation rather than storing them.

Reformer demonstrated processing up to 1 million words on a single 16GB GPU. Unlike Longformer and BigBird's fixed patterns, LSH attention is data-dependent — bucketing dynamically adapts to the input, making it more flexible.

#### 1.2 Dynamic and Learned Sparse Attention (2024–2025)

By 2024, the field shifted decisively from fixed sparsity patterns to dynamic, query-aware sparsity:

**Quest (Tang et al., 2024, MIT HAN Lab)** introduced query-aware page-level sparsity. Instead of static token selection, Quest scores KV-cache pages against the current query using min/max key statistics, retrieving only the most relevant subset. This achieves O(L) time complexity (where L ≪ N) while preserving the full KV cache in HBM — a compute reduction without memory reduction.

**SeerAttention (Gao et al., NeurIPS 2025)** learns block-level attention sparsity via a **learnable gating module** inspired by Mixture-of-Experts (MoE). The gate pools Q and K tensors along the sequence dimension, processes them through learnable linear layers, and produces gating scores predicting block-level sparsity. Combined with a custom block-sparse FlashAttention kernel, SeerAttention achieves 90% sparsity at 32K context with minimal perplexity loss and 5.67× speedup over FlashAttention-2. Trained via lightweight self-distillation on pre-trained LLMs.

**Native Sparse Attention (NSA)** from DeepSeek introduced a hardware-aligned sparse attention design with three branches: compressive attention (for coarse-grained global patterns), sliding window attention (for local context), and token selection based on compressed key representations. A learned gating mechanism dynamically balances these branches.

**DELTA (2025)** exploits cross-layer attention correlation: attention patterns evolve gradually across layers, so DELTA computes full attention only in a few key "Δ-layers" and reuses selected high-recall token subsets in intermediate sparse layers — training-free.

### 2. Kernel-Based Linear Attention

Linear attention reformulates the attention computation through the lens of kernel methods. The key insight is that softmax attention can be expressed as:

$$\text{Att}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

By viewing the softmax as a kernel and approximating it via random feature maps φ: ℝᵈ → ℝᵐ (where m ≪ n), the computation can be rearranged:

$$\text{Att}(Q, K, V) ≈ \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}$$

Computing φ(K)ᵀV first (an m×d matrix) reduces complexity to O(n·m·d) — linear in sequence length.

#### 2.1 Performer / FAVOR+ (Choromanski et al., ICLR 2021)

The **Performer** introduced **FAVOR+** (Fast Attention Via positive Orthogonal Random features), which remains a foundational baseline in 2024–2025 research:

- **Random Features**: Draws m random vectors ω ~ N(0, I_d) and constructs feature maps φ(x) using exponentiated dot products: φ(x) = h(x)/√m · [exp(ω₁ᵀx), ..., exp(ωₘᵀx)]
- **Positive Features**: Unlike earlier sin/cos-based random features that could produce negative values (causing instability), FAVOR+ uses all-positive feature maps. Two variants exist: SM⁺ (using exp) and SMʰʸᵖ⁺ (using both exp(u) and exp(-u) for further variance reduction). The variance of these estimators tends to 0 as the approximated kernel values approach 0.
- **Orthogonal Features**: Random vectors are orthogonalized (e.g., via Gram-Schmidt) to decorrelate Monte Carlo samples, providing lower estimation variance and tighter uniform convergence bounds for the same number of features.
- **Causal Attention**: Prefix-sum computations maintain the lower-triangular constraint for autoregressive models without materializing the attention matrix.

Performer provides provably unbiased estimates with strong theoretical guarantees and achieves true O(n) time and memory. It generalizes to any kernelizable attention (ReLU-based, regularized softmax, etc.).

#### 2.2 Key Advances in Linear Attention (2024–2025)

**RALA: Rank-Augmented Linear Attention (Fan et al., CVPR 2025)** addressed the fundamental low-rank limitation of linear attention. The authors conducted rank analysis from two perspectives — the KV buffer and the output features — demonstrating that the performance gap between linear and softmax attention is driven by the low-rank nature of linear attention's feature map. RALA rivals softmax attention performance while maintaining linear complexity, achieving 84.4% Top-1 accuracy on ImageNet-1k with only 26M parameters and 4.6G FLOPs in the RAVLT vision architecture.

**Bridging Softmax and Linear Attention (Han et al., NeurIPS 2024)** identified two critical weaknesses preventing naive linear attention from matching softmax: (1) lack of injective mapping — different queries can produce identical attention weights in linear attention; (2) poor locality modeling — unlike softmax's natural focus on nearby tokens due to exponential scaling. By designing unique query embeddings and adding locality mechanisms, they demonstrated linear attention can outperform softmax on vision tasks while staying linear.

**Lightning Attention** has become the dominant linear attention variant for language modeling, featuring input-dependent gating for selective memory retention and improved training stability. It serves as the linear attention backbone in MiniCPM-SALA.

### 3. Low-Rank Approximations

Low-rank methods exploit the empirical observation that the n×n attention matrix is inherently low-rank, projecting keys and values to a fixed lower dimension.

#### 3.1 Linformer (Wang et al., 2020)

The **Linformer** proved theoretically that self-attention is low-rank and can be approximated by projecting the n×d key and value matrices to k×d matrices (where k ≪ n) using learned projection matrices E, F ∈ ℝᵏˣⁿ:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q(EK)^T}{\sqrt{d}}\right) \cdot FV$$

This reduces complexity from O(n²) to O(nk), where k is a fixed constant independent of sequence length. The key theoretical result shows that the attention matrix can be well-approximated by a low-rank matrix due to the spectral properties of the softmax kernel. In practice, Linformer achieves competitive performance with standard Transformers while being significantly faster for long sequences.

#### 3.2 Nyströmformer (Xiong et al., AAAI 2021)

**Nyströmformer** approximates the full n×n softmax attention matrix using the Nyström method for low-rank matrix approximation:

1. Select m "landmark" points from the input sequence (m ≪ n).
2. Compute the m×n kernel matrix between landmarks and all tokens.
3. Compute the m×m kernel matrix among landmarks and its pseudoinverse.
4. Reconstruct the full n×n attention matrix as the product of these smaller matrices using the Nyström approximation.

This achieves O(n) complexity without learnable projection parameters, making it simpler to deploy than Linformer. The approximation quality depends on the choice of landmarks, which are typically selected via segmental max-pooling along the sequence.

#### 3.3 LoLCATs: Linearizing Pre-trained Transformers (Zhang et al., ICLR 2025)

**LoLCATs** (Low-rank Linear Conversion via Attention Transfer) represents a practical breakthrough for large-scale deployment. Instead of training linear-attention models from scratch, LoLCATs converts pre-trained Transformers into subquadratic models via a two-step process:

1. **Attention Transfer**: Replace softmax attention with learnable linear attention, training only ~0.2% of parameters to mimic softmax outputs via MSE loss.
2. **Low-Rank Adaptation (LoRA)**: Fine-tune QKVO projection weights on ~40M tokens to recover remaining quality.

Key results include:
- First linearized 70B and 405B parameter LLMs (~50× larger than prior linear models)
- 77.8% of original quality retained on 5-shot MMLU for Llama 3.1 70B
- 20+ point improvement on 5-shot MMLU over prior linearizing methods
- Training budget: ~5 hours on a single A100 (7B/8B), 18 hours on 8×H100 (70B)

### 4. Hybrid Architectures (2025–2026): The Best of Both Worlds

The 2025–2026 research landscape has converged on **hybrid architectures** that combine sparse and linear attention, achieving the best accuracy-efficiency trade-off.

#### 4.1 MiniCPM-SALA (Feb 2026)

**MiniCPM-SALA** (Sparse Attention and Linear Attention) from OpenBMB is the first large-scale trained sparse-linear hybrid attention model. Key features:

- **Architecture**: 25% sparse layers (InfLLM-V2, for high-fidelity retrieval) + 75% linear layers (Lightning Attention, for global efficiency) in a 1:3 ratio, with optimal layer placement determined algorithmically.
- **HyPE (Hybrid Positional Encoding)**: RoPE applied to linear layers for position-sensitive memory; RoPE removed from sparse layers to prevent long-distance information decay.
- **QK-Normalization** and **Output Gates** for training stability in hybrid architectures.
- **HALO (Hybrid Attention via Layer Optimization)**: Converts pre-trained Transformers via continual training (~2T tokens), reducing training costs by ~75% vs. training from scratch (~8T tokens).

**Results**: 3.5× inference speedup over Qwen3-8B at 256K tokens on a single A6000D GPU; supports 1M-token context where full-attention models go OOM. Maintains general capabilities (knowledge, math, coding) comparable to full-attention models while showing substantial advantages on long-context benchmarks.

#### 4.2 SPLA: Block Sparse + Linear Attention (Jan 2026)

**SPLA** combines block-sparse attention with **Residual Linear Attention (RLA)**. Rather than discarding unselected tokens (as in pure sparse approaches), SPLA compresses them via linear attention — using sparse for "peaks" and linear for the "tail." Block selection uses a second-order Taylor expansion for principled ranking. SPLA closes the performance gap with dense attention on 256K context (RULER benchmark) while maintaining sparse decoding efficiency.

#### 4.3 Sparse Mamba / SSM + Sparse Attention Hybrids

**Samba (Ren et al., 2024, Microsoft)** interleaves Mamba SSM layers with Sliding Window Attention (SWA), where SSM compresses sequences into recurrent hidden states and attention layers maintain precise recall of recent memories. Demonstrated efficient unlimited-context language modeling.

**Taipan (Nguyen et al., 2024)** introduces Selective Attention Layers (SALs) with Mamba-2, identifying tokens that need long-range interactions and pruning unimportant features. Extends accurate prediction to 1M tokens.

**Hydra (Chaudhary et al., 2025, UC Berkeley)** integrates Mamba SSM, Sparsified Global Attention (SGA with intermittent application and budgeted global tokens), Mixture-of-Experts, and dual memory systems — a blueprint for comprehensive modular long-context architectures.

### 5. FlashAttention: Hardware-Efficient Exact Attention

While not strictly a sparsity or linearization method, **FlashAttention** (Dao et al., 2022–2024) deserves mention as it has become the de facto standard for efficient attention computation. FlashAttention achieves 2–4× speedups over standard attention with zero accuracy loss through IO-aware tiling and recomputation:

- **FlashAttention (2022)**: Fuses attention operations into a single CUDA kernel, using tiling to minimize HBM reads/writes and recomputing the softmax in backward pass rather than storing the attention matrix.
- **FlashAttention-2 (2023)**: Improved parallelism over sequence length dimension, better work partitioning between warps, reducing non-matmul FLOPs.
- **FlashAttention-3 (2024)**: Leverages Hopper GPU architecture features (WGMMA instructions, TMA for asynchronous data movement, producer-consumer asynchrony) for further speedups.

FlashAttention kernels now serve as the backbone for block-sparse implementations like SeerAttention and FSA (Flash Sparse Attention), enabling hardware-efficient sparse attention.

### 6. KV Cache Compression (Complementary Approach)

A closely related research direction is KV cache compression, which reduces memory (rather than compute) for long-context inference:

- **Eigen Attention (2024)**: SVD-based projection of K, V into low-rank space; up to 40% KV reduction, 60% latency reduction, post-training deployable.
- **SALS (NeurIPS 2025)**: Combines low-rank projection with sparse token selection in latent space; 6.4× KV compression, 5.7× attention speedup. Handles RoPE's rank-increasing effects by compressing pre-RoPE.
- **Multi-Head Latent Attention (MLA)**: DeepSeek's approach of joint low-rank KV compression into shared latent representations, enabling extreme compression.

---

## Important Papers & References

### Foundational Methods (2020–2021)

1. **Beltagy, I., Peters, M.E., & Cohan, A. (2020).** *Longformer: The Long-Document Transformer.* arXiv:2004.05150. — Introduced sliding window + dilated + global attention, achieving O(n) complexity and SOTA on long-document NLP tasks.

2. **Zaheer, M., Guruganesh, G., et al. (2020).** *Big Bird: Transformers for Longer Sequences.* NeurIPS 2020. arXiv:2007.14062. — Combined random + window + global attention with theoretical proofs of universal approximation and Turing completeness.

3. **Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020).** *Reformer: The Efficient Transformer.* ICLR 2020 (Oral). arXiv:2001.04451. — LSH-based dynamic sparse attention + reversible residual layers; processes 1M tokens on single GPU.

4. **Choromanski, K., Likhosherstov, V., et al. (2021).** *Rethinking Attention with Performers.* ICLR 2021. arXiv:2009.14794. — FAVOR+ mechanism: positive orthogonal random features for provably unbiased O(n) attention approximation. Foundation for kernel-based linear attention.

5. **Wang, S., Li, B.Z., et al. (2020).** *Linformer: Self-Attention with Linear Complexity.* arXiv:2006.04768. — Proved self-attention is low-rank; projected K,V to fixed dimension k ≪ n, achieving O(nk) complexity.

6. **Xiong, Y., Zeng, Z., et al. (2021).** *Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention.* AAAI 2021. arXiv:2102.03902. — Nyström method for low-rank attention approximation using landmark points; O(n) complexity without learnable projections.

7. **Katharopoulos, A., Vyas, A., et al. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020. — First to show that linearized attention with causal masking can be viewed as an RNN, enabling O(n) training and O(1) inference.

### Dynamic Sparse Attention (2024–2025)

8. **Tang, J., Zhao, Y., et al. (2024).** *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* arXiv:2406.10774. — Query-aware page-level dynamic sparsity using min/max key statistics; O(L) time with full KV cache preserved.

9. **Gao, Y., Zeng, Z., et al. (2025).** *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs.* NeurIPS 2025. arXiv:2410.13276. — Learned block-level sparsity via MoE-inspired gating; 90% sparsity at 32K context, 5.67× speedup.

10. **Yuan, Z., et al. (2025).** *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* DeepSeek. — Three-branch architecture (compressive + sliding window + token selection) with learned gating; hardware-aligned implementation.

### Linear Attention Advances (2024–2025)

11. **Fan, Q., Huang, Z., & He, K. (2025).** *Breaking the Low-Rank Dilemma of Linear Attention.* CVPR 2025. — Rank-Augmented Linear Attention (RALA) rivals softmax while maintaining O(n); 84.4% ImageNet Top-1 with 26M params.

12. **Han, D., et al. (2024).** *Bridging the Gap Between Softmax and Linear Attention.* NeurIPS 2024. — Addressed injectivity and locality limitations; linear attention can outperform softmax on vision tasks.

13. **Zhang, M., Arora, S., et al. (2024).** *LoLCATs: On Low-Rank Linearizing of Large Language Models.* ICLR 2025. arXiv:2410.10254. — Two-step conversion of pre-trained Transformers to linear attention; first linearized 70B and 405B LLMs; 77.8% MMLU retention on Llama 3.1 70B.

### Hybrid Architectures (2024–2026)

14. **MiniCPM Team (2026).** *MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling.* arXiv:2602.11761. — First large-scale sparse-linear hybrid (9B params); 3.5× speedup, 1M-token support on consumer GPUs, 75% training cost reduction via HALO.

15. **Ren, L., Liu, Y., et al. (2024).** *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* arXiv:2406.07522. — Mamba SSM + Sliding Window Attention interleaving for unlimited context.

16. **Nguyen, E., et al. (2024).** *Taipan: Efficient and Expressive State Space Language Models with Selective Attention.* arXiv:2410.18572. — Selective Attention Layers with Mamba-2; accurate prediction to 1M tokens.

17. **Anonymous (2026).** *SPLA: Block Sparse Plus Linear Attention for Long Context Modeling.* arXiv:2601.22379. — Residual linear attention for unselected tokens; closes gap with dense attention on 256K context.

### Hardware-Efficient Implementations

18. **Dao, T., Fu, D., et al. (2022).** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. — IO-aware tiling; 2–4× speedup with zero accuracy loss.

19. **Dao, T. (2023).** *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* — Improved parallelism over sequence dimension.

20. **Shah, J., Bikshandi, G., et al. (2024).** *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision.* — Leverages Hopper GPU features for further speedups.

### Surveys and Empirical Comparisons

21. **HKUST-GZ (2025).** *Sparse Attention in Large Language Models: A Survey.* — Structured taxonomy of static vs. dynamic sparse attention methods; KV cache management focus.

22. **Cheng, Y., et al. (2025).** *Long-Context Efficient Transformers: A Comprehensive Survey of Techniques, Applications, and Future Directions.* TechRxiv. — Broad survey covering sparse, kernel-based, and memory-augmented approaches.

---

## Open Questions & Future Directions

### Current Limitations

1. **Linear Attention's Associative Recall Gap**: Despite significant advances (LoLCATs, RALA), linear attention still struggles with precise token-level associative recall compared to softmax attention. The fundamental tension between compressing information into a fixed-size recurrent state and maintaining perfect recall of arbitrary past tokens remains unresolved. Hybrid approaches (SPLA, MiniCPM-SALA) mitigate this by using sparse attention for retrieval, but do not eliminate it.

2. **Training Stability of Hybrid Models**: Multi-component systems (SSM + attention + MoE) require careful training curricula. The interaction between different attention mechanisms within a single model can create optimization challenges — QK-Normalization and output gates in MiniCPM-SALA are ad-hoc fixes rather than principled solutions.

3. **Selection Fidelity in Dynamic Sparse Attention**: Block-sparse methods that dynamically select which KV blocks to attend to face a fundamental challenge when attention distributions shift rapidly (e.g., during reasoning chains). The optimal selection granularity (block size, number of blocks) is task-dependent and currently determined heuristically.

4. **KV Cache Storage Bottleneck**: While sparse attention reduces compute, it still requires storing the full KV cache. True end-to-end linear scaling requires both compute sparsity AND memory compression — a challenge that methods like SALS, Eigen Attention, and MLA are beginning to address.

5. **Scaling Laws for Efficient Attention**: The scaling behavior of models with different attention mechanisms is not well-characterized. Do sparse and linear models follow the same scaling laws as dense Transformers? Early evidence from LoLCATs and MiniCPM-SALA suggests they can match at current scales (≤405B), but the trend at larger scales is unknown.

6. **Pre-training from Scratch vs. Conversion**: LoLCATs and MiniCPM-SALA demonstrate that converting pre-trained Transformers is significantly cheaper than training efficient architectures from scratch. However, this ties the architecture to the pre-trained model's design choices and may limit ultimate efficiency.

### Active Research Frontiers

1. **Extreme Sparsity (50–250×)**: Recent work (2026) shows that modern LLMs maintain quality at 50× sparsity during inference, and models exhibit greater sparsity robustness with increased scale. The theoretical finding that dense attention is mathematically impossible at long context (embedding bottleneck theorem when d < N−1) suggests sparsity is not an approximation but a necessity for million-token contexts. Training-time sparsity induction (rather than relying on emergent sparsity) is the next frontier.

2. **State Space Models as Linear Attention**: The theoretical connection between SSMs and linear attention, formalized through the structured state space duality (SSD) framework in Mamba-2, has opened new avenues. SSMs can be viewed as linear attention with a specific parameterization, suggesting that advances in one domain transfer to the other.

3. **Hardware-Aligned Design**: NSA and FlashAttention-3 demonstrate that co-designing attention mechanisms with hardware characteristics (GPU memory hierarchy, tensor core utilization) yields larger practical speedups than algorithmic complexity improvements alone. This "hardware-algorithm co-design" philosophy is becoming a dominant paradigm.

4. **Training-Aware Sparse Architectures**: Models trained from scratch with sparsity-inducing objectives may develop fundamentally different attention patterns than those converted post-hoc. The optimal training recipe for sparse/linear attention is an open question.

5. **Long-Context Evaluation**: Current benchmarks (RULER, LongBench, Needle-in-a-Haystack) test limited aspects of long-context understanding. More comprehensive evaluation frameworks that test retrieval, reasoning, aggregation, and multi-hop dependencies over long ranges are needed.

6. **Multimodal Efficient Attention**: Extending efficient attention to vision-language and video models introduces additional challenges (spatial redundancy, temporal dependencies, heterogeneous modality importance) that are not addressed by text-only methods.

### Theoretical Open Problems

- **Expressiveness characterization**: Under what conditions can linear attention provably match softmax attention for all downstream tasks? The injectivity result (Han et al., 2024) and low-rank dilemma analysis (Fan et al., 2025) are steps toward this, but a complete characterization remains elusive.
- **Optimal hybrid ratios**: Is there a theoretically optimal ratio of sparse to linear attention layers for a given compute budget, or is this purely empirical?
- **Attention entropy dynamics**: How does attention entropy evolve across layers in sparse and linear models? Understanding this could inform more principled sparsity pattern design.

---

## Relevance to Main Topic

Sparse and linear attention mechanisms are central to the broader research landscape of efficient long-context language modeling. This sub-topic directly connects to several other research areas:

### Connection to State Space Models and Alternative Architectures

Linear attention's relationship with State Space Models (Mamba, Mamba-2) has been formalized through the structured state space duality (SSD) framework. This reveals that SSMs are essentially linear attention with a specific semi-separable matrix parameterization. The convergence of these two lines of research suggests that future architectures will likely blend Transformer-style attention, SSM-style recurrence, and sparse selection mechanisms — a trend already visible in Samba, Taipan, MiniCPM-SALA, and Hydra.

### Connection to Mixture-of-Experts (MoE)

Dynamic sparse attention — particularly learned gating mechanisms like those in SeerAttention and NSA — shares deep structural similarities with MoE routing. Both use learned gates to selectively activate subsets of a larger model, trading increased capacity for minimal additional compute. The integration of MoE with efficient attention (as in Routing Mamba, Hydra) represents a natural synthesis of these ideas.

### Connection to Positional Encoding

The choice of positional encoding interacts critically with attention sparsity patterns. Rotary Position Embeddings (RoPE) increase key variance and effective rank, making compression harder — this observation has led to RoPE-aware compression methods (SALS, EliteKV) and hybrid encoding schemes (HyPE in MiniCPM-SALA). Understanding this interaction is essential for designing effective sparse/linear attention systems.

### Connection to Inference Optimization

Efficient attention mechanisms are a critical enabler for deploying LLMs with long context windows on resource-constrained hardware. The ability to process 1M-token contexts on consumer GPUs (MiniCPM-SALA, SSA) opens new applications in document analysis, codebase understanding, and multi-turn conversational agents. This connects to broader infrastructure concerns about serving costs, KV cache management, and batch processing efficiency.

### Answer to the Core Research Question

**Which sparse/linear attention paradigm offers the best accuracy-efficiency trade-off for long-context language modeling in 2024–2026?**

The evidence from 2024–2026 research points to a clear answer: **Hybrid architectures combining sparse attention for high-fidelity retrieval with linear attention for efficient global compression offer the best accuracy-efficiency trade-off.** This conclusion is supported by:

1. **MiniCPM-SALA** demonstrates 3.5× speedup with full-attention-comparable quality at 9B scale.
2. **SPLA** closes the gap with dense attention on 256K context benchmarks using residual linear attention for the sparse "tail."
3. **Samba and Taipan** show that SSM + sparse attention interleaving enables accurate prediction to 1M+ tokens.
4. **LoLCATs** proves that converting pre-trained Transformers to linear attention is viable even at 405B scale, but the resulting models still benefit from hybrid sparse components for retrieval.

The optimal architecture appears to be a **layer-level hybrid** where ~20–30% of layers use sparse attention (sliding window + learned token selection) and ~70–80% use linear attention (or SSM-based recurrence), augmented with: (a) hybrid positional encoding (RoPE on linear layers, no RoPE on sparse), (b) output gates and QK normalization for stability, and (c) efficient KV cache compression for the sparse layers. The conversion approach (Transformer → hybrid via continual training) reduces training costs by ~75% compared to training from scratch, making this architecture practically deployable.

For the near future (2026–2027), the trend points toward **extreme sparsity (50–250×)** during inference, supported by the theoretical finding that dense attention over million-token contexts is fundamentally impossible. The frontier is shifting from "how to approximate dense attention" to "how to design models that are natively efficient" — a paradigm shift that will shape the next generation of language models.
