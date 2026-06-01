# KV-Cache Optimization and Inference Efficiency

## Overview

The key-value (KV) cache is the dominant memory bottleneck in autoregressive large language model (LLM) inference. During decoding, every generated token requires attending to all prior tokens — the KV cache stores the intermediate key and value activations to avoid recomputing them, but its size scales as **2 × batch_size × sequence_length × num_kv_heads × head_dim × num_layers × precision_bytes**. At long context lengths (128K+ tokens) and large batch sizes, the KV cache can consume tens of gigabytes of GPU memory, far exceeding the model weights themselves. This makes it the primary constraint on serving throughput: it limits batch size, caps maximum context length, and transforms the decoding step from compute-bound to memory-bandwidth-bound.

The research community has attacked this bottleneck from multiple complementary angles. **Architectural innovations** like Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce the number of KV heads, sharing them across query heads. **Memory management** approaches like PagedAttention (vLLM) apply virtual memory principles to eliminate KV cache fragmentation. **Quantization** techniques compress KV tensors to INT8 or INT4 precision, halving or quartering memory footprint. **Speculative decoding** uses draft models to generate multiple tokens per forward pass, amortizing the KV cache load cost. Most recently, **attention sink** theory revealed that keeping only a handful of initial tokens plus a sliding window enables stable streaming over millions of tokens. These techniques are orthogonal and multiplicative: their combination in production systems like vLLM with GQA-native models and INT8 KV quantization can achieve **5–10× throughput improvements** over naive serving.

The central research question — how GQA, quantization, and PagedAttention combine to maximize throughput — has a clear answer emerging from 2024–2025 literature: GQA reduces the absolute size of the KV cache by 4–8× (depending on group count), PagedAttention eliminates ~60–80% memory waste from fragmentation, and INT8 quantization provides an additional ~2× compression with negligible accuracy loss. Together they enable batch sizes 5–10× larger on the same hardware, directly translating to proportional throughput gains. The Opt-GPTQ framework (Kong et al., 2025) explicitly demonstrates this three-way combination, while production systems like vLLM v0.6.0 achieve 2.7× throughput improvements over prior versions by integrating multi-step scheduling alongside these core techniques.

---

## Key Methods & Approaches

### 1. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

**Problem:** In standard Multi-Head Attention (MHA), each of *h* query heads has its own dedicated key and value head, so the KV cache stores *h* full sets of K and V tensors. For large models like Llama 2 70B with 64 query heads and 128-dim heads across 80 layers, the FP16 KV cache for a single 4096-token sequence consumes ~2.5 GB — dwarfing compute requirements.

**MQA (Shazeer, 2019):** All query heads share a single KV head. This reduces the KV cache by a factor of *h* (e.g., 64× reduction for 64-head models). While dramatically memory-efficient, MQA can degrade attention quality since all heads are forced to use the same key-value representations, limiting the model's ability to attend to different aspects of the input.

**GQA (Ainslie et al., 2023):** A middle ground — query heads are partitioned into *g* groups, with each group sharing one KV head. When *g = 1*, GQA = MQA; when *g = h*, GQA = MHA. Typical configurations use 4–8 groups, achieving 4–8× KV cache reduction with quality nearly indistinguishable from MHA. GQA is now the practical standard: Llama 3 (all sizes), Llama 2 70B, Gemma, Mistral, and many other leading models adopt GQA with group counts varying by model size.

**Beyond GQA — Cross-Layer Attention (CLA):** Brandon et al. (NeurIPS 2024) propose extending KV sharing *across adjacent layers*, not just within a layer. CLA2 pairs consecutive layers to share one KV projection, achieving 2× further reduction on top of GQA/MQA with negligible perplexity increase at 1B and 3B scales. CLA is orthogonal to GQA and can be applied as a post-training modification.

**QCQA (Joshi et al., 2024):** Addresses a key GQA limitation — KV heads are grouped arbitrarily (mean-pooled or randomly) rather than based on which heads are most mutually compatible. QCQA uses an evolutionary algorithm to find optimal query-head groupings, achieving 20% higher accuracy than standard GQA at the same cache size, or 40% less cache for equal accuracy, tested on Llama 2 7B.

**DeepSeek MLA (Multi-Head Latent Attention, 2024):** A paradigm shift from sharing to compression. Rather than caching per-head K/V vectors, MLA stores a single low-rank latent vector per token and reconstructs K/V on-the-fly via learned projection matrices. With a latent dimension of ~1/8 the full KV dimension, MLA achieves 6–8× compression. The clever "matrix merging" trick fuses Q and K projections so explicit K reconstruction is never needed for attention computation. The MHA2MLA framework (Ji et al., ACL 2025) enables retrofitting MLA onto any pretrained transformer with only 0.3–0.6% fine-tuning data, achieving 92.19% KV cache reduction on Llama 2 7B with only 0.5–1% LongBench degradation.

### 2. KV Cache Quantization

**Motivation:** KV tensors are activation values, not model weights, so their distribution changes with every input. This makes post-training quantization more challenging than weight quantization — key channels within a single head can have dynamic ranges spanning two orders of magnitude, and outlier channels can cause catastrophic information loss under naive quantization.

**INT8 KV Quantization:** Per-channel (per-head) asymmetric quantization is the production standard. Each key/value channel gets its own scale and zero-point, computed online without calibration data. LMDeploy benchmarks show INT8 KV is essentially lossless: on LLaMA 2 7B, INT8 KV achieves 1.27× throughput over FP16 with MMLU scores of 64.00 (vs. 63.91 FP16) and GSM8K of 69.75 (vs. 70.13). This is effectively a "free lunch" for most deployments.

**INT4 KV Quantization:** Twice the compression of INT8 but with measurable quality impact. LMDeploy benchmarks show 1.39× throughput over FP16 (1.09× over INT8) but GSM8K drops from 70.13 to 66.87. Keys are more sensitive than values — per-channel INT8 for keys combined with per-group INT4 for values is a common production compromise.

**Advanced Quantization Techniques:**
- **TurboQuant (Google/ICLR 2026):** Uses Hadamard transform rotation to "smear" outlier magnitudes across dimensions, followed by 4-bit quantization for keys (3-bit Lloyd-Max + 1-bit QJL residual) and 2-bit for values. Averages ~3 bits per KV element with ~5× memory compression.
- **Runtime-Certified Quantization (arXiv:2605.20868, May 2026):** Tiered KV cache with INT8 keys + INT4 values in VRAM, FP16 backup in CPU RAM. Adaptive precision selection promotes high-attention blocks to FP16, with per-head per-step error bounds and guaranteed recovery to exact dense attention when bounds are exceeded.
- **Adaptive Tiered Caches:** Three-tier systems (FP16 → INT8 → INT4 → evicted) with per-token importance scoring based on cumulative attention, recency, cross-head variance, and semantic distinctiveness.

**Key vs. Value Asymmetry:** A critical finding across quantization literature: keys are significantly more sensitive to quantization than values. Key quantization errors distort the attention distribution multiplicatively (through softmax), while value errors propagate linearly through the weighted sum. Production systems universally apply more conservative quantization to keys.

### 3. PagedAttention and Virtual Memory-Inspired Memory Management (vLLM)

**The Fragmentation Problem:** Traditional LLM serving systems allocate contiguous GPU memory for each request's KV cache, pre-sized to the maximum possible sequence length. Since actual output lengths vary widely, this leads to 60–80% internal fragmentation — allocated but unused memory that cannot be reassigned. External fragmentation from varying allocation lifetimes further compounds the waste, typically leaving only ~40% of GPU memory actually usable.

**PagedAttention (Kwon et al., SOSP 2023):** The core insight is treating the KV cache like virtual memory. The cache is divided into fixed-size blocks (typically 16–256 tokens each), and a block table maps logical token positions to physical GPU memory blocks. This enables:
- **Non-contiguous storage:** A single sequence's KV cache can be scattered across physical memory, eliminating external fragmentation.
- **On-demand allocation:** Blocks are allocated only as tokens are actually generated, eliminating pre-allocation waste.
- **Memory sharing:** Multiple sequences with the same prefix (e.g., system prompt, beam search candidates) can share physical KV blocks via copy-on-write semantics.
- **GPU-CPU swapping:** When GPU memory is exhausted, infrequently accessed blocks can be evicted to CPU RAM and reloaded on demand.

**Performance:** PagedAttention raises GPU memory utilization from ~40% to >90%, supporting 2–4× larger batch sizes. Combined with the contiguous memory savings, vLLM achieves up to 24× throughput over naive HuggingFace Transformers serving on Llama 2 7B under high concurrency.

**Production Evolution (vLLM v0.6.0, Sept 2024):** The v0.6.0 release integrated multi-step scheduling (amortizing CPU overhead by scheduling multiple decode steps at once), chunked prefill (breaking long prefill computations into chunks interleaved with decode), and async output processing. On Llama 3 8B with a single H100, this achieved 2.7× throughput over v0.5.3 and **5× reduction in inter-token latency**. On Llama 3 70B with 4× H100, 1.8× throughput improvement.

**Hybrid Chunked Prefill (2025):** Addresses a key trade-off in chunked prefill — while it dramatically improves inter-token latency under high load, it can increase time-to-first-token (TTFT) under low concurrency. The hybrid approach adaptively switches between continuous and chunked prefill based on decode traffic, achieving 10–20% lower TTFT at low concurrency while maintaining high-load throughput.

### 4. Speculative Decoding and Attention Mechanism Interaction

**Core Idea:** Autoregressive decoding is memory-bandwidth-bound because each forward pass loads the entire KV cache to generate a single token. Speculative decoding uses a lightweight "draft" model to propose *k* candidate tokens cheaply, then verifies all *k* in a single forward pass of the full model. If the draft is reasonably accurate, throughput increases proportionally to the acceptance rate.

**The Counter-Intuitive Finding (MagicDec, Chen et al., 2024):** Prior wisdom held that speculative decoding is only useful at small batch sizes (where decoding is memory-bound). At large batch sizes, model parameter loading was thought to dominate, making decoding compute-bound. MagicDec disproved this for long contexts: at long sequence lengths, KV cache loading dominates parameter loading *regardless of batch size*. This means speculative decoding becomes *more* effective at larger batch sizes with long contexts — exactly the regime where throughput matters most.

**Key Approaches:**
- **Self-Speculation with StreamingLLM (MagicDec):** Uses the target model itself as the draft model but with a small fixed-size sliding window KV cache. The draft runs cheaply (KV cache ~1–2K tokens regardless of full context length), while verification uses the complete cache. Achieves 2× speedup on Llama 2 7B 32K.
- **TriForce (Sun et al., 2024):** Hierarchical speculation — a small model with StreamingLLM sparse attention handles the model-loading bottleneck, then the original model with retrieval-based sparse attention handles the KV-cache bottleneck. Up to 2.31× on Llama 2 128K.
- **QuantSpec (2024):** Combines INT4/INT8 KV cache quantization with speculative decoding. Uses quantized KV cache in draft attention for maximum bandwidth savings. 2.49× speedup at 128K context.
- **SparseSpec (2024):** Selectively loads only top-k attention tokens during drafting using PillarAttn, with zero-overhead token scoring by reusing verification-phase attention scores. 2.13× on reasoning models.

**Attention Arithmetic Intensity Analysis:** The key insight linking speculative decoding to KV cache optimization: model parameter operations have arithmetic intensity proportional to batch size *b* (can become compute-bound), but KV cache operations have intensity proportional to the GQA ratio *g* (always ~1–8), meaning they remain memory-bound regardless of batch size. As sequence length *n* grows, the fraction of time spent loading the KV cache approaches 1.0 — making techniques that amortize KV cache loads (speculative decoding) or reduce cache size (GQA, quantization, eviction) increasingly valuable.

### 5. Attention Sinks and Streaming LLM

**The Discovery (Xiao et al., ICLR 2024):** When analyzing LLM attention patterns, the authors found that a disproportionate amount of attention mass is consistently allocated to the very first few tokens of any sequence — regardless of their semantic content. This "attention sink" phenomenon arises from the softmax requirement that attention weights sum to 1: when no token is particularly relevant, the model "dumps" excess probability mass onto tokens visible to all subsequent positions (the initial tokens). Strikingly, when these initial tokens are evicted from the KV cache (e.g., by a sliding window), model performance collapses immediately — even if the evicted tokens were semantically irrelevant.

**StreamingLLM:** Maintains only 4 initial "sink" tokens plus a rolling window of recent tokens. Total KV cache is O(sink_tokens + window_size), constant regardless of stream length. This enables stable, unbounded streaming generation on sequences exceeding 4 million tokens with Llama 2, MPT, Falcon, and Pythia — without any fine-tuning. Achieves 22.2× speedup over sliding window with recomputation. Adding a dedicated placeholder "sink token" during pre-training further improves streaming stability.

**Limitations:** StreamingLLM does *not* extend the effective context window — the model can only "see" window_size recent tokens. It is ideal for streaming chatbots and multi-turn dialogue (where only recent context matters) but not for long-document summarization or retrieval over distant context.

### 6. KV Cache Eviction Policies

Beyond static window approaches, dynamic eviction policies selectively retain important tokens:

| Method | Importance Metric | Key Innovation |
|--------|-------------------|----------------|
| **H2O** (Zhang et al., 2023) | Cumulative attention scores | Heavy-Hitter phenomenon: small fraction of tokens dominate attention; formulates as dynamic submodular optimization |
| **FastGen** (Ge et al., 2023) | Attention pattern profiling | Five fundamental attention structures; per-head adaptive policy assignment |
| **SnapKV** (Li et al., 2024) | Observation window scores | Important prompt tokens identified during a short observation window remain stable throughout generation |
| **L2Compress** (Devoto et al., 2024) | L₂ norm of key embeddings | Counter-intuitive: low L₂ norm → high attention importance; evict high-norm tokens |
| **Attention-Gate** (Zeng et al., Oct 2024) | Learned global context | Trainable per-token, per-head, per-layer eviction flags; evicts *before* attention computation to save FLOPs |
| **Keyformer** (Adnan et al., 2024) | Gumbel-adjusted scores | Accounts for softmax distribution distortion after token removal |
| **Value-Aware** (Guo et al., EMNLP 2024) | Value state norms | "Attention Score is Not All You Need" — value states carry independent importance signals |

**H2O Limitations Addressed by 2024 Work:**
- Attention bias toward initial/recent tokens distorts importance → Attention-Gate uses global structure
- Uniform eviction ratio across heads → per-head, per-layer granularity in newer methods
- Wasted computation on evicted tokens → pre-MHA eviction in Attention-Gate
- Heuristic, not learned → light fine-tuning in Attention-Gate (4×4090 GPUs, 5K samples)

### 7. FlashAttention and Kernel-Level Optimization

**FlashAttention-3 (Dao et al., NeurIPS 2024):** While FlashAttention-1/2 focused on reducing HBM reads via tiling and recomputation, FA3 is purpose-built for NVIDIA Hopper (H100) GPUs, exploiting three new hardware features:
- **Warp specialization:** Producer warps load data asynchronously via TMA while consumer warps compute WGMMA matrix multiplies on Tensor Cores — hiding memory latency behind computation.
- **Asynchronous softmax:** Overlaps softmax with WGMMA using a 2-stage pipeline, breaking the sequential dependency between softmax and GEMM that limited FA2.
- **FP8 with incoherent processing:** Multiplies Q/K by random Hadamard matrices before FP8 quantization to "smear" outlier magnitudes, reducing quantization error by 2.6×.

FA3 achieves **75–85% H100 utilization** for FP16 attention (vs. ~35% for FA2) and up to **1.2 PFLOPS/s** in FP8 — near the GPU's theoretical peak. This transforms attention from a severe bottleneck into a well-utilized operation, complementing the memory reduction techniques above.

---

## Important Papers & References

1. **Shazeer, N. (2019).** "Fast Transformer Decoding: One Write-Head Is All You Need." *arXiv:1911.02150*. — Introduced Multi-Query Attention (MQA), demonstrating that sharing a single KV head across all query heads dramatically reduces memory with minimal quality loss.

2. **Ainslie, J., Lee-Thorp, J., de Jong, M., et al. (2023).** "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP 2023*. — Introduced Grouped-Query Attention as a tunable middle ground between MHA and MQA; demonstrated uptraining from MHA checkpoints with minimal tokens.

3. **Kwon, W., Li, Z., Zhuang, S., et al. (2023).** "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. — The vLLM paper introducing PagedAttention: virtual-memory-inspired KV cache management achieving near-zero memory waste and 2–24× throughput improvements.

4. **Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024).** "Efficient Streaming Language Models with Attention Sinks." *ICLR 2024*. — Discovered the attention sink phenomenon; proposed StreamingLLM for stable generation over 4M+ tokens with constant KV cache.

5. **Brandon, W., Mishra, A., Nrusimha, A., et al. (2024).** "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention." *NeurIPS 2024*. — Extended KV sharing across layers (CLA2), achieving 2× further reduction on top of MQA/GQA with negligible perplexity cost.

6. **Joshi, R., Garg, S., et al. (2024).** "QCQA: Quality and Capacity-aware Grouped Query Attention." *arXiv:2406.10247*. — Evolutionary algorithm for optimal query-head grouping in GQA, achieving 20% higher accuracy or 40% KV cache savings vs. standard GQA.

7. **Dao, T., Shah, J., et al. (2024).** "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." *NeurIPS 2024 (Spotlight)*. — Hopper-optimized attention kernel achieving 75–85% H100 utilization; introduced warp specialization, async softmax, and incoherent FP8 processing.

8. **Chen, Z., et al. (2024).** "MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding." *arXiv:2408.11049*. — Showed that long-context large-batch decoding is KV-cache-memory-bound, making speculative decoding viable at scale; introduced self-speculation with StreamingLLM draft.

9. **Sun, Z., et al. (2024).** "TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding." — Hierarchical speculation combining sparse attention for both model-loading and KV-cache bottlenecks; 2.31× on 128K context.

10. **Zhang, Z., Sheng, Y., et al. (2023).** "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." *NeurIPS 2023*. — Formulated KV cache eviction as dynamic submodular optimization; identified Heavy-Hitter phenomenon where few tokens dominate attention.

11. **Ge, S., et al. (2023).** "Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs." *arXiv:2310.03629*. — FastGen: identified five attention patterns and applied per-head adaptive compression strategies.

12. **DeepSeek-AI. (2024).** "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." — Introduced Multi-Head Latent Attention (MLA), achieving 6–8× KV cache compression through low-rank joint compression.

13. **Ji, T., Guo, B., et al. (2025).** "Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs." *ACL 2025*. — MHA2MLA: framework to retrofit MLA onto any pretrained LLM with minimal fine-tuning; 92.19% KV cache reduction on Llama 2 7B.

14. **Kong, J., et al. (2025).** "Opt-GPTQ: An Optimized GPTQ Combining Sparse Attention and Quantization Techniques." *arXiv:2505.02351*. — Explicitly combined GQA + GPTQ quantization + PagedAttention in a unified framework integrated with vLLM.

15. **Hooper, C., et al. (2024).** "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." — Pushed KV cache quantization to 2–4 bits with specialized handling for key outliers.

16. **Zeng, X., et al. (2024).** "In-context KV-Cache Eviction for LLMs via Attention-Gate." *arXiv:2410.12876*. — Learnable, context-aware KV eviction operating before MHA computation, with per-token, per-head, per-layer granularity.

17. **Li, Y., et al. (2024).** "SnapKV: LLM Knows What You are Looking for Before Generation." — Observation-window-based KV selection showing important prompt tokens remain stable across generation.

18. **Guo, Z., et al. (2024).** "Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters." *EMNLP 2024*. — Demonstrated that value states carry independent importance signals beyond attention scores for KV cache eviction decisions.

---

## Open Questions & Future Directions

### 1. End-to-End System Co-Design
Most current research optimizes individual components (attention architecture, memory management, quantization, scheduling) in isolation. The Opt-GPTQ paper is a notable exception in explicitly combining GQA + quantization + PagedAttention, but the space of joint optimizations is largely unexplored. How should group count, quantization precision, and block size be jointly configured given a specific GPU's memory bandwidth, compute throughput, and on-chip memory hierarchy? Optimal configurations almost certainly depend on model size, sequence length distribution, and request arrival patterns — but systematic frameworks for deriving them are lacking.

### 2. MLA vs. GQA: Compression Frontier
MLA achieves dramatically more compression than GQA (6–8× vs. 4–8×), but with higher computational overhead from the on-the-fly KV reconstruction. The hardware-centric analysis from KU Leuven (2025) shows MLA can push attention workloads from memory-bandwidth-bound toward compute-bound — but on compute-rich GPUs like H100, the extra FLOPs from reconstruction are cheap compared to memory savings. On bandwidth-constrained edge devices, the tradeoff may flip. Comprehensive Pareto analyses across hardware targets are needed.

### 3. Quantization Error Accumulation in Long Sequences
While INT8 KV quantization is near-lossless at typical sequence lengths (4K–8K), error accumulation over 128K+ token sequences is not well characterized. The runtime-certified quantization approach (2026) with FP16 fallback is a promising direction, but its overhead in production serving (CPU-GPU transfers, online error monitoring) needs real-world benchmarking.

### 4. Unified Eviction-Quality Theory
KV cache eviction methods currently optimize ad-hoc metrics (cumulative attention, L₂ norms, learned gates) without a unified theoretical framework. The Heavy-Hitter submodular formulation in H2O is a step toward formalization, but it doesn't account for value-state importance, cross-head interactions, or the effect of quantization on eviction decisions. A unified theory connecting eviction policy to bound on output quality degradation would enable provably safe compression.

### 5. Speculative Decoding + Quantization Interactions
Speculative decoding's draft model benefits from quantized KV caches (as in QuantSpec), but the interaction between draft accuracy, quantization error, and verification overhead is not fully characterized. At very long sequences where KV cache loading dominates, aggressive quantization may be optimal; at shorter sequences, the draft model's quality may matter more. Dynamic precision switching during speculation is an open research area.

### 6. Attention Sinks: From Phenomenon to Design Principle
The attention sink discovery explains *why* StreamingLLM works, but it hasn't yet been systematically exploited in architecture design. Could future models be pre-trained with explicit sink mechanisms (beyond simple placeholder tokens) that make KV eviction safer? Could the sink property be engineered to support *selective* long-range attention — retaining important distant tokens while safely evicting others?

### 7. Disaggregated Prefill and Decode
Emerging serving architectures physically separate prefill (compute-bound, benefits from many GPUs) from decode (memory-bound, benefits from large KV cache capacity). This disaggregation changes the optimization landscape: prefill nodes may prefer MLA's compression (reducing memory for cached KV sent to decode nodes), while decode nodes may prefer simpler GQA (avoiding reconstruction overhead). The TPLA paper (2025) addresses tensor parallelism in this setting, but optimal architecture assignment across disaggregated clusters is unsolved.

### 8. Beyond Transformers
All techniques discussed assume the standard transformer attention mechanism. Emerging architectures (Mamba, RWKV, linear attention, Titans) fundamentally avoid the KV cache by design through state-space models or linearized attention. As these architectures mature and scale, the entire KV cache optimization problem may become less relevant — but for now, transformers dominate production deployment, and the optimization techniques above represent the best path to efficient serving.

---

## Relevance to Main Topic

KV-cache optimization is arguably the single most impactful area for improving LLM inference efficiency in production. The techniques surveyed here — architectural compression (GQA, MLA), memory management (PagedAttention), numerical compression (quantization), and algorithmic acceleration (speculative decoding, attention sinks, eviction) — are all deployed in production systems today and directly address the primary constraint on serving throughput.

The research question at the heart of this survey — *how do GQA + quantization + PagedAttention combine to maximize throughput?* — has a well-supported answer: they are multiplicative and complementary. GQA reduces the absolute volume of KV data that must be stored (4–8×), PagedAttention eliminates the memory waste from fragmentation (~2–3× effective capacity gain), and quantization compresses whatever remains (2× for INT8, 4× for INT4). A system deploying all three can support **10–30× larger effective batch sizes** than a naive MHA + contiguous allocation + FP16 system on identical hardware. Since LLM serving throughput scales near-linearly with batch size in the memory-bound regime, this translates directly to **5–10× throughput improvements** in practice.

For the broader research pipeline, these findings have several implications:
- **Architecture decisions at training time** (GQA group count, MLA adoption) have outsized effects on serving costs that persist for the lifetime of the model.
- **Serving infrastructure** (vLLM, TensorRT-LLM, SGLang) is evolving rapidly and should be considered a moving target — optimization interactions that work today may be superseded by framework improvements.
- **The combination of techniques matters more than any individual one** — papers that study optimizations in isolation may underestimate real-world benefits when techniques are layered.
- **Hardware matters:** The optimal configuration of GQA groups, quantization precision, and block size depends on the specific GPU's memory bandwidth, compute throughput, and on-chip cache hierarchy. H100-optimal configurations may not be B200-optimal.

The field is moving toward a holistic "inference compiler" paradigm where model architecture, quantization scheme, memory management, and scheduling policy are co-optimized for a specific hardware target and workload distribution — rather than tuned independently. The Opt-GPTQ and TPLA papers represent early steps in this direction, and we expect it to be a major research theme through 2026–2027.
