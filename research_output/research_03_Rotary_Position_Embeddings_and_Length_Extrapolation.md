# Rotary Position Embeddings and Length Extrapolation

## Overview

Rotary Position Embedding (RoPE) has emerged as the dominant positional encoding scheme for modern large language models, adopted by virtually all major architectures including LLaMA, Qwen, Mistral, GPT-NeoX, DeepSeek, and Phi. Introduced by Su et al. (2021) in "RoFormer: Enhanced Transformer with Rotary Position Embedding," RoPE encodes position information by applying a rotation matrix to query and key vectors before computing attention scores. The key design insight is that the dot product between RoPE-encoded queries and keys depends only on the *relative* position difference (n − m), while the encoding itself operates on *absolute* positions — achieving the best of both worlds: absolute position encoding with relative position semantics.

The fundamental limitation of RoPE-based models is that they fail catastrophically when processing sequences longer than their training context length. This occurs because higher-dimensional RoPE frequency components encounter rotation angles never seen during pre-training — an out-of-distribution (OOD) problem. For instance, a model trained at 4K context will see perplexity explode when fed 8K+ sequences, with the highest-frequency RoPE dimensions (those with the smallest rotation angles) being most severely affected. This has motivated an intense research effort to develop methods that can extend pre-trained RoPE-based LLMs to 32K, 128K, or even 1M+ token context windows with minimal or no additional fine-tuning.

The research landscape has evolved rapidly from simple position interpolation (Chen et al., 2023) through NTK-aware frequency scaling (2023), to the sophisticated YaRN method (Peng et al., ICLR 2024) that combines piecewise frequency handling with attention temperature calibration. More recent advances include LongRoPE2 (Microsoft, 2025) with evolutionary search for optimal per-dimension rescaling factors, and training-free approaches like DPE (dimension-wise positional embedding manipulation, 2025) that can achieve 128K context with zero fine-tuning. Parallel to these practical advances, theoretical work has deepened our understanding: the Lie group/algebraic framework (Liu & Zhou, 2025) provides a unified mathematical foundation for all RoPE variants, while Men et al. (NeurIPS 2024) proved that the RoPE base frequency imposes an absolute lower bound on achievable context length. Perhaps most provocatively, Kazemnejad et al. (NeurIPS 2023) showed that NoPE — transformers with no position encoding at all — can outperform explicit encodings including RoPE on length generalization tasks, challenging fundamental assumptions about the necessity of position information.

---

## Key Methods & Approaches

### 1. Mathematical Foundations of RoPE

**Core Formulation.** For a query vector **q** at position *m* and key vector **k** at position *n*, RoPE applies a block-diagonal rotation matrix *R*ₘ constructed from 2D rotation blocks:

$$R_{\Theta,m} = \bigoplus_{i=1}^{d/2} \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}$$

where frequencies follow a geometric progression: θᵢ = *b*^(-2i/d), typically with base *b* = 10,000. The critical property is that the attention dot product depends only on relative position:

$$(R_m q_m)^\top(R_n k_n) = q_m^\top R_{n-m} k_n$$

This means RoPE inherently provides relative position encoding while operating on absolute positions, requiring no learned parameters and being fully compatible with KV-caching for efficient autoregressive decoding.

**Long-Term Decay Property.** The inner product between RoPE-encoded vectors decays (in an oscillating fashion) as relative distance increases, giving the model a natural inductive bias toward attending to nearby tokens. This decay arises from the interference pattern of the sum of cosines across frequency components: Σcos((m−n)θᵢ). Higher frequencies (small i) capture fine-grained local patterns; lower frequencies (large i) capture long-range structure. Men et al. (NeurIPS 2024) formalized this as the "long-term decay of attending to similar tokens" and proved that the RoPE base *b* imposes an **absolute lower bound** on achievable context length — if *b* is too small, the model exhibits "superficial long-context ability" (low perplexity but inability to retrieve information).

**Lie Algebraic Unification (2025).** Liu & Zhou (arXiv:2504.06308, 2025) provided the first rigorous mathematical framework grounding RoPE in Lie group and Lie algebra theory. They identified two fundamental properties — **relativity** (attention depends only on x₂ − x₁) and **reversibility** (distinct positions map to distinct rotation matrices, guaranteeing injectivity) — and proved that all valid RoPE schemes must be constructed from generators lying in the basis of a Maximal Abelian Subalgebra (MASA) of the special orthogonal Lie algebra 𝔰𝔬(n). The generator matrices Bᵢ must be skew-symmetric, linearly independent, and pairwise commuting ([Bᵢ, Bⱼ] = 0). This framework unifies 1D RoPE, 2D RoPE, MRoPE (multimodal), and learnable variants like STRING and ComRoPE under a single theoretical roof.

### 2. NTK-Aware Scaling, YaRN, and Position Interpolation

**Position Interpolation (PI).** The simplest approach, introduced by Chen et al. (2023): uniformly scale all position indices by factor *s* = L_target / L_train, mapping all positions back into the training range. While conceptually simple and effective with fine-tuning (~1000 steps), PI uniformly compresses all frequency dimensions, destroying the high-frequency resolution needed for local token discrimination. This causes degradation on short-context tasks even after fine-tuning.

**NTK-Aware Scaling.** Instead of scaling position indices, scale the RoPE base frequency: *b'* = *b* · s^(d/(d−2)). This nonlinearly adjusts frequencies — high-frequency dimensions are barely changed (preserving local resolution), while low-frequency dimensions are heavily scaled (accommodating longer-range positions). The method is inspired by Neural Tangent Kernel theory (hence "NTK-aware") and can achieve 2-4× extension *without any fine-tuning*. However, all dimensions use the same scaling formula, and extreme extensions (>4×) remain problematic.

**NTK-by-Parts.** A refinement that classifies each RoPE dimension by its wavelength λᵢ = 2π/θᵢ relative to the training length L:
- **High frequency** (λᵢ ≪ L, ratio r > β): These dimensions are well-trained; **no interpolation** — pure extrapolation.
- **Low frequency** (λᵢ ≫ L, ratio r < α): These dimensions are under-trained; **full interpolation** (θᵢ/s).
- **Mid frequency** (α ≤ r ≤ β): **Linear ramp** between interpolation and extrapolation.

Typical hyperparameters: α = 1, β = 32.

**YaRN (Yet another RoPE extensioN).** Peng et al. (ICLR 2024) combined NTK-by-Parts with a critical innovation: **attention temperature scaling**. When RoPE frequencies are modified, the softmax attention distribution becomes artificially "sharpened" — this entropy collapse causes the model to focus too narrowly and lose retrieval ability. YaRN corrects this by scaling the attention logits:

$$\text{Attention} = \text{softmax}\left(\frac{QK^T}{t\sqrt{|D|}}\right), \quad \sqrt{\frac{1}{t}} = 0.1\ln(s) + 1$$

For a 32× extension (s=32), √(1/t) ≈ 1.347, gently softening the attention distribution to match pre-extension entropy levels. This temperature factor can be absorbed into query/key normalization for zero inference overhead and full FlashAttention compatibility.

**YaRN Results:**
- 10× fewer fine-tuning tokens vs. PI; ~2.5× fewer training steps
- LLaMA-2 7B @ 32K: perplexity 2.77 (vs. 3.57 for PI)
- 128K context: 99.4% PassKey retrieval accuracy
- Adopted by LLaMA 3.1, Qwen 3, Mistral Large, and DeepSeek V3
- **Dynamic YaRN** variant achieves 2×+ extension with zero fine-tuning

### 3. Dynamic NTK and Code-Based RoPE Adaptations

**Dynamic NTK.** The key insight is that the scale factor *s* should adapt to the *actual* input length at inference time rather than being a fixed hyperparameter. The scaling function uses elastic sub-linear growth:

$$S(l') = \max\left(1, \gamma \cdot (l'/L)^{\kappa}\right)$$

with typical values γ = 0.5, κ = 0.7. This avoids sudden spectral changes when sequences modestly exceed the training length. When the input is within the training range (l' ≤ L), s = 1 and behavior is unchanged. As sequences grow longer, the scale factor increases smoothly.

**HuggingFace Implementation.** Starting from Transformers v4.35+, Dynamic NTK and YaRN are natively supported via the `rope_scaling` config:

```python
config.rope_scaling = {
    "type": "dynamic",    # Dynamic NTK
    "factor": 2.0         # 4K → 8K
}
# or
config.rope_scaling = {
    "type": "yarn",
    "factor": 8.0,        # 4K → 32K
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
}
```

**LongRoPE2 (Microsoft, 2025).** A major advance for near-lossless context scaling. Key innovations:
1. **RoPE Training Deficiency Hypothesis**: Higher RoPE dimensions are systematically under-trained because they rarely experience full rotation cycles during pre-training. This shifts the *true* critical dimension lower than theory predicts (e.g., from dimension 35 → 30 for LLaMA3-8B).
2. **Evolutionary Search with "Needle-Driven" Perplexity**: Instead of using average perplexity (which masks long-context failures), the search optimizes using a "needle-driven" metric that focuses on answer-token loss in passkey retrieval tasks.
3. **Mixed Context Window Training**: The model simultaneously trains on short sequences (using original RoPE) and long sequences (using rescaled RoPE), with cross-document attention masking to prevent interference.

Results: LLaMA3-8B → 128K with **98.5%** short-context retention, using only 10B tokens (80× fewer than Meta's 800B-token approach). Surpasses Meta's LLaMA3.1-8B-128K on RULER. Adopted in Microsoft's Phi-4-mini and Phi-4-multimodal.

**DPE (Dimension-wise Positional Embeddings Manipulation, April 2025).** A training-free approach that detects each RoPE dimension's "effective length" and rescales only the problematic dimensions. Results: LLaMA3-8B → 128K without training; LLaMA3.1-70B + DPE outperforms GPT-4-128K on RULER.

**ParallelComp (February 2025).** Training-free method using parallel chunked attention with calibration to mitigate attention sink phenomena. Achieves 4K → 128K on a single A100 80GB with 23.5× prefilling acceleration and 91.17% of GPT-4's long-context performance.

### 4. Comparison with ALiBi and NoPE

**ALiBi (Attention with Linear Biases).** Press et al. (ICLR 2022) proposed replacing explicit position embeddings with a static, non-learned linear bias added directly to attention scores:

$$\text{softmax}(q_i K^\top + m \cdot [-(i-1), \dots, -2, -1, 0])$$

where *m* is a head-specific slope (geometric sequence: 2⁻¹, 2⁻², …, 2⁻ⁿ for *n* heads). This provides an inductive bias toward recency without any learned or computed position representations.

Key ALiBi findings:
- Models trained at 1,024 tokens extrapolate to 10,000+ tokens without fine-tuning
- 1.3B ALiBi model at L=1024 matches sinusoidal model at L=2048, while training ~11% faster with ~11% less memory
- Used in BLOOM (BigScience)
- **Limitation**: The rigid linear recency bias cannot flexibly model diverse attention patterns; performance on long-range dependency tasks lags behind well-tuned RoPE+YaRN at equivalent compute budgets

**NoPE (No Position Encoding).** Kazemnejad et al. (NeurIPS 2023) delivered the surprising result that decoder-only transformers *without any positional encoding* can outperform RoPE, ALiBi, sinusoidal, and T5-relative-bias on length generalization tasks. The authors proved NoPE can theoretically represent both absolute and relative positions (causal attention masks provide implicit positional information through the triangular structure), and empirically found that NoPE attention patterns most closely resemble T5's Relative PE — encouraging attention to both short- and long-range positions.

Subsequent work by Wang et al. (ACL 2024) showed that NoPE's length generalization failure is caused by **attention distribution distraction** — attention heads become too uniformly spread (high entropy) as sequences lengthen. Their solution, head-based softmax temperature scaling (only 704 trainable parameters for a 1B model, 0.03% of pretraining data), outperforms RoPE-based zero-shot NTK extension and approaches fine-tuned YaRN performance.

**Comparative Assessment:**

| Method | Extrapolation | Short-Context | Computational Cost | Flexibility |
|--------|:---:|:---:|:---:|:---:|
| RoPE + YaRN | ★★★★★ (128K+) | ★★★★☆ | Medium (light FT) | ★★★★★ |
| RoPE + Dynamic NTK | ★★★☆☆ (~16K) | ★★★★★ | Zero | ★★★☆☆ |
| ALiBi | ★★★★☆ (10K+) | ★★★★☆ | Zero (built-in) | ★★☆☆☆ |
| NoPE + Temp Scale | ★★★★☆ (32K+) | ★★★★☆ | Minimal (704 params) | ★★★★☆ |

The emerging consensus is that RoPE + YaRN/LongRoPE2 represents the best overall trade-off for production systems, while NoPE with temperature scaling offers a compelling minimalist alternative for scenarios where position encoding overhead is undesirable.

### 5. Attention Score Concentration and Entropy

A unifying theme across 2024 research is that **attention entropy control** is the fundamental mechanism underlying length generalization, transcending specific position encoding choices.

**The Entropy Problem.** As sequence length grows:
- Softmax attention entropy approaches its theoretical maximum Θ(log n), causing attention weights to become nearly uniform
- This "attention dilution" means the model cannot effectively focus on relevant tokens
- Simultaneously, embedding vectors collapse toward their low-frequency (DC) components due to self-attention acting as a low-pass filter (Zhou et al., 2024 — "Length-Induced Embedding Collapse")
- The combination produces the characteristic perplexity explosion at out-of-distribution lengths

**Temperature Scaling as Universal Fix.** Zhang et al. (2024, "Extending LLMs' Context Window with 100 Samples") demonstrated that attention entropy stabilization — achieved through RoPE base frequency adjustment combined with dynamic attention logit scaling — enables extension to 16K with only 100 training samples and 6 steps. Their "Entropy-ABF" method directly targets entropy maintenance as the optimization objective.

Zhong et al. (2024, "Understanding RoPE Extensions from an Attention Perspective") identified three critical findings:
1. Maintaining attention patterns consistent with pre-training distributions improves extrapolation
2. Large attention uncertainty (entropy spikes) directly correlates with retrieval failures in Needle-in-a-Haystack tests
3. Longer continual pre-training reduces attention uncertainty by giving the model more experience with diverse attention distributions

**Beyond Softmax.** Recent work proposes replacing softmax entirely: α-entmax and ASEntmax (Adaptive-Scalable Entmax, 2024) maintain bounded, low-entropy attention patterns even at extreme lengths by producing sparse attention distributions instead of dense softmax. This approach remains experimental but points toward fundamentally different attention mechanisms for ultra-long contexts.

---

## Important Papers & References

### Foundational Works

1. **Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"** — arXiv:2104.09864, 2021. The original RoPE paper. Introduces rotation-based position encoding achieving relative position semantics through absolute position operations. Establishes the core mathematical framework used by virtually all modern LLMs.

2. **Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"** — ICLR 2022, arXiv:2108.12409. Introduces ALiBi, demonstrating that static linear attention biases enable length extrapolation without learned position embeddings. Adopted in BLOOM.

3. **Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers"** — NeurIPS 2023, arXiv:2305.19466. Landmark study showing NoPE (no position encoding) outperforms RoPE, ALiBi, and other explicit encodings on length generalization. Proves NoPE can theoretically represent both absolute and relative positions.

### Context Window Extension Methods

4. **Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation"** — arXiv:2306.15595, 2023. Introduces Position Interpolation (PI), the first practical method for extending pre-trained RoPE model context windows by linearly scaling position indices.

5. **Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models"** — ICLR 2024, arXiv:2309.00071. Combines NTK-by-parts frequency handling with attention temperature scaling. The current industry standard, adopted by LLaMA 3.1, Qwen 3, Mistral Large, and DeepSeek V3. Achieves 128K context with 10× fewer fine-tuning tokens than PI.

6. **Zhang et al., "Extending LLMs' Context Window with 100 Samples"** — arXiv:2401.07004, 2024. Introduces Entropy-ABF, demonstrating that attention entropy stabilization enables 4K→16K extension with only 100 samples and 6 training steps. Reframes length generalization as an entropy control problem.

7. **An et al., "Training-Free Long-Context Scaling of Large Language Models (ChunkLlama)"** — ICML 2024, arXiv:2402.17463. Dual-chunk attention mechanism for training-free context extension, compatible with PI, NTK, and YaRN. Achieves 100K+ context on LLaMA 3 70B.

### Advanced RoPE Scaling (2024-2025)

8. **Men et al., "Base of RoPE Bounds Context Length"** — NeurIPS 2024, arXiv:2405.14591. Proves the RoPE base frequency imposes an absolute lower bound on achievable context length. Reveals "superficial long-context ability" where models maintain low perplexity but lose information retrieval capability if the base is too small.

9. **Shang et al., "LongRoPE2: Near-Lossless LLM Context Window Scaling"** — arXiv:2502.20082, 2025 (ICML). Microsoft's evolutionary search approach to per-dimension RoPE rescaling. Achieves 128K with 98.5% short-context retention using only 10B tokens. Adopted in Phi-4-mini and Phi-4-multimodal.

10. **Zhong et al., "Understanding the RoPE Extensions of Long-Context LLMs: An Attention Perspective"** — arXiv:2406.13282, 2024. Systematic study of how attention patterns change during context extension. Identifies attention entropy spikes as the mechanistic cause of retrieval failures in long contexts.

### Theoretical Foundations

11. **Liu & Zhou, "Rethinking RoPE: A Mathematical Blueprint for N-dimensional Positional Encoding"** — arXiv:2504.06308, 2025. Rigorous Lie group/algebraic framework unifying all RoPE variants. Proves valid RoPE schemes must be constructed from Maximal Abelian Subalgebras of 𝔰𝔬(n).

12. **Wang et al., "Length Generalization of Causal Transformers without Position Encoding"** — ACL 2024, arXiv:2404.12224. Identifies attention distribution distraction as NoPE's failure mode. Introduces head-based softmax temperature scaling (704 parameters) that outperforms RoPE-based zero-shot NTK extension.

### Related Phenomena

13. **Zhou et al., "Length-Induced Embedding Collapse in Transformer-based Models"** — arXiv:2410.24200, 2024. Theoretical analysis showing self-attention acts as a low-pass filter, causing embedding collapse in long sequences. Proposes TempScale as a plug-and-play fix.

14. **ORoPE: Optimal RoPE Extension via Bayesian Optimization** — Natural Language Processing Journal, 2025. Training-free method using Bayesian optimization to search for optimal per-dimension frequency weights. Achieves 41.2% improvement over PI/NTK/YaRN at 32K on proxy tasks.

---

## Open Questions & Future Directions

### 1. The NoPE Challenge to RoPE Orthodoxy
The finding that transformers without any position encoding can match or exceed RoPE-based models on length generalization raises fundamental questions. If causal attention masks provide sufficient positional information, why does RoPE help at all? Current evidence suggests RoPE's primary benefit may be training efficiency (faster convergence) rather than fundamental representational capacity. The interaction between position encoding choice, training data scale, and length generalization ability remains poorly understood and is an active research frontier.

### 2. The Ultimate Limits of Length Extrapolation
While methods like LongRoPE2 and DPE can achieve 128K with near-lossless short-context performance, the theoretical limits remain unclear. Men et al. (2024) provided a lower bound based on RoPE base frequency, but the interaction between base frequency, model dimension, training data distribution, and maximum effective context length is not fully characterized. Open questions include: Is there a hard information-theoretic limit on context length for a given model size? Can techniques like RingAttention or Infini-attention circumvent RoPE's fundamental constraints?

### 3. Attention Entropy as a Unified Framework
Multiple independent lines of work in 2024 converged on attention entropy as the key mechanism underlying length generalization failure. This suggests a potential unification: all position encoding methods (RoPE, ALiBi, NoPE) may succeed or fail based on their ability to maintain stable attention entropy distributions across sequence lengths. Developing a rigorous information-theoretic framework connecting position encoding design to attention entropy dynamics is a promising direction that could yield principled design rules rather than empirical heuristics.

### 4. Multi-Modal and Multi-Dimensional RoPE
As models become increasingly multi-modal, extensions like MRoPE (multimodal RoPE) that encode text position, 2D spatial coordinates, and temporal indices in a unified rotational framework are critical. The Lie-algebraic framework of Liu & Zhou (2025) provides the mathematical foundation, but practical challenges remain: How should frequency allocation be balanced across modalities? How do we prevent modality interference in the shared positional space? These questions will grow in importance as vision-language-action models become prevalent.

### 5. Training-Free vs. Fine-Tuning Approaches
The rapid improvement in training-free methods (DPE, ParallelComp, ORoPE) raises the question of whether fine-tuning will remain necessary for context extension. Current training-free methods approach but don't quite match the best fine-tuned approaches (LongRoPE2) on comprehensive benchmarks like RULER. The gap is narrowing, however, and the economic advantage of zero-training solutions is compelling. The next breakthrough may come from combining training-free rescaling with extremely lightweight adaptation (e.g., 0.01% parameter updates).

### 6. Evaluation Methodology
A critical methodological issue is that standard perplexity averaging can mask long-context failures — a model may maintain low average perplexity while completely failing at retrieving specific information from long contexts (Men et al.'s "superficial long-context ability"). The shift toward "needle-driven" evaluation metrics (LongRoPE2) and comprehensive benchmarks like RULER represents important progress, but the field still lacks standardized protocols for measuring true long-context capability versus superficial statistical matching.

### 7. Hardware-Aware RoPE Design
Current RoPE implementations are optimized for GPU tensor cores, but the emergence of alternative hardware (TPUs, specialized attention accelerators, edge devices) may motivate hardware-specific RoPE variants. Frequency allocation strategies that minimize KV-cache memory bandwidth, rotation-free approximations for resource-constrained settings, and dynamic precision allocation across RoPE dimensions are underexplored areas with practical significance.

---

## Relevance to Main Topic

Rotary Position Embeddings and length extrapolation are central to the broader research agenda of making large language models capable of processing and reasoning over very long contexts. This connects to the main research topic in several critical ways:

**Architecture Design.** RoPE is the de facto standard positional encoding for modern LLMs. Any architectural decisions about attention mechanisms, context processing strategies, or model scaling must contend with RoPE's properties — its long-term decay characteristics, its OOD failure mode, and its interaction with training dynamics. Understanding RoPE is not optional; it is prerequisite knowledge for LLM architecture research.

**Practical Deployment.** The ability to extend pre-trained models from 4K to 128K+ with minimal fine-tuning is one of the highest-impact capabilities in applied LLM work. The methods surveyed here — from Dynamic NTK (drop-in, no training, 2-4× extension) through YaRN (light fine-tuning, 8-32× extension) to LongRoPE2 (moderate fine-tuning, near-lossless 32× extension) — form a practical toolkit that every LLM practitioner needs. The decision tree is increasingly clear: for 2× extension, use Dynamic NTK with zero training; for 4-8×, use YaRN with ~400 steps; for 16-32× with strict short-context preservation requirements, use LongRoPE2.

**Theoretical Understanding.** The 2024-2025 convergence of multiple independent lines of work on attention entropy as the key mechanism is a significant scientific development. It suggests that position encoding design, attention mechanism choice, and length generalization are all manifestations of a deeper principle: the need to maintain stable information-theoretic properties of attention distributions across varying sequence lengths. This principle may guide the design of next-generation architectures that transcend current context-length limitations.

**Open Challenges.** Despite rapid progress, fundamental questions remain unanswered. Why does NoPE work at all? What is the true information-theoretic limit on context length for a given model capacity? Can training-free methods close the remaining gap with fine-tuned approaches? These questions place RoPE and length extrapolation research at the frontier of our understanding of transformer architectures, making it a rich area for both theoretical investigation and practical engineering.

---

## Summary: Best Practices for 4K → 128K+ Extension

Based on the current research landscape (as of mid-2025), the recommended approach for extending a pre-trained RoPE-based LLM from 4K to 128K+ with minimal fine-tuning is:

1. **Start with Dynamic NTK** (zero training) to assess baseline extensibility. If the model handles 2-4× extension acceptably, it has good "RoPE health."

2. **For 8-16× extension**, apply **YaRN** with piecewise frequency scaling (α=1, β=32) and attention temperature scaling (√(1/t) = 0.1·ln(s) + 1). Fine-tune for ~400-600 steps on mixed-length data (short + long sequences) using FlashAttention-2. This should require ~0.1% of original pre-training data.

3. **For 32× extension (4K→128K)** with strict short-context preservation requirements, use **LongRoPE2** methodology: evolutionary search for per-dimension rescaling factors guided by needle-driven perplexity, plus mixed context window training. This requires ~10B tokens of fine-tuning data.

4. **For zero-training 128K**, evaluate **DPE** (dimension-wise positional embedding manipulation) or **ParallelComp** (calibrated chunked attention). These are newer methods with less extensive validation but compelling results.

5. **Evaluation**: Always test with comprehensive benchmarks (RULER, Needle-in-a-Haystack, InfiniteBench) — not just average perplexity. Monitor attention entropy distributions to detect superficial long-context capability.

6. **Infrastructure**: Always use FlashAttention-2 (or equivalent) for memory-efficient long-sequence processing. Consider GQA (Grouped Query Attention) if modifying architecture is feasible.

The field is moving extremely fast: methods considered state-of-the-art in early 2024 (simple PI + fine-tuning) are now considered baseline approaches. Practitioners should monitor the training-free methods (DPE, ORoPE) as these may soon match fine-tuned quality while eliminating the data and compute requirements of adaptation.
