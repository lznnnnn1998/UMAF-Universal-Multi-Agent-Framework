# Attention in Multimodal and Cross-Modal Architectures

## Overview

Multimodal large language models (MLLMs) face a fundamental architectural challenge: how should visual, textual, audio, and video tokens interact within a unified neural architecture? The past three years (2022–2025) have witnessed a vigorous architectural debate between two dominant paradigms — **embedding-space alignment** (self-attention over concatenated tokens, exemplified by LLaVA) and **cross-attention-based alignment** (gated cross-attention layers interleaved with frozen LLM layers, pioneered by Flamingo). Each approach makes fundamentally different trade-offs between training data efficiency, inference throughput, parameter count, and multimodal reasoning quality.

The self-attention (decoder-only) paradigm treats visual tokens as a prefix to text tokens, feeding both through the LLM's standard self-attention layers. This approach is remarkably data-efficient — LLaVA-1.5 trains a simple 2-layer MLP projector in ~108 GPU-hours — but suffers from quadratic inference cost in the number of visual tokens (O((V+T)²)), since every token attends to every other token and the LLM's FFN layers process all visual tokens. The cross-attention paradigm instead inserts dedicated gated cross-attention (XATTN) layers between frozen LLM transformer blocks, where text tokens (as queries) attend to visual tokens (as keys/values). Flamingo demonstrated this approach with a tanh-gating mechanism initialized to zero, allowing the model to start identical to the frozen LLM and gradually incorporate visual signals. Cross-attention models enjoy ~3× higher training throughput (NVIDIA's NVLM finding) because visual tokens bypass the LLM's self-attention and FFN layers, but historically required massive noisy pretraining data (Flamingo used ~43M interleaved image-text web pages).

The 2024–2025 research trajectory reveals a clear **convergence toward hybrid approaches** that combine the best of both paradigms. Key validated findings include: (1) visual→visual self-attention is highly redundant — eliminating it can improve both accuracy and efficiency (NAAViT/SAISA achieved +1.2% accuracy with 66% fewer FLOPs); (2) simple token compression (adaptive pooling, 2D convolution) consistently outperforms learned semantic compression (Q-Former) because "double abstraction" is harmful — the LLM already performs semantic extraction, so the connector should only do compression; (3) preserving spatial information is critical for OCR and fine-grained understanding tasks; (4) gating mechanisms that dynamically weight modality contributions prevent modality bias and attention collapse; and (5) training-free token pruning at inference time can reduce visual token count by 50–90% with <1% accuracy loss (FastV, AIM, HoloV).

The central research question — "What cross-attention fusion strategy achieves the best multimodal understanding with minimal computational overhead?" — points toward **NVLM-H's hybrid design** (thumbnail → self-attention for global reasoning; tile details → cross-attention for efficiency) combined with **training-free token compression** (FastV-style early-layer pruning) as the current Pareto-optimal frontier. However, 2025 has also seen radical alternatives emerge: SAISA's NAAViT proves that eliminating visual-visual attention entirely can improve results, and GeminiFusion achieves linear-complexity O(n) pixel-wise fusion competitive with cross-attention. The field is far from settled.

---

## Key Methods & Approaches

### 1. Cross-Attention Designs in Vision-Language Models

#### 1.1 Flamingo (DeepMind, 2022): The Cross-Attention Pioneer

Flamingo (Alayrac et al., NeurIPS 2022) established the cross-attention paradigm for MLLMs with three key architectural innovations:

**Perceiver Resampler**: A learned bottleneck that compresses a variable number of visual encoder outputs (from a frozen NFNet trained with CLIP) into a fixed number of visual tokens (typically 64). The resampler uses cross-attention where learned latent queries attend over visual features. Unlike the standard Perceiver IO, Flamingo concatenates the keys and values computed from the latent queries with those from the visual features — enabling the latent queries to attend both to input features and their own representations, which ablations showed was marginally better.

**Gated Cross-Attention (XATTN-Dense) Layers**: Inserted at every transformer layer of the frozen LLM. Each layer computes:
```
XATTN(Q_text, K_vis, V_vis) = tanh(α) × CrossAttention(Q_text, K_vis, V_vis) + (1 - tanh(α)) × Q_text
```
where α is a learnable scalar initialized to zero. This tanh-gating mechanism ensures the model behaves identically to the frozen LLM at initialization, gradually incorporating visual signals during training. Ablation findings: gating contributed +8% to performance, and per-layer cross-attention (inserted at every layer, not just every other) was optimal.

**Training Data Strategy**: Flamingo was trained on M3W (Multi-Modal MassiveWeb), ~43M interleaved image-text web pages, plus image-text pairs and video-text pairs. The interleaved few-shot training data was critical — ablations showed it contributed +17% to performance.

**Key Limitation**: Flamingo's cross-attention layers introduce many new learnable parameters per LLM layer, requiring massive training data. Moreover, the Perceiver Resampler loses spatial information, degrading OCR performance.

#### 1.2 LLaVA (2023–2024): The Self-Attention Counter-Revolution

LLaVA (Liu et al., 2023, updated through LLaVA-1.5 and LLaVA-NeXT in 2024) took the opposite approach, proving that a simple MLP projector concatenating visual tokens into the LLM's text embedding space could work remarkably well:

```
Visual Tokens = MLP(CLIP_ViT(image))   # 576 tokens from 24×24 grid
Input = [Visual_Tokens, Text_Tokens]   # Concatenation
Output = LLM(Input)                    # Standard self-attention
```

LLaVA's key insight was that **data efficiency matters more than architectural sophistication** — a simple MLP projector trained on high-quality instruction data (558K + 665K examples) could outperform Flamingo's complex cross-attention design on many benchmarks. Training cost: ~108 GPU-hours for LLaVA-1.5-7B.

**Critical findings from 2024–2025 analysis of LLaVA-style architectures**:

**LLaViT** (Nov 2025) revealed that visual tokens at LLM input are poorly aligned with the text embedding space — cosine similarity between visual and word embeddings is ~0.1, and the LLM actively translates visual representations into text across its layers. LLaViT proposed three modifications: (a) **separate QKV projections for visual tokens** (copied from text QKV then independently trained, applied across all layers) to compartmentalize visual adaptation; (b) **bidirectional attention on visual tokens** — removing the causal mask when both query and key are visual tokens, since visual patches have no inherent sequential order; (c) multi-depth CLIP features. Ablation: removing visual attention degraded vision-centric performance by 14.4 points (3B model). A 3B LLaViT outperformed 7B LLaVA-1.5.

**Delta-LLaVA** (2025) proposed "alignment before interaction" — spatial downsampling (576→144 tokens) with a low-rank DeltaProjection (W_proj = W_base + UV^T), achieving 55% throughput increase and 81% FLOPs reduction while improving accuracy through more semantically rich aligned representations.

**SAISA / NAAViT** (Feb 2025, Chinese Academy of Sciences): The most provocative finding — **eliminating visual→visual self-attention entirely** (NAAViT: No Attention Among Visual Tokens) actually **improved** accuracy. SAISA aligns visual features directly into NAAViT self-attention block input spaces. Using the same LLaVA-1.5 config: 66% fewer inference FLOPs, 26% lower training budget, superior accuracy across benchmarks. This directly contradicts LLaViT's finding that visual attention is critical — the contradiction likely stems from how "no attention" is implemented and how features are aligned.

#### 1.3 BLIP-2 → BLIP-3 Evolution (Salesforce, 2023–2024)

**BLIP-2** (Li et al., 2023) introduced the **Q-Former** (Querying Transformer): a lightweight BERT-style transformer with learnable query tokens that cross-attend to frozen ViT features. Trained with three losses: Image-Text Matching (ITM), Image-Text Contrastive (ITC), and Language Modeling (LM). The Q-Former's output tokens were then concatenated with text tokens and fed into a frozen LLM.

**Q-Former Limitations** (revealed by DeCo, 2024):
- **"Double Abstraction"**: Q-Former abstracts visual patches into semantic concepts, then the LLM abstracts again from those concepts — redundant and harmful since LLMs can extract semantics from raw visual features
- **Training Difficulty**: Q-Former needs massive data to train effectively; under LLaVA-scale data (558K+665K), it underperforms an MLP
- **Spatial Information Loss**: Resampling shuffles token positions, destroying spatial layout critical for OCR
- **Token Redundancy**: Multiple query tokens often converge on the same visual region, wasting capacity
- **Apple MM1's Conclusion** (2024): with sufficient training resources, Q-Former shows no advantage over average pooling

**BLIP-3 / xGen-MM** (Aug 2024): Abandoned the Q-Former entirely. Replaced with a simpler **Perceiver Resampler** (still cross-attention based, but used purely for token compression with a single autoregressive loss — no ITM/ITC). The resampler has channel dimension 1152 and compresses visual tokens by ~5× (e.g., 729→128 tokens per image patch). Key improvements: supports multi-image inputs (Q-Former couldn't), uses any-resolution encoding with per-patch resampling, and trains end-to-end with a single objective. Model weights open-sourced under Phi-3-mini backbone.

#### 1.4 NVLM (NVIDIA, Sep 2024): Three-Way Architecture Comparison

NVIDIA's NVLM 1.0 (arXiv 2409.11402) performed the most systematic architecture comparison to date, training three variants on identical data:

| Architecture | Type | Training Throughput | Text-Only Performance | OCR Quality |
|---|---|---|---|---|
| **NVLM-D** | Decoder-only (LLaVA-style) | Baseline (1×) | +4.3 over backbone | Best |
| **NVLM-X** | Cross-attention (Flamingo-style) | ~3× higher | Guaranteed no degradation (LLM frozen) | Good (no Perceiver) |
| **NVLM-H** | **Hybrid** (novel) | Between D and X | Competitive | Best overall |

**NVLM-H Hybrid Design**: The thumbnail (global view) is concatenated with text tokens and processed through self-attention layers (enabling global multimodal reasoning). Image tiles with tile tags are fed through gated cross-attention layers (enabling efficient fine-grained detail processing). This balances reasoning capability with computational efficiency.

**Key finding contradicting prior belief**: Dataset quality and task diversity > scale, even during pretraining. Cross-attention models do NOT necessarily need massive noisy pretraining data if high-quality curated data is used — a finding that partially rehabilitates the cross-attention paradigm.

**1-D Tile-Tagging**: Text-based tile tags inserted before corresponding image tokens in the decoder — ablation showed this achieved best accuracy for both OCR and multimodal reasoning.

**Dynamic High-Resolution (DHR) Encoder**: InternViT-6B with dynamic aspect ratio matching (6 options), 448×448 non-overlapping tiles, pixel reshuffle reducing 32×32→16×16 patches (1024→256 tokens/tile, 4× dimension increase).

### 2. Spatial and Temporal Attention in Video Understanding

#### 2.1 TimeSformer (Bertasius et al., 2021): The Foundation

TimeSformer demonstrated that **divided space-time attention** is both efficient and effective for video understanding, establishing principles that later influenced video-LLM architectures:

**Divided Space-Time (divST)**: Within each transformer block, temporal attention (each patch attends to patches at the same spatial location across all frames) and spatial attention (each patch attends to other patches within the same frame) are applied separately in alternating fashion. This reduces complexity from O(T²S²) for joint space-time attention to O(T² + S²).

**Key Results** (Kinetics-400 top-1): Divided Space-Time 77.92% > Joint Space-Time 77.01% > Space-Only 76.93%. A crucial finding: **temporal attention is critical** for video understanding, and separating attention across dimensions is more efficient than computing joint attention.

**Scalability**: TimeSformer can process videos over one minute long (96+ frames), impossible for joint attention models.

#### 2.2 Video-LLaMA (Zhang et al., 2023): Q-Former for Video

Video-LLaMA adapted BLIP-2's Q-Former approach for video by adding temporal positional embeddings to frame-level ViT features before Q-Former aggregation. An Audio Q-Former branch (using ImageBind encoder) ran in parallel. Both produced "soft prompts" concatenated into the frozen LLaMA LLM.

**Critical Limitation**: The Q-Former's resampling shuffles token positions, destroying the spatial-temporal order that autoregressive LLMs depend on. As VideoLLaMA 2 authors noted: "Q-Former does not preserve spatial-temporal order in output visual tokens."

#### 2.3 VideoLLaMA 2 (Cheng et al., Jun 2024): STC Connector

VideoLLaMA 2 (DAMO Academy, Alibaba) replaced the Q-Former with a **Spatial-Temporal Convolution (STC) Connector** guided by three principles:

1. **Preserve spatial-temporal order** — avoid resampler architectures; use only convolution or pooling
2. **Reduce token count** — insert 3D downsampling operators for compression
3. **Alleviate information loss** — insert RegStage blocks (strong conv blocks from Radosavovic et al.) before and after downsampling

**STC Architecture**: `RegStage → 3D Conv Downsample (reduces T, H, W) → RegStage → Compact tokens`

**Experimental comparison** (average across MV-Bench, EgoSchema, ActivityNet-QA):
- 2D Pool: 43.0 avg score
- 3D Conv + RegStage (STC): **45.1 avg score** — using 50% fewer tokens (576 vs 1152)
- 3D designs consistently outperformed 2D on long-video understanding (EgoSchema), confirming that **early fusion of frame-level features** is critical

**Audio Branch**: BEATs encoder + MLP projection, jointly trained with video. Achieved SOTA on Clotho-AQA (60.6).

#### 2.4 Extreme Token Efficiency for Video

**LLaMA-VID** (2023): Dual-token design — 2 tokens per frame (context token + content token), enabling processing of hour-long videos.

**BLIP-3-Video / xGen-MM-Vid** (Oct 2024): Only 32 tokens per video using: Perceiver Resampler (cross-attention, 729→128 per frame), Spatio-temporal Attentional Pooling (TokenLearner with learnable soft-selection via MLP + softmax), and Token Turing Machines (4 transformer layers with persistent memory processing frames sequentially).

**DyCoke** (CVPR 2025): Training-free dynamic compression specifically for video — Temporal Token Merging (merges similar tokens across adjacent frames exploiting temporal redundancy) + Dynamic KV Cache Pruning (prunes low-attention tokens per decoding step with reactivation capability). Achieves 1.5× speedup and 1.4× memory reduction while **improving** performance.

### 3. Early Fusion vs. Late Fusion via Attention

#### 3.1 Theoretical Taxonomy

The traditional classification of fusion into early (input-level), middle (feature-level), and late (decision-level) is increasingly obsolete for transformer-based multimodal models, where attention mechanisms interweave representation learning, fusion, and decision-making. A 2024 ACM survey proposed a new taxonomy based on mechanism: encoder-decoder, attention mechanism, GNN, and generative methods.

However, the practical distinction remains useful for analyzing when cross-modal signals are injected:

| Fusion Strategy | When | Mechanism | Compute Cost | Cross-Modal Richness |
|---|---|---|---|---|
| **Early Fusion** | Input tokens | Concatenation + self-attention | O((V+T)²) | Highest (every token sees every other) |
| **Middle/Deep Fusion** | Intermediate layers | Gated cross-attention layers | O(V×T) per cross-attn layer | High (controlled by gating) |
| **Late Fusion** | Final layers / output | Separate branches → combine logits | O(T²) + O(V²) separate | Lowest (no token-level interaction) |
| **Hybrid Fusion** | Multiple levels | Thumbnail self-attn + tile cross-attn (NVLM-H) | Trades between extremes | Best balance |

#### 3.2 Empirical Comparisons

**Bamikole (2024 Master's Thesis, NCI)**: Compared three transformer fusion strategies on CMU-MOSI:
- Early concatenation: fastest training
- Cross-modal attention: smallest parameter size
- Hierarchical modal attention: **best performance** (lowest MAE: 0.0111, highest correlation: 0.5509) but largest and slowest

**Google/Lund University (2024)**: On VQA with T5+ViT — late fusion wins with moderate-to-large decoder sizes; early fusion only better for very small decoders; early fusion is consistently slower.

**NVIDIA NVLM (2024)**: The most comprehensive comparison — NVLM-D (early fusion via self-attention on concatenated tokens) achieved best overall benchmark scores but lowest training throughput; NVLM-X (middle fusion via cross-attention) achieved ~3× higher throughput but slightly lower OCR performance (no Perceiver Resampler preserved spatial info); NVLM-H (hybrid) achieved the best balance.

#### 3.3 The Convergence: When to Inject Cross-Modal Signals

The research consensus from 2024–2025 suggests:

- **For global reasoning and semantic understanding**: Concatenate a low-resolution global view with text tokens and use self-attention throughout. This enables rich cross-modal reasoning.
- **For fine-grained details (OCR, spatial relationships)**: Use cross-attention with high-resolution tile features, preserving spatial structure without exploding sequence length.
- **For efficiency**: Insert cross-attention at every LLM layer (Flamingo-style), not just selected layers — Flamingo's ablation showed per-layer was optimal.
- **For training data efficiency**: Use self-attention with simple MLP projection (LLaVA route) — fewer new parameters to train.
- **For guaranteed text-only performance preservation**: Use cross-attention with frozen LLM (NVLM-X route) — text capabilities cannot degrade.

### 4. Efficiency Challenges: Token Compression and Pruning

#### 4.1 The Core Problem

A 224×224 image processed by CLIP ViT-L/14 produces 256 patch tokens. High-resolution images can produce thousands. In self-attention architectures, this creates O(V²) self-attention overhead and FFN computation on all visual tokens. For video, the problem compounds: 32 frames × 256 tokens = 8,192 visual tokens.

#### 4.2 Architectural Compression Methods

**Q-Former** (BLIP-2, 2023): Learnable queries → cross-attention over visual tokens → fixed compressed representation. **Status**: Largely deprecated for images (DeCo 2024, MM1 2024) but still used for audio (Audio Q-Former, MMCE-Qformer) and some video applications.

**Perceiver Resampler** (Flamingo, 2022; BLIP-3, 2024): Similar cross-attention bottleneck but simpler — single autoregressive loss, no multi-objective training. Used in BLIP-3 at channel dim 1152, 5× compression ratio. More scalable than Q-Former but still requires training.

**2D Adaptive Average Pooling** (DeCo, PLLaVA, 2024): The surprising winner — a parameter-free spatial downsampling (e.g., `AdaptiveAvgPool2d(12,12)` reducing 24×24→12×12) outperforms learned Q-Former compression. DeCo showed +0.9% MLLM benchmarks, +7.1% visual localization, +2.9% VQA at the same 4:1 compression ratio. Key insight: **decouple compression from semantic abstraction** — let pooling do compression, let the LLM do semantics.

**STC Connector** (VideoLLaMA 2, 2024): RegStage conv blocks + 3D conv downsampling. Preserves spatial-temporal structure better than attention-based resampling.

**Delta-LLaVA Low-Rank Projection** (2025): W_proj = W_base + UV^T (low-rank), with Multi-Head Convolutional Attention (MHCA, O(N) via group depthwise convs) and Windowed Cross-Attention. 55% throughput increase, 81% FLOPs reduction.

**LLaVA-Meteor Top-Down Compression** (May 2025): Mamba-based SSM for O(N) global context propagation + dual-expert token scoring (visual saliency + task relevance) → Top-K selection. 75–95% compression.

#### 4.3 Training-Free Inference-Time Pruning

These methods require **no fine-tuning** — they are plug-and-play for any existing MLLM:

**FastV** (ECCV 2024 Oral, PKU): Discovered that visual tokens become highly inefficient in deep LLM layers. Prunes redundant visual tokens based on attention scores in early layers (layer 2+), dropping ~50% of tokens. Key finding: "An image is worth 1/2 tokens after layer 2."

**FasterVLM / VisPruner** (ICCV 2025): Uses [CLS] attention from the vision encoder (not text-visual cross-attention) to rank token importance before they enter the LLM. Key insight: text-visual attention suffers from "attention shift" and "attention dispersion" — it misaligns with actual visual token importance. CLS attention is more concentrated and reliable.

**AIM** (ICCV 2025): Two-stage: (a) iterative token merging based on embedding similarity before the LLM, (b) progressive token pruning within LLM layers based on multimodal importance. 7× FLOPs reduction, +4.6 on MLVU long video understanding.

**HoloV** (NeurIPS 2025): Critiques attention-based pruning — attention tends to over-retain local salient clusters, creating position bias. Uses semantic diversity (variance of intra-crop similarity) + CLS attention with adaptive per-crop token budgets. At 88.9% pruning, retains 95.8% of original performance, while FastV drops to 80–90%.

**iLLaVA** (Dec 2024): Gradually merges redundant tokens while recycling beneficial information from pruned tokens into survivors (avoids direct context loss). Nearly 2× throughput, 50% memory reduction, only 0.2–0.5% performance drop.

#### 4.4 Composite Attention: The EE-MLLM Approach

**EE-MLLM** (Aug 2024) proposed a **composite attention mechanism** that eliminates visual-token self-attention entirely without introducing extra parameters like Flamingo's cross-attention layers. It reuses existing LLM layer weights for vision-language alignment, achieving both data efficiency (like LLaVA) and compute efficiency (like Flamingo). Key result: prefill time 79ms vs. LLaVA's 277ms on H800 GPU — a 3.5× speedup with competitive benchmark performance.

### 5. Shared vs. Modality-Specific Attention Weights

#### 5.1 The Modality Bias Problem

A critical challenge in multimodal attention is **modality bias** — one modality (typically text) dominates attention regardless of its informativeness for the current input. **RollingQ** (ICML 2024) identified a self-reinforcing cycle: biased modality accumulates higher attention scores → receives more optimization gradient → further widens the distribution gap. They proposed an Attention Imbalance Rate (AIR) metric and a Rolling Query mechanism that rotates attention queries toward underutilized modalities.

#### 5.2 Design Approaches

**Fully Shared Attention** (LLaVA, NVLM-D): Visual and text tokens share the same QKV projections and attention mechanism. Simple, data-efficient, but no modality-specific processing and risks modality bias. The LLM "translates" visual representations into text concepts across layers (LLaViT finding).

**Separate QKV for Visual Tokens** (LLaViT, Nov 2025): Copies text QKV parameters into visual-specific parameters, trained during pretraining while text QKV stays frozen. Applied across all LLM layers. This compartmentalizes visual adaptation, preventing degradation of language capabilities. Query computation becomes modality-conditional.

**Gated Cross-Attention** (Flamingo, NVLM-X): Text tokens use dedicated cross-attention heads to attend to visual K/V, with learnable gating controlling how much visual signal to incorporate. Modality-specific in that visual features serve different roles (K/V only) than text (Q). LLM's original self-attention is unchanged.

**Dual-Transformer Architectures** (AEFNet, HBridge, 2025): Separate transformer branches for intra-modal (within-modality) processing + cross-modal (inter-modal) fusion transformers. HBridge keeps modality-specific shallow and deep layers separate, only bridging intermediate layers for semantic alignment — reducing attention sharing by >40%.

**Adaptive Gating Mechanisms**:

- **NeuroFusionX** (2025): Learns instance-wise gate vectors g ∈ (0,1)³ from fused state, dynamically down-weighting unreliable modalities per example. Trained with modality dropout (p=0.2).
- **Gated Recursive Fusion — GRF** (July 2025): Processes modalities sequentially through a recurrent pipeline with a shared multimodal context vector. Uses a GRU-inspired Gated Fusion Unit (GFU) to dynamically control retain/overwrite/blend decisions. Reduces fusion complexity from O(n²) to O(n).
- **RollingQ** (ICML 2024): Rotates attention queries toward anchors favoring underutilized modalities, preventing attention collapse.

**GeminiFusion** (ICML 2024): Combines intra-modal and inter-modal attention at pixel level with layer-adaptive noise for per-layer fusion control. Linear complexity O(n). SOTA on semantic segmentation (DELIVER 66.9 mIoU, SUN-RGBD 54.6 mIoU).

#### 5.3 Key Architectural Insight

The emerging best practice is **asymmetric sharing**: allow rich intra-modal processing (modality-specific attention or separate QKV projections in early and late layers) while enabling controlled cross-modal interaction (bridged intermediate layers or gated cross-attention). Full weight sharing (LLaVA baseline) is maximally data-efficient but suffers at extreme scale; fully separate branches (dual-transformers) are maximally expressive but parameter-heavy. HBridge's decoupled shallow/deep layers with bridged intermediates represents a promising middle ground.

---

## Important Papers & References

1. **Flamingo: a Visual Language Model for Few-Shot Learning** — Alayrac, Donahue, et al. (DeepMind), NeurIPS 2022. Pioneered gated cross-attention for MLLMs with frozen LLMs; introduced Perceiver Resampler and tanh-gating mechanism. Established the cross-attention paradigm.

2. **Visual Instruction Tuning (LLaVA)** — Liu et al. (UW-Madison, Microsoft), NeurIPS 2023. Demonstrated that a simple MLP projector + high-quality instruction data could match or exceed complex cross-attention designs. Sparked the self-attention paradigm for MLLMs.

3. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** — Li et al. (Salesforce), ICML 2023. Introduced Q-Former bridging frozen ViT and LLM with multi-objective training (ITM+ITC+LM). Established token-level fusion.

4. **xGen-MM (BLIP-3): A Family of Open Large Multimodal Models** — Xue et al. (Salesforce), arXiv 2408.08872, Aug 2024. Abandoned Q-Former for Perceiver Resampler with single autoregressive loss. Supports multi-image and interleaved inputs. Open-source under Phi-3-mini.

5. **NVLM: Open Frontier-Class Multimodal LLMs** — Dai et al. (NVIDIA), arXiv 2409.11402, Sep 2024. Most systematic architecture comparison to date: decoder-only (NVLM-D), cross-attention (NVLM-X), and novel hybrid (NVLM-H). Showed cross-attention has ~3× higher training throughput and dataset quality matters more than scale.

6. **DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models** — Yao et al. (Peking University), arXiv 2405.20985, 2024. Demonstrated Q-Former's "double abstraction" is redundant and harmful. Adaptive pooling + MLP outperforms Q-Former at same compression ratio with zero additional parameters.

7. **Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)** — Bertasius, Wang, Torresani (Facebook AI), ICML 2021. Proved divided space-time attention is more efficient and effective than joint attention for video. Foundation for video transformer architectures.

8. **VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs** — Cheng et al. (DAMO Academy, Alibaba), arXiv 2406.07476, Jun 2024. Replaced Q-Former with STC Connector (3D conv + RegStage) preserving spatial-temporal order. Joint video-audio training with BEATs encoder.

9. **FastV: An Image is Worth 1/2 Tokens After Layer 2** — Chen et al. (PKU), ECCV 2024 (Oral). Discovered visual tokens become inefficient in deep LLM layers. Training-free plug-and-play pruning achieving 50% token reduction with minimal performance loss.

10. **SAISA: Towards Multimodal Large Language Models with Both Training and Inference Efficiency** — Anonymous (Chinese Academy of Sciences), arXiv 2502.02458, Feb 2025. NAAViT eliminates visual-visual self-attention entirely and improves accuracy. 66% fewer FLOPs, 26% lower training budget.

11. **EE-MLLM: A Data-Efficient and Compute-Efficient Multimodal Large Language Model** — He et al., arXiv 2408.11795, Aug 2024. Composite attention mechanism reusing LLM weights for vision-language alignment with no extra parameters. 79ms prefill vs. 277ms for LLaVA.

12. **AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning** — Zhong et al., ICCV 2025. Two-stage training-free compression: iterative embedding-similarity merging + progressive pruning. 7× FLOPs reduction.

13. **DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models** — CVPR 2025. Temporal Token Merging + Dynamic KV Cache Pruning for video LLMs. 1.5× speedup with performance improvement.

14. **HoloV: Beyond Attention-Based Visual Token Pruning** — NeurIPS 2025. Semantic diversity-based pruning with adaptive per-crop budgets. At 88.9% pruning, retains 95.8% performance vs. 80–90% for FastV.

15. **GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer** — ICML 2024. Linear-complexity O(n) intra+inter-modal attention fusion with layer-adaptive noise. SOTA on multimodal segmentation.

16. **RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer** — ICML 2024. Identified attention collapse as self-reinforcing cycle; proposed Rolling Query mechanism with Attention Imbalance Rate metric.

17. **LLaViT: LLMs as Extended Vision Transformers** — arXiv 2511.10301, Nov 2025. Showed visual tokens are poorly aligned with text embedding space; proposed separate visual QKV projections + bidirectional visual attention.

18. **Perceiver IO: A General Architecture for Structured Inputs & Outputs** — Jaegle et al. (DeepMind), ICLR 2022. Introduced asymmetric cross-attention bottleneck enabling processing of arbitrary modalities with depth decoupled from input size. Foundation for Flamingo's Perceiver Resampler.

19. **Delta-LLaVA** — 2025. Low-rank DeltaProjection with Multi-Head Convolutional Attention achieving 55% throughput increase and 81% FLOPs reduction. "Alignment before interaction" principle.

20. **HBridge: Decoupled Shallow and Deep Layers for Multimodal Transformers** — Nov 2025. Asymmetric H-shaped architecture keeping modality-specific shallow/deep layers separate, bridging only intermediate layers. >40% reduction in attention sharing.

---

## Open Questions & Future Directions

### 1. The Visual Attention Necessity Paradox

The field faces a fundamental contradiction: LLaViT (Nov 2025) shows removing visual→visual self-attention causes a 14.4-point accuracy drop (3B model), while SAISA/NAAViT (Feb 2025) shows removing it **improves** accuracy. This likely stems from implementation differences — how features are aligned when attention is removed matters enormously. Resolving this paradox requires systematic ablations controlling for feature alignment quality, LLM backbone, training data, and task distribution. The answer may depend on whether "no visual attention" is compensated by stronger cross-modal alignment or richer per-token visual features.

### 2. Unified Architecture Search

No existing work systematically searches the full design space of: (a) which layers get cross-attention vs. self-attention, (b) what compression ratio per layer, (c) shared vs. separate QKV projections per modality per layer, (d) gating mechanism design, (e) token merging/pruning schedule. NVLM-H's hybrid design (thumbnail→self-attn, tiles→cross-attn) was hand-designed. Neural architecture search (NAS) over these decisions could discover more optimal configurations.

### 3. Beyond Static Fusion Strategies

Nearly all current methods use fixed fusion strategies — cross-attention at every layer, or self-attention on all tokens, or a static hybrid. The GRF approach (sequential recurrent fusion with learned gating) and RollingQ (dynamic query rotation) point toward **input-dependent dynamic routing** — where the model decides per-example which modalities to attend to, at which layers, and how strongly. This is underexplored.

### 4. Modality-Specific Scaling Laws

The optimal fusion strategy likely depends on the relative information density of modalities. For an image with dense spatial information (OCR, medical imaging), preserving token-level detail via cross-attention or high-resolution tiles is critical. For a simple icon or diagram, extreme compression may suffice. No current work systematically characterizes how fusion strategy optimality varies with input visual complexity.

### 5. The Role of Pretraining Data

NVLM-2024 showed cross-attention models don't need the massive noisy data Flamingo used — high-quality curated data suffices. But the interaction between architecture choice and pretraining data scale/quality remains poorly understood. Self-attention models (LLaVA) are proven data-efficient; cross-attention models might become equally data-efficient with better training recipes, or there may be a fundamental trade-off (more architectural parameters → more data needed).

### 6. Multimodal Attention for >2 Modalities

Most research focuses on vision+text. The VideoLLaMA 2 audio+video+text design and GeminiFusion's arbitrary-modal fusion (RGB+depth+LiDAR+event) are early steps. As models incorporate touch, proprioception, sensor data, and more, the O(n²) pairwise fusion problem becomes acute. GRF's O(n) recurrent fusion and GeminiFusion's O(n) pixel-wise fusion are promising directions.

### 7. Interpretable Cross-Modal Attention

The NOTICE paper (NAACL 2025) revealed mechanistic differences — BLIP's cross-attention heads implement object detection and image grounding, while LLaVA's self-attention heads only perform outlier suppression without grounding. Understanding exactly what cross-attention heads compute could inform better architectural designs. Current designs are empirically driven; mechanistic understanding could enable principled design.

### 8. Inference-Time Adaptation

All current token pruning methods (FastV, AIM, HoloV) are training-free and use heuristics (attention scores, embedding similarity, semantic diversity). Could a small learned router network predict which tokens to prune per example? This would add negligible overhead but could significantly improve pruning quality, especially for out-of-distribution inputs.

### 9. Video-Specific Efficiency

Video understanding amplifies all efficiency challenges (thousands of tokens per clip). Video-specific methods (DyCoke, BLIP-3-Video's 32-token representation, LLaMA-VID's 2-token-per-frame) are nascent. The temporal dimension offers unique compression opportunities (redundancy across frames) that image methods cannot exploit. Dedicated video fusion architectures remain underexplored relative to image-text models.

### 10. The Ultimate Minimal Architecture

The trajectory from Flamingo (complex cross-attention) → LLaVA (simple MLP) → SAISA (no visual attention) → EE-MLLM (zero extra parameters, composite attention) suggests a trend toward architectural minimalism. The limiting case — can we achieve competitive multimodal understanding with zero architectural modifications to a pretrained LLM, using only clever token representation? — is an open and provocative question.

---

## Relevance to Main Topic

Attention in multimodal and cross-modal architectures sits at the intersection of two of the most active research areas in AI: attention mechanism design (how should information flow between tokens?) and multimodal representation learning (how should different modalities interact?). The findings surveyed here have implications beyond vision-language models:

**For attention mechanism theory**: The discovery that visual→visual self-attention is highly redundant (NAAViT, FastV) challenges the assumption that more attention is always better. The attention mechanism's value depends on the information structure of the input — for modalities with high spatial redundancy (images), pruning attention can improve both efficiency and accuracy. This insight may generalize to other redundant-modality settings.

**For architecture design principles**: The decoupling principle proven by DeCo — "let compression do compression, let semantics do semantics" — is a general architectural lesson. Complex learned modules that conflate multiple functions (Q-Former's simultaneous compression and semantic abstraction) underperform simpler modules with clean functional separation. This principle applies broadly to neural architecture design.

**For efficiency research**: Training-free token pruning methods (FastV, AIM, HoloV) achieving 50–90% token reduction with <1% accuracy loss represent a major practical advance. These techniques can be applied to any deployed MLLM without retraining, making them immediately impactful for production systems.

**For the cross-attention vs. self-attention debate**: The 2024–2025 research has largely resolved this as a false dichotomy. The optimal approach is hybrid — using self-attention for global reasoning on compressed representations and cross-attention for efficient access to fine-grained details. NVLM-H's design and the broader trend toward adaptive, input-dependent fusion strategies suggest the field is converging on architectures that flexibly combine both mechanisms rather than choosing one.

**For the research question** — "What cross-attention fusion strategy achieves the best multimodal understanding with minimal computational overhead?" — the evidence points to a three-component strategy: (1) **hybrid self- + cross-attention** with a low-resolution global view processed via self-attention and high-resolution tiles accessed via gated cross-attention (NVLM-H pattern); (2) **training-free token compression** (FastV-style early-layer pruning or HoloV-style diversity-based selection) to reduce visual tokens by 50–90% before they reach compute-intensive deep layers; and (3) **simple compression connectors** (adaptive pooling or lightweight convolutions, not learned semantic compressors like Q-Former) to minimize training cost while preserving spatial information. This combination achieves near-SOTA accuracy with 3–7× computational reduction compared to naive self-attention over all visual tokens at all layers.

---

*Research conducted June 2026. Key sources include peer-reviewed publications from NeurIPS, ICML, ICLR, ECCV, ICCV, CVPR; arXiv preprints from 2024–2025; and technical reports from DeepMind, NVIDIA, Salesforce, Meta, Microsoft, Alibaba, and academic institutions.*
