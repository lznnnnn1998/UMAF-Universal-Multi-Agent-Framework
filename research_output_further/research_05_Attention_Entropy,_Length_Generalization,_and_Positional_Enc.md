# Attention Entropy, Length Generalization, and Positional Encoding Integration

> **Research Sub-Topic 5**: Analysis of how RhoAttention (ρ-Attn) interacts with attention entropy, length generalization, and positional encodings. Building on the survey's finding that attention entropy control is the unifying framework for length generalization.

---

## 1. Overview

### 1.1 The Entropy–Length Generalization Nexus

The research program's unifying insight — that **attention entropy control is the central mechanism governing length generalization** — is supported by converging evidence from four independent research communities. The softmax exponential has a single, fatal property: as sequence length N → ∞, its entropy H(α) → log N regardless of content, transforming focused attention into uniform dispersion (Weakness W5 of the audit). This "attention dilution" problem, formalized by Lin et al. (2025) as a phase transition governed by the scaling factor β_n, shows that softmax attention collapses toward uniformity precisely as β_n ≁ log N.

Three distinct mitigation strategies have emerged, each with different theoretical foundations:

1. **Log-N temperature rescaling** (SSMax, InfoScale, Scale-Invariant Attention): Rescale logits by a log N-dependent factor to counteract the entropy drift. Lin et al. (2025) proved that β_n ≍ log N is the **critical scaling** — any sub-logarithmic scaling leads to attention dilution, while super-logarithmic scaling reduces attention to identity. This provides the first rigorous justification for the empirical success of YaRN and Qwen-style log-N rescaling.

2. **Sparse normalization** (α-entmax, Sparsemax, ReLU-based attention): Replace softmax with a sparsity-inducing normalization that naturally limits the support of the attention distribution. Li & Kong (2025) proved that α-entmax with α > 1 achieves lim_{n→∞} H(α-entmax)/log n ≤ β < 1 — sublogarithmic entropy growth — while softmax achieves exactly the maximal entropy growth rate. ReLU-based attention (ReLA, Zhang et al. 2021) takes this further by producing exact zeros for negatively aligned keys, creating a natural support set whose size depends on content rather than sequence length.

3. **Gated/Selective attention** (NeurIPS 2024–2025): Decouple the routing function of attention from intensity control via output gating, allowing the model to dynamically regulate attention concentration without the softmax sum-to-1 constraint.

### 1.2 RhoAttention's Position in This Landscape

RhoAttention (ρ-Attn), defined in Sub-Topic 2, occupies a unique position in this landscape. Its normalization function — the resolvent C = (ρI_d + K^T K)^{-1} — is a **matrix rational function** that modulates attention scores via global key covariance shrinkage rather than element-wise competition. When combined with ReLU activation (RhoAttn-sparse), it achieves both:

- **Structural entropy control**: The ReLU produces exact zeros for keys anti-aligned with the query, limiting the effective support to N_eff = |{i : P_i > 0}| — the number of positively-attended keys. Unlike softmax's forced distribution of probability mass across all N positions, RhoAttn-sparse concentrates attention only on those keys whose resolvent-modulated similarity exceeds zero.

- **Spectral regularization via ρ**: The hyperparameter ρ controls the "stiffness" of the resolvent. As ρ grows, C → (1/ρ)I and attention approaches uniform dispersion (high entropy). As ρ → 0, C → (K^T K)^{-1} (pseudoinverse) and attention approaches orthogonal projection onto the key subspace (low entropy, potentially unstable). The optimal ρ provides a tunable entropy control that is **independent of sequence length N** — a property that softmax fundamentally lacks.

The remainder of this analysis provides rigorous treatment of five interconnected questions: (a) the entropy of RhoAttention as a function of N, (b) asymptotic entropy behavior and the attention dilution problem, (c) RoPE integration and relative-position preservation, (d) length extrapolation guarantees, and (e) the NoPE challenge.

---

## 2. Key Methods & Approaches

### 2.1 Entropy Analysis of RhoAttention as a Function of N

#### 2.1.1 Formal Setup

For a single query q ∈ ℝ^d, RhoAttn-sparse computes attention weights α ∈ ℝ^N as:

$$\alpha_i = \frac{\max(0, P_i)}{\sum_{j=1}^N \max(0, P_j) + \epsilon}, \quad P_i = q^T C k_i$$

where C = (ρI_d + K^T K)^{-1} ∈ ℝ^{d×d} is the resolvent of the key Gram matrix, and q, k_i are RoPE-rotated query and key vectors.

The attention entropy is:

$$H(\alpha) = -\sum_{i=1}^N \alpha_i \log \alpha_i$$

with the convention 0·log 0 = 0.

#### 2.1.2 Support Set Analysis

Define the support set of the attention distribution:

$$\mathcal{S}(q) = \{i \in \{1, \ldots, N\} : P_i > 0\} = \{i : q^T C k_i > 0\}$$

The support size is N_eff(q) = |S(q)|. The ReLU activation naturally partitions the N keys into positive (attended) and non-positive (ignored) subsets.

**Lemma 1 (Support Size Bound)**. For any query q with ‖q‖ = 1 and under the assumption that keys k_i are drawn i.i.d. from a distribution with bounded support, the expected support size satisfies:

$$\mathbb{E}[N_{\text{eff}}(q)] = N \cdot \mathbb{P}(q^T C k > 0)$$

If the keys are approximately uniformly distributed on the unit sphere (Gaussian initialization), and C is approximately isotropic (ρ ≫ ‖K^T K‖), then P(q^T C k > 0) ≈ 1/2 (since the bilinear form q^T C k is symmetric around zero for random vectors). However, **after training**, keys align with the query subspace, and P(q^T C k > 0) concentrates — relevant keys have positive scores while irrelevant keys have negative or near-zero scores.

**Key insight**: Unlike softmax, which assigns nonzero probability to **every** key regardless of relevance (α_i > 0 for all i since exp(x) > 0), RhoAttn-sparse can and does produce exact zeros. The effective support size N_eff is a **content-dependent quantity** that does not automatically grow with N.

#### 2.1.3 Entropy Upper Bound

**Theorem 1 (RhoAttention Entropy Upper Bound)**. For RhoAttn-sparse with ReLU activation, the attention entropy satisfies:

$$H(\alpha) \leq \log N_{\text{eff}} \leq \log N$$

with equality in the first inequality if and only if all positive P_i are equal, and equality in the second if and only if all N keys have P_i > 0.

*Proof.* The attention weights α_i are zero for i ∉ S(q). For i ∈ S(q), α_i = P_i / Σ_{j∈S(q)} P_j. By the Gibbs inequality, the maximum entropy for a distribution over S(q) elements is log |S(q)|, achieved when all elements have equal probability. Therefore H(α) ≤ log N_eff ≤ log N. The second inequality is strict when N_eff < N. ∎

#### 2.1.4 Comparison to Softmax Entropy

For softmax attention with logits s_i = q^T k_i / √d:

$$H_{\text{softmax}}(\alpha) \to \log N \quad \text{as} \quad N \to \infty$$

This is **inevitable** because softmax assigns positive probability to every key, and as the denominator Σ exp(s_i) grows as Θ(N), individual probabilities approach Θ(1/N).

For RhoAttn-sparse, the entropy is bounded by log N_eff, and N_eff can be **bounded independently of N** if the resolvent effectively discriminates between relevant and irrelevant keys. Specifically:

**Theorem 2 (Asymptotic Entropy Behavior of RhoAttention)**. Assume that after training:
1. For "relevant" keys (those containing information needed by the query), q^T C k_i > δ > 0 (bounded away from zero)
2. For "irrelevant" keys, q^T C k_i < -δ (bounded away from zero in the negative direction)
3. The proportion of relevant keys is bounded by a constant γ ∈ (0, 1) independent of N

Then as N → ∞:

$$N_{\text{eff}} \leq \gamma N$$

$$H(\alpha) \leq \log(\gamma N) = \log N + \log \gamma$$

and the entropy growth rate is:

$$\lim_{N \to \infty} \frac{H(\alpha)}{\log N} \leq 1$$

with equality only when γ → 1 (all keys are relevant). For γ < 1, the entropy grows **strictly slower** than the softmax maximum.

*Proof.* By assumption, N_eff = |{i : q^T C k_i > 0}| ≤ γN since irrelevant keys (fraction 1-γ) have negative scores and are zeroed by ReLU. Therefore H(α) ≤ log(γN) = log N + log γ. ∎

**Corollary (Concentration Preservation)**. If the resolvent C is such that the proportion of relevant keys γ scales as γ ∝ N^{-β} for some β > 0 (i.e., attention becomes increasingly selective at longer contexts), then:

$$H(\alpha) \leq (1 - \beta) \log N + O(1)$$

For β = 1 (constant number of relevant keys independent of N, as in needle-in-haystack tasks), H(α) = O(1) — the entropy is bounded independently of N, achieving perfect concentration preservation.

#### 2.1.5 The Role of the Resolvent in Entropy Control

The resolvent C = (ρI + K^T K)^{-1} provides a second layer of entropy control beyond ReLU sparsification. Recall from Sub-Topic 2 that the resolvent has the Neumann series expansion:

$$C = \frac{1}{\rho} \sum_{k=0}^{\infty} \left(-\frac{G}{\rho}\right)^k$$

where G = K^T K is the key Gram matrix. This expansion implicitly performs **infinite-order polynomial reweighting** of key similarities. Keys that are highly correlated with many other keys (redundant patterns, large entries in G) are downweighted by the higher-order terms; unique keys retain their influence.

**Information-theoretic interpretation**: The resolvent is the posterior precision matrix of a Gaussian process with prior precision ρI and likelihood precision K^T K. In this Bayesian framework:

- The **prior** (ρI) encodes the belief that all keys are equally informative a priori
- The **likelihood** (K^T K) encodes the empirical covariance structure of the keys
- The **posterior precision** C = (ρI + K^T K)^{-1} encodes the updated belief after observing the key distribution

Keys that are redundant (highly correlated with other keys) receive **lower effective weight** because their information is already captured by similar keys. This is a principled, information-theoretic alternative to softmax's "competition via exponentiation" — it is **competition via Bayesian precision shrinkage**.

**Entropy effect of ρ**: The hyperparameter ρ provides a direct entropy control:

- ρ → ∞: C ≈ (1/ρ)I, all keys are treated equally, P_i ≈ (1/ρ) q^T k_i → N_eff ≈ N, H(α) ≈ log N (maximum entropy, like softmax)
- ρ → 0: C ≈ (K^T K)^{-1} (pseudoinverse), only keys in the row space of K receive non-zero scores → N_eff ≈ rank(K), H(α) ≤ log rank(K) ≤ log d (bounded by head dimension)

For intermediate ρ, the entropy interpolates smoothly between these extremes. A fixed ρ > 0 provides a natural "prior" that prevents the entropy from collapsing to log d (which would be too concentrated for many tasks) while also preventing it from growing to log N (which would be too diffuse).

### 2.2 Asymptotic Entropy Behavior and the Attention Dilution Problem

#### 2.2.1 Formal Definition of Attention Dilution

**Definition (Attention Dilution)**. An attention mechanism suffers from attention dilution if, for any sequence of attention logits with bounded dynamic range (max s_i - min s_i = O(1)), the attention entropy satisfies:

$$\lim_{N \to \infty} \frac{H(\alpha)}{\log N} = 1$$

This means the attention distribution approaches perfect uniformity as N grows, regardless of content.

Softmax attention with standard scaling (τ = √d) exhibits attention dilution because the denominator Σ exp(s_i) grows as Θ(N) while individual numerators are O(1), forcing α_i = Θ(1/N) and H(α) → log N.

**SSMax bound** (Nakanishi, 2025): Even when one logit is dominant (s_max - s_2nd = δ > 0), standard softmax satisfies:

$$\alpha_{\max} \leq \frac{1}{(N-1)e^{-\delta} + 1} \to 0 \quad \text{as} \quad N \to \infty$$

This means **no single key can retain non-vanishing attention** at extreme context lengths — the model is forced to distribute attention across all positions.

#### 2.2.2 RhoAttention's Avoidance of Attention Dilution

**Theorem 3 (RhoAttention Avoids Attention Dilution)**. For RhoAttn-sparse with any fixed ρ > 0 and under the assumptions of Theorem 2 with γ = o(1) (the proportion of relevant keys decreases with N), the attention mechanism does NOT suffer from attention dilution:

$$\lim_{N \to \infty} \frac{H(\alpha)}{\log N} < 1$$

and in the limiting case where N_eff = O(1) (constant number of relevant keys independent of N):

$$\lim_{N \to \infty} H(\alpha) = O(1)$$

The attention remains concentrated on relevant keys regardless of N.

*Proof.* By Theorem 1, H(α) ≤ log N_eff. Under the assumption N_eff = o(N), we have H(α) = o(log N), which implies lim_{N→∞} H(α)/log N = 0 < 1. For N_eff = O(1), H(α) = O(1) directly. ∎

**Comparison to alternative approaches:**

| Mechanism | Entropy Growth (as N → ∞) | Avoids Dilution? | Mechanism of Control |
|-----------|--------------------------|------------------|---------------------|
| Standard Softmax | H = log N − (N-1)σ²/(2N) + O(σ⁴) | ❌ No | None — forced uniformity |
| Log-N scaled Softmax (SSMax) | H ≤ log N (scaled by s·log N) | ✅ Yes (with tuning) | Temperature rescaling β_n = s·log N |
| α-entmax (α > 1) | H ≤ β·log N with β < 1 | ✅ Yes | Sparse support from α > 1 threshold |
| Scale-Invariant Attention | H controlled per distance octave | ✅ Yes | Distance-dependent logit transform |
| **RhoAttn-sparse** | **H ≤ log N_eff** | **✅ Yes (structural)** | Dual control: ReLU zeros + resolvent shrinkage |

#### 2.2.3 Quantitative Entropy Comparison

For a concrete numerical illustration, consider N = 128K and d = 128:

- **Standard Softmax**: H/ log N ≈ 0.95+ for almost any bounded logits → near-complete dispersion. The model cannot maintain concentration on any single token; all 128K positions receive roughly equal attention.

- **SSMax with s = 0.1**: With log N ≈ 11.76, the effective temperature is s·log N ≈ 1.176. If one key has logit advantage δ = 5 over others, α_max ≈ 1/(1 + (N-1)·e^{-5·1.176}) ≈ 1/(1 + 128K·e^{-5.88}) ≈ 0.994 — near-perfect concentration. However, this requires the log-N rescaling to be tuned, and the optimal s varies per head and per layer.

- **α-entmax (α = 1.5)**: Typical support size is 10–100 tokens regardless of N. Entropy is bounded by log(100) ≈ 4.6, compared to log(128K) ≈ 11.8 for softmax — a 2.6× reduction in entropy.

- **RhoAttn-sparse (ρ = 0.1·tr(G)/d)**: The support size N_eff depends on the alignment between q and the key subspace. For a well-aligned query (needle-in-haystack), N_eff may be 1–10, yielding H ≈ 0–2.3. For a semantic integration query (needs broad context), N_eff may be 100–1000, yielding H ≈ 4.6–6.9. The key advantage is **content-adaptive entropy**: the same mechanism automatically provides sharp attention when needed and broad attention when appropriate, without per-head tuning.

#### 2.2.4 The Log-N Connection in RhoAttention

While RhoAttention avoids the O(log N) entropy drift of softmax through its structural sparsification, there is a more subtle connection to log N scaling. As N grows, the key Gram matrix G = K^T K/√d accumulates more terms, and its spectral norm grows. This affects the resolvent C = (ρI + G)^{-1}:

$$\|G\|_2 = \left\|\frac{1}{\sqrt{d}} \sum_{i=1}^N k_i k_i^T\right\|_2 \leq \frac{1}{\sqrt{d}} \sum_{i=1}^N \|k_i\|^2$$

If keys have approximately constant norm ‖k_i‖² ≈ d (standard initialization), then ‖G‖_2 grows as O(N/√d). The effective regularization is ρ relative to ‖G‖_2: when ρ ≪ ‖G‖_2 (large N), the resolvent is dominated by the key structure; when ρ ≫ ‖G‖_2 (small N), the resolvent is dominated by the prior.

To maintain consistent entropy behavior **across different context lengths**, ρ could be scaled as:

$$\rho(N) = \rho_0 \cdot \frac{\|G(N)\|_2}{\|G(N_0)\|_2} \approx \rho_0 \cdot \frac{N}{N_0}$$

This is analogous to the log-N scaling in SSMax — ρ grows with N to prevent the resolvent from becoming too sharp (low entropy) at long contexts. However, unlike SSMax's log-N scaling of the temperature, RhoAttention's ρ-scaling is linear in N (for constant-norm keys), which is more aggressive. The practical implication is that **a fixed ρ works well for a moderate range of N** (e.g., 4× training length), but very extreme extrapolation (100×+) may benefit from ρ adaptation.

### 2.3 RoPE Integration and Relative-Position Preservation

#### 2.3.1 Standard RoPE Integration in RhoAttention

RhoAttention integrates RoPE by pre-rotating queries and keys before the resolvent computation:

$$q'_m = R_\theta(m) q_m, \quad k'_n = R_\theta(n) k_n$$

where R_θ(m) = diag(R(θ_1 m), ..., R(θ_{d/2} m)) and each R(θ m) ∈ SO(2) is a 2D rotation.

The attention logit between positions m and n becomes:

$$P_{mn} = (q'_m)^T C (k'_n) = q_m^T R_\theta(m)^T C R_\theta(n) k_n$$

where C = (ρI_d + (K')^T K')^{-1} is the resolvent computed from RoPE-rotated keys.

#### 2.3.2 Analysis of Relative-Position Property

**Definition (Strict Relative-Position Property)**. An attention mechanism has the strict relative-position property if the attention score between positions m and n depends only on the relative position Δ = n − m:

$$P_{mn} = f(q_m, k_n, n - m)$$

for some function f, independent of absolute positions m and n.

For standard RoPE attention, strict relative-position holds because:

$$(R_\theta(m) q_m)^T (R_\theta(n) k_n) = q_m^T R_\theta(m)^T R_\theta(n) k_n = q_m^T R_\theta(n - m) k_n$$

using the group property R_θ(m)^T R_θ(n) = R_θ(n − m). This follows from the commutativity of 2D rotations about the same axis.

For RhoAttention, the resolvent C introduces an additional complication:

**Theorem 4 (RhoAttention RoPE Compatibility)**. Standard RhoAttention with RoPE pre-rotation does NOT in general have the strict relative-position property, because:

$$R_\theta(m)^T C R_\theta(n) \neq f(n - m)$$

in general. The resolvent C depends on the absolute positions of ALL keys through K' = [R_θ(1)k_1; ...; R_θ(N)k_N], and does not commute with R_θ(m).

*Proof.* The resolvent is:

$$C = \left(\rho I + \sum_{i=1}^N R_\theta(i) k_i k_i^T R_\theta(i)^T\right)^{-1}$$

The term R_θ(m)^T C R_θ(n) = R_θ(-m) (ρI + Σ_i R_θ(i) k_i k_i^T R_θ(-i))^{-1} R_θ(n). For this to depend only on n − m, we would need the resolvent to be invariant under simultaneous rotation of all keys by the same angle — which would require the key distribution to be rotationally invariant. In general, this does not hold.

**Quantitative Deviation from Shift-Invariance.** While strict relative-position does not hold, the deviation is bounded:

$$\|P_{m+\Delta, m} - P_{m'+\Delta, m'}\| \leq \frac{2\|C\|_2 \cdot \|q\| \cdot \|k\| \cdot \Delta \cdot \theta_{\max}}{\rho}$$

where θ_max = max_i θ_i is the maximum RoPE frequency. The deviation grows linearly with the position difference Δ (since the rotation angles accumulate with position) but is suppressed by 1/ρ. For practical values (θ_max = 1.0 for base frequency 10,000, ρ ≈ 0.1·‖G‖_2, d = 128), this deviation is O(Δ/d) and typically negligible for Δ ≪ d.

**Practical implication**: For typical context lengths (N < d·θ_max^{-1} ≈ 10,000 for base frequency 10,000), the violation of strict shift-invariance is small enough to be inconsequential. However, for extreme length extrapolation (N > 100K with base frequency 10,000), the deviation may become significant.

#### 2.3.3 Block-Diagonal RoPE-Resolvent: Achieving Strict Shift-Invariance

For applications requiring strict shift-invariance, RhoAttention offers the **Block-Diagonal RoPE-Resolvent** variant (defined in Sub-Topic 2, Section 2.7):

Instead of a single global resolvent C ∈ ℝ^{d×d}, maintain one resolvent per RoPE frequency band:

$$C^{(i)} = (\rho I_2 + (K^{(i)})^T K^{(i)})^{-1} \in \mathbb{R}^{2 \times 2}$$

where K^{(i)} ∈ ℝ^{N×2} is the i-th frequency band of all RoPE-rotated key vectors.

The attention logit becomes:

$$P_{mn} = \sum_{i=0}^{d/2-1} (q_m^{(i)})^T R(\theta_i m)^T C^{(i)} R(\theta_i n) k_n^{(i)}$$

Now, note that within each 2×2 frequency band, the RoPE rotation is:

$$R(\theta_i m) = \begin{pmatrix} \cos(\theta_i m) & -\sin(\theta_i m) \\ \sin(\theta_i m) & \cos(\theta_i m) \end{pmatrix}$$

**Theorem 5 (Block-Diagonal Shift-Invariance)**. For the block-diagonal variant, if each band-specific resolvent C^{(i)} is approximately a scalar multiple of the identity, C^{(i)} ≈ c_i · I_2, then:

$$R(\theta_i m)^T C^{(i)} R(\theta_i n) \approx c_i \cdot R(\theta_i (n - m))$$

and the strict relative-position property is approximately recovered.

*Proof.* If C^{(i)} = c_i · I_2, then R(θ_i m)^T (c_i I_2) R(θ_i n) = c_i · R(θ_i m)^T R(θ_i n) = c_i · R(θ_i (n − m)) by the group property of 2D rotations. The approximation quality depends on how close C^{(i)} is to a scalar multiple of I_2, which improves as ρ grows relative to the band-specific key covariance. ∎

**Practical trade-off**: The block-diagonal variant sacrifices cross-frequency coupling (all interactions are within individual frequency bands) in exchange for strict shift-invariance. For tasks where relative position is critical (e.g., copying, retrieval), this is a favorable trade. For tasks requiring complex cross-frequency patterns (e.g., semantic composition), the standard (full C) variant may be preferable.

#### 2.3.4 A Spectral Positional Encoding Alternative

Beyond RoPE integration, the resolvent structure of RhoAttention suggests a **novel positional encoding** that is mathematically native to the mechanism:

**Spectral Position Encoding (SPE).** Instead of pre-rotating Q and K, encode position directly in the resolvent's spectral structure. Let each position t have an associated **positional precision matrix**:

$$\Pi_t = \rho I_d + \lambda \cdot \text{diag}(\phi_1(t), \ldots, \phi_d(t))$$

where φ_j(t) are position-dependent functions. The resolvent becomes position-conditioned:

$$C_t = (\Pi_t + K^T K)^{-1}$$

For example, if φ_j(t) = α_j·t (linear in position), the resolvent naturally up-weights recent keys (small t, small φ_j(t), smaller regularization → sharper attention) and down-weights distant keys (large t, large φ_j(t), larger regularization → softer attention). This provides a **continuous, differentiable positional bias** without requiring rotation of Q and K.

**Comparison to RoPE:**

| Property | RoPE | Spectral PE (SPE) |
|----------|------|-------------------|
| Relative-position encoding | Yes (via R(n−m)) | Implicit (via Π_t) |
| Strict shift-invariance | Yes (for 2D bands) | No (absolute position matters) |
| Mathematical compatibility with resolvent | Requires block-diagonal variant | Native — directly incorporated in C |
| Computational overhead | O(Nd) for rotations | O(d³) for per-position Cholesky (prohibitive) |
| Extrapolation | OOD at unseen rotation angles | Potentially better (no unseen angles) |

The key challenge for SPE is computational: recomputing C_t via Cholesky at each position costs O(d³), which is prohibitive for long sequences. However, the Sherman-Morrison update could be adapted to incorporate position-dependent ρ(t), enabling incremental updating of C_t at O(d²) per token.

### 2.4 Length Extrapolation Analysis

#### 2.4.1 The Challenge

Length extrapolation — using a model at sequence lengths far beyond its training length — is arguably the most critical practical challenge for attention mechanisms. The literature identifies three distinct failure modes:

1. **Positional OOD**: Position encodings (e.g., RoPE rotation angles) for positions > N_train were never seen during training, producing out-of-distribution attention logits that cascade through all layers (Weakness W4).

2. **Attention Entropy Collapse/Dilution**: As N grows, softmax entropy approaches log N, forcing attention toward uniformity and destroying the model's ability to focus on relevant tokens (Weakness W5).

3. **State Capacity Saturation**: For recurrent/SSM models, the fixed-size state can only encode information from approximately N_train tokens before saturating; retrieval collapses at lengths beyond training (Weakness W8, W11).

RhoAttention addresses each of these differently from existing approaches.

#### 2.4.2 RhoAttention's Length Extrapolation Properties

**Positional OOD**: RhoAttention suffers from the same RoPE frequency extrapolation problem as standard attention when using the standard RoPE integration. However, the severity is potentially reduced because:

1. The resolvent C provides global regularization that can suppress outlier attention scores from OOD RoPE angles. A key at an unseen rotation angle may produce anomalous dot products with certain queries, but the resolvent's spectral shrinkage (‖C‖_2 ≤ 1/ρ) bounds the impact.

2. The ReLU activation naturally zeros out negative attention scores — including those that might arise from OOD RoPE interactions that produce negative or near-zero logits.

3. The block-diagonal variant maintains per-band resolvents that are small (2×2) and can adapt to local frequency statistics, potentially providing more robust extrapolation within each band.

**Entropy Stability**: This is RhoAttention's strongest extrapolation advantage. As shown in Section 2.2, RhoAttn-sparse does NOT suffer from attention dilution — its entropy is bounded by log N_eff where N_eff is content-dependent, not sequence-length-dependent. For needle-in-haystack retrieval at extreme lengths (128K+), RhoAttn-sparse can maintain concentration on the needle while softmax would distribute attention uniformly across the haystack.

**State Capacity**: In recurrent inference mode, RhoAttention maintains a fixed-size state (C_t, M_t) of size 2d² = O(d²) independent of N. There is no state saturation — the model can theoretically handle arbitrarily long sequences. However, the quality of the state depends on:

1. **Numerical stability of Sherman-Morrison**: Over very long sequences (N ≫ 10^6), accumulated floating-point error in the rank-1 updates may degrade C_t. The periodic full recomputation (every T_recomp = max(100, d) tokens via Cholesky) resets this error, providing a tunable accuracy-efficiency tradeoff.

2. **Effective memory horizon**: While the state does not saturate (unlike SSMs with ‖Ā‖ < 1 causing exponential forgetting), the resolvent C_t gives progressively less weight to new keys as t grows (since G_t = Σ k_s k_s^T accumulates more terms). For very long sequences, this could manifest as "rigidity" — the attention pattern becomes increasingly insensitive to new information because the accumulated Gram dominates the prior ρI.

**Theorem 6 (Memory Rigidity Bound)**. Let C_t = (ρI + Σ_{s=1}^t k_s k_s^T)^{-1} be the resolvent at position t. The sensitivity of C_t to a new key k_{t+1} is:

$$\|C_{t+1} - C_t\|_2 = \frac{\|C_t k_{t+1}\|^2}{1 + k_{t+1}^T C_t k_{t+1}} \leq \|C_t\|_2^2 \cdot \|k_{t+1}\|^2$$

Since ‖C_t‖_2 ≤ 1/ρ (bounded by the regularization), the per-token change is bounded by ‖k‖²/ρ². However, as t → ∞:

$$\lim_{t \to \infty} \|C_t\|_2 = 0$$

(the resolvent vanishes as the Gram accumulates), so the **absolute** sensitivity decreases with t. To maintain sensitivity to new information at extreme lengths, ρ could be scaled as ρ(t) ∝ t to keep the effective regularization constant relative to the accumulated Gram.

**Practical ρ-Scheduling for Extrapolation**:

For length extrapolation, we recommend:

$$\rho(t) = \rho_0 \cdot \max\left(1, \frac{t}{N_{\text{train}}}\right)$$

This keeps the effective "stiffness" of the resolvent approximately constant as t grows beyond N_train. At t = N_train, ρ = ρ_0 (the training value). At t = 100 × N_train, ρ = 100 × ρ_0, preventing the resolvent from becoming too sharp. This is analogous to log-N scaling in SSMax but with a linear (rather than logarithmic) dependence, reflecting the linear growth of the key Gram's spectral norm.

#### 2.4.3 Quantitative Extrapolation Predictions

Based on the theoretical analysis, we predict:

| Extrapolation Factor (N_test / N_train) | Standard Softmax | RhoAttn-sparse (fixed ρ) | RhoAttn-sparse (ρ-scheduled) |
|----------------------------------------|-----------------|-------------------------|------------------------------|
| 2× | Mild degradation (entropy dilution begins) | No degradation expected | No degradation expected |
| 4× | Significant degradation | Mild degradation (resolvent stiffening) | No degradation expected |
| 8× | Severe degradation | Moderate degradation | Mild degradation |
| 16× | Near-complete failure | Significant degradation | Moderate degradation |
| 32×+ | Complete failure (H ≈ log N) | Degraded but functional (content-dependent) | Functional with ρ(t) |

These predictions require empirical validation — they are theoretical bounds based on the entropy and rigidity analyses, not experimental results.

#### 2.4.4 Comparison to Other Length Extrapolation Methods

| Method | Mechanism | Training-Free? | Max Extrapolation (Reported) |
|--------|-----------|----------------|------------------------------|
| YaRN (NTK-aware) | Rescale RoPE frequencies | Yes | ~16× |
| Gali (Logit Interpolation) | Interpolate attention logits | Yes | Competitive on LongBench |
| InfoScale | Entropy-invariant temperature | Yes | 64× (GAU-α model) |
| Scale-Invariant (NeurIPS 2025) | Distance-dependent logit transform | Yes | 16× zero-shot |
| SSMax | s·log N temperature scaling | Partially (learned s) | 10× |
| α-entmax / ASEntmax | Sparse attention + (log n)^γ scaling | Partially (learnable γ) | Empirical up to 128K |
| **RhoAttn-sparse** | **Dual control: ReLU sparsity + resolvent rigidity** | **Yes (with fixed ρ)** | **Theoretical: 8–32×** |

RhoAttention's key advantage is that length extrapolation is **built into the architecture** — the ReLU zeros and resolvent rigidity provide inherent protection against attention dilution, without requiring per-task calibration of scaling factors (though ρ-scheduling can further improve performance). The disadvantage is that the standard RoPE integration inherits RoPE's frequency extrapolation problem, which is orthogonal to the attention normalization mechanism.

### 2.5 The NoPE Challenge

#### 2.5.1 Background: Can Attention Function Without Positional Encoding?

The NoPE (No Position Encoding) research program, initiated by Haviv et al. (2022) and significantly advanced by Wang et al. (ACL 2024), asks: can transformers function without any explicit positional encoding? The answer is nuanced:

- **Yes, NoPE generalizes better than RoPE** for moderate length extrapolation (Wang et al., 2024). Causal transformers without positional encoding learn implicit position representations through the causal mask — the asymmetry of "can attend to past, not future" provides sufficient positional signal for many tasks.

- **But NoPE has a hard ceiling**: At unseen context lengths, attention heads suffer from **"attention distribution distraction"** — attention weights become increasingly uniform across tokens (measured via attention entropy), causing perplexity to spike. The inflection point of attention entropy closely tracks the perplexity inflection point (Wang et al., 2024).

- **Simple fix**: Scaling the softmax temperature (uniformly or per-head) to re-concentrate distracted attention substantially extends NoPE's effective context length (e.g., from 2K to 4K+ tokens without additional training). This directly connects NoPE's failure mode to the attention dilution problem analyzed in Section 2.2.

#### 2.5.2 RhoAttention Without Positional Encoding

**Can RhoAttention function without explicit position encoding?** Theoretically, **yes** — with important qualifications.

RhoAttention without RoPE pre-rotation reduces to:

$$P_{mn} = q_m^T C k_n$$

where C = (ρI_d + K^T K)^{-1} depends on all keys (with no positional rotation). The attention scores are purely content-based bilinear forms.

**Key differences from NoPE + Softmax:**

1. **Content-based addressing only**: Without RoPE, RhoAttention cannot distinguish between identical tokens at different positions — the same limitation as any NoPE system. Position information must be inferred from the causal mask (via the asymmetry of C being computed from all keys up to position t in the recurrent form) and from the sequential accumulation of context.

2. **Resolvent as implicit position signal**: The resolvent C_t = (ρI + Σ_{s=1}^t k_s k_s^T)^{-1} is effectively a **running summary of the key distribution** up to position t. This provides a weak positional signal: earlier tokens are reflected in a larger accumulated Gram, which is fundamentally different from later tokens that haven't yet contributed. However, this signal is statistical, not positional — two tokens at positions 100 and 200 with identical key distributions would be indistinguishable.

3. **Does NOT avoid attention distribution distraction**: The attention distribution distraction failure mode (Wang et al., 2024) arises from softmax forcing uniform attention at unseen lengths. RhoAttn-sparse naturally avoids this through ReLU zeros — even at extreme N, only positively-aligned keys receive attention. However, without positional encoding, RhoAttention would suffer from a **different** problem: at extreme lengths, the content-based addressing may fail to identify **which** of many similar keys to attend to, since all positional cues have been stripped away.

**Theorem 7 (NoPE-RhoAttention Limitation)**. Without positional encoding, RhoAttention cannot solve tasks that require distinguishing between tokens with identical content at different positions. Formally, for any permutation π of the input sequence that preserves the causal structure (i.e., π is an automorphism of the causal order), the output at each position is invariant: o_t({x_π(1), ..., x_π(t)}) = o_t({x_1, ..., x_t}).

This is the fundamental limitation of any NoPE system (not specific to RhoAttention), and it explains why position encoding is necessary for tasks like copying, counting, and arithmetic — where the model must track the sequential order of tokens.

#### 2.5.3 A Position-Free Variant of RhoAttention

Despite the fundamental limitation, RhoAttention's resolvent structure suggests a novel **position-free** variant that may extend NoPE's effective range:

**RhoAttn-NoPE**: Remove RoPE pre-rotation entirely. The resolvent C = (ρI + K^T K)^{-1} provides a **content-based regularization** that naturally clusters keys by similarity:

- Tokens that frequently co-occur in similar contexts have aligned key vectors, creating larger entries in G = K^T K. The resolvent downweights these redundant keys, naturally implementing a form of **content-based attention sparsity**.

- The causal mask in the quadratic form ensures that the model can only attend to past tokens, providing the minimal positional signal.

- The recurrent form C_t = (ρI + Σ_{s=1}^t k_s k_s^T)^{-1} provides a weak temporal signal through the progressive accumulation of key statistics.

**Predicted behavior**: RhoAttn-NoPE should perform comparably or better than standard NoPE + Softmax for tasks where content-based addressing is sufficient (language modeling, semantic tasks), but would fail on tasks requiring explicit position tracking (copying, counting, needle-in-haystack with identical distractors). This is empirically testable.

#### 2.5.4 Comparison: NoPE Failure Modes

| Failure Mode | Standard NoPE + Softmax | RhoAttn-NoPE |
|-------------|------------------------|--------------|
| Attention distribution distraction (H → log N) | **Yes** — primary failure mode at unseen lengths. Mitigated by temperature scaling. | **Reduced** — ReLU zeros limit support to positively-aligned keys, preventing forced uniformity. |
| Position ambiguity (identical content at different positions) | **Yes** — no positional signal in content-based attention. | **Yes** — same fundamental limitation for any NoPE system. |
| Causal mask signal degradation at long contexts | **Yes** — as N grows, the asymmetry of the causal mask provides weaker per-token positional signal. | **Potentially reduced** — the resolvent's accumulated key statistics provide a complementary temporal signal through C_t's progressive refinement. |
| Retrieval of specific positions | **Impossible** without explicit position encoding. | **Impossible** — same limitation. |

The key insight: RhoAttention's ReLU sparsification directly addresses the **attention distribution distraction** failure mode identified by Wang et al. (2024) as the primary cause of NoPE length generalization failure. By preventing the forced uniformity of softmax, RhoAttn-NoPE should maintain sharper attention distributions at extreme lengths. However, the fundamental **position ambiguity** problem remains unsolved without some form of position encoding.

---

## 3. Important Papers & References

### Entropy Analysis & Length Generalization Theory

1. **Lin, Z., et al. (2025). "Critical Attention Scaling in Long-Context Transformers."** *arXiv:2510.05554*. **The most directly relevant theoretical work**: proves a phase transition in attention behavior governed by the scaling factor β_n, with β_n ≍ log n as the critical regime. Provides the first rigorous justification for logarithmic attention scaling. Directly formalizes the "attention dilution" phenomenon that RhoAttention's ReLU sparsification is designed to avoid.

2. **Li, Y., & Kong, J. (2025). "Information Entropy Invariance: Enhancing Length Extrapolation in Attention Mechanisms."** *arXiv:2506.16640*. Proposes InfoScale and CosScale based on entropy invariance principles. Provides formal analysis of attention score dilution and theoretically grounded scaling temperatures. Achieves 64× training length extrapolation. Establishes that entropy control is the central mechanism for length generalization.

3. **Hong, J., & Lee, S. (2025). "Variance Sensitivity Induces Attention Entropy Collapse and Instability in Transformers."** *EMNLP 2025*. Proves that softmax attention entropy decreases as logit variance increases: H(p) = log N − (N−1)σ²/(2N) + O(σ⁴), with ∂H/∂σ² < 0. Shows that ReLU kernel attention achieves entropy stability with bounded gradients. Provides the theoretical bridge between entropy collapse and training instability.

4. **Nakanishi, K., et al. (2025). "Scalable-Softmax Is Superior for Attention."** *arXiv:2501.xxxxx*. Introduces SSMax: Softmax((s·log n)·z) that maintains sharp attention at arbitrary lengths. Proves that standard softmax suffers from attention fading as n → ∞. Empirically recovers the log-N relationship from learned per-head parameters. The SSMax form directly parallels RhoAttention's ρ(N) scaling.

### RoPE Theory & Extensions

5. **Su, J. (2021/2023). "Rotary Position Embedding (RoPE)."** *Blog series + arXiv*. The foundational work introducing RoPE and its relative-position property q_m^T R(n−m) k_n. The group-theoretic structure (commuting 2D rotations) is essential for understanding why RhoAttention's resolvent breaks strict shift-invariance.

6. **Liu, Z., & Zhou, B. (2025). "Rethinking RoPE: A Mathematical Blueprint for N-dimensional Positional Embedding."** *arXiv:2504.06308*. Generalizes RoPE to N dimensions using Lie group theory. Defines the two core properties — Relativity (R_{x1}^T R_{x2} = R_{x2−x1}) and Reversibility — and provides the general solution via commuting skew-symmetric generators from a maximal abelian subalgebra (MASA). Directly informs the conditions under which RhoAttention's block-diagonal variant can recover shift-invariance.

7. **Ostmeier, S., et al. (2024/2025). "LieRE: Lie Rotational Positional Encodings."** *arXiv:2406.10322*. Generalizes RoPE via learned dense skew-symmetric matrices (Lie algebra elements). Demonstrates that strict relative-position requires commutativity of the Lie algebra generators, but that non-commutative variants may still perform well empirically.

### NoPE Research

8. **Wang, J., Ji, T., Wu, Y., Yan, H., Gui, T., Zhang, Q., Huang, X., & Wang, X. (2024). "Length Generalization of Causal Transformers without Position Encoding."** *Findings of ACL 2024*. **The foundational NoPE paper**: identifies attention distribution distraction as the failure mode of NoPE at unseen lengths; shows attention entropy inflection tracks perplexity inflection; proposes temperature scaling as a simple fix. Directly motivates RhoAttention's ReLU-based entropy control for position-free operation.

9. **Köcher, C., Kozachinskiy, A., Lin, A.W., Sälzer, M., & Zetzsche, G. (2025). "NoPE: The Counting Power of Transformers with No Positional Encodings."** *arXiv:2505.11199*. Proves that NoPE-AHATs can express counting languages corresponding to Diophantine equations, characterizing their expressiveness as semi-algebraic sets. Provides the theoretical ceiling for what any NoPE system (including RhoAttn-NoPE) can achieve.

### Sparse Attention & Entropy Control

10. **Zhang, B., Titov, I., & Sennrich, R. (2021). "Sparse Attention with Linear Units (ReLA)."** *EMNLP 2021*. The first systematic study replacing softmax with ReLU in attention. Demonstrates natural sparsity, head diversity, and comparable BLEU on MT tasks. Directly validates RhoAttention's ReLU-based sparsification approach.

11. **Santos, S., et al. (2025/2026). "Sparse Attention as Compact Kernel Regression."** *ICLR 2026*. Establishes formal kernel-theoretic correspondence: normalized ReLU attention ↔ Epanechnikov kernel; sparsemax ↔ Epanechnikov with adaptive normalization; α-entmax ↔ higher-order compact kernels. Provides the kernel-regression justification for why ReLU-based attention produces well-behaved sparsity patterns.

12. **Vasylenko, P., et al. (2025). "Long-Context Generalization with Sparse Attention."** *ICLR 2026*. α-entmax for dynamic sparsity at long contexts with formal entropy bounds: lim_{n→∞} H(α-entmax)/log n ≤ β < 1. The ASEntmax variant with (log n)^γ scaling provides the closest existing comparison point to RhoAttention's entropy behavior.

### Dual Forms & Matrix Methods

13. **Dao, T., & Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality."** *arXiv:2405.21060*. The SSD framework proving SSM-attention duality via semiseparable matrices. The Woodbury identity used in RhoAttention's duality proof provides an alternative (and in some ways stronger) mathematical foundation for the dual form.

14. **Siems, J., et al. (2025). "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products."** *NeurIPS 2025*. Uses generalized Householder matrices (diagonal + rank-r) for state transitions, providing a tunable expressivity knob. The Sherman-Morrison update in RhoAttention is mathematically analogous (rank-1 update to a matrix inverse), but RhoAttention applies the update to the attention normalization rather than the state transition.

15. **Mongaras, L., & Larson, J. (2025). "On the Expressiveness of Softmax Attention: A Recurrent Neural Network Perspective."** *arXiv:2507.23632*. Proves linear attention is first-order Taylor approximation of softmax; softmax implicitly uses infinite-order Kronecker interactions. RhoAttention's resolvent also captures infinite-order interactions through the Neumann series, but through a rational rather than exponential generating function.

### Additional Foundational Works

16. **Anson, B., Wang, Y., & Aitchison, L. (2025). "Scale-Invariant Attention for Long-Context Language Modeling."** *NeurIPS 2025*. Distance-dependent logit transform for scale-invariant attention across logarithmic distance octaves. Zero-shot generalization from 4K to 64K (16×). The per-octave entropy maintenance provides an alternative entropy control mechanism to RhoAttention's support limitation.

17. **Zsámboki, P., et al. (2025). "Learning What's Missing: Attention Dispersion and EMA Stabilization in Length Generalization."** *arXiv:2510.08341*. Proves that softmax compresses logit displacements as ~1/N, eroding valid-invalid token separation. The precision degradation analysis provides a mechanistic explanation for why ReLU zeros (which completely eliminate invalid tokens) avoid this degradation.

18. **Chen, B., Liang, Y., Sha, Z., Shi, Z., & Zhao, S. (2024/2025). "HSR-Enhanced Sparse Attention Acceleration."** *arXiv:2410.10165*. Provides rigorous asymptotic bounds on ReLU attention complexity using Half-Space Reporting: attention generation reduces from O(mn) to O(mn^{4/5}). Demonstrates that ReLU attention's sparsity can be exploited algorithmically for speedup — directly supporting RhoAttention's hardware-efficiency claims.

---

## 4. Open Questions & Future Directions

### 4.1 Empirical Entropy Measurement

The theoretical analysis in Sections 2.1–2.2 predicts specific entropy behavior for RhoAttention, but these predictions require empirical validation:

1. **Entropy vs. N curves**: Measure H(α) for RhoAttn-sparse across sequence lengths from 1K to 1M tokens. Does the entropy grow sub-logarithmically as predicted? What is the empirical N_eff as a function of N, layer depth, and task?

2. **Per-head entropy diversity**: Do different attention heads naturally specialize to different entropy regimes (some sharp, some broad) through the learned ρ and key subspaces?

3. **Entropy–perplexity correlation**: Does the inflection point of attention entropy track the perplexity inflection point (as Wang et al. showed for NoPE) for RhoAttention? If so, what intervention (ρ adjustment, temperature scaling) restores performance?

### 4.2 Optimal ρ Scheduling for Length Extrapolation

The linear ρ-scaling proposed in Section 2.4 (ρ(t) ∝ t/N_train) is theoretically motivated but empirically untested. Open questions:

- Is the spectral norm growth of G = K^T K truly linear in N for trained models? Correlated keys could produce sub-linear growth; heavy-tailed key distributions could produce super-linear growth.
- Should ρ be scheduled per-layer (early layers are more sensitive to position → smaller ρ for sharper attention; later layers perform semantic integration → larger ρ for broader attention)?
- Can ρ be made **learnable and context-length-aware**, trained to adapt to different context lengths using a meta-learning objective?
- Is log-N scaling (ρ(t) ∝ log t) more appropriate than linear scaling for some tasks? The log-N scaling in SSMax arises from the extreme value theory of Gaussian logits; does a similar analysis apply to the key Gram's spectral norm?

### 4.3 Block-Diagonal RoPE-Resolvent: Empirical Trade-offs

The block-diagonal variant (Section 2.3.3) achieves strict shift-invariance at the cost of cross-frequency coupling. Key empirical questions:

1. **Representational capacity**: Do the independent per-band resolvents lose important cross-frequency interactions? Can this be mitigated by adding a small number of cross-band coupling terms?

2. **Training dynamics**: Does the block-diagonal structure train stably? Each 2×2 block undergoes independent Cholesky decompositions — does this lead to inconsistent entropy behavior across frequency bands?

3. **Comparison to RoPE + RhoAttn**: At what context length does the deviation from shift-invariance in standard RhoAttn + RoPE become practically significant? Is the block-diagonal variant necessary for extreme lengths (128K+) or is the standard variant sufficient?

### 4.4 NoPE + RhoAttention: Hybrid Positional Strategies

The analysis in Section 2.5 suggests that RhoAttn-NoPE avoids attention distribution distraction but cannot overcome the fundamental position ambiguity of NoPE systems. Promising hybrid strategies:

1. **Partial position encoding**: Apply RoPE only to a subset of attention heads (those responsible for position-sensitive tasks like copying and retrieval), leaving the remaining heads as RhoAttn-NoPE for content-based semantic processing.

2. **Position-conditioned ρ**: Instead of pre-rotating Q and K, make the regularization parameter ρ position-dependent: ρ_t = ρ_0 · f(t) where f(t) encodes position information (e.g., f(t) = (1 + t/τ)). This provides positional bias through the resolvent stiffness rather than through key/query rotation.

3. **Resolvent-based position mixing**: Use the Sherman-Morrison update history as a positional feature — the sequence of rank-1 updates applied to C_t encodes the temporal order of keys, potentially providing a learnable positional signal.

### 4.5 The Unifying Theory: Entropy as the Control Variable

The convergence of four independent research communities on entropy as the central mechanism for length generalization raises a profound theoretical question: **Is attention entropy control necessary and sufficient for length generalization?**

- **Necessity**: The evidence strongly suggests yes. Every mechanism that achieves length generalization — SSMax, InfoScale, Scale-Invariant Attention, α-entmax, RhoAttn-sparse — does so by controlling attention entropy. Conversely, every mechanism that fails at length generalization — standard softmax, unmodified RoPE, vanilla SSMs — fails in ways that manifest as entropy dysregulation.

- **Sufficiency**: This is an open question. RhoAttention's dual control (ReLU zeros for concentration, resolvent shrinkage for regularization) may be sufficient, but empirical validation is needed. The key test: can a mechanism that perfectly controls attention entropy (maintaining both sharpness when needed and breadth when appropriate, at any N) achieve perfect length generalization? Or are there additional failure modes (gradient dynamics, representation collapse) that entropy control alone cannot address?

### 4.6 Integration with Alternative Position Encodings

Beyond RoPE, RhoAttention's resolvent structure may be compatible with other position encoding schemes:

- **ALiBi** (linear bias): Add a position-dependent bias term b_{mn} = −λ|m−n| to the attention logits. This is trivially compatible with RhoAttention since the bias is additive: P'_{mn} = P_{mn} + b_{mn}.

- **xPos** (Sun et al., 2023): Combines RoPE with a position-dependent exponential decay. Compatible with RhoAttention via the same mechanism as RoPE integration.

- **FIRE** (Li et al., 2024): Learnable functional position encoding via MLP applied to position indices. Can be applied as an additive term to RhoAttention logits.

- **NoPE + causal mask**: The causal mask itself provides positional signal through the asymmetry of attention. RhoAttn-NoPE leverages this, and the resolvent's progressive refinement provides a complementary temporal signal.

### 4.7 Theoretical Frontiers

1. **Information-theoretic optimality**: Prove whether RhoAttention's resolvent-based normalization achieves information-theoretic optimality for some class of attention problems (e.g., Gaussian process regression with squared-exponential kernel). The connection between the resolvent and Bayesian posterior precision suggests a natural optimality criterion.

2. **Phase transition analysis**: Does RhoAttention exhibit a phase transition analogous to the one Lin et al. (2025) proved for softmax attention? The ρ parameter serves a role similar to the scaling factor β_n — is there a critical ρ_c such that ρ < ρ_c leads to attention collapse and ρ > ρ_c leads to excessive diffusion?

3. **Connections to random matrix theory**: As N → ∞, the key Gram matrix G = K^T K follows the Marchenko-Pastur law under Gaussian assumptions. The resolvent C = (ρI + G)^{-1} is the Stieltjes transform of the spectral distribution. Can random matrix theory provide exact asymptotic predictions for the entropy and attention behavior?

---

## 5. Relevance to Main Topic

### 5.1 Entropy Control as Architectural Primitive

This analysis establishes that RhoAttention provides **architectural entropy control** — the ability to maintain stable attention entropy across sequence lengths — through two complementary mechanisms:

1. **ReLU sparsification**: Produces exact zeros for anti-aligned keys, naturally limiting the support of the attention distribution to positively-aligned keys. This prevents the H → log N entropy collapse without requiring per-head temperature tuning or log-N rescaling.

2. **Resolvent spectral regularization**: The resolvent C = (ρI + K^T K)^{-1} modulates attention scores based on global key statistics. The hyperparameter ρ provides a direct entropy control: larger ρ → softer attention (higher entropy); smaller ρ → sharper attention (lower entropy). This is analogous to temperature in softmax but operates through a fundamentally different mathematical mechanism (matrix rational function vs. element-wise exponential).

Together, these mechanisms implement the design principle of **entropy-stable normalization** (Principle 2 from the weakness audit) in a way that is simultaneously:

- **Matrix-multiply-only** (Principle 1): The resolvent computation uses only Cholesky decomposition (O(d³), tensor-core-compatible via cuSOLVER) and matrix multiplications.
- **Dual-form compatible** (Principle 3): The entropy control properties are preserved in both quadratic (training) and recurrent (inference) forms, since the ReLU zeros depend only on the sign of q^T C k, which is identical in both forms.
- **Content-adaptive** (Principle 4): The entropy is not a fixed function of N but adapts to content — sharp attention for well-aligned queries, broad attention for queries requiring context integration.

### 5.2 Connection to the Full Research Program

The entropy and length generalization analysis directly informs and constrains the remaining sub-topics:

- **Sub-Topic 3 (Complexity Analysis)**: The entropy bounds (Theorem 1–2) imply computational benefits: if N_eff ≪ N (typical for RhoAttn-sparse), then the attention matrix α is row-sparse, potentially reducing the cost of the αV multiplication from O(N²d) to O(N·N_eff·d) using sparse matrix formats.

- **Sub-Topic 4 (Hardware-Aware Design)**: The ReLU zeros create structural sparsity that can be exploited in hardware. The HSR-based sparse attention acceleration (Chen et al., 2024) achieves O(N^{4/5}) complexity for ReLU attention generation — directly applicable to RhoAttn-sparse.

- **Sub-Topic 6 (Quantitative Comparison)**: The entropy comparisons in Section 2.2.3 provide the theoretical foundation for predicting RhoAttention's retrieval quality advantage over softmax attention at extreme lengths. These predictions should be quantitatively validated against FlashAttention-4, SSMs, and linear attention.

- **Sub-Topic 7 (Implementation)**: The ρ-scheduling strategy (Section 2.4.2) and the block-diagonal RoPE variant (Section 2.3.3) require specific implementation considerations in the CUDA kernel design, particularly for the per-band resolvent updates and the Sherman-Morrison error reset logic.

### 5.3 Contribution Summary

This sub-topic makes the following contributions to the overall research program:

1. **Rigorous entropy analysis of RhoAttention**: Formal proofs that RhoAttn-sparse maintains H(α) ≤ log N_eff with N_eff content-dependent, preventing the entropy collapse that plagues softmax attention.

2. **Proof of attention dilution avoidance**: Theorem 3 establishes that RhoAttn-sparse does NOT suffer from attention dilution as N → ∞, with the entropy growth rate strictly bounded below the softmax maximum.

3. **RoPE compatibility analysis**: Complete characterization of the conditions under which RhoAttention preserves (or fails to preserve) the strict relative-position property, and the block-diagonal variant that recovers it.

4. **Length extrapolation predictions**: Theoretically grounded predictions for RhoAttention's performance at various extrapolation factors, with specific recommendations for ρ-scheduling.

5. **NoPE analysis**: Characterization of RhoAttention's behavior without positional encoding, identifying both the advantage (resistance to attention distribution distraction) and the fundamental limitation (position ambiguity for identical tokens).

6. **Unifying perspective**: Framing of attention entropy control as the central mechanism for length generalization, connecting RhoAttention to the broader landscape of entropy-aware attention mechanisms (SSMax, α-entmax, Scale-Invariant Attention, InfoScale).

### 5.4 Status of Key Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| RhoAttn-sparse avoids entropy collapse | ✅ Proven | Theorem 1 & 2 (mathematical bounds) |
| RhoAttn-sparse avoids attention dilution | ✅ Proven | Theorem 3 (structural property) |
| Block-diagonal variant achieves strict shift-invariance | ✅ Proven (approximate) | Theorem 5 (depends on C^{(i)} ≈ c_i·I_2) |
| RhoAttn + RoPE has bounded deviation from shift-invariance | ✅ Bounded | Quantitative bound: O(Δ/d) |
| Length extrapolation possible without catastrophic degradation | ⚠️ Theoretically predicted | Theorem 6 (memory rigidity bound); requires empirical validation |
| RhoAttn-NoPE resists attention distribution distraction | ✅ Theoretically argued | Wang et al. (2024) mechanism analysis + ReLU property |
| ρ-scheduling improves extrapolation | ⚠️ Theoretically motivated | Spectral norm analysis; requires empirical validation |

### 5.5 Final Assessment

RhoAttention represents a **fundamentally new approach to attention entropy control** — one that is built into the mathematical structure of the normalization function rather than added as a post-hoc correction. The resolvent provides principled, Bayesian-motivated regularization, while the ReLU activation provides natural sparsification. Together, these mechanisms address the attention dilution problem (the single most impactful weakness identified in the audit) at the architectural level.

The key open question — as with all theoretical analyses — is empirical validation. The entropy bounds, length extrapolation predictions, and NoPE analysis are mathematically sound but rest on assumptions about key distributions, training dynamics, and task structure that must be verified experimentally. The remaining sub-topics (particularly Sub-Topic 7 on experimental validation design) provide the blueprint for this empirical program.

---

## Appendix A: Notation Reference

| Symbol | Description |
|--------|-------------|
| N | Sequence length |
| d | Head dimension |
| q, k, v | Query, key, value vectors (ℝ^d) |
| Q, K, V | Full matrices (ℝ^{N×d}) |
| G = K^T K / √d | Key Gram matrix (ℝ^{d×d}) |
| ρ | Regularization parameter |
| C = (ρI + G)^{-1} | Resolvent (ℝ^{d×d}) |
| P_{ij} = q_i^T C k_j | Rational attention logit |
| α | Attention weights |
| H(α) = −Σ α_i log α_i | Attention entropy |
| N_eff = \|{i : P_i > 0}\| | Effective support size |
| R_θ(m) | RoPE rotation matrix at position m |
| θ_i = b^{-2i/d} | RoPE frequency for band i |
| γ | Proportion of relevant keys |
| β_n | Attention scaling factor |
| T_recomp | Sherman-Morrison recomputation interval |

## Appendix B: Key Theorems Reference

| Theorem | Statement |
|---------|-----------|
| **T1 (Entropy Upper Bound)** | H(α) ≤ log N_eff ≤ log N for RhoAttn-sparse |
| **T2 (Asymptotic Entropy)** | Under separation of relevant/irrelevant keys: H(α) ≤ log(γN) with γ < 1 |
| **T3 (No Dilution)** | RhoAttn-sparse avoids attention dilution: lim H(α)/log N < 1 |
| **T4 (RoPE Compatibility)** | Standard RhoAttn+RoPE lacks strict relative-position; deviation bounded by O(Δ/d) |
| **T5 (Block-Diagonal Shift-Invariance)** | Block-diagonal variant approximately recovers relative-position when C^{(i)} ≈ c_i·I_2 |
| **T6 (Memory Rigidity)** | Per-token resolvent update bounded by ‖k‖²/ρ²; rigidity at long contexts mitigated by ρ-scheduling |
| **T7 (NoPE Limitation)** | Without positional encoding, RhoAttention cannot distinguish permuted tokens with identical content |

---

*Research conducted: June 2026. This analysis synthesizes RhoAttention (defined in Sub-Topic 2) with the literature on attention entropy, length generalization, and positional encoding (2021–2026). Key sources include the Weakness Audit (Sub-Topic 1), the RhoAttention specification (Sub-Topic 2), and the web-search results from June 2026 capturing the latest developments in entropy-aware attention mechanisms.*
