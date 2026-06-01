# Research Report: Hardware-Aware Algorithm Design — Tiling, Memory Hierarchy, and Kernel Architecture

## Overview

Hardware-aware algorithm design for attention mechanisms has undergone a paradigm shift from the naive "materialize everything in HBM" approach of standard attention toward a sophisticated, multi-level orchestration of data movement, computation, and synchronization across the GPU memory hierarchy. The foundational insight — first crystallized by FlashAttention (Dao et al., 2022) — is that attention is fundamentally **I/O-bound**, not compute-bound: the quadratic `O(N²)` attention matrix must transit the narrow HBM ↔ SRAM bus multiple times in a naive implementation, while SRAM has ~12× higher bandwidth. By tiling the computation so that all intermediate dot-products, exponentials, and reductions remain in SRAM, FlashAttention reduced HBM traffic by 4–16× and eliminated the `O(N²)` memory footprint.

The subsequent evolution through FlashAttention-2 (2023), FlashAttention-3 (2024, NeurIPS Spotlight), and FlashAttention-4 (2025) tracks the co-evolution of algorithm design with GPU hardware generations — from Ampere (A100) through Hopper (H100) to Blackwell (B200/GB200). This progression reveals a fundamental principle: **algorithm-hardware co-design is no longer optional**. Each GPU generation introduces asymmetric scaling of hardware units — tensor core throughput doubles while shared memory bandwidth, register file capacity, and special function units (SFUs) scale much more slowly. Kernels must be restructured around these asymmetries, using warp specialization, asynchronous pipelining, software emulation of slow hardware units, and new memory tiers (TMEM) to approach peak utilization.

For a novel attention mechanism — particularly one incorporating nonlinearities beyond standard softmax — the hardware-aware design problem decomposes into six interdependent sub-problems: (a) deriving block sizes `B_r`, `B_c` as functions of head dimension `d` and SRAM capacity `M`, accounting for the mechanism's specific compute and memory footprint; (b) mapping data across HBM, SRAM, registers, and TMEM (on Blackwell) with explicit load/store/recompute schedules; (c) warp-specialized kernel pseudocode mapping nonlinear operations to either tensor cores (if expressible as matmuls), FMA units (via polynomial approximation), or SFU/MUFU units; (d) asynchronous pipelining with TMA to hide data movement latency; (e) a backward-pass strategy balancing intermediate storage against recomputation; and (f) low-precision (FP8/FP4) compatibility, critically addressing whether the mechanism's nonlinearity creates a bottleneck on the MUFU.EX2 unit.

---

## Key Methods & Approaches

### (a) Tiling Strategy: Block Size Derivation

The tiling strategy partitions the `Q` (query), `K` (key), and `V` (value) matrices into blocks that fit entirely within on-chip SRAM. The canonical derivation from FlashAttention determines block sizes `B_r` (Q rows) and `B_c` (K/V columns) from the SRAM capacity constraint.

**Standard softmax attention:** During the tiled inner loop, SRAM must simultaneously hold:
1. Q block: `B_r × d` elements
2. K block: `B_c × d` elements  
3. V block: `B_c × d` elements
4. Output O accumulator: `B_r × d` elements
5. Score matrix S: `B_r × B_c` elements (intermediate)
6. Online softmax statistics: `m` (row max) and `ℓ` (row sum), `B_r` elements each

Assuming `B_r = B_c = B` and FP16 precision (2 bytes/element), the dominant SRAM usage is:
```
SRAM_usage ≈ 4Bd + B²  (in elements)
           = 8Bd + 2B²  (in bytes) for FP16
```

Setting `SRAM_usage ≤ M` yields the constraint:
```
B_c = ⌈M / (4d)⌉      B_r = min(⌈M / (4d)⌉, d)
```

**Concrete example (A100, d=128, FP16):** `M = 192 KB = 196,608 bytes`. Then `B_c ≈ 196,608 / (4 × 128 × 2) ≈ 192`, practically clamped to 128–192. For typical head dimensions `d ∈ {64, 128, 256}`, `B_r` and `B_c` range from 64–256.

**Impact of mechanism-specific compute patterns on tiling:** A novel attention mechanism modifies this constraint in two ways:

1. **Additional SRAM-resident tensors:** If the mechanism introduces extra per-tile state (e.g., gating values, learned temperature parameters, bias terms, or recurrent state for linear attention variants), the effective `d` in the denominator increases, reducing `B_c`. Specifically, if the mechanism requires an additional tensor of size `B_r × B_c` (matching the score matrix), the constraint becomes:
```
SRAM_usage ≈ 4Bd + kB² ≤ M
```
where `k` is the number of `B_r × B_c` intermediate tensors (`k=1` for standard softmax; `k=2` if the mechanism needs both S and a transformed version). For `k=2` at d=128, `B_c` drops from ~192 to ~144.

2. **Nonlinearity that operates element-wise on S in SRAM:** Operations like `exp()`, `gelu()`, `relu()`, or custom activation functions on the score matrix are applied while S resides in SRAM, so they don't change the memory footprint — but they DO change the **compute balance**. If the nonlinearity is expensive (e.g., requiring MUFU.EX2 for exponential), it may dominate the inner loop latency. Conversely, a mechanism using a cheaper nonlinearity (sigmoid, ReLU, or polynomial-approximated exp) shifts the bottleneck back to the matmuls.

3. **Multi-stage tiling (FlashAttention-4, TFLA):** When the mechanism uses a chunked or recurrent formulation (as in linear attention, xLSTM, or state-space models), tiling acquires a second level: an outer loop over chunks (each producing an intermediate state) and an inner tiled loop within each chunk. The chunk size `C` is now a hyperparameter bounded by `M / (state_size)` where `state_size` grows with the mechanism's recurrent state dimension.

**Block size design for a novel mechanism — practical procedure:**
1. Enumerate all tensors that must be SRAM-resident during the inner loop
2. Express total SRAM usage as `f(B_r, B_c, d, precision_bytes)`
3. Solve `f(B_r, B_c) ≤ M` for maximal `B_r`, `B_c` subject to alignment constraints (typically multiples of 16 or 32 for warp-level operations)
4. If `B_c` falls below ~32, the kernel becomes occupancy-limited (too few warps can be scheduled), and the mechanism should be restructured to reduce SRAM pressure
5. For Blackwell with TMEM available, intermediate tensors (S, P) can be moved to TMEM rather than SRAM, relaxing the SRAM constraint to:
```
SRAM_usage ≈ 3Bd  (Q, K, V blocks only)
TMEM_usage ≈ B² + 2Bd  (S, P, O accumulators)
```
This effectively doubles the usable block size on Blackwell.

---

### (b) Memory Hierarchy: HBM → SRAM → Registers → TMEM

The GPU memory hierarchy on modern NVIDIA architectures has four tiers, each with distinct capacity, bandwidth, and latency characteristics:

| Memory Tier | A100 (Ampere) | H100 (Hopper) | B200 (Blackwell) | Bandwidth | Scope |
|---|---|---|---|---|---|
| **HBM** (global) | 40–80 GB | 80 GB | 192 GB | 1.5–2.0 TB/s (A100), 3.35 TB/s (H100), 8.0 TB/s (B200) | All SMs |
| **L2 Cache** | 40 MB | 50 MB | 96 MB | ~4 TB/s (shared) | All SMs |
| **SRAM** (shared memory) | 192 KB/SM | 228 KB/SM | 228 KB/SM | ~19 TB/s | Per SM |
| **L1 Cache** | 192 KB/SM (configurable with SRAM) | 256 KB/SM | 256 KB/SM | ~19 TB/s | Per SM |
| **Registers** | 65,536 × 32-bit/SM | 65,536 × 32-bit/SM | 65,536 × 32-bit/SM | ~80 TB/s (estimated) | Per SM |
| **TMEM** (tensor memory) | N/A | N/A | 256 KB/SM | Direct tensor core connection | Per SM |

**Data residency strategy for standard FlashAttention:**

**HBM (always resident):** Full Q, K, V, dO matrices; final output O; stored LSE statistics per row.

**SRAM (loaded per tile):** Q tile (`B_r × d`), K tile (`B_c × d`), V tile (`B_c × d`), O accumulator (`B_r × d`).

**Registers (per-thread):** Online softmax running statistics (`m`, `ℓ`); warp-level MMA accumulators (on Ampere/Hopper where accumulator is in registers).

**TMEM (Blackwell only):** Score matrix S (`B_r × B_c`), softmax probabilities P (`B_r × B_c`, overlapping S), O accumulator partial sums, backward-pass intermediates (dS, dP, dV, dK accumulators). TMEM is tightly coupled to tensor cores via the `tcgen05` scalar dispatch mechanism — the tensor engine reads operands directly from TMEM and writes results back to TMEM **without consuming register file bandwidth or warp issue slots**.

**When data is loaded, stored, and recomputed:**

| Phase | Load from HBM → SRAM | SRAM computation | Store SRAM → HBM |
|---|---|---|---|
| Forward: Q | Once (if `B_r` covers all Q) or tiled | — | — |
| Forward: K, V | Each tile, via TMA | — | — |
| Forward: S = QK^T | — | MMA on tensor cores (operands in SRAM, result → TMEM or registers) | — |
| Forward: softmax | — | Element-wise on S in SRAM/TMEM | — |
| Forward: O += PV | — | MMA (P in TMEM/SRAM, V in SRAM → O accumulator) | O accumulator → HBM (once at end) |
| Forward: LSE | — | Scalar per row | LSE → HBM (once at end, for backward pass) |
| Backward: dO, O, LSE | dO, O, LSE → SRAM | — | — |
| Backward: Q, K, V | Q, K, V tiles → SRAM | — | — |
| Backward: S, P | — | **Recomputed** from Q, K, LSE (never stored forward) | — |
| Backward: dP = V @ dO^T | — | MMA | — |
| Backward: dS = P ⊙ (dP − rowsum(P ⊙ dP)) | — | Element-wise | — |
| Backward: dQ, dK, dV | — | MMA accumulations | dQ, dK, dV → HBM |

**Recomputation policy:** The score matrix S and softmax probabilities P are **never written to HBM** — they are recomputed during the backward pass from Q, K, and the stored LSE scalar per row. This trades ~30% additional forward computation (re-running the matmul and softmax) for eliminating O(N²) HBM storage. On Blackwell, the recomputation cost is further amortized because the S and P recomputation runs at full tensor core speed using TMEM.

---

### (c) Kernel Design: Forward Pass Pseudocode

The following pseudocode describes a warp-specialized forward kernel for Blackwell-class GPUs (SM100). It targets a novel attention mechanism with a **generic nonlinearity** `σ(·)` applied to the score matrix (replacing or augmenting softmax). The design assumes the nonlinearity is expressible as either:
- **Path A (tensor-core-friendly):** A sequence of matrix multiplications (e.g., quadratic forms, MLP on attention scores), or
- **Path B (FMA-friendly):** An element-wise function that can be approximated by a low-degree polynomial on FMA units, bypassing MUFU.EX2.

```
Algorithm: NovelAttention Forward Kernel (Blackwell SM100)

Constants:
  N:       sequence length
  d:       head dimension (typically 64–256)
  tile_m:  Q tile rows = 128  (B_r)
  tile_n:  K/V tile cols = 128 (B_c)
  dtype:   FP16 (forward), FP32 (accumulation)

Warp roles (CTA = 16 warps = 512 threads, 128 threads/warp):
  Warps 0–3:   softmax0     — compute σ(S[0]), track stats (192 regs each)
  Warps 4–7:   softmax1     — compute σ(S[1]), track stats (192 regs each)
  Warps 8–11:  correction   — rescale O accumulators (96 regs each)
  Warp 12:     mma          — issue UMMA instructions (32 regs)
  Warp 13:     load         — issue TMA loads for Q, K, V (32 regs)
  Warp 14:     epilogue     — issue TMA stores for O (32 regs)
  Warp 15:     unused       — idle or auxiliary (24 regs)

TMEM layout (512 columns total = 256 KB):
  S[0]:   offset 0,   size 128 cols  — score matrix for Q[0] tile
  S[1]:   offset 128, size 128 cols  — score matrix for Q[1] tile  
  P[0]:   offset 64,  size 64 cols   — σ(S[0]) probabilities (overlaps S[0])
  P[1]:   offset 192, size 64 cols   — σ(S[1]) probabilities (overlaps S[1])
  O[0]:   offset 256, size 128 cols  — output accumulator for Q[0]
  O[1]:   offset 384, size 128 cols  — output accumulator for Q[1]

SMEM layout:
  smem_Q[0]:  tile_m × d  FP16  — Q tile 0
  smem_Q[1]:  tile_m × d  FP16  — Q tile 1 (double-buffered)
  smem_K:     tile_n × d  FP16  — K tile (ping-pong staged)
  smem_V:     tile_n × d  FP16  — V tile (ping-pong staged)

Named barriers:
  mbar_load_KV:    signals K/V tile ready in SMEM (2 stages)
  mbar_S_ready:    signals S computed and in TMEM
  mbar_P_ready:    signals σ(S) computed and P in TMEM  
  mbar_O_ready:    signals O accumulator updated
  mbar_epi_ready:  signals O ready for TMA store

================================================================
PROLOGUE (executed once)
================================================================
// Load Q blocks into SMEM (persistent across KV loop)
load_warp: TMA_load_async(Q[0:tile_m-1],     → smem_Q[0])
load_warp: TMA_load_async(Q[tile_m:2*tile_m-1] → smem_Q[1])  // if q_stage==2
// Initialize O accumulators in TMEM to 0
// Initialize online statistics: m = -∞, ℓ = 0 (or mechanism-specific equivalents)
// Note: if σ is not softmax, the "statistics" vector depends on the mechanism
//   For a generic σ, we may need a running normalizer, gating state, or bias tracker

================================================================
MAIN LOOP: for j = 0 to ceil(N / tile_n) - 1  (K/V tile index)
================================================================

  // ---- STAGE 0: Async load K_j, V_j (load warp) ----
  load_warp:
    TMA_load_async(K[j*tile_n : (j+1)*tile_n], → smem_K[stage])
    TMA_load_async(V[j*tile_n : (j+1)*tile_n], → smem_V[stage])
    mbarrier_arrive(mbar_load_KV, stage)

  // ---- STAGE 1: Compute S[0] = Q[0] @ K^T (mma warp on Q tile 0) ----
  mma_warp:
    mbarrier_wait(mbar_load_KV, 0)
    // Issue UMMA: smem_Q[0] (FP16) × smem_K (FP16) → TMEM S[0] (FP32)
    tcgen05.mma(S[0], smem_Q[0], smem_K)
    // Apply causal or local attention mask (element-wise on S[0] in TMEM)
    if causal:  S[0][i < k] = -∞   // via element-wise store to TMEM
    mbarrier_arrive(mbar_S_ready, 0)

  // ---- STAGE 2a: Nonlinearity σ(S[0]) (softmax0 warps) ----
  softmax0_warps (warps 0–3):
    mbarrier_wait(mbar_S_ready, 0)
    
    // === MECHANISM-SPECIFIC NONLINEARITY ===
    // Option A: If σ is softmax or exp-based:
    if mechanism == "softmax":
      // Software-emulated exp2 on FMA units (avoids MUFU.EX2 bottleneck):
      //   Given x = S[0][i], compute exp2(x) via:
      //     x_int = floor(x)
      //     x_frac = x - x_int  
      //     exp2(x_frac) ≈ P(x_frac) where P is degree-3 minimax polynomial
      //       P(t) = 1 + t*(0.693147 + t*(0.240226 + t*0.055505))
      //     exp2(x) = exp2(x_frac) * 2^x_int  (integer shift)
      //   This matches BF16 accuracy on 99% of inputs (Dao et al., 2025, FA4)
      //
      // Online softmax rescaling:
      //   m_new = max(m_old, row_max(S[0]))
      //   ℓ_new = ℓ_old * exp2(m_old - m_new) + row_sum(exp2(S[0] - m_new))
      //   P[0] = exp2(S[0] - m_new)  → TMEM offset 64
      //   if |m_new - m_old| > threshold: trigger rescale flag
      
    // Option B: If σ is a learned or arbitrary activation (e.g., GELU, Swish, sigmoid):
    else if mechanism == "polynomial_activation":
      // Map σ to a degree-k polynomial evaluated on FMA units:
      //   For GELU: σ(x) ≈ x/2 * (1 + tanh(√(2/π)*(x + 0.044715*x³)))
      //   Can be approximated as degree-5 polynomial
      //   P[0] = a₀ + x*(a₁ + x*(a₂ + x*(a₃ + x*(a₄ + x*a₅))))
      //   5 FMAs per element — fully saturates FMA pipeline, no MUFU needed
      
    // Option C: If σ requires matrix multiplication (e.g., quadratic form):
    else if mechanism == "quadratic":
      // σ(S) = S ⊙ (S @ W)  or  σ(S) = (S @ W₁) ⊙ σ₀(S @ W₂)
      // Break into additional UMMA calls: S @ W → TMEM temp → element-wise ⊙ S
      
    // Option D: If mechanism unavoidably needs MUFU.EX2:
    else:
      // Direct exp2() via MUFU.EX2 instruction — BUT mitigate bottleneck by:
      //   1. Using MUFU only for a subset of values (e.g., top-k sparse mask)
      //   2. Overlapping MUFU with concurrent MMA on other warp groups
      //   3. Quantizing to FP8 before MUFU to reduce operand count
      
    mbarrier_arrive(mbar_P_ready, 0)

  // ---- STAGE 2b: Same for Q[1] tile (softmax1 warps, parallel) ----
  // (Structurally identical to Stage 2a, on different TMEM region)
  // Runs concurrently with Stage 2a if q_stage == 2

  // ---- STAGE 3: MMA P @ V (mma warp) ----
  mma_warp:
    mbarrier_wait(mbar_P_ready, stage % 2)
    // UMMA: P (FP16 in TMEM) × V (FP16 in SMEM) → O_tmp (FP32 in TMEM O[stage])
    tcgen05.mma(O[stage], P_in_TMEM, smem_V)
    mbarrier_arrive(mbar_O_ready, stage)

  // ---- STAGE 4: Correction / Accumulation (correction warps) ----
  correction_warps (warps 8–11):
    mbarrier_wait(mbar_O_ready, stage)
    // Mechanism-dependent rescaling:
    //   For softmax: O[stage] = (O_old * ℓ_old + O_tmp * ℓ_new) / ℓ_total
    //   For generic σ: depends on whether σ has a normalization property
    //   Conditional: only rescale if running statistics changed significantly
    //     (FA4 reports ~10× reduction in rescaling ops via conditional approach)
    
    // Update running normalizer and O accumulator
    // If σ produces unnormalized scores, apply explicit normalization here
    mbarrier_arrive(mbar_epi_ready, stage)

  stage = 1 - stage  // toggle ping-pong buffer

================================================================
EPILOGUE (after all KV tiles)
================================================================
epilogue_warp (warp 14):
  // If mechanism has normalization: O_final = O_acc / normalizer
  // Apply output projection, dropout, residual connection if fused
  TMA_store_async(O_acc → global memory O)
  barrier_sync()  // ensure all writes complete before kernel exit
```

**Key design considerations for the nonlinearity mapping:**

1. **Expressibility as matmuls (Path A):** If the mechanism's nonlinearity can be expressed as a sequence of matrix multiplications (e.g., `σ(S) = relu(S @ W₁ + b₁) @ W₂`, quadratic forms like `(QK^T)²`, or gated attention where gating is a learned linear projection), it can run entirely on tensor cores. This is ideal — tensor cores are the fastest compute units and are not a bottleneck on any GPU generation.

2. **FMA polynomial approximation (Path B):** For element-wise nonlinearities (softmax, GELU, Swish, tanh), the state of the art (FlashAttention-4) uses **degree-3 minimax polynomial approximation of exp2** evaluated on FMA units. This matches BF16 precision on 99% of inputs while completely bypassing the MUFU.EX2 bottleneck (16 ops/clock/SM, shared with other transcendental instructions). The polynomial `P(t) = 1 + t·(c₁ + t·(c₂ + t·c₃))` requires only 3 FMAs per element. The FMA pipeline can sustain ~128 FMAs/clock/SM (vs. 16 MUFU ops/clock), providing an 8× throughput advantage for the nonlinearity step.

3. **Forced MUFU.EX2 path (Fallback):** If the nonlinearity genuinely requires `exp()` and cannot be polynomial-approximated at acceptable precision, the MUFU bottleneck must be mitigated through: (a) **sparse evaluation** — compute exp only for top-k scores (the rest are masked to 0 or -∞), reducing MUFU calls by ~90% for long sequences; (b) **temporal overlap** — schedule MUFU on one warp group while another warp group runs MMA (exploiting that MUFU and tensor cores use different execution pipelines); (c) **precision reduction** — quantize inputs to FP8 before MUFU, halving operand bandwidth.

---

### (d) Asynchronous Pipelining: TMA and Warp Specialization

**TMA (Tensor Memory Accelerator):** Introduced in Hopper (SM90) and enhanced in Blackwell (SM100), TMA is a dedicated hardware unit for bulk asynchronous data transfer between global memory (HBM) and shared memory (SRAM). Key advantages over Ampere's `cp.async`:

| Feature | `cp.async` (Ampere) | TMA (Hopper/Blackwell) |
|---|---|---|
| Thread participation | All warps issue loads | **Single thread** issues entire transfer |
| Bandwidth efficiency | ~70–80% peak | **~90–95% peak** |
| Address computation | Software (per-thread pointer arithmetic) | Hardware (tensor descriptor: base + strides) |
| Multi-dimensional tiles | Manual index math | **Native ND tensor descriptors** |
| Synchronization | Per-thread tracking | **Hardware mbarrier** |
| Register pressure | High (per-thread pointers) | **Low** (single descriptor per tensor) |
| Cluster multicast | Not available | **Yes** (one TMA load → multiple CTAs' SMEM) |

**Warp specialization pattern:** The canonical design (FA3, FA4, CUTLASS Blackwell FMHA) divides a CTA's 16 warps into specialized roles that run **concurrently** on different execution pipelines:

- **Producer warp (Load):** Issues TMA commands. Does nothing else — just advances the pipeline state machine, issues `TMA_load_async`, and signals barriers. Uses minimal registers (24–32), freeing registers for consumer warps via `setmaxnreg`.
- **Consumer warps (MMA, Softmax, Correction):** Execute the actual computation. MMA warp issues `tcgen05.mma` (scalar dispatch on Blackwell — single thread triggers the tensor engine, which then runs independently without consuming warp issue slots). Softmax warps handle the nonlinearity on FMA or MUFU units. Correction warps handle online rescaling.
- **Epilogue warp:** Issues TMA stores back to global memory.

**Multi-stage asynchronous pipeline (Blackwell FMHA):** The forward kernel operates a **7-pipeline system** coordinated by named barriers:

```
PipelineQ:   load_warp →(TMA)→ smem_Q    →(barrier)→ mma_warp
PipelineKV:  load_warp →(TMA)→ smem_K,V  →(barrier)→ mma_warp
PipelineS:   mma_warp  →(UMMA)→ TMEM S   →(barrier)→ softmax_warps
PipelineC:   softmax_warps →(FMA)→ stats →(barrier)→ correction_warps
PipelineO:   mma_warp  →(UMMA)→ TMEM O   →(barrier)→ correction_warps
PipelineE:   correction_warps → SMEM O    →(barrier)→ epilogue_warp →(TMA)→ HBM
OrderSoftmax: softmax0 ↔ softmax1 ordering (prevents TMEM write conflicts)
```

**Ping-pong double buffering:** While softmax0 warps compute `σ(S[0])`, the mma warp computes `S[1] = Q[1] @ K^T` and softmax1 warps prepare to process it. Meanwhile, the load warp prefetches the next K/V tile. At steady state, **three operations execute concurrently**: TMA load of tile `n+1`, softmax on tile `n`, and MMA accumulation for tile `n-1`. This hides both memory latency and nonlinearity latency.

**Register reallocation with `setmaxnreg`:** Hopper/Blackwell support per-warp register limits. Producer warps use only 24–40 registers, MMA warp uses 32–40 (since TMEM eliminates accumulator register pressure), and softmax/correction warps use 96–192 registers (for element-wise loop unrolling). This asymmetric distribution ensures that no warp group starves for registers while the total per-SM register budget (65,536 × 32-bit) is fully utilized.

---

### (e) Backward Pass Strategy: Store vs. Recomputation

**Standard attention backward pass:** Requires the attention weight matrix `P = softmax(QK^T/√d)` to compute gradients. Without optimization, storing P costs `O(N²)` memory — prohibitive for long sequences.

**FlashAttention recomputation strategy:** Only two quantities are stored from the forward pass:
1. **Output O:** shape `(batch, heads, N, d)` — FP16, `O(Nd)` memory
2. **Log-Sum-Exp (LSE) per row:** shape `(batch, heads, N)` — FP32, `O(N)` memory, where `LSE_i = log(∑_j exp(S_ij))`

The backward pass then **recomputes** P on-the-fly block-by-block:
```
For each K/V block j:
    Load K_j, V_j from HBM
    For each Q block i:
        S_ij = Q_i @ K_j^T                    // recompute scores
        P_ij = exp2(S_ij - LSE_log2_i)        // reconstruct softmax via stored LSE
        dP = V_j @ dO_i^T                     // first gradient
        dS = P_ij ⊙ (dP - rowsum(P_ij ⊙ dP))  // softmax gradient
        dV_j += P_ij^T @ dO_i                  // accumulate dV
        dK_j += dS^T @ Q_i                     // accumulate dK
        dQ_i += dS @ K_j                       // accumulate dQ
```

The recomputation cost is ~30% additional forward computation (one extra matmul pass), but eliminates the `O(N²)` memory footprint entirely.

**Mechanism-specific backward pass considerations:**

1. **Non-softmax nonlinearities:** If the mechanism uses a nonlinearity `σ` different from softmax, the gradient computation changes. For a generic element-wise `σ` applied to scores S:
```
P = σ(S)                                   // forward nonlinearity
dS = σ'(S) ⊙ (dP - correction_term)        // backward: element-wise gradient of σ
```
If `σ` is a polynomial, `σ'` is also a polynomial (one degree lower) — same FMA-friendly pattern. If `σ` has a closed-form gradient (sigmoid: `σ'(x) = σ(x)(1-σ(x))`), the backward can reuse the forward-computed P. If `σ` has no simple gradient (e.g., a learned function), P must be stored or recomputed.

2. **What to store vs. recompute for a novel mechanism:**

| Quantity | Store? | Reason |
|---|---|---|
| Q, K, V | Store (HBM, needed for recomputation) | Small: O(Nd) each |
| O (output) | Store (HBM, needed for downstream layers) | Small: O(Nd) |
| S = QK^T | **Recompute** | Large: O(N²), recomputable from Q, K |
| P = σ(S) | **Recompute** (unless σ is non-invertible from stored statistics) | Recomputable from S and stored normalizer |
| Normalizer statistics | **Store** (HBM, scalar per row) | Small: O(N). For softmax: LSE. For generic σ: running normalizer |
| Gating variables (if any) | **Store** if they depend on Q,K,V | Depends on mechanism |
| Intermediate gradients (dS, dP) | **Recompute** on-the-fly in TMEM | Never stored to HBM |

3. **Critical difference from standard attention:** If the novel mechanism's `σ` is **not invertible** from stored statistics (e.g., a learned nonlinearity with internal state), then either: (a) the forward P must be stored (increasing memory to `O(N²)`), which defeats FlashAttention's main advantage; or (b) the mechanism must be designed so that P is deterministically recomputable from stored statistics of size `O(Nd)` or `O(N)`. This is a **hard design constraint** for any attention variant aiming for long-context training.

4. **Backward pass TMEM orchestration (Blackwell):** FA4's backward kernel uses TMEM for all five intermediate matrices (S, dP, dV, dK, dS, dQ), with aggressive buffer overlapping since they have non-overlapping lifetimes. A single MMA warp executes all five MMAs sequentially (S recomputation, dP, dV, dK, dQ), reducing synchronization complexity. The **2-CTA MMA mode** on Blackwell allows pairs of CTAs to jointly execute the `dS @ K` MMA for dQ computation, halving atomic reduction operations.

5. **Deterministic backward pass:** FA4 introduces a deterministic backward mode with ~15% overhead using semaphore-locked reduction ordering with shortest-processing-time-first scheduling. This is critical for reproducible reinforcement learning training where nondeterministic gradient accumulation causes divergence.

---

### (f) FP8/FP4 Compatibility and the MUFU.EX2 Bottleneck

**Low-precision attention landscape:**

| Precision | TFLOPS (H100) | TFLOPS (B200) | Memory reduction vs. FP16 | Key challenge |
|---|---|---|---|---|
| FP16/BF16 | 990 / 990 | 2,250 / 2,250 | Baseline | Standard reference |
| FP8 (E4M3/E5M2) | 1,979 | 4,500 | 2× | Limited dynamic range (±448 max for E4M3); needs per-block scaling |
| FP4 (E2M1) | N/A | 9,000 (B200 only) | 4× | Extremely limited range; requires sophisticated scaling |
| INT8 | 1,979 | 4,500 | 2× | Integer-only; not suitable for softmax exponentials |

**Block quantization for FP8 attention:** The standard approach (FlashAttention-3, MOSS, COAT) uses **per-block scaling factors** to maintain numerical accuracy in FP8:

1. **For matrix multiplications (QK^T, PV):** Operands are stored in FP8 with per-block (128-element) scaling factors. The matmul runs in FP8 → FP32 accumulation. This works well because matmuls are numerically robust to quantization — the dot product averages out per-element errors.

2. **For the softmax/nonlinearity:** This is the hard part. The exponential function amplifies small quantization errors in the score matrix S. FlashAttention-3 addresses this through **incoherent processing**: applying a random Hadamard transform (or random orthogonal rotation) to Q and K before quantization, which spreads outlier magnitudes evenly across dimensions. This reduces quantization error by ~2.6× compared to direct FP8 quantization. Specifically:
```
Q_rot = Q @ H_d     // H_d is a d×d Hadamard matrix (or random orthogonal)
K_rot = K @ H_d
// Quantize Q_rot, K_rot to FP8
// During attention: S = Q_rot @ K_rot^T / √d
// The rotation ensures no single dimension dominates the dot product
```

**FP4 compatibility:** Blackwell introduces native FP4 (E2M1) tensor core support, doubling throughput over FP8. However, FP4 has only 16 representable values and an extremely limited range (max ~6.0). For attention, FP4 is viable only with:
- **Per-vector scaling factors** (every 32–64 elements)
- **Incoherent processing** (Hadamard rotation to flatten outlier distribution)
- **Mixed-precision**: compute the nonlinearity (σ) in FP16 or FP8, only the matmuls in FP4
- **NVFP4 KVCache** optimization (NVIDIA Blackwell): stores KV cache in FP4 format, reducing memory bandwidth for long-context inference

**The MUFU.EX2 bottleneck and asymmetric scaling:**

The MUFU.EX2 instruction computes `2^x` (base-2 exponential) on NVIDIA's Special Function Units (SFUs). This is the hardware primitive underlying `exp()`, `exp2()`, `expf()`, and by extension, softmax. The bottleneck arises from **asymmetric hardware scaling**:

| Hardware unit | Hopper (H100) throughput | Blackwell (B200) throughput | Scaling factor |
|---|---|---|---|
| Tensor cores (FP16 MMA) | 990 TFLOPS | 2,250 TFLOPS | **2.27×** |
| Shared memory bandwidth | ~1.5 TB/s per SM | ~1.5 TB/s per SM | **~1.0×** (unchanged) |
| MUFU.EX2 (exp2) | 16 ops/clock/SM | 16 ops/clock/SM | **~1.0×** (unchanged) |
| FMA units | 128 FMAs/clock/SM | 128 FMAs/clock/SM | **~1.0×** (unchanged) |

As tensor core throughput doubles generation-over-generation while MUFU throughput stays constant, the softmax exponential becomes an increasingly dominant fraction of total attention latency. On Blackwell at FP8 precision, the matmuls complete ~4× faster than on Hopper (FP8), but softmax runs at the same speed — softmax can consume >50% of the forward pass latency for long sequences.

**Blackwell Ultra (GB300) mitigation:** NVIDIA partially addressed this in Blackwell Ultra by doubling SFU throughput (MUFU.EX2 goes from ~4,900 Gop/s to ~9,700 Gop/s). But this is a hardware fix — not available on standard Blackwell (GB200) or Hopper (H100).

**Software mitigation strategies (relevant to novel mechanism design):**

1. **FMA-based polynomial approximation (FA4 approach):** Replace `exp2(x)` with a degree-3 minimax polynomial evaluated on FMA units:
```
exp2(x) ≈ 2^floor(x) * P(x - floor(x))
where P(t) = 1 + t * (c₁ + t * (c₂ + t * c₃))
c₁ ≈ 0.6931471805599453   // ln(2)
c₂ ≈ 0.2402265069591007
c₃ ≈ 0.0555041086648216
```
This uses 3 FMAs per element instead of 1 MUFU.EX2. With FMA throughput at 128 ops/clock vs. MUFU at 16 ops/clock, the effective throughput is **24× higher** (128/16 × 3 FMAs per element = 24), though the element latency is 3× longer. The key insight: FMA units are abundant and underutilized during the softmax phase (since MMA runs on tensor cores), while MUFU units are scarce.

2. **Design the nonlinearity to avoid exp() entirely:** If the mechanism uses ReLU, sigmoid, tanh, GELU, or any other activation with a polynomial approximation, it completely bypasses the MUFU bottleneck. For instance:
   - Sigmoid: `σ(x) = 1/(1+exp(-x))` — still needs exp, but can use the same FMA polynomial trick
   - ReLU: `σ(x) = max(0, x)` — trivial, no MUFU needed
   - GELU: approximable as degree-5 polynomial on FMA
   - **Custom polynomial nonlinearity:** `σ(x) = a₀ + a₁x + a₂x² + a₃x³` — 3 FMAs, fully saturates FMA

3. **Sparse/selective exp evaluation:** For mechanisms that only need exp on a subset of scores (e.g., top-k attention, sparse attention with top-k selection), compute exp only on the selected elements using MUFU, and set the rest to 0. This reduces MUFU calls proportionally to the sparsity ratio (e.g., top-256 out of 8192 → 32× reduction).

4. **Mixed-precision exp:** Reduce the precision of the exp input from FP32 to FP16 or FP8, reducing the data volume flowing through MUFU. Accept a small accuracy penalty (typically <0.1% perplexity increase).

**Does the proposed mechanism avoid reliance on MUFU.EX2?**

This is the central question. The answer depends on the mechanism's nonlinearity:

- **YES (MUFU-free):** If the mechanism uses: (a) a polynomial activation, (b) ReLU/gated linear units, (c) matrix-multiplication-based gating (e.g., `σ(S) = S ⊙ (S @ W_gate)` where gating is a learned projection), (d) sigmoid/tanh approximated by polynomials, or (e) any nonlinearity with a degree ≤ 5 polynomial approximation on FMA. These designs avoid MUFU entirely and can run at tensor-core-limited speed.

- **PARTIAL (MUFU-mitigated):** If the mechanism needs exp() but only: (a) on a sparse subset (top-k), (b) at reduced precision (FP16/BF16 input instead of FP32), or (c) with aggressive FMA polynomial substitution on all but the most extreme values. The FMA polynomial can handle 99% of inputs, falling back to MUFU only when the polynomial error exceeds BF16 epsilon.

- **NO (MUFU-dependent):** If the mechanism **requires** high-precision exponential on all `N²` score elements with no sparsity or approximation. In this case, the bottleneck is real and unavoidable on current hardware. The only mitigations are: (a) target Blackwell Ultra hardware with doubled SFU throughput, (b) accept longer latency (the softmax will simply take proportionally longer), or (c) restructure the mechanism to compute the nonlinearity in a lower-dimensional space (e.g., per-head rather than per-pair).

**Recommendation for novel mechanism design:** Design the nonlinearity to be **FMA-friendly from the start**. If the mechanism's mathematical properties require something like an exponential, consider whether: (a) a polynomial approximation preserves the essential mathematical behavior (for attention, the key property of softmax is competition via exponentiation — a degree-3 polynomial exp2 approximation preserves this to >99% fidelity), or (b) the exponential can be applied in a lower-dimensional space (per-token rather than per-token-pair), reducing the bottleneck from `O(N²)` to `O(N)`.

---

## Important Papers & References

| # | Paper | Authors | Venue | Year | Key Contribution |
|---|---|---|---|---|---|
| 1 | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, Christopher Ré | NeurIPS | 2022 | Introduced IO-aware tiling for attention; block sizes `B_r, B_c = O(M/d)`; online softmax; O(N²) memory elimination |
| 2 | **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | Tri Dao | arXiv | 2023 | Swapped loop order in backward pass; parallelized over Q blocks; reduced non-matmul FLOPs; stored only LSE (not m and ℓ separately); ~2× speedup |
| 3 | **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision** | Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao | NeurIPS (Spotlight) | 2024 | Warp specialization with producer/consumer warp groups; TMA async data movement; WGMMA instructions; FP8 with incoherent Hadamard processing; 75% H100 utilization |
| 4 | **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling** | Tri Dao et al. | MLSys (Oral) | 2026 (published Mar 2025) | Blackwell-native kernel; TMEM utilization; software-emulated exp2 via degree-3 polynomial on FMA; conditional softmax rescaling (~10× fewer ops); 2-CTA MMA mode; deterministic backward; 71% B200 utilization; up to 2.7× faster than Triton |
| 5 | **CUTLASS: CUDA Templates for Linear Algebra Subroutines** | NVIDIA (Vijay Thakkar, Pradeep Ramani et al.) | Open-source (GitHub) | Ongoing (v3.x 2025) | Reference implementation of warp-specialized FMHA kernels for Hopper/Blackwell; 7-pipeline architecture; CuTe DSL; WoGMMA/UMMA patterns |
| 6 | **Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels** | (Anonymous, NeurIPS 2025) | NeurIPS | 2025 | Two-level sequence tiling for linear attention; breaks SRAM chunk size limit; applied to mLSTM; outperforms FlashAttention and Mamba |
| 7 | **FlexAttention: A Compiler-Driven Programming Model for Attention** | Horace He, Driss Guessous, Yanbo Liang, Joy Dong | PyTorch Blog / arXiv | 2024 | Compiler stack for automatic attention kernel generation; user provides score_mod function; design space exploration for tile shapes |
| 8 | **Lightning Attention-2: A Constant-Speed Attention Mechanism** | MiniMax (Li et al.) | arXiv | 2024 (updated 2025) | Left-product kernel trick: `O = Q(K^TV)`; true linear complexity; powers MiniMax-M1 456B model with 4M-token context |
| 9 | **MOSS: Efficient and Accurate FP8 LLM Training with Microscaling and Automatic Scaling** | Yu Zhang, Hui-Ling Zhen, Mingxuan Yuan, Bei Yu | arXiv:2511.05811 | 2025 | Two-level microscaling for FP8 activations; automatic scaling via Adam optimizer dynamics; SNR analysis; 34% throughput gain |
| 10 | **Making Softmax More Efficient with NVIDIA Blackwell Ultra** | NVIDIA Technical Blog | Developer Blog | 2025 | Analysis of MUFU.EX2 bottleneck; GB300 doubling SFU throughput; ~35% improvement in FP8 forward pass; NVFP4 KVCache optimization |
| 11 | **FP8-Flow-MoE: A Casting-Free FP8 Recipe without Double Quantization Error** | Fengjuan Wang et al. | arXiv:2511.02302 | 2025 | Scaling-aware transpose for FP8; eliminates double quantization error; evaluated on 671B DeepSeek-V3 MoE; 21% throughput gain |
| 12 | **COAT: Compressing Optimizer States and Activations for Memory-Efficient FP8 Training** | (Multiple authors) | MLSys | 2024 | Per-group activation quantization with compander; mixed-granularity FP8; 1.54× memory reduction; 1.43× speedup vs. BF16 |
| 13 | **SoftEx: Polynomial-Corrected Schraudolph's Method for Efficient Softmax** | Andrea Belano et al. | DATE | 2025 | Hardware softmax accelerator; Schraudolph's exp2 method + piecewise polynomial mantissa correction; 0.14% mean relative error; 10.8× speedup on RISC-V |
| 14 | **HARA: A Unified Framework for Hardware-Efficient Non-Linearity in Transformers** | Tusheng Wang et al. | Under review | 2025 | Replaces GELU, Softmax, LayerNorm with unified ReLU-polynomial architecture; >60% silicon area reduction for nonlinearity; <0.1% accuracy loss |
| 15 | **UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling** | (Multiple authors) | ACM | 2025 | Three-level hierarchical tiling (node → device → kernel); ILP workload allocation; 5.5× speedup on 64 GPUs |

---

## Open Questions & Future Directions

1. **Automated block size optimization:** Current tiling strategies use analytical formulas `B_c ≈ M/(4d)` derived from static SRAM capacity. However, the optimal block size also depends on: (a) the mechanism's specific compute-to-memory ratio (nonlinearity cost shifts the roofline), (b) GPU occupancy constraints (too-large blocks reduce the number of concurrent CTAs), and (c) wave quantization effects (tail waves waste SM cycles). An automated block size optimizer (like FlexAttention's design space exploration) that takes the mechanism's compute DAG as input and outputs Pareto-optimal `(B_r, B_c)` would be valuable. Current compilers (Triton, FlexAttention) do this only for softmax; extending to arbitrary nonlinearities is an open problem.

2. **Beyond polynomial approximation for exp:** The degree-3 polynomial for exp2 used in FA4 matches BF16 on 99% of inputs, but the remaining 1% of values can cause training instability in long-context regimes where errors accumulate across sequence positions. Higher-degree polynomials or rational function approximations (Padé approximants) could extend the accurate range, but at additional FMA cost. The optimal accuracy-vs-cost tradeoff for different sequence lengths and precisions is unexplored.

3. **FP4 training for attention:** FP4 attention training is experimentally unexplored as of 2025. The main barrier is the nonlinearity: with only 16 representable values, the softmax (or any exponential-based mechanism) loses all precision. Incoherent processing helps for the matmuls, but a fundamentally different approach may be needed for the nonlinearity — perhaps a learned lookup table, a quantized polynomial, or a mechanism that operates natively in log-space.

4. **Non-softmax mechanisms and backward pass storage:** While the recomputation strategy for softmax is well-understood (store LSE, recompute P), non-softmax mechanisms may have different information-theoretic requirements. For a mechanism that cannot reconstruct P from `O(N)` stored statistics, new checkpointing strategies are needed. The key question: **what is the minimal sufficient statistic for reconstructing the attention weights of an arbitrary nonlinearity?** This is related to the concept of sufficient statistics in exponential families (softmax is the canonical link for the categorical distribution), and generalizing this to non-exponential-family mechanisms is an open problem.

5. **Hardware-software co-design for future GPUs:** The trend of asymmetric scaling (tensor cores doubling while SFUs stagnate) will continue with future architectures (Rubin, etc.). This suggests two design principles for attention mechanisms: (a) maximize the fraction of FLOPs that can run on tensor cores (matmul-heavy designs), and (b) minimize reliance on non-tensor-core units (SFU, shared memory, register file). Mechanisms that are "tensor-core-native" — expressible almost entirely as sequences of matrix multiplications — will have a growing advantage over mechanisms requiring significant scalar/element-wise compute.

6. **Multi-GPU and distributed tiling:** UltraAttn (2025) and FlexLA (2025) demonstrate that tiling strategies extend naturally to multi-GPU settings, but the optimal tradeoff between communication volume, compute load balance, and memory hierarchy utilization across nodes is not fully characterized. For multi-node training with model parallelism, the optimal tiling strategy may differ from the single-GPU case — e.g., it may be better to use larger blocks on each GPU to reduce cross-GPU communication even if it slightly reduces per-GPU utilization.

7. **Deterministic training at scale:** FA4's deterministic mode (15% overhead) is a major step, but the overhead may grow with sequence length and model size. Whether deterministic attention can be achieved at <5% overhead for trillion-parameter models is unknown.

---

## Relevance to Main Topic

This sub-topic — hardware-aware algorithm design — is the **implementation bridge** between a novel attention mechanism's mathematical formulation (sub-topic 2) and its practical deployment at scale. The key connections are:

1. **Mathematical formulation → Tiling feasibility:** The mechanism's compute DAG determines what must be SRAM-resident during the inner loop. If the mechanism introduces additional `B_r × B_c` tensors, block sizes shrink, reducing arithmetic intensity. This creates a **design feedback loop**: if the mathematical formulation requires too much SRAM per tile, either the mechanism must be simplified or the block size penalty must be accepted (with corresponding throughput loss).

2. **Nonlinearity → Kernel mapping:** The mechanism's specific nonlinearity `σ(·)` must be mapped to one of: tensor cores (ideal, if matmul-expressible), FMA polynomial approximation (good, bypasses MUFU), or MUFU (bottleneck, must be mitigated). **This is likely the single most important hardware-aware design decision** for any novel attention mechanism. A mechanism using ReLU, polynomial activation, or learned linear gating can achieve near-peak hardware utilization; a mechanism requiring high-precision exp2 on all token pairs will be MUFU-bound and run at a fraction of peak throughput.

3. **Backward pass storage → Training feasibility:** If the mechanism's P (attention weights) cannot be reconstructed from `O(N)` stored statistics, long-context training becomes memory-prohibitive. The mechanism must either: (a) be designed so P is deterministically recomputable (as softmax is from LSE), or (b) accept the `O(N²)` memory cost, limiting practical sequence lengths to ~4K–8K tokens.

4. **Low-precision compatibility → Deployment efficiency:** FP8 and FP4 support are increasingly essential for both training throughput and inference latency. A mechanism that is numerically robust to quantization (via incoherent processing, per-block scaling, or log-space computation) will deploy more efficiently than one requiring FP16 or FP32 throughout.

5. **The MUFU.EX2 verdict:** The most critical recommendation from this analysis is: **design the mechanism to avoid reliance on MUFU.EX2**. The asymmetric scaling between tensor cores (2.27× per generation) and SFUs (unchanged) means that any MUFU-dependent mechanism will become an increasingly severe bottleneck on future hardware. Three pathways to MUFU independence: (a) use FMA-based polynomial approximation of exp2 (3 FMAs, handles 99% of inputs at BF16 precision), (b) replace softmax/exp with an FMA-friendly alternative (polynomial, ReLU, sigmoid approximation, or learned gating), or (c) apply the nonlinearity in a lower-dimensional space (per-token rather than per-token-pair) to reduce the bottleneck from `O(N²)` to `O(N)`.

In summary, hardware-aware design is not an afterthought — it is a **first-class constraint** that should shape the mathematical formulation of any novel attention mechanism from the outset. The most successful attention variants of the next generation will be those whose compute patterns align with the asymmetric trajectory of GPU hardware: abundant tensor cores, stagnant SFUs, and increasingly deep memory hierarchies.

---

*Research conducted: June 2026. Sources include peer-reviewed papers (NeurIPS, MLSys, DATE, ACM), preprints (arXiv), technical documentation (NVIDIA CUTLASS, NVIDIA Technical Blog), and open-source repositories (Dao-AILab/flash-attention, NVIDIA/cutlass).*
