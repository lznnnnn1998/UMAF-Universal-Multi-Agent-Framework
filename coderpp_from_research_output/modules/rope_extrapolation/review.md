# Code Review: rope_extrapolation

## Issues Found

### Bug 1: `build_dpe_attention_mask` column range allows cross-chunk attention (dpe.py:158) — **HIGH**
The loop that grants chunk-local attention used `mask[start:end, non_global_start:end] = 0.0`, but `non_global_start:end` grows with each iteration. For chunk 2 and beyond, this grants non-global tokens access to ALL prior chunk non-global tokens, breaking the DPE locality constraint.

**Example**: With `seq_len=6, chunk_size=3, num_global_tokens=1`:
- Chunk 1 (tokens 1-3): `mask[1:4, 1:4] = 0.0` — correct.
- Chunk 2 (tokens 4-5): `mask[4:6, 1:6] = 0.0` — **wrong!** Tokens 4-5 can see chunk 1 tokens 1-3.

The docstring example was also incorrect (showing only 2 tokens in a chunk_size=3 chunk, and comparing `cos[0]` with `cos[0]` in the test).

### Bug 2: `DPERoPE.__init__` max_pos doesn't account for `shift` (dpe.py:278) — **MEDIUM**
`max_pos = max(chunk_size, num_global_tokens)` ignores the `shift` parameter. Position IDs computed by `compute_dpe_position_ids` can be up to `(chunk_size - 1) + shift`. If `shift > 0`, `self.cos`/`self.sin` buffer indexing would go out of bounds in `forward()`.

### Bug 3: `YaRNScaler` docstring temperature formula mismatch (extrapolation.py:225) — **LOW**
Docstring said `t = min(1.0, (scale)^{0.25})`, suggesting temperature is capped at 1.0. The actual code uses `min(self.scale ** 0.25, 32.0)`, which correctly allows temperature > 1 (as the YaRN paper prescribes). The old docstring would mislead readers into thinking temperature is always ≤ 1.

### Bug 4: `test_dpe_chunk_position_cycle` is a no-op test (test_rope.py:639) — **LOW**
The test compared `cos[0]` with `cos[8 % 8]` = `cos[0]`, which is an identity check that always passes. `dpe_precompute_cos_sin` returns tables of shape `(chunk_size, dim)`, so there's no `cos[8]` entry to compare. The test never actually verified the chunk-position cycling property.

## Fixes Applied

1. **dpe.py:158**: Changed `mask[start:end, non_global_start:end]` → `mask[start:end, start:end]`. Each chunk's non-global tokens now correctly attend only to the global tokens and tokens within their own chunk.

2. **dpe.py:129-136**: Updated the docstring example to match the correct behavior (first chunk has 3 non-global tokens with `chunk_size=3`, and tokens self-attend within their chunk).

3. **dpe.py:278**: Changed `max_pos = max(chunk_size, num_global_tokens)` → `max_pos = max(chunk_size + shift, num_global_tokens)`. Added explanatory comment.

4. **extrapolation.py:224-234**: Fixed YaRNScaler docstring temperature formula from `min(1.0, scale^{0.25})` → `min(scale^{0.25}, 32.0)` to match the code and the YaRN paper.

5. **test_rope.py:633-648**: Replaced the no-op table comparison with actual functional tests using `apply_dpe_rotary` at different offsets (0 vs chunk_size, 3 vs chunk_size+3) to verify that chunk-local position cycling produces identical embeddings.

## Final Test Results

```
============================== 93 passed in 0.68s ==============================
```

All 93 tests pass. Verified with additional manual checks:
- `build_dpe_attention_mask` correctly blocks cross-chunk non-global attention
- `DPERoPE` with `shift=50` works without index errors
- DPE chunk-position cycling produces identical embeddings at matching chunk-local positions

## Verdict

**REVIEW_PASSED** — All identified bugs fixed, all tests pass, and the code is solid.
