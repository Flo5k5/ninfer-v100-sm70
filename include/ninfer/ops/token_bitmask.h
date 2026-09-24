#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Number of I32 words of one token bitmask over a token domain: one bit per token, token v in
// bit (v % 32) of word v / 32.
[[nodiscard]] constexpr std::int32_t token_bitmask_words(std::int32_t token_domain) noexcept {
    return (token_domain + 31) / 32;
}

/**
 * Op: apply_token_bitmask
 *
 * Math / indexing:
 *   For row b < B, column j < W, and logits row v < physical_rows, with
 *   active_b = clamp(mask_columns[b], 0, W):
 *     logits[v,j,b] = -inf     when j < active_b and (v >= token_domain or bit v of
 *                              bitmask[:,j,b] is clear);
 *     logits[v,j,b] unchanged  otherwise.
 *   Columns at or beyond active_b, including every column of a row whose count is zero or
 *   negative, are left untouched. A set bit never raises a logit: the Op only excludes tokens.
 *
 * Logical shapes:
 *   logits is contiguous BF16 [physical_rows,W,B] with B>=1 and W>=1. bitmask is contiguous I32
 *   [token_bitmask_words(token_domain),C,R] with C>=W and R>=B: column j of row b reads the word
 *   vector at ((b*C)+j)*words, so a caller may keep one stable [words,C,R] buffer for every
 *   execution shape. mask_columns is contiguous I32 with at least B elements. token_domain is in
 *   [1,physical_rows]. Bits of the last word at or beyond token_domain are ignored. logits does not
 *   overlap bitmask or mask_columns.
 *
 * Numeric:
 *   Excluded logits become the BF16 bit pattern of negative infinity (0xff80). Every other logit
 *   keeps its exact bits. A column whose mask allows no token is left with no finite logit, and
 *   argmax and sample() then return token 0 for it: callers keep at least one token allowed in
 *   every active column.
 *
 * Effects:
 *   Updates logits in place; bitmask and mask_columns are read-only.
 *
 * Execution:
 *   A captured call stays valid for any mask_columns and bitmask contents present at replay; the
 *   tensor addresses and shapes are those of the capture.
 *
 * Workspace:
 *   None.
 */
void apply_token_bitmask(Tensor& logits, const Tensor& bitmask, const Tensor& mask_columns,
                         std::int32_t token_domain, cudaStream_t stream);

} // namespace ninfer::ops
