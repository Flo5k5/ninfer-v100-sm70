// ninfer::ops - apply_token_bitmask wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/token_bitmask.h"

#include "ops/launcher/token_bitmask.h" // detail::apply_token_bitmask_launch

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void apply_token_bitmask(Tensor& logits, const Tensor& bitmask, const Tensor& mask_columns,
                         std::int32_t token_domain, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("apply_token_bitmask: logits must be BF16");
    }
    if (bitmask.dtype != DType::I32 || mask_columns.dtype != DType::I32) {
        throw std::invalid_argument("apply_token_bitmask: bitmask and mask_columns must be I32");
    }
    if (!logits.is_contiguous() || !bitmask.is_contiguous() || !mask_columns.is_contiguous()) {
        throw std::invalid_argument("apply_token_bitmask: tensors must be contiguous");
    }
    if (logits.ne[3] != 1 || bitmask.ne[3] != 1) {
        throw std::invalid_argument(
            "apply_token_bitmask: logits must be [physical_rows,W,B] and bitmask [words,C,R]");
    }
    if (mask_columns.ne[1] != 1 || mask_columns.ne[2] != 1 || mask_columns.ne[3] != 1) {
        throw std::invalid_argument("apply_token_bitmask: mask_columns must be rank-1");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "apply_token_bitmask: token_domain must be in [1, logits.ne[0]]");
    }
    if (bitmask.ne[0] != token_bitmask_words(token_domain)) {
        throw std::invalid_argument(
            "apply_token_bitmask: bitmask rows must hold one bit per token of the domain");
    }
    if (bitmask.ne[1] < logits.ne[1] || bitmask.ne[2] < logits.ne[2] ||
        mask_columns.ne[0] < logits.ne[2]) {
        throw std::invalid_argument(
            "apply_token_bitmask: bitmask and mask_columns must cover every logits column and row");
    }
    if (logits.data == nullptr || bitmask.data == nullptr || mask_columns.data == nullptr) {
        throw std::invalid_argument("apply_token_bitmask: tensor data must be non-null");
    }
    if (overlaps(logits, bitmask) || overlaps(logits, mask_columns)) {
        throw std::invalid_argument(
            "apply_token_bitmask: logits must not overlap bitmask or mask_columns");
    }

    detail::apply_token_bitmask_launch(logits, bitmask, mask_columns, token_domain, stream);
}

} // namespace ninfer::ops
