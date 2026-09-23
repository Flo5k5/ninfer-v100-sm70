// ninfer::ops - qk_rmsnorm_rope wrapper: fused route for the fixed text geometries, composed
// rmsnorm + rmsnorm + rope for everything else.
#include "ninfer/ops/qk_rmsnorm_rope.h"

#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ops/launcher/qk_rmsnorm_rope.h"

#include <cmath>
#include <cstdint>

namespace ninfer::ops {
namespace {

bool aligned4(const Tensor& tensor) {
    return tensor.data != nullptr && (reinterpret_cast<std::uintptr_t>(tensor.data) & 3U) == 0;
}

bool heads_tensor(const Tensor& tensor, std::int32_t heads, std::int32_t tokens) {
    return tensor.dtype == DType::BF16 && tensor.ne[0] == 256 && tensor.ne[1] == heads &&
           tensor.ne[2] == tokens && tensor.ne[3] == 1 && tensor.is_contiguous() && aligned4(tensor);
}

bool gain_tensor(const Tensor& tensor) {
    return tensor.dtype == DType::BF16 && tensor.ne[0] == 256 && tensor.ne[1] == 1 &&
           tensor.ne[2] == 1 && tensor.ne[3] == 1 && tensor.is_contiguous() && aligned4(tensor);
}

bool disjoint(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin + lhs.bytes() <= rhs_begin || rhs_begin + rhs.bytes() <= lhs_begin;
}

bool fused_geometry(const Tensor& q, const Tensor& k, const Tensor& q_norm, const Tensor& k_norm,
                    float eps, const Tensor& positions, int rotary_dim, float theta,
                    const Tensor& q_out, const Tensor& k_out) {
    const std::int32_t tokens = q.ne[2];
    const std::int32_t q_heads = q.ne[1];
    const std::int32_t k_heads = k.ne[1];
    const bool heads = (q_heads == 24 && k_heads == 4) || (q_heads == 16 && k_heads == 2);
    return heads && tokens > 0 && rotary_dim == 64 && theta == 1.0e7F && eps > 0.0F &&
           std::isfinite(eps) && heads_tensor(q, q_heads, tokens) &&
           heads_tensor(k, k_heads, tokens) && heads_tensor(q_out, q_heads, tokens) &&
           heads_tensor(k_out, k_heads, tokens) && gain_tensor(q_norm) && gain_tensor(k_norm) &&
           positions.dtype == DType::I32 && positions.ne[0] == tokens &&
           (positions.ne[1] == 1 || positions.ne[1] == 3) && positions.ne[2] == 1 &&
           positions.ne[3] == 1 && positions.is_contiguous() && positions.data != nullptr &&
           disjoint(q, q_out) && disjoint(k, k_out) && disjoint(q, k_out) && disjoint(k, q_out) &&
           disjoint(q_out, k_out);
}

} // namespace

void qk_rmsnorm_rope(const Tensor& q, const Tensor& k, const Tensor& q_norm_weight,
                     const Tensor& k_norm_weight, float eps, const Tensor& positions,
                     int rotary_dim, float theta, Tensor& q_out, Tensor& k_out,
                     cudaStream_t stream) {
    if (fused_geometry(q, k, q_norm_weight, k_norm_weight, eps, positions, rotary_dim, theta,
                       q_out, k_out)) {
        detail::qk_rmsnorm_rope_fused_launch(positions, q, k, q_norm_weight, k_norm_weight, q_out,
                                             k_out, eps, stream);
        return;
    }
    rmsnorm(q, q_norm_weight, eps, true, q_out, stream);
    rmsnorm(k, k_norm_weight, eps, true, k_out, stream);
    rope(positions, rotary_dim, theta, q_out, k_out, stream);
}

} // namespace ninfer::ops
