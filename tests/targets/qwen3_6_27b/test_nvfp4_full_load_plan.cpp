// Host-only binding check for the Qwen3.8 NVFP4 "full A" profile: MLP 0-63 and the output head in
// NVFP4, the token embedding, attention and GDN in FP8. Binding reads the directory and a few small
// host objects (divisors, proposal ids); nothing here touches a device. The bound formats must also
// give the text residual stream its FP32 form, which Variant must then accept for the profile.
//
//   NINFER_QWEN3_8_27B_NVFP4_FULL_A_WEIGHTS=<stage A artifact>
//   NINFER_QWEN3_8_27B_NVFP4_WEIGHTS=<mixed NVFP4/FP8 artifact it was derived from>   (optional)

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

std::filesystem::path environment_path(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::filesystem::path() : std::filesystem::path(value);
}

ninfer::targets::qwen3_6::StartupFeatures mtp_features() {
    return {
        .vision        = false,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

bool valid_nvfp4(const WeightPlan& weight) {
    const float weight_divisor = std::bit_cast<float>(weight.weight_scale_divisor_bits);
    const float input_divisor  = std::bit_cast<float>(weight.input_scale_divisor_bits);
    return weight.format == NumericFormat::NVFP4 && std::isfinite(weight_divisor) &&
           weight_divisor > 0.0F && std::isfinite(input_divisor) && input_divisor > 0.0F;
}

std::uint64_t device_bytes(const std::filesystem::path& path, WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    return bind_artifact(binder, profile, mtp_features()).materialization.device_capacity_bytes;
}

// linear_add updates an FP32 residual stream from FP8 and NVFP4 weights only.
bool fp32_residual_projection(NumericFormat format) {
    return format == NumericFormat::FP8_E4M3FN_ROW_BF16S || format == NumericFormat::NVFP4;
}

// The FP32 residual stream needs an FP32 form of every op that writes it: the FP8 embedding gather
// and, in each layer, the attention or GDN output projection and the MLP down projection. The
// output head reads the normed BF16 hidden state, not the stream. Bindings that meet this criterion
// must be accepted by Variant's per-profile guard, which Engine startup checks.
int verify_fp32_residual_form(const BindingPlan& bindings) {
    if (bindings.token_embedding.format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
        std::cerr << "token embedding is not FP8\n";
        return 1;
    }
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& plan = bindings.text_layers[layer];
        const WeightPlan& mixer_output =
            plan.is_full_attention ? plan.attention.output : plan.gdn.output;
        if (!fp32_residual_projection(mixer_output.format) ||
            !fp32_residual_projection(plan.mlp.down.format)) {
            std::cerr << "layer " << layer << " writes the residual stream without an FP32 form\n";
            return 1;
        }
    }
    if (!Variant::fp32_residual_supported(WeightsProfile::Qwen38Nvfp4FullA)) {
        std::cerr << "Variant rejects an FP32 residual for full A, whose bindings support it\n";
        return 1;
    }
    return 0;
}

int verify_full_a(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::Qwen38Nvfp4FullA) {
        std::cerr << "identity '" << reader.identity().model_id << "/"
                  << reader.identity().weights_id << "' did not resolve to Qwen38Nvfp4FullA\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::Qwen38Nvfp4FullA, mtp_features());
    const BindingPlan& bindings = plan.bindings;
    if (!valid_nvfp4(bindings.output_head)) {
        std::cerr << "output head is not NVFP4 with valid divisors\n";
        return 1;
    }
    std::size_t nvfp4_mlp = 0;
    for (const TextLayerPlan& layer : bindings.text_layers) {
        nvfp4_mlp += valid_nvfp4(layer.mlp.gate_up) && valid_nvfp4(layer.mlp.down) ? 1 : 0;
        if (layer.is_full_attention) {
            const auto& fused = std::get<FusedAttentionProjectionPlan>(layer.attention.projection);
            if (fused.query_key_gate_value.format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
                layer.attention.output.format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
                std::cerr << "attention left FP8\n";
                return 1;
            }
        } else {
            const auto& fused = std::get<FusedGdnInputProjectionPlan>(layer.gdn.input_projection);
            if (fused.query_key_value_z.format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
                layer.gdn.output.format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
                std::cerr << "GDN left FP8\n";
                return 1;
            }
        }
    }
    if (nvfp4_mlp != kTextLayers) {
        std::cerr << "expected every MLP in NVFP4, found " << nvfp4_mlp << '\n';
        return 1;
    }
    if (const int result = verify_fp32_residual_form(bindings); result != 0) { return result; }
    std::cout << "full A: " << plan.materialization.device_objects.size() << " device objects, "
              << plan.materialization.device_capacity_bytes << " device bytes (MTP, optimized "
              << "proposal head)\n";

    ninfer::artifact::Binder dflash2_binder(reader);
    try {
        (void)bind_artifact(dflash2_binder, WeightsProfile::Qwen38Nvfp4FullA,
                            {.vision        = false,
                             .speculative   = ninfer::SpeculativeBackend::DFlash2,
                             .proposal_head = ninfer::ProposalHead::Full});
        std::cerr << "DFlash2 with an NVFP4 output head was accepted\n";
        return 1;
    } catch (const std::invalid_argument& error) {
        std::cout << "DFlash2 refused: " << error.what() << '\n';
    }
    return 0;
}

} // namespace

int main() {
    const std::filesystem::path full_a =
        environment_path("NINFER_QWEN3_8_27B_NVFP4_FULL_A_WEIGHTS");
    if (full_a.empty() || !std::filesystem::is_regular_file(full_a)) {
        std::cerr << "skip: set NINFER_QWEN3_8_27B_NVFP4_FULL_A_WEIGHTS\n";
        return 77;
    }
    try {
        if (const int result = verify_full_a(full_a); result != 0) { return result; }
        const std::filesystem::path mixed = environment_path("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS");
        if (!mixed.empty() && std::filesystem::is_regular_file(mixed)) {
            const std::uint64_t before = device_bytes(mixed, WeightsProfile::Qwen38Nvfp4);
            const std::uint64_t after  = device_bytes(full_a, WeightsProfile::Qwen38Nvfp4FullA);
            std::cout << "device weight bytes: mixed " << before << ", full A " << after
                      << ", saved " << static_cast<std::int64_t>(before - after) << '\n';
            if (after >= before) {
                std::cerr << "full A does not reduce device weight bytes\n";
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "binding failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
