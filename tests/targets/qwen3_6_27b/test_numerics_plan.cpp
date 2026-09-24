// Startup validation and workspace planning of the text numerics options
// (EngineOptions::text_residual and prefill_attention) on the 27B package. Planning needs a CUDA
// Volta device for the compute-capability check, but no artifact.
//
// The FP32 residual stream is accepted only for qwen3.8-27b/nvfp4 without DFlash, and unknown
// option values are rejected. The residual is the root allocation of the prefill chunk, the
// ordinary decode batch (TextContext::ordinary_decode_batch) and the MTP lookup verify aggregate
// (TextContext::target_verify_batch_impl), so planning it in FP32 must grow each of these phases
// by exactly two more bytes per element. The prefill plan must also reserve the staging of the
// selected attention kernel.

#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/layouts.h"

#include "core/device.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using ninfer::targets::qwen3_6_27b::Package;
using WeightsProfile = Package::WeightsProfile;
using WorkspacePlan  = ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime::WorkspacePlan;

constexpr std::size_t kHidden = 5120;

ninfer::EngineOptions production_options() {
    ninfer::EngineOptions options;
    options.max_context                      = 32768;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(32768);
    options.prefill_chunk                    = 2048;
    options.kv_cache                         = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 4;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.use_cuda_graph                   = false;
    options.context_cache.device_state_slots = 1;
    return options;
}

WorkspacePlan workspace_plan(ninfer::DeviceContext& device, const ninfer::EngineOptions& options,
                             WeightsProfile profile) {
    auto planner              = Package::make_sequence_planner(device, options, profile);
    const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
    const Package::SequencePlan plan = std::move(planner).finalize(pages);
    return plan.impl_->workspace;
}

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int expect_rejection(ninfer::DeviceContext& device, const ninfer::EngineOptions& options,
                     WeightsProfile profile, const std::string& fragment,
                     const std::string& label) {
    try {
        (void)workspace_plan(device, options, profile);
    } catch (const std::invalid_argument& error) {
        return check(std::string(error.what()).find(fragment) != std::string::npos,
                     label + " was rejected for another reason: " + error.what());
    }
    std::cerr << label << " was accepted\n";
    return 1;
}

int verify_rejections(ninfer::DeviceContext& device) {
    int failures = 0;
    ninfer::EngineOptions fp32 = production_options();
    fp32.text_residual         = ninfer::TextResidualStorage::Float32;
    for (const auto& [profile, name] :
         {std::pair{WeightsProfile::Qwen36GroupwiseInt, "qwen3.6-27b/groupwise-int"},
          std::pair{WeightsProfile::Qwen36Nvfp4, "qwen3.6-27b/nvfp4"},
          std::pair{WeightsProfile::Qwen38GroupwiseInt, "qwen3.8-27b/groupwise-int"}}) {
        failures += expect_rejection(device, fp32, profile, "FP32 text residual",
                                     std::string("an FP32 residual for ") + name);
    }

    ninfer::EngineOptions dflash      = fp32;
    dflash.speculative.backend        = ninfer::SpeculativeBackend::DFlash2;
    dflash.speculative.draft_tokens   = 7;
    dflash.speculative.proposal_head  = ninfer::ProposalHead::Full;
    failures += expect_rejection(device, dflash, WeightsProfile::Qwen38Nvfp4, "DFlash",
                                 "an FP32 residual with DFlash2");

    ninfer::EngineOptions unknown_residual = production_options();
    unknown_residual.text_residual         = static_cast<ninfer::TextResidualStorage>(7);
    failures += expect_rejection(device, unknown_residual, WeightsProfile::Qwen38Nvfp4,
                                 "text residual", "an unknown text residual storage");
    ninfer::EngineOptions unknown_kernel = production_options();
    unknown_kernel.prefill_attention     = static_cast<ninfer::PrefillAttentionKernel>(9);
    failures += expect_rejection(device, unknown_kernel, WeightsProfile::Qwen38Nvfp4,
                                 "prefill attention", "an unknown prefill attention kernel");

    try {
        (void)workspace_plan(device, fp32, WeightsProfile::Qwen38Nvfp4);
    } catch (const std::exception& error) {
        std::cerr << "an FP32 residual with MTP on qwen3.8-27b/nvfp4 was rejected: "
                  << error.what() << '\n';
        ++failures;
    }
    return failures;
}

int verify_residual_sizing(ninfer::DeviceContext& device) {
    int failures = 0;
    const auto compare = [&](ninfer::EngineOptions options, const char* label,
                             const std::function<std::size_t(const WorkspacePlan&)>& phase,
                             std::size_t columns) {
        options.text_residual   = ninfer::TextResidualStorage::BFloat16;
        const WorkspacePlan bf16 = workspace_plan(device, options, WeightsProfile::Qwen38Nvfp4);
        options.text_residual   = ninfer::TextResidualStorage::Float32;
        const WorkspacePlan fp32 = workspace_plan(device, options, WeightsProfile::Qwen38Nvfp4);
        const std::size_t expected = kHidden * columns * 2;
        std::cout << label << ": BF16 " << phase(bf16) << " FP32 " << phase(fp32)
                  << " bytes, expected growth " << expected << '\n';
        failures += check(phase(fp32) == phase(bf16) + expected,
                          std::string(label) + ": the FP32 residual was not planned");
    };

    // Prefill: the chunk-wide residual root.
    compare(production_options(), "prefill chunk", [](const WorkspacePlan& plan) {
        return plan.text_prefill;
    }, 2048);
    compare(production_options(), "MTP prefill chunk", [](const WorkspacePlan& plan) {
        return plan.mtp_prefill;
    }, 2048);
    ninfer::EngineOptions scoring = production_options();
    scoring.purpose               = ninfer::EnginePurpose::CausalScoring;
    scoring.max_context           = 4096;
    scoring.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    scoring.prefill_chunk         = 1024;
    scoring.speculative           = {};
    compare(scoring, "causal scoring chunk", [](const WorkspacePlan& plan) {
        return plan.text_prefill;
    }, 1024);

    // Ordinary decode, one sequence: [hidden, 1]. (With several batch sizes the round is the
    // largest of their plans, and the small-T attention split can make a smaller batch the peak.)
    ninfer::EngineOptions ordinary = production_options();
    ordinary.speculative           = {};
    compare(ordinary, "ordinary decode", [](const WorkspacePlan& plan) {
        return plan.ordinary_round;
    }, 1);

    // MTP lookup verify at the largest batch: [hidden, 8 * lookup width].
    ninfer::EngineOptions verify = production_options();
    verify.max_concurrency       = 8;
    verify.max_context           = 4096;
    verify.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(32768);
    compare(verify, "MTP lookup verify batch 8", [](const WorkspacePlan& plan) {
        return plan.mtp_round;
    }, 8 * ninfer::targets::qwen3_6::kMtpLookupMaximumWidth);
    return failures;
}

int verify_prefill_attention_sizing(ninfer::DeviceContext& device) {
    // A context long enough that the staged keys and values of the wide prefill attention (two
    // FP16 [256, 4, keys] planes) exceed every other stage of the prefill chunk.
    const auto prefill = [&](ninfer::PrefillAttentionKernel kernel) {
        ninfer::EngineOptions options = production_options();
        options.max_context           = 196608;
        options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(196608);
        options.prefill_attention     = kernel;
        return workspace_plan(device, options, WeightsProfile::Qwen38Nvfp4).text_prefill;
    };
    const std::size_t automatic = prefill(ninfer::PrefillAttentionKernel::Automatic);
    const std::size_t splitd    = prefill(ninfer::PrefillAttentionKernel::SplitD);
    const std::size_t flash     = prefill(ninfer::PrefillAttentionKernel::Flash);
    const std::size_t reference = prefill(ninfer::PrefillAttentionKernel::Reference);
    std::cout << "prefill workspace: auto " << automatic << " splitd " << splitd << " flash "
              << flash << " reference " << reference << '\n';
    // The 27B geometry resolves automatic to split-D; the direct kernel stages nothing.
    return check(automatic == splitd && flash != splitd && reference < splitd &&
                     reference < flash,
                 "the prefill plan does not follow the attention kernel selection");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        ninfer::DeviceContext device(0);
        int failures = verify_rejections(device);
        failures += verify_residual_sizing(device);
        failures += verify_prefill_attention_sizing(device);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen3_6_27b numerics planning\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "qwen3_6_27b numerics planning: " << error.what() << '\n';
        return 1;
    }
}
