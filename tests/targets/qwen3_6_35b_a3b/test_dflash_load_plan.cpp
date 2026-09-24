#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <vector>

namespace {

using ninfer::targets::qwen3_6_35b_a3b::detail::ArtifactLoadPlan;
using ninfer::targets::qwen3_6_35b_a3b::detail::kDFlashLayers;

// Feature projection, context norm and final norm, plus eight objects per DFlash layer.
constexpr std::size_t kDFlashObjects = 3 + 8 * kDFlashLayers;

std::filesystem::path artifact_path() {
    if (const char* env = std::getenv("NINFER_QWEN3_6_35B_A3B_WEIGHTS");
        env != nullptr && *env != '\0') {
        return env;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "out/qwen3_6_35b_a3b.ninfer";
}

ninfer::targets::qwen3_6::StartupFeatures load_features(bool vision,
                                                        ninfer::SpeculativeBackend speculative) {
    return {
        .vision        = vision,
        .speculative   = speculative,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

std::vector<ninfer::artifact::ObjectHandle> dflash_objects(const ArtifactLoadPlan& plan) {
    const auto& dflash = plan.bindings.dflash;
    std::vector<ninfer::artifact::ObjectHandle> out = {
        dflash.feature_projection, dflash.context_norm, dflash.final_norm};
    for (const auto& layer : dflash.layers) {
        out.insert(out.end(), {layer.input_norm, layer.query_key_value, layer.query_norm,
                               layer.key_norm, layer.attention_output, layer.post_attention_norm,
                               layer.gate_up, layer.down});
    }
    return out;
}

std::size_t distinct_dflash_objects(const ArtifactLoadPlan& plan) {
    std::set<std::size_t> indices;
    for (const auto handle : dflash_objects(plan)) { indices.insert(handle.index); }
    return indices.size();
}

std::size_t resident_dflash_objects(const ArtifactLoadPlan& plan) {
    const auto& placed = plan.materialization.device_objects;
    std::size_t resident = 0;
    for (const auto handle : dflash_objects(plan)) {
        const bool on_device = std::any_of(placed.begin(), placed.end(), [&](const auto& placement) {
            return placement.object == handle;
        });
        resident += on_device ? 1 : 0;
    }
    return resident;
}

} // namespace

int main() {
    const std::filesystem::path path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: real 35B artifact is unavailable at " << path << '\n';
        return 77;
    }

    ninfer::artifact::Reader reader(path);
    // Every plan also keeps the six frontend resources and the proposal token ids, which the
    // binder validates, on the host. The DFlash objects are distinct, validated but nonresident
    // with MTP, and resident with DFlash.
    {
        ninfer::artifact::Binder binder(reader);
        const auto plan = ninfer::targets::qwen3_6_35b_a3b::detail::bind_artifact(
            binder, load_features(true, ninfer::SpeculativeBackend::Mtp));
        if (plan.materialization.object_count != 940 ||
            plan.materialization.device_objects.size() != 883 ||
            plan.materialization.host_objects.size() != 7 ||
            plan.materialization.device_capacity_bytes != 22'360'207'360ULL ||
            distinct_dflash_objects(plan) != kDFlashObjects || resident_dflash_objects(plan) != 0) {
            std::cerr << "MTP+Vision materialization plan changed resident weights\n";
            return 1;
        }
    }
    {
        ninfer::artifact::Binder binder(reader);
        const auto plan = ninfer::targets::qwen3_6_35b_a3b::detail::bind_artifact(
            binder, load_features(false, ninfer::SpeculativeBackend::DFlash));
        if (plan.materialization.object_count != 940 ||
            plan.materialization.device_objects.size() != 586 ||
            plan.materialization.host_objects.size() != 7 ||
            plan.materialization.device_capacity_bytes != 21'591'653'888ULL ||
            resident_dflash_objects(plan) != kDFlashObjects) {
            std::cerr << "DFlash-only materialization plan is incomplete: device_objects="
                      << plan.materialization.device_objects.size()
                      << " device_bytes=" << plan.materialization.device_capacity_bytes << '\n';
            return 1;
        }
    }
    {
        ninfer::artifact::Binder binder(reader);
        const auto plan = ninfer::targets::qwen3_6_35b_a3b::detail::bind_artifact(
            binder, load_features(true, ninfer::SpeculativeBackend::DFlash));
        if (plan.materialization.object_count != 940 ||
            plan.materialization.device_objects.size() != 919 ||
            plan.materialization.host_objects.size() != 7 ||
            plan.materialization.device_capacity_bytes != 21'872'326'656ULL ||
            resident_dflash_objects(plan) != kDFlashObjects) {
            std::cerr << "DFlash+Vision materialization plan is incomplete: device_objects="
                      << plan.materialization.device_objects.size()
                      << " device_bytes=" << plan.materialization.device_capacity_bytes << '\n';
            return 1;
        }
    }
    return 0;
}
