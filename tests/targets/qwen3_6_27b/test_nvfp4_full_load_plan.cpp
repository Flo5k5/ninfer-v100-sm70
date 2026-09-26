// Host-only binding check for the derived Qwen3.8 NVFP4 profiles, one per run:
//   full A (no argument): MLP 0-63 and the output head in NVFP4, the token embedding, attention
//     and GDN in FP8;
//   full B (--full-b): full A with the attention and GDN input projections in NVFP4 too;
//   full C (--full-c): full B with the attention and GDN output projections in NVFP4 too.
// Binding reads the directory and a few small host objects (divisors, proposal ids); nothing here
// touches a device. The bound formats must also give the text residual stream its FP32 form, which
// Variant must then accept for the profile.
//
//   NINFER_QWEN3_8_27B_NVFP4_FULL_A_WEIGHTS=<full A artifact>
//   NINFER_QWEN3_8_27B_NVFP4_FULL_B_WEIGHTS=<full B artifact>                         (--full-b)
//   NINFER_QWEN3_8_27B_NVFP4_FULL_C_WEIGHTS=<full C artifact>                         (--full-c)
//   NINFER_QWEN3_8_27B_NVFP4_WEIGHTS=<mixed NVFP4/FP8 artifact full A derives from>   (optional)
//
// When the artifact a profile derives from is also set (the mixed artifact for full A, full A for
// full B, full B for full C), the run checks that the profile needs fewer device weight bytes.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

constexpr NumericFormat kFp8 = NumericFormat::FP8_E4M3FN_ROW_BF16S;

// A v3 leaf of an NVFP4 parent and the parent rows it covers (docs/maintainer/
// qwen3.8-27b-artifact.md, section 8.1).
struct Leaf {
    std::string_view name;
    std::uint64_t first_row;
    std::uint64_t rows;
};

// Columns of the input parents ([14336,5120], [16384,5120]) and of the outputs ([5120,6144]).
constexpr std::uint64_t kInputColumns  = 5120;
constexpr std::uint64_t kOutputColumns = 6144;

constexpr std::array<Leaf, 4> kAttentionLeaves{{
    {.name = "query", .first_row = 0, .rows = 6144},
    {.name = "key", .first_row = 6144, .rows = 1024},
    {.name = "gate", .first_row = 7168, .rows = 6144},
    {.name = "value", .first_row = 13312, .rows = 1024},
}};

constexpr std::array<Leaf, 4> kGdnLeaves{{
    {.name = "query", .first_row = 0, .rows = 2048},
    {.name = "key", .first_row = 2048, .rows = 2048},
    {.name = "value", .first_row = 4096, .rows = 6144},
    {.name = "z", .first_row = 10240, .rows = 6144},
}};

// An attention or GDN output is not fused: its one leaf is the whole object.
constexpr std::array<Leaf, 1> kOutputLeaf{{{.name = "output", .first_row = 0, .rows = 5120}}};

// Bindings that meet the FP32 residual criterion below must be accepted by Variant's per-profile
// guard, which Engine startup checks. Checked at compile time, so a build that drops a profile from
// the guard fails even where this test cannot run (no artifact).
static_assert(Variant::fp32_residual_supported(WeightsProfile::Qwen38Nvfp4) &&
                  Variant::fp32_residual_supported(WeightsProfile::Qwen38Nvfp4FullA) &&
                  Variant::fp32_residual_supported(WeightsProfile::Qwen38Nvfp4FullB) &&
                  Variant::fp32_residual_supported(WeightsProfile::Qwen38Nvfp4FullC),
              "the four Qwen3.8 NVFP4 profiles must allow the FP32 text residual");

// Only full C prepacks its NVFP4 attention and GDN outputs at load; the Qwen3.6-27B nvfp4 profile,
// whose outputs are NVFP4 too, keeps them checkpoint-native.
static_assert(prepacks_nvfp4_mixer_outputs(WeightsProfile::Qwen38Nvfp4FullC) &&
                  !prepacks_nvfp4_mixer_outputs(WeightsProfile::Qwen36Nvfp4),
              "only the full-c profile prepacks its NVFP4 mixer outputs");

// A derived profile under test, and the artifact it derives from.
struct DerivedProfile {
    const char* label;
    WeightsProfile profile;
    const char* variable;
    // Formats of the attention and GDN input and output projections; every other checked role is
    // shared.
    NumericFormat input_projection;
    NumericFormat output_projection;
    WeightsProfile parent_profile;
    const char* parent_variable;
};

constexpr DerivedProfile kFullA{
    .label             = "full A",
    .profile           = WeightsProfile::Qwen38Nvfp4FullA,
    .variable          = "NINFER_QWEN3_8_27B_NVFP4_FULL_A_WEIGHTS",
    .input_projection  = kFp8,
    .output_projection = kFp8,
    .parent_profile    = WeightsProfile::Qwen38Nvfp4,
    .parent_variable   = "NINFER_QWEN3_8_27B_NVFP4_WEIGHTS",
};

constexpr DerivedProfile kFullB{
    .label             = "full B",
    .profile           = WeightsProfile::Qwen38Nvfp4FullB,
    .variable          = "NINFER_QWEN3_8_27B_NVFP4_FULL_B_WEIGHTS",
    .input_projection  = NumericFormat::NVFP4,
    .output_projection = kFp8,
    .parent_profile    = WeightsProfile::Qwen38Nvfp4FullA,
    .parent_variable   = "NINFER_QWEN3_8_27B_NVFP4_FULL_A_WEIGHTS",
};

constexpr DerivedProfile kFullC{
    .label             = "full C",
    .profile           = WeightsProfile::Qwen38Nvfp4FullC,
    .variable          = "NINFER_QWEN3_8_27B_NVFP4_FULL_C_WEIGHTS",
    .input_projection  = NumericFormat::NVFP4,
    .output_projection = NumericFormat::NVFP4,
    .parent_profile    = WeightsProfile::Qwen38Nvfp4FullB,
    .parent_variable   = "NINFER_QWEN3_8_27B_NVFP4_FULL_B_WEIGHTS",
};

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

bool has_format(const WeightPlan& weight, NumericFormat format) {
    return format == NumericFormat::NVFP4 ? valid_nvfp4(weight) : weight.format == format;
}

std::uint64_t device_bytes(const std::filesystem::path& path, WeightsProfile profile) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    return bind_artifact(binder, profile, mtp_features()).materialization.device_capacity_bytes;
}

// linear_add updates an FP32 residual stream from FP8 and NVFP4 weights only.
bool fp32_residual_projection(NumericFormat format) {
    return format == kFp8 || format == NumericFormat::NVFP4;
}

// The FP32 residual stream needs an FP32 form of every op that writes it: the FP8 embedding gather
// and, in each layer, the attention or GDN output projection and the MLP down projection. The input
// projections and the output head read normed BF16 hidden states, not the stream. The guard side of
// the contract is the static_assert above.
int verify_fp32_residual_form(const BindingPlan& bindings) {
    if (bindings.token_embedding.format != kFp8) {
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
    return 0;
}

// The v3 leaves of an NVFP4 parent, `columns` wide. Each binds one part of the bound parent object,
// the rows of section 8.1 (all of them for an output): the binder only checks that the leaves
// follow each other and cover the object, not where each one ends, so leaves with the wrong row
// counts would pass it. Each leaf has at least one Use carrying the activation input divisor, as
// the binder requires; every divisor found is a scalar FP32 object, finite and positive,
// bit-identical across the leaves, and the value the binder bound for the parent.
int verify_leaves(const ninfer::artifact::Reader& reader, const std::string& group,
                  std::span<const Leaf> leaves, std::uint64_t columns, const WeightPlan& parent) {
    const ninfer::artifact::Directory& directory = reader.directory();
    std::optional<std::uint32_t> shared;
    for (const Leaf& leaf : leaves) {
        const std::string parameter = group + std::string(leaf.name);
        const auto binding          = directory.bindings.find(parameter);
        if (binding == directory.bindings.end() || binding->second.parts.size() != 1) {
            std::cerr << parameter << ": not bound to one part of its parent\n";
            return 1;
        }
        const ninfer::artifact::Part& part = binding->second.parts.front();
        const std::uint64_t begin          = leaf.first_row * columns;
        const std::uint64_t end            = (leaf.first_row + leaf.rows) * columns;
        if (!(part.object == parent.object) || part.begin != begin || part.end != end) {
            std::cerr << parameter << ": binds elements [" << part.begin << "," << part.end
                      << ") instead of rows [" << leaf.first_row << ","
                      << leaf.first_row + leaf.rows << ") x " << columns
                      << " of the bound parent\n";
            return 1;
        }
        std::size_t divisors = 0;
        for (auto use = directory.uses.lower_bound({parameter, std::string()});
             use != directory.uses.end() && use->first.first == parameter; ++use) {
            const auto aux = use->second.auxiliaries.find("activation_input_divisor");
            if (aux == use->second.auxiliaries.end()) { continue; }
            ++divisors;
            if (aux->second.parts.size() != 1) {
                std::cerr << parameter << ": input divisor spans several objects\n";
                return 1;
            }
            const std::vector<std::byte> bytes = reader.read_object(aux->second.parts[0].object);
            if (bytes.size() != sizeof(std::uint32_t)) {
                std::cerr << parameter << ": input divisor is not one FP32 word\n";
                return 1;
            }
            const std::uint32_t bits = std::to_integer<std::uint32_t>(bytes[0]) |
                                       (std::to_integer<std::uint32_t>(bytes[1]) << 8U) |
                                       (std::to_integer<std::uint32_t>(bytes[2]) << 16U) |
                                       (std::to_integer<std::uint32_t>(bytes[3]) << 24U);
            const float value        = std::bit_cast<float>(bits);
            if (!std::isfinite(value) || value <= 0.0F) {
                std::cerr << parameter << ": input divisor " << value << " is not positive\n";
                return 1;
            }
            if (shared.has_value() && *shared != bits) {
                std::cerr << parameter << ": input divisor differs from the other leaves of "
                          << group << '\n';
                return 1;
            }
            shared = bits;
        }
        if (divisors == 0) {
            std::cerr << parameter << ": no Use carries an activation input divisor\n";
            return 1;
        }
    }
    if (!shared.has_value() || *shared != parent.input_scale_divisor_bits) {
        std::cerr << group << ": the bound input divisor is not the leaves' divisor\n";
        return 1;
    }
    return 0;
}

// Formats per role: the input and output projections in the profile's formats (with their leaf rows
// and divisors when NVFP4), every MLP and the output head in NVFP4.
int verify_formats(const ninfer::artifact::Reader& reader, const BindingPlan& bindings,
                   const DerivedProfile& tested) {
    if (!valid_nvfp4(bindings.output_head)) {
        std::cerr << "output head is not NVFP4 with valid divisors\n";
        return 1;
    }
    std::size_t nvfp4_mlp        = 0;
    std::size_t attention_inputs = 0;
    std::size_t gdn_inputs       = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& plan = bindings.text_layers[layer];
        const std::string prefix  = "text/layers/" + std::to_string(layer) + "/";
        nvfp4_mlp += valid_nvfp4(plan.mlp.gate_up) && valid_nvfp4(plan.mlp.down) ? 1 : 0;
        const bool full = plan.is_full_attention;
        const WeightPlan& input =
            full ? std::get<FusedAttentionProjectionPlan>(plan.attention.projection)
                       .query_key_gate_value
                 : std::get<FusedGdnInputProjectionPlan>(plan.gdn.input_projection)
                       .query_key_value_z;
        const WeightPlan& output = full ? plan.attention.output : plan.gdn.output;
        if (!has_format(input, tested.input_projection) ||
            !has_format(output, tested.output_projection)) {
            std::cerr << "layer " << layer << ": " << (full ? "attention" : "GDN")
                      << " input or output projection has the wrong format\n";
            return 1;
        }
        const std::string group = prefix + (full ? "attention/" : "gdn/");
        const std::span<const Leaf> input_leaves =
            full ? std::span<const Leaf>(kAttentionLeaves) : std::span<const Leaf>(kGdnLeaves);
        if (tested.input_projection == NumericFormat::NVFP4) {
            const int leaves = verify_leaves(reader, group, input_leaves, kInputColumns, input);
            if (leaves != 0) { return leaves; }
        }
        if (tested.output_projection == NumericFormat::NVFP4) {
            const int leaves = verify_leaves(reader, group, kOutputLeaf, kOutputColumns, output);
            if (leaves != 0) { return leaves; }
        }
        if (full) {
            ++attention_inputs;
        } else {
            ++gdn_inputs;
        }
    }
    if (nvfp4_mlp != kTextLayers || attention_inputs != kFullAttentionLayers ||
        gdn_inputs != kGdnLayers) {
        std::cerr << "expected " << kTextLayers << " NVFP4 MLPs, " << kFullAttentionLayers
                  << " attention and " << kGdnLayers << " GDN inputs; found " << nvfp4_mlp << ", "
                  << attention_inputs << " and " << gdn_inputs << '\n';
        return 1;
    }
    return 0;
}

int verify_profile(const DerivedProfile& tested, const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != tested.profile) {
        std::cerr << "identity '" << reader.identity().model_id << "/"
                  << reader.identity().weights_id << "' did not resolve to " << tested.label
                  << '\n';
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(binder, tested.profile, mtp_features());
    if (const int result = verify_formats(reader, plan.bindings, tested); result != 0) {
        return result;
    }
    if (const int result = verify_fp32_residual_form(plan.bindings); result != 0) { return result; }
    if (plan.bindings.prepack_nvfp4_mixer_outputs != prepacks_nvfp4_mixer_outputs(tested.profile)) {
        std::cerr << "the binding plan does not carry the profile's mixer output prepack\n";
        return 1;
    }
    std::cout << tested.label << ": " << plan.materialization.device_objects.size()
              << " device objects, " << plan.materialization.device_capacity_bytes
              << " device bytes (MTP, optimized proposal head)\n";

    ninfer::artifact::Binder dflash2_binder(reader);
    try {
        (void)bind_artifact(dflash2_binder, tested.profile,
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

// The derived profile must need fewer device weight bytes than the artifact it derives from.
int verify_smaller_than_parent(const DerivedProfile& tested, const std::filesystem::path& path) {
    const std::filesystem::path parent = environment_path(tested.parent_variable);
    if (parent.empty()) {
        std::cout << "skip device bytes: " << tested.parent_variable << " not set\n";
        return 0;
    }
    if (!std::filesystem::is_regular_file(parent)) {
        std::cout << "skip device bytes: " << tested.parent_variable << " does not name a file\n";
        return 0;
    }
    // NINFER_QWEN3_8_27B_NVFP4_WEIGHTS also names the artifact of the real FP32 tests, which may be
    // a derived one.
    if (Package::resolve_weights(ninfer::artifact::Reader(parent).identity()) !=
        tested.parent_profile) {
        std::cout << "skip device bytes: " << tested.parent_variable
                  << " does not name the artifact " << tested.label << " derives from\n";
        return 0;
    }
    const std::uint64_t before = device_bytes(parent, tested.parent_profile);
    const std::uint64_t after  = device_bytes(path, tested.profile);
    std::cout << "device weight bytes: parent " << before << ", " << tested.label << ' ' << after
              << ", saved " << static_cast<std::int64_t>(before - after) << '\n';
    if (after >= before) {
        std::cerr << tested.label << " does not reduce device weight bytes\n";
        return 1;
    }
    return 0;
}

// The profile a run checks: full A without an argument, full B or full C with its flag.
const DerivedProfile* selected_profile(int argc, char** argv) {
    if (argc == 1) { return &kFullA; }
    if (argc != 2) { return nullptr; }
    const std::string_view flag(argv[1]);
    if (flag == "--full-b") { return &kFullB; }
    if (flag == "--full-c") { return &kFullC; }
    return nullptr;
}

} // namespace

int main(int argc, char** argv) {
    const DerivedProfile* selected = selected_profile(argc, argv);
    if (selected == nullptr) {
        std::cerr << "usage: " << argv[0] << " [--full-b|--full-c]\n";
        return 2;
    }
    const DerivedProfile& tested         = *selected;
    const std::filesystem::path artifact = environment_path(tested.variable);
    if (artifact.empty() || !std::filesystem::is_regular_file(artifact)) {
        std::cerr << "skip: set " << tested.variable << '\n';
        return 77;
    }
    try {
        if (const int result = verify_profile(tested, artifact); result != 0) { return result; }
        return verify_smaller_than_parent(tested, artifact);
    } catch (const std::exception& error) {
        std::cerr << "binding failed: " << error.what() << '\n';
        return 1;
    }
}
