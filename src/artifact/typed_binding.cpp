#include "artifact/typed_binding.h"

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/tensor.h"
#include "core/weight_view.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::artifact {

namespace {

constexpr std::string_view kInputDivisorSuffix = "/input_scale_divisor";
constexpr std::string_view kActivationInputDivisor = "activation_input_divisor";

DType dtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16: return DType::BF16;
    case NumericFormat::FP32: return DType::FP32;
    case NumericFormat::I32: return DType::I32;
    default: throw std::logic_error("quantized format has no direct dtype");
    }
}

std::string quote_name(std::string_view text) { return "'" + std::string(text) + "'"; }

// ---------------------------------------------------------------------------------------------
// Naming: v2 (flat, fused) -> v3 (logical, one per sub-tensor)
// ---------------------------------------------------------------------------------------------

void replace_all(std::string& text, std::string_view from, std::string_view to) {
    for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

// Pure renames (one v2 name -> one v3 binding with the same coverage).
std::string canonical_v3_name(std::string_view name) {
    if (name == "text/draft_head") { return "proposal/head"; }
    if (name == "text/draft_head_token_ids") { return "proposal/token_ids"; }
    std::string out(name);
    if (out.starts_with("mtp/layer/")) { out = "mtp/layers/0/" + out.substr(sizeof("mtp/layer/") - 1); }
    if (out.starts_with("vision/")) {
        replace_all(out, "/norm1/", "/norm1_");
        replace_all(out, "/norm2/", "/norm2_");
        replace_all(out, "/norm/", "/norm_");
    }
    constexpr std::string_view kSharedDown = "/moe/shared_down";
    if (out.ends_with(kSharedDown)) {
        out.replace(out.size() - kSharedDown.size(), kSharedDown.size(), "/moe/shared/down");
    }
    return out;
}

// Routed experts of each Qwen3.6-35B-A3B MoE block. A v2 routed parent is expert-major
// (docs/maintainer/qwen3.6-35b-a3b-artifact.md, section 9.2): every role of expert 0, then every
// role of expert 1, and so on.
constexpr std::size_t kRoutedExperts = 256;

std::vector<std::string> routed_expert_leaves(std::initializer_list<std::string_view> roles) {
    std::vector<std::string> out;
    out.reserve(kRoutedExperts * roles.size());
    for (std::size_t expert = 0; expert < kRoutedExperts; ++expert) {
        const std::string prefix = "experts/" + std::to_string(expert) + "/";
        for (const auto role : roles) { out.push_back(prefix + std::string(role)); }
    }
    return out;
}

// A fused v2 tensor is the concatenation, in this order, of these v3 leaves.
std::vector<std::string> fused_leaves(std::string_view suffix) {
    if (suffix == "query_key_gate_value") { return {"query", "key", "gate", "value"}; }
    if (suffix == "query_key_value_z") { return {"query", "key", "value", "z"}; }
    if (suffix == "query_key_value") { return {"query", "key", "value"}; }
    if (suffix == "query_key") { return {"query", "key"}; }
    if (suffix == "gate_value") { return {"gate", "value"}; }
    if (suffix == "value_z") { return {"value", "z"}; }
    if (suffix == "gate_up") { return {"gate", "up"}; }
    if (suffix == "a_b_projection") { return {"a_projection", "b_projection"}; }
    if (suffix == "qkv") { return {"query", "key", "value"}; }
    if (suffix == "qkv_bias") { return {"query_bias", "key_bias", "value_bias"}; }
    // MoE: router rows, then the shared-expert score row; shared gate rows, then shared up rows.
    if (suffix == "router_shared_gate") { return {"router", "shared_score"}; }
    if (suffix == "shared_gate_up") { return {"shared/gate", "shared/up"}; }
    if (suffix == "routed_gate_up") { return routed_expert_leaves({"gate", "up"}); }
    if (suffix == "routed_down") { return routed_expert_leaves({"down"}); }
    return {};
}

// Expands a canonical v3 name into the v3 logical bindings that compose it.
std::vector<std::string> logical_names(const Directory& directory, std::string_view v3_name) {
    if (directory.bindings.contains(v3_name)) { return {std::string(v3_name)}; }
    const auto slash = v3_name.rfind('/');
    if (slash == std::string_view::npos) { return {}; }
    auto leaves = fused_leaves(v3_name.substr(slash + 1));
    for (auto& leaf : leaves) { leaf.insert(0, v3_name.substr(0, slash + 1)); }
    return leaves;
}

std::uint64_t object_elements(const Directory& directory, ObjectHandle handle) {
    return weight_element_count(directory.tensor(handle).shape);
}

// A v3 Binding must cover exactly one whole physical object to be exposed as a v2 tensor (the
// target packages address whole objects, never slices).
ObjectHandle whole_object_of(const Directory& directory, const Binding& binding,
                             std::string_view label) {
    if (binding.parts.size() != 1) {
        throw ArtifactError(std::string(label) + ": v3 Binding spans several objects");
    }
    const Part& part = binding.parts.front();
    if (part.begin != 0 || part.end != object_elements(directory, part.object)) {
        throw ArtifactError(std::string(label) + ": v3 Binding is a slice, not a whole object");
    }
    return part.object;
}

const Binding& require_binding(const Directory& directory, std::string_view v3_name,
                               std::string_view v2_name) {
    const auto found = directory.bindings.find(v3_name);
    if (found == directory.bindings.end()) {
        throw ArtifactError("missing logical parameter " + quote_name(v2_name) + " (v3 " +
                            quote_name(v3_name) + ")");
    }
    return found->second;
}

// Resolves a v2 tensor name to the physical v3 object that its leaves cover end to end.
ObjectHandle resolve_tensor_object(const Directory& directory, std::string_view v2_name) {
    const std::string v3_name = canonical_v3_name(v2_name);
    const auto leaves         = logical_names(directory, v3_name);
    if (leaves.empty()) {
        throw ArtifactError("missing logical parameter " + quote_name(v2_name) + " (v3 " +
                            quote_name(v3_name) + ", no fused mapping)");
    }
    if (leaves.size() == 1) {
        return whole_object_of(directory, require_binding(directory, leaves.front(), v2_name),
                               v2_name);
    }
    ObjectHandle object;
    std::uint64_t cursor = 0;
    for (const auto& leaf : leaves) {
        const Binding& binding = require_binding(directory, leaf, v2_name);
        if (binding.parts.size() != 1) {
            throw ArtifactError(quote_name(leaf) + ": fused member spans several objects");
        }
        const Part& part = binding.parts.front();
        if (cursor == 0) {
            object = part.object;
        } else if (!(part.object == object)) {
            throw ArtifactError(quote_name(v2_name) + ": fused members live in different objects (" +
                                quote_name(leaf) + ")");
        }
        if (part.begin != cursor) {
            throw ArtifactError(quote_name(v2_name) + ": fused member " + quote_name(leaf) +
                                " is not contiguous with its predecessor");
        }
        cursor = part.end;
    }
    if (cursor != object_elements(directory, object)) {
        throw ArtifactError(quote_name(v2_name) + ": fused members do not cover the whole object");
    }
    return object;
}

// ".../<group>/<x>_projection/input_scale_divisor" -> v2 name of the consuming weight.
std::string divisor_owner_name(std::string_view v2_name) {
    std::string owner(v2_name.substr(0, v2_name.size() - kInputDivisorSuffix.size()));
    constexpr std::string_view kProjection = "_projection";
    if (!owner.ends_with(kProjection)) {
        throw ArtifactError(quote_name(v2_name) + ": unrecognised input divisor name");
    }
    owner.erase(owner.size() - kProjection.size());
    const auto slash = owner.rfind('/');
    if (slash == std::string::npos) {
        throw ArtifactError(quote_name(v2_name) + ": unrecognised input divisor name");
    }
    const std::string_view x = std::string_view(owner).substr(slash + 1);
    if (x == "input") {
        const std::string prefix = owner.substr(0, slash + 1);
        if (prefix.ends_with("attention/")) { return prefix + "query_key_gate_value"; }
        if (prefix.ends_with("gdn/")) { return prefix + "query_key_value_z"; }
        throw ArtifactError(quote_name(v2_name) + ": input projection group is unknown");
    }
    return owner; // gate_up, down, output: already the v2 weight name
}

std::uint32_t read_u32_le_bytes(std::span<const std::byte> bytes, std::string_view label) {
    if (bytes.size() != 4) {
        throw ArtifactError(std::string(label) + ": expected one FP32 word, got " +
                            std::to_string(bytes.size()) + " bytes");
    }
    return read_u32_le(bytes.data());
}

// Resolves a v2 input divisor to the v3 `activation_input_divisor` auxiliary. A fused v2 weight
// (gate_up) has one v3 leaf per sub-tensor, each with its own auxiliary; the Volta kernel consumes
// only one, so they must be identical.
ObjectHandle resolve_input_divisor(Binder& binder, std::string_view v2_name) {
    const Directory& directory = binder.reader().directory();
    const std::string owner    = divisor_owner_name(v2_name);
    const auto leaves          = logical_names(directory, canonical_v3_name(owner));
    if (leaves.empty()) {
        throw ArtifactError(quote_name(v2_name) + ": no v3 parameter for " + quote_name(owner));
    }
    std::vector<ObjectHandle> divisors;
    for (const auto& leaf : leaves) {
        std::vector<ObjectHandle> found;
        for (auto it = directory.uses.lower_bound({leaf, std::string()});
             it != directory.uses.end() && it->first.first == leaf; ++it) {
            const auto aux = it->second.auxiliaries.find(kActivationInputDivisor);
            if (aux != it->second.auxiliaries.end()) {
                found.push_back(whole_object_of(directory, aux->second, v2_name));
            }
        }
        // A parameter read from several inputs (the output head: text, MTP and DFlash2 hidden
        // states) may carry one divisor per Use, provided they all name the same object.
        if (found.empty()) {
            throw ArtifactError(quote_name(v2_name) + ": no Use of " + quote_name(leaf) +
                                " carries an activation input divisor");
        }
        const bool shared =
            std::all_of(found.begin(), found.end(),
                        [&](const ObjectHandle& handle) { return handle == found.front(); });
        if (!shared) {
            throw ArtifactError(quote_name(v2_name) + ": the " + std::to_string(found.size()) +
                                " Uses of " + quote_name(leaf) +
                                " name different activation input divisor objects");
        }
        divisors.push_back(found.front());
    }
    const std::uint32_t reference =
        read_u32_le_bytes(binder.host_object(divisors.front()), v2_name);
    for (std::size_t i = 1; i < divisors.size(); ++i) {
        const std::uint32_t bits = read_u32_le_bytes(binder.host_object(divisors[i]), v2_name);
        if (bits != reference) {
            throw ArtifactError(quote_name(v2_name) + ": fused members carry different input divisors (" +
                                quote_name(leaves[i]) + ")");
        }
    }
    return divisors.front();
}

void validate_geometry(const Reader& reader, ObjectHandle handle, std::string_view v2_name,
                       NumericFormat format, std::span<const std::uint64_t> shape) {
    const WeightGeometry& geometry = reader.geometry(handle);
    if (geometry.format != qtype_for(format)) {
        throw ArtifactError(quote_name(v2_name) + ": artifact format " +
                            std::string(format_name(geometry.format)) + " differs from requested " +
                            std::string(format_name(format)));
    }
    if (!shape.empty() && !std::equal(shape.begin(), shape.end(), geometry.shape.begin(),
                                      geometry.shape.end())) {
        std::string expected;
        std::string actual;
        for (const auto dim : shape) { expected += std::to_string(dim) + ","; }
        for (const auto dim : geometry.shape) { actual += std::to_string(dim) + ","; }
        throw ArtifactError(quote_name(v2_name) + ": artifact shape [" + actual +
                            "] differs from requested [" + expected + "]");
    }
}

ObjectHandle resolve_and_validate(Binder& binder, std::string_view name, NumericFormat format,
                                  std::span<const std::uint64_t> shape) {
    const ObjectHandle handle = name.ends_with(kInputDivisorSuffix)
                                    ? resolve_input_divisor(binder, name)
                                    : resolve_tensor_object(binder.reader().directory(), name);
    validate_geometry(binder.reader(), handle, name, format, shape);
    return handle;
}

void apply_placement(Binder& binder, ObjectHandle handle, TensorPlacement placement) {
    switch (placement) {
    case TensorPlacement::Device:
    case TensorPlacement::Any:
        binder.require_device(handle);
        break;
    case TensorPlacement::Host:
        (void)binder.host_object(handle);
        break;
    case TensorPlacement::ValidateOnly:
        break;
    }
}

const WeightParent& resident_parent(const MaterializedArtifact& materialized, ObjectHandle handle) {
    if (materialized.has_device(handle)) { return materialized.device_parent(handle); }
    return materialized.host_parent(handle);
}

WeightView whole_view(const WeightParent& parent, Shape shape) {
    WeightView view;
    view.shape = std::move(shape);
    view.parts.push_back({&parent, 0, parent.geometry.elements});
    return view;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------------------------

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16: return "BF16";
    case NumericFormat::FP32: return "FP32";
    case NumericFormat::NVFP4: return "NVFP4";
    case NumericFormat::FP8_E4M3: return "FP8_E4M3FN_ROW_BF16S";
    case NumericFormat::Q4: return "Q4G64_F16S";
    case NumericFormat::Q5: return "Q5G64_F16S";
    case NumericFormat::Q6: return "Q6G64_F16S";
    case NumericFormat::W8: return "W8G32_F16S";
    case NumericFormat::I32: return "I32";
    }
    return "unknown";
}

QType qtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16: return QType::BF16;
    case NumericFormat::FP32: return QType::FP32;
    case NumericFormat::I32: return QType::INT32;
    case NumericFormat::Q4: return QType::Q4_G64_FP16;
    case NumericFormat::Q5: return QType::Q5_G64_FP16;
    case NumericFormat::Q6: return QType::Q6_G64_FP16;
    case NumericFormat::W8: return QType::Q8_G32_FP16;
    case NumericFormat::NVFP4: return QType::NVFP4;
    case NumericFormat::FP8_E4M3: return QType::FP8_E4M3FN_ROW_BF16;
    }
    throw std::logic_error("unhandled numeric format");
}


// ---------------------------------------------------------------------------------------------
// Binding
// ---------------------------------------------------------------------------------------------

ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                         std::initializer_list<std::uint64_t> shape, TensorPlacement placement) {
    const ObjectHandle handle = resolve_and_validate(
        binder, name, format, std::span<const std::uint64_t>(shape.begin(), shape.size()));
    apply_placement(binder, handle, placement);
    return handle;
}

ObjectHandle bind_device_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                std::initializer_list<std::uint64_t> shape) {
    return bind_tensor(binder, name, format, shape, TensorPlacement::Device);
}

ObjectHandle bind_raw_resource(Binder& binder, std::string_view name) {
    // v2: "frontend/<role>"; v3: each component carries its own resources (tokenizer under
    // "text", preprocessor under "vision").
    const auto slash            = name.rfind('/');
    const std::string_view role = slash == std::string_view::npos ? name : name.substr(slash + 1);
    const Directory& directory  = binder.reader().directory();
    std::vector<std::string_view> candidates;
    if (slash != std::string_view::npos && name.substr(0, slash) != "frontend") {
        candidates.push_back(name.substr(0, slash));
    }
    candidates.push_back("text");
    for (const auto& [component_name, component] : directory.components) {
        if (std::find(candidates.begin(), candidates.end(), component_name) == candidates.end()) {
            candidates.push_back(component_name);
        }
    }
    for (const auto candidate : candidates) {
        const auto component = directory.components.find(candidate);
        if (component != directory.components.end() &&
            component->second.resources.contains(role)) {
            return binder.resource(candidate, role);
        }
    }
    throw ArtifactError("missing resource " + quote_name(name) + " in every component");
}

std::uint32_t nvfp4_weight_divisor_bits(Binder& binder, ObjectHandle handle) {
    const Reader& reader           = binder.reader();
    const WeightGeometry& geometry = reader.geometry(handle);
    if (geometry.format != QType::NVFP4) {
        throw ArtifactError(reader.directory().tensor(handle).id + ": not an NVFP4 object");
    }
    std::array<std::byte, 4> word{};
    reader.read_into(checked_add(reader.directory().tensor(handle).offset, geometry.divisor_offset,
                                 "NVFP4 weight divisor offset"),
                     word);
    return read_u32_le(word.data());
}

// ---------------------------------------------------------------------------------------------
// Materialization
// ---------------------------------------------------------------------------------------------

Tensor materialized_tensor(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::initializer_list<std::int32_t> internal_shape) {
    const WeightParent& parent = resident_parent(materialized, handle);
    if (parent.geometry.format != qtype_for(format)) {
        throw ArtifactError("materialized tensor format differs from its binding");
    }
    Tensor out = weight_tensor(whole_view(parent, parent.geometry.shape), internal_shape);
    if (out.dtype != dtype_for(format)) {
        throw ArtifactError("materialized tensor dtype differs from its binding");
    }
    return out;
}

::ninfer::Weight materialized_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                                     NumericFormat format, std::int32_t rows,
                                     std::int32_t columns) {
    if (format == NumericFormat::NVFP4) {
        throw std::invalid_argument(
            "materialized_weight: NVFP4 requires target-validated weight and input divisors");
    }
    const WeightParent& parent = materialized.device_parent(handle);
    if (parent.geometry.format != qtype_for(format)) {
        throw ArtifactError("materialized weight format differs from its binding");
    }
    return native_weight(
        whole_view(parent, {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)}),
        0.0F);
}

::ninfer::Weight materialized_nvfp4_weight(const MaterializedArtifact& materialized,
                                           ObjectHandle handle, std::int32_t rows,
                                           std::int32_t columns, float input_scale_divisor) {
    const WeightParent& parent = materialized.device_parent(handle);
    if (parent.geometry.format != QType::NVFP4) {
        throw ArtifactError("materialized NVFP4 weight has another format");
    }
    if (!std::isfinite(input_scale_divisor) || input_scale_divisor <= 0.0F) {
        throw ArtifactError("NVFP4 input divisor must be finite and positive");
    }
    return native_weight(
        whole_view(parent, {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)}),
        input_scale_divisor);
}

std::span<const std::byte> host_bytes(Binder& binder, ObjectHandle handle) {
    return binder.host_object(handle);
}

} // namespace ninfer::artifact
