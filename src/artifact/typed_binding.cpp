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
// Nommage : v2 (plat, fusionné) -> v3 (logique, par sous-tenseur)
// ---------------------------------------------------------------------------------------------

void replace_all(std::string& text, std::string_view from, std::string_view to) {
    for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

// Renommages purs (un nom v2 -> un binding v3 de même couverture).
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
    return out;
}

// Un tenseur fusionné v2 = la concaténation, dans cet ordre, de ces feuilles v3.
std::vector<std::string_view> fused_leaves(std::string_view suffix) {
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
    return {};
}

// Développe un nom v3 canonique en la liste des bindings logiques v3 qui le composent.
std::vector<std::string> logical_names(const Directory& directory, std::string_view v3_name) {
    if (directory.bindings.contains(v3_name)) { return {std::string(v3_name)}; }
    const auto slash = v3_name.rfind('/');
    if (slash == std::string_view::npos) { return {}; }
    const auto leaves = fused_leaves(v3_name.substr(slash + 1));
    std::vector<std::string> out;
    out.reserve(leaves.size());
    for (const auto leaf : leaves) {
        out.emplace_back(std::string(v3_name.substr(0, slash + 1)) + std::string(leaf));
    }
    return out;
}

std::uint64_t object_elements(const Directory& directory, ObjectHandle handle) {
    return weight_element_count(directory.tensor(handle).shape);
}

// Un Binding v3 doit couvrir intégralement un seul objet physique pour être exposable
// comme tenseur v2 (les porteurs adressent des objets entiers, jamais des tranches).
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

// Résout un nom de tenseur v2 vers l'objet physique v3 que ses feuilles couvrent bout à bout.
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

// ".../<group>/<x>_projection/input_scale_divisor" -> nom v2 du poids consommateur.
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
    return owner; // gate_up, down, output : déjà le nom v2 du poids
}

std::uint32_t read_u32_le_bytes(std::span<const std::byte> bytes, std::string_view label) {
    if (bytes.size() != 4) {
        throw ArtifactError(std::string(label) + ": expected one FP32 word, got " +
                            std::to_string(bytes.size()) + " bytes");
    }
    return read_u32_le(bytes.data());
}

// Résout un diviseur d'entrée v2 vers l'auxiliaire `activation_input_divisor` v3. Un poids
// fusionné v2 (gate_up) a une feuille v3 par sous-tenseur, chacune avec son auxiliaire ; le
// kernel Volta n'en consomme qu'un, on exige donc qu'ils soient identiques.
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
        if (found.size() != 1) {
            throw ArtifactError(quote_name(v2_name) + ": expected exactly one Use of " + quote_name(leaf) +
                                " with an activation input divisor, found " +
                                std::to_string(found.size()));
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
    // v2 : "frontend/<role>" ; v3 : chaque composant porte ses propres ressources
    // (tokenizer sous "text", preprocessor sous "vision").
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
// Matérialisation
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
