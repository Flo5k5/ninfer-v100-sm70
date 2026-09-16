#include "artifact/typed_binding.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/tensor.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace ninfer::artifact {

namespace {

std::pair<std::string_view, std::string_view> split_name(std::string_view name) {
    // v3 : les ressources frontend vivent sous "text" pas sous "frontend"
    if (name.starts_with("frontend/")) {
        return {"text", name.substr(sizeof("frontend/") - 1)};
    }
    const auto slash = name.find('/');
    if (slash == std::string_view::npos) {
        return {"default", name};
    }
    return {name.substr(0, slash), name.substr(slash + 1)};
}

std::string_view strip_role(std::string_view name) {
    const auto slash = name.rfind('/');
    return slash == std::string_view::npos ? name : name.substr(slash + 1);
}

} // namespace

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16: return "BF16";
    case NumericFormat::FP32: return "FP32";
    case NumericFormat::NVFP4: return "NVFP4";
    case NumericFormat::FP8_E4M3: return "FP8_E4M3";
    case NumericFormat::Q4: return "Q4";
    case NumericFormat::Q5: return "Q5";
    case NumericFormat::Q6: return "Q6";
    case NumericFormat::W8: return "W8";
    case NumericFormat::I32: return "I32";
    }
    return "unknown";
}

std::string_view layout_name(StorageLayout layout) noexcept {
    switch (layout) {
    case StorageLayout::ContiguousLeV1: return "contiguous_le_v1";
    case StorageLayout::RowSplitK128V1: return "row_split_k128_v1";
    case StorageLayout::BlockScaleK16M128x4V1: return "block_scale_k16_m128x4_v1";
    case StorageLayout::RowScaleV1: return "row_scale_v1";
    }
    return "unknown";
}

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    (void)format;
    BlockScaleGeometry g;
    g.rows           = shape.size() > 0 ? shape[0] : 0;
    g.columns        = shape.size() > 1 ? shape[1] : 0;
    g.groups_per_row = (g.columns + 31) / 32; // K16 M128 x4 : deux groupes par tuile de 32
    g.k_tiles        = (g.columns + 511) / 512;
    // NVFP4 : 4 bits par poids + FP8 par groupe de 16.
    g.code_plane_bytes      = (g.rows * g.columns + 1) / 2;
    g.scale_plane_offset    = g.code_plane_bytes;
    g.scale_plane_bytes     = g.rows * ((g.columns + 15) / 16);
    g.weight_divisor_offset = g.scale_plane_offset + g.scale_plane_bytes;
    g.encoded_bytes         = g.weight_divisor_offset + 4;
    return g;
}

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    (void)format;
    RowScaleGeometry g;
    g.rows              = shape.size() > 0 ? shape[0] : 0;
    g.columns           = shape.size() > 1 ? shape[1] : 0;
    g.code_plane_bytes  = g.rows * g.columns; // W8 : 1 octet par poids
    g.scale_plane_offset = g.code_plane_bytes;
    g.scale_plane_bytes = g.rows * 2; // FP16 par ligne
    g.encoded_bytes     = g.scale_plane_offset + g.scale_plane_bytes;
    return g;
}

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    (void)format;
    RowSplitGeometry g;
    g.rows          = shape.size() > 0 ? shape[0] : 0;
    g.columns       = shape.size() > 1 ? shape[1] : 0;
    g.tile_columns  = 128;
    g.tile_count    = (g.columns + g.tile_columns - 1) / g.tile_columns;
    g.tile_bytes    = g.rows * g.tile_columns; // W8 par tuile
    g.encoded_bytes = g.tile_count * g.tile_bytes;
    return g;
}

ObjectHandle require_tensor(Binder& binder, std::string_view name, NumericFormat format,
                            StorageLayout layout, std::span<const std::uint64_t> shape) {
    (void)format;
    (void)layout;
    (void)shape;
    const auto [component, role] = split_name(strip_role(name));
    return binder.resource(component, role);
}

ObjectHandle require_resource(Binder& binder, std::string_view name, ResourceEncoding encoding) {
    (void)encoding;
    const auto [component, role] = split_name(strip_role(name));
    return binder.resource(component, role);
}

ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                         std::initializer_list<std::uint64_t> shape, TensorPlacement placement) {
    (void)format;
    (void)shape;
    (void)placement;
    const auto [component, role] = split_name(name);
    return binder.resource(component, role);
}

ObjectHandle bind_device_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                std::initializer_list<std::uint64_t> shape) {
    return bind_tensor(binder, name, format, shape, TensorPlacement::Device);
}

ObjectHandle bind_raw_resource(Binder& binder, std::string_view name) {
    const auto [component, role] = split_name(name);
    return binder.resource(component, role);
}

Tensor materialized_tensor(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::initializer_list<std::int32_t> internal_shape) {
    (void)format;
    const auto bytes = materialized.host_bytes(handle);
    void* data       = const_cast<void*>(static_cast<const void*>(bytes.data()));
    std::vector<std::int32_t> dims(internal_shape);
    if (dims.empty()) {
        dims.push_back(static_cast<std::int32_t>(bytes.size() / sizeof(std::uint16_t)));
    }
    if (dims.size() == 1) {
        return Tensor(data, DType::BF16, {dims[0]});
    }
    return Tensor(data, DType::BF16, {dims[0], dims[1]});
}

::ninfer::Weight materialized_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::int32_t rows, std::int32_t columns) {
    (void)format;
    const auto bytes = materialized.host_bytes(handle);
    Weight w{};
    w.ndim     = 2;
    w.shape[0] = rows;
    w.shape[1] = columns;
    w.n        = rows;
    w.k        = columns;
    w.qtype    = QType::BF16_CTRL;
    w.qdata    = bytes.data();
    w.payload  = bytes.data();
    w.payload_bytes = static_cast<std::uint64_t>(bytes.size());
    return w;
}

} // namespace ninfer::artifact

namespace ninfer::artifact {

// Raccourcis que les porteurs utilisent directement sur Binder :
// remplacés par sed dans les sources .cpp vers les _compat ci-dessous.

} // namespace ninfer::artifact
