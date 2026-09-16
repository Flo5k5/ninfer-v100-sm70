// Compat v2 -> v3 : types et helpers de l'API artifact v2 du port Volta,
// réimplémentés par-dessus le module artifact v3 de l'amont.
// Les porteurs Volta (src/targets/**) consomment cette surface ; le moteur
// v3 (schema/binder/materializer) reste inchangé en dessous.
#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>

namespace ninfer::artifact {

class MaterializedArtifact;

enum class NumericFormat : std::uint16_t {
    BF16 = 0, FP32 = 1, NVFP4 = 2, FP8_E4M3 = 3, Q4 = 4, Q5 = 5, Q6 = 6, W8 = 7, I32 = 8,
    Q4G64_F16S = Q4, Q5G64_F16S = Q5, Q6G64_F16S = Q6, W8G32_F16S = W8,
    FP8_E4M3FN_ROW_BF16S = FP8_E4M3,
    BF16_CTRL = BF16, FP32_CTRL = FP32,
};

enum class TensorPlacement : std::uint8_t { Device = 0, Host = 1, Any = 2, ValidateOnly = 3 };

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;

struct PayloadSpan {
    std::span<const std::byte> data;
};

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowScaleGeometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowSplitGeometry {
    std::uint64_t rows             = 0;
    std::uint64_t columns          = 0;
    std::uint64_t tile_columns     = 0;
    std::uint64_t tile_count       = 0;
    std::uint64_t tile_bytes       = 0;
    std::uint64_t encoded_bytes    = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

// Binder v2 surface (requêtes par nom plat).
[[nodiscard]] ObjectHandle require_tensor(Binder& binder, std::string_view name,
                                          NumericFormat format, StorageLayout layout,
                                          std::span<const std::uint64_t> shape);
[[nodiscard]] ObjectHandle require_resource(Binder& binder, std::string_view name,
                                            ResourceEncoding encoding);

// Helpers de binding typé v2.
[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement = TensorPlacement::Device);
[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);
[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

#include "core/tensor.h"
[[nodiscard]] ::ninfer::Weight materialized_weight(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::int32_t rows, std::int32_t columns);


// Extensions Binder v2 (le port Volta les appelle directement sur Binder).
struct PayloadSpanV2 {
    std::span<const std::byte> data;
};

} // namespace ninfer::artifact

namespace ninfer::artifact {
// Helpers v2 appelés depuis les porteurs Volta.
inline PayloadSpanV2 binder_payload_v2(Binder& binder, ObjectHandle handle) {
    return PayloadSpanV2{binder.host_object(handle)};
}
inline const void* materialized_device_data_v2(const MaterializedArtifact& materialized,
                                               ObjectHandle handle) {
    return materialized.host_bytes(handle).data();
}
} // namespace ninfer::artifact
// Extensions v2 sur Binder (méthodes appelées directement par les porteurs Volta).
// Ces méthodes wrapent l'API v3 (Binder::parameter / Binder::resource).
namespace ninfer::artifact {

inline ObjectHandle Binder_require_tensor_compat(Binder& binder, std::string_view name,
                                                  NumericFormat format, StorageLayout layout,
                                                  std::span<const std::uint64_t> shape) {
    (void)format;
    (void)layout;
    (void)shape;
    const auto slash = name.rfind('/');
    std::string_view role = slash == std::string_view::npos ? name : name.substr(slash + 1);
    std::string_view comp = slash == std::string_view::npos ? std::string_view("default")
                                                             : name.substr(0, slash);
    return binder.resource(comp, role);
}

inline void Binder_materialize_on_device_compat(Binder& binder, ObjectHandle handle) {
    binder.require_device(handle);
}

inline std::span<const std::byte> Binder_payload_compat(Binder& binder, ObjectHandle handle) {
    return binder.host_object(handle);
}

} // namespace ninfer::artifact
