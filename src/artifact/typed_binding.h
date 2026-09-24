// v2-to-v3 compatibility: the v2 artifact API that the Volta target packages (src/targets/**)
// consume, reimplemented on top of upstream's v3 artifact module.
//
// A flat v2 name ("text/layers/3/mlp/gate_up") resolves to the one physical v3 object that the
// matching v3 logical bindings ("text/layers/3/mlp/gate" + ".../mlp/up") cover end to end. The
// v2 NVFP4 input divisors (".../gate_up_projection/input_scale_divisor") resolve through the
// `activation_input_divisor` auxiliaries of the v3 Uses. Materialization delegates to
// core/weight_view (native_weight / weight_tensor), which produces the existing Weight/Tensor ABI.
#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/tensor.h"

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

std::string_view format_name(NumericFormat format) noexcept;
[[nodiscard]] QType qtype_for(NumericFormat format);

// Typed v2 binding: resolves the flat name, checks format and shape, and requests residency.
[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement = TensorPlacement::Device);
[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);
[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

// Host bytes of an already bound object (small objects: draft ids, divisors). The object stays
// held on the host until materialization.
[[nodiscard]] std::span<const std::byte> host_bytes(Binder& binder, ObjectHandle handle);

// NVFP4 weight divisor, read directly from the file (the last 4 bytes of the object) without
// bringing the whole object to the host.
[[nodiscard]] std::uint32_t nvfp4_weight_divisor_bits(Binder& binder, ObjectHandle handle);

[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

[[nodiscard]] ::ninfer::Weight materialized_weight(const MaterializedArtifact& materialized,
                                                   ObjectHandle handle, NumericFormat format,
                                                   std::int32_t rows, std::int32_t columns);

// NVFP4: complete Weight (codes, swizzled scales, divisors) from the device parent.
[[nodiscard]] ::ninfer::Weight materialized_nvfp4_weight(const MaterializedArtifact& materialized,
                                                         ObjectHandle handle, std::int32_t rows,
                                                         std::int32_t columns,
                                                         float input_scale_divisor);

} // namespace ninfer::artifact
