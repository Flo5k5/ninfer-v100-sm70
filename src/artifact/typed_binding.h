// Compat v2 -> v3 : surface de l'API artifact v2 consommée par les porteurs Volta
// (src/targets/**), réimplémentée par-dessus le module artifact v3 de l'amont.
//
// Principe : un nom plat v2 ("text/layers/3/mlp/gate_up") est résolu vers l'objet physique
// v3 unique que couvrent, bout à bout, les bindings logiques v3 correspondants
// ("text/layers/3/mlp/gate" + ".../mlp/up"). Les diviseurs d'entrée NVFP4 v2
// (".../gate_up_projection/input_scale_divisor") sont résolus via les auxiliaires
// `activation_input_divisor` des Uses v3. La matérialisation délègue à core/weight_view
// (native_weight / weight_tensor), qui produit l'ABI Weight/Tensor existante.
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

// Binding typé v2 : résolution du nom plat, validation format/forme, demande de résidence.
[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement = TensorPlacement::Device);
[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);
[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

// Octets hôte d'un objet déjà lié (petits objets : ids de draft, diviseurs). L'objet reste
// retenu côté hôte jusqu'à la matérialisation.
[[nodiscard]] std::span<const std::byte> host_bytes(Binder& binder, ObjectHandle handle);

// Diviseur de poids NVFP4 : lu directement dans le fichier (4 octets en queue d'objet),
// sans rapatrier l'objet entier côté hôte.
[[nodiscard]] std::uint32_t nvfp4_weight_divisor_bits(Binder& binder, ObjectHandle handle);

[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

[[nodiscard]] ::ninfer::Weight materialized_weight(const MaterializedArtifact& materialized,
                                                   ObjectHandle handle, NumericFormat format,
                                                   std::int32_t rows, std::int32_t columns);

// NVFP4 : Weight complet (codes, scales swizzlés, diviseurs) depuis le parent device.
[[nodiscard]] ::ninfer::Weight materialized_nvfp4_weight(const MaterializedArtifact& materialized,
                                                         ObjectHandle handle, std::int32_t rows,
                                                         std::int32_t columns,
                                                         float input_scale_divisor);

} // namespace ninfer::artifact
