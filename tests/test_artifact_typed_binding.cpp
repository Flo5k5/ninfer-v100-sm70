#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::Binder;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectHandle;
using ninfer::artifact::Reader;
using ninfer::artifact::TensorPlacement;
using ninfer::test::artifact_fixture::Json;
using ninfer::test::artifact_fixture::write_fixture;

constexpr std::uint64_t kExperts = 256;
// Every fixture matrix is BF16 with two columns, and each logical role of an expert is one row.
constexpr std::uint64_t kColumns = 2;

Json rows(const std::string& object, std::uint64_t first, std::uint64_t count) {
    return {{"parts", Json::array({{{"object", object},
                                    {"range", {first * kColumns, (first + count) * kColumns}}}})}};
}

// One Text and one MTP MoE block laid out like the published v3 artifact: each v2 MoE parent is
// one object, and its v3 logical parameters are row slices of it in v2 row order.
Json moe_directory() {
    Json objects         = Json::array();
    Json bindings        = Json::object();
    std::uint64_t offset = 0;
    const auto add       = [&](const std::string& id, std::uint64_t rows) {
        const std::uint64_t bytes = rows * kColumns * 2;
        objects.push_back({{"id", id},
                           {"kind", "tensor"},
                           {"shape", {rows, kColumns}},
                           {"format", "bf16"},
                           {"layout", "contiguous_le_v1"},
                           {"offset", offset},
                           {"bytes", bytes}});
        offset = (offset + bytes + 255) / 256 * 256;
    };
    for (const std::string block : {"text/layers/0/moe/", "mtp/layers/0/moe/"}) {
        const std::string parent = "weight/" + block;
        add(parent + "router_shared_gate", kExperts + 1);
        bindings[block + "router"]       = rows(parent + "router_shared_gate", 0, kExperts);
        bindings[block + "shared_score"] = rows(parent + "router_shared_gate", kExperts, 1);
        add(parent + "routed_gate_up", 2 * kExperts);
        add(parent + "routed_down", kExperts);
        for (std::uint64_t expert = 0; expert < kExperts; ++expert) {
            const std::string name  = block + "experts/" + std::to_string(expert) + "/";
            bindings[name + "gate"] = rows(parent + "routed_gate_up", 2 * expert, 1);
            bindings[name + "up"]   = rows(parent + "routed_gate_up", 2 * expert + 1, 1);
            bindings[name + "down"] = rows(parent + "routed_down", expert, 1);
        }
        add(parent + "shared_gate_up", 2);
        bindings[block + "shared/gate"] = rows(parent + "shared_gate_up", 0, 1);
        bindings[block + "shared/up"]   = rows(parent + "shared_gate_up", 1, 1);
        add(parent + "shared_down", 1);
        bindings[block + "shared/down"] = {{"object", parent + "shared_down"}};
    }
    return {
        {"components", {{"text", {{"config", Json::object()}}}}},
        {"objects", objects},
        {"bindings", bindings},
        {"uses", Json::array()},
        {"metadata", {{"name", "qwen3.6-35b-a3b"}}},
    };
}

ObjectHandle bind_bf16(Binder& binder, std::string_view name,
                       std::initializer_list<std::uint64_t> shape) {
    return ninfer::artifact::bind_tensor(binder, name, NumericFormat::BF16, shape,
                                         TensorPlacement::ValidateOnly);
}

void test_moe_parents_resolve() {
    auto fixture = write_fixture(moe_directory(), "typed_binding_moe");
    const Reader reader(fixture.path);
    Binder binder(reader);
    // The binder addresses the MTP block as mtp/layer/, which v3 names mtp/layers/0/.
    for (const auto& [v2_block, v3_block] : {std::pair{"text/layers/0/moe/", "text/layers/0/moe/"},
                                             std::pair{"mtp/layer/moe/", "mtp/layers/0/moe/"}}) {
        const std::string parent = std::string("weight/") + v3_block;
        const auto expect = [&](std::string_view role, std::initializer_list<std::uint64_t> shape) {
            const std::string name = std::string(v2_block) + std::string(role);
            if (!(bind_bf16(binder, name, shape) == reader.find(parent + std::string(role)))) {
                throw std::runtime_error(name + " did not resolve to its v3 parent object");
            }
        };
        expect("router_shared_gate", {kExperts + 1, kColumns});
        expect("routed_gate_up", {2 * kExperts, kColumns});
        expect("routed_down", {kExperts, kColumns});
        expect("shared_gate_up", {2, kColumns});
        expect("shared_down", {1, kColumns});
    }
}

// The Volta MoE kernels read the whole v2 parent, so a v3 artifact whose logical parameters do not
// cover it in v2 row order must be rejected rather than bound.
void expect_rejected(const std::string& label, Json directory, std::string_view v2_name,
                     std::initializer_list<std::uint64_t> shape, std::string_view needle) {
    auto fixture = write_fixture(std::move(directory), "typed_binding_rejected");
    const Reader reader(fixture.path);
    Binder binder(reader);
    try {
        (void)bind_bf16(binder, v2_name, shape);
    } catch (const ArtifactError& error) {
        if (std::string_view(error.what()).find(needle) == std::string_view::npos) {
            throw std::runtime_error(label + ": unexpected error: " + error.what());
        }
        return;
    }
    throw std::runtime_error(label + " was accepted");
}

void test_moe_row_order_is_enforced() {
    constexpr std::string_view block = "text/layers/0/moe/";
    const std::string parent         = "weight/text/layers/0/moe/";
    const auto expert                = [&](std::uint64_t index, std::string_view role) {
        return std::string(block) + "experts/" + std::to_string(index) + "/" + std::string(role);
    };

    Json gate_major = moe_directory();
    for (std::uint64_t index = 0; index < kExperts; ++index) {
        gate_major["bindings"][expert(index, "gate")] = rows(parent + "routed_gate_up", index, 1);
        gate_major["bindings"][expert(index, "up")] =
            rows(parent + "routed_gate_up", kExperts + index, 1);
    }
    expect_rejected("gate-major routed gate/up", std::move(gate_major),
                    "text/layers/0/moe/routed_gate_up", {2 * kExperts, kColumns},
                    "experts/0/up' is not contiguous");

    Json up_first = moe_directory();
    for (std::uint64_t index = 0; index < kExperts; ++index) {
        up_first["bindings"][expert(index, "gate")] =
            rows(parent + "routed_gate_up", 2 * index + 1, 1);
        up_first["bindings"][expert(index, "up")] = rows(parent + "routed_gate_up", 2 * index, 1);
    }
    expect_rejected("up before gate", std::move(up_first), "text/layers/0/moe/routed_gate_up",
                    {2 * kExperts, kColumns}, "experts/0/gate' is not contiguous");

    Json missing = moe_directory();
    missing["bindings"].erase(expert(100, "down"));
    expect_rejected("missing expert", std::move(missing), "text/layers/0/moe/routed_down",
                    {kExperts, kColumns},
                    "missing logical parameter 'text/layers/0/moe/routed_down' "
                    "(v3 'text/layers/0/moe/experts/100/down')");

    Json short_router = moe_directory();
    short_router["bindings"][std::string(block) + "router"] =
        rows(parent + "router_shared_gate", 0, kExperts - 1);
    short_router["bindings"][std::string(block) + "shared_score"] =
        rows(parent + "router_shared_gate", kExperts - 1, 1);
    expect_rejected("partial router coverage", std::move(short_router),
                    "text/layers/0/moe/router_shared_gate", {kExperts + 1, kColumns},
                    "do not cover the whole object");

    Json sliced_down                                            = moe_directory();
    sliced_down["bindings"][std::string(block) + "shared/down"] = {
        {"parts",
         Json::array({{{"object", parent + "shared_down"}, {"range", {0, kColumns - 1}}}})}};
    expect_rejected("sliced shared down", std::move(sliced_down), "text/layers/0/moe/shared_down",
                    {1, kColumns}, "slice, not a whole object");
}

} // namespace

int main() {
    try {
        test_moe_parents_resolve();
        test_moe_row_order_is_enforced();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
