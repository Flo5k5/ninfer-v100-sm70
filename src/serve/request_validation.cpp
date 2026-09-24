#include "serve/request_validation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

ninfer::ResponseFormat json_schema_response_format(const RequestJson& object, const char* key,
                                                   std::string location, const char* param) {
    if (!object.contains(key) || !object.at(key).is_object()) {
        bad_request(location + " must be a JSON Schema object", param);
    }
    return ninfer::ResponseFormat{.kind            = ninfer::ResponseFormatKind::JsonSchema,
                                  .schema_json     = object.at(key).dump(),
                                  .schema_location = std::move(location)};
}

void reject_unknown_members(const RequestJson& object,
                            std::initializer_list<std::string_view> allowed,
                            const std::string& field, const char* param) {
    for (auto member = object.begin(); member != object.end(); ++member) {
        if (member.value().is_null() ||
            std::find(allowed.begin(), allowed.end(), member.key()) != allowed.end()) {
            continue;
        }
        bad_request("unsupported " + field + " member: " + member.key(), param,
                    "parameter_not_supported");
    }
}

ninfer::ResponseFormat openai_json_schema_format(const RequestJson& object,
                                                 std::initializer_list<std::string_view> enclosing,
                                                 const std::string& field, const char* param) {
    constexpr std::string_view kMembers[] = {"name", "description", "schema", "strict"};
    for (auto member = object.begin(); member != object.end(); ++member) {
        const bool known =
            std::find(std::begin(kMembers), std::end(kMembers), member.key()) !=
                std::end(kMembers) ||
            std::find(enclosing.begin(), enclosing.end(), member.key()) != enclosing.end();
        if (!known && !member.value().is_null()) {
            bad_request("unsupported " + field + " member: " + member.key(), param,
                        "parameter_not_supported");
        }
    }
    for (const char* key : {"name", "description"}) {
        if (object.contains(key) && !object.at(key).is_null() && !object.at(key).is_string()) {
            bad_request(field + "." + key + " must be a string", param);
        }
    }
    if (object.contains("strict") && !object.at("strict").is_null() &&
        !object.at("strict").is_boolean()) {
        bad_request(field + ".strict must be a boolean", param);
    }
    return json_schema_response_format(object, "schema", field + ".schema", param);
}

} // namespace ninfer::serve
