#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// The JSON Schema response format held by object[key], which must be a JSON object. Only its
// shape is checked here: the Engine admits and compiles the schema and rejects what it cannot
// enforce. `location` names the schema in those errors; `param` names the request field.
[[nodiscard]] ninfer::ResponseFormat json_schema_response_format(const RequestJson& object,
                                                                 const char* key,
                                                                 std::string location,
                                                                 const char* param);

// Rejects the members of `object` other than `allowed` that are not null, with the code
// parameter_not_supported: the Engine would otherwise ignore them, and a constraint the request
// asked for would silently not apply. `field` names the object in the message.
void reject_unknown_members(const RequestJson& object,
                            std::initializer_list<std::string_view> allowed,
                            const std::string& field, const char* param);

// The JSON Schema response format of the OpenAI APIs, described by `object`: `schema` (a JSON
// Schema object), `name` and `description` (strings) and `strict` (a boolean, accepted either way:
// the Engine rejects a schema it cannot enforce rather than relax it), besides the `enclosing`
// members of the format object when the definition sits in it. Other members are rejected.
// `field` names `object` in errors; its schema is `<field>.schema`.
[[nodiscard]] ninfer::ResponseFormat
openai_json_schema_format(const RequestJson& object,
                          std::initializer_list<std::string_view> enclosing,
                          const std::string& field, const char* param);

} // namespace ninfer::serve
