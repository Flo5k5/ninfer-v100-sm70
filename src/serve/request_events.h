#pragma once

#include "serve/generation_service.h"
#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

struct RequestLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;           // the served public model ID
    std::string requested_model; // log_safe_model_alias() of the client's model field
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    bool enable_thinking  = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<RequestedReasoningEffort> requested_reasoning_effort;
    std::optional<ninfer::ReasoningEffort> resolved_reasoning_effort;
    bool preserve_thinking                 = false;
    bool preserve_thinking_semantic_change = false;
    ninfer::ResolvedSamplingParameters sampling;
    double acquisition_seconds = 0.0;
    ninfer::PromptPreparationStats preparation;
};

// Handler-supplied request identity. The views must remain valid while the handler builds its log
// contexts, which copy only log-safe values: never the client's model field itself.
struct RequestLogMetadata {
    std::string_view served_model;
    std::string_view requested_model;
    bool stream                            = false;
    bool output_tokens_explicit            = false;
    bool preserve_thinking_semantic_change = false;
};

// A parsed generation request that failed during synchronous preparation. It intentionally has a
// separate shape because sampler and prompt semantics may not have resolved.
struct RequestRejectionLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;           // the served public model ID
    std::string requested_model; // log_safe_model_alias() of the client's model field
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    std::optional<RequestedReasoningEffort> requested_reasoning_effort;
    ApiError error;
    std::string cause; // internal rejections only; see internal_failure_cause()
};

enum class RequestFailurePhase : std::uint8_t {
    Prepare,
    Generation,
    ResponseRender,
    ResponseStore,
    Transport,
    Http,
};

enum class RequestFailureClass : std::uint8_t {
    ClientInput,
    ClientDisconnected,
    Overload,
    Timeout,
    Unavailable,
    Upstream,
    Internal,
};

// Log-safe failure description shared by the operational and JSONL sinks. It deliberately has no
// free-text message: API error messages and exception text can quote prompts, generated output,
// tool arguments, or media locations, so no log record may carry them. Type, code, parameter, and
// cause are server-defined identifiers; the complete message is returned only to the client.
struct RequestFailure {
    RequestFailurePhase phase          = RequestFailurePhase::Generation;
    RequestFailureClass classification = RequestFailureClass::Internal;
    int http_status                    = 0;
    std::string error_type;
    std::string error_code;
    std::string param;
    std::string cause; // internal failures only; see internal_failure_cause()
};

struct ThroughputReport {
    double interval_seconds               = 0.0;
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t committed_decode_tokens = 0;
    std::uint64_t decode_rounds           = 0;
    std::uint64_t decode_row_rounds       = 0;
    ninfer::RuntimeStats previous;
    ninfer::RuntimeStats current;
};

// Anthropic clients may send any non-empty string as `model`, including text a user typed or a
// value as large as the request body. Logs therefore keep it only when it has the shape of a model
// identifier, at most 128 characters from [A-Za-z0-9._:@/-], and record "other" otherwise.
[[nodiscard]] std::string log_safe_model_alias(std::string_view requested_model);

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const RequestLogMetadata& metadata,
                                           const PreparedRequest& prepared);
RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              const RequestLogMetadata& metadata,
                                                              ApiError error);
// A preparation rejection caused by an unexpected exception keeps only its content-free cause.
RequestRejectionLogContext make_request_rejection_log_context(
    std::uint64_t id, std::string protocol, const GenerationRequest& request,
    const RequestLogMetadata& metadata, ApiError error, const std::exception& cause);

// Content-free name of the exception behind an internal failure, derived from its dynamic type and
// never from what(): `out_of_memory`, `json_error_<id>` for a JSON library exception,
// `system_error_<code>`, `invalid_argument`, `out_of_range`, `length_error`, `logic_error`,
// `api_error`, `runtime_error`, or `exception`. An exception that wraps another through
// std::nested_exception reports the cause of the wrapped one.
[[nodiscard]] std::string internal_failure_cause(const std::exception& exception);

[[nodiscard]] RequestFailure make_request_failure(RequestFailurePhase phase, const ApiError& error);
[[nodiscard]] RequestFailure make_generation_request_failure(const ApiError& error);
[[nodiscard]] RequestFailure make_internal_request_failure(RequestFailurePhase phase,
                                                           const std::exception& exception);
// For a caught object that is not a std::exception; its cause is `unknown`.
[[nodiscard]] RequestFailure make_unknown_internal_request_failure(RequestFailurePhase phase);
[[nodiscard]] RequestFailure make_client_disconnected_failure(RequestFailurePhase phase);

} // namespace ninfer::serve
