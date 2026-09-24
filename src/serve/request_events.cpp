#include "serve/request_events.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <new>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::size_t kMaximumModelAliasBytes = 128;

bool is_model_identifier_character(char value) noexcept {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
           (value >= '0' && value <= '9') || value == '.' || value == '_' || value == ':' ||
           value == '@' || value == '/' || value == '-';
}

} // namespace

std::string log_safe_model_alias(std::string_view requested_model) {
    if (requested_model.empty() || requested_model.size() > kMaximumModelAliasBytes ||
        !std::all_of(requested_model.begin(), requested_model.end(),
                     is_model_identifier_character)) {
        return "other";
    }
    return std::string(requested_model);
}

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const RequestLogMetadata& metadata,
                                           const PreparedRequest& prepared) {
    RequestLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = std::string(metadata.served_model);
    context.requested_model                    = log_safe_model_alias(metadata.requested_model);
    context.stream                             = metadata.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = metadata.output_tokens_explicit;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.enable_thinking                    = prepared.enable_thinking;
    context.thinking_budget                    = prepared.thinking_budget;
    context.requested_reasoning_effort         = request.reasoning_effort;
    context.resolved_reasoning_effort          = prepared.effective_reasoning_effort;
    context.preserve_thinking                  = prepared.preserve_thinking;
    context.preserve_thinking_semantic_change  = metadata.preserve_thinking_semantic_change;
    context.sampling                           = prepared.sampling;
    context.acquisition_seconds                = prepared.acquisition_seconds;
    context.preparation                        = prepared.preparation;
    return context;
}

RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              const RequestLogMetadata& metadata,
                                                              ApiError error) {
    RequestRejectionLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = std::string(metadata.served_model);
    context.requested_model                    = log_safe_model_alias(metadata.requested_model);
    context.stream                             = metadata.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = metadata.output_tokens_explicit;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.requested_reasoning_effort         = request.reasoning_effort;
    context.error                              = std::move(error);
    return context;
}

RequestRejectionLogContext make_request_rejection_log_context(
    std::uint64_t id, std::string protocol, const GenerationRequest& request,
    const RequestLogMetadata& metadata, ApiError error, const std::exception& cause) {
    RequestRejectionLogContext context = make_request_rejection_log_context(
        id, std::move(protocol), request, metadata, std::move(error));
    context.cause = internal_failure_cause(cause);
    return context;
}

std::string internal_failure_cause(const std::exception& exception) {
    if (const auto* wrapper = dynamic_cast<const std::nested_exception*>(&exception);
        wrapper != nullptr && wrapper->nested_ptr() != nullptr) {
        try {
            wrapper->rethrow_nested();
        } catch (const std::exception& nested) {
            return internal_failure_cause(nested);
        } catch (...) { return "unknown"; }
    }
    if (dynamic_cast<const std::bad_alloc*>(&exception) != nullptr) { return "out_of_memory"; }
    if (const auto* json = dynamic_cast<const nlohmann::json::exception*>(&exception)) {
        return "json_error_" + std::to_string(json->id);
    }
    if (const auto* system = dynamic_cast<const std::system_error*>(&exception)) {
        return "system_error_" + std::to_string(system->code().value());
    }
    if (dynamic_cast<const std::invalid_argument*>(&exception) != nullptr) {
        return "invalid_argument";
    }
    if (dynamic_cast<const std::out_of_range*>(&exception) != nullptr) { return "out_of_range"; }
    if (dynamic_cast<const std::length_error*>(&exception) != nullptr) { return "length_error"; }
    if (dynamic_cast<const std::logic_error*>(&exception) != nullptr) { return "logic_error"; }
    if (dynamic_cast<const ApiException*>(&exception) != nullptr) { return "api_error"; }
    if (dynamic_cast<const std::runtime_error*>(&exception) != nullptr) { return "runtime_error"; }
    return "exception";
}

RequestFailure make_request_failure(RequestFailurePhase phase, const ApiError& error) {
    RequestFailureClass classification = RequestFailureClass::Internal;
    if (error.status == 499 || error.code == "client_disconnected") {
        classification = RequestFailureClass::ClientDisconnected;
    } else if (error.status == 429 || error.status == 529) {
        classification = RequestFailureClass::Overload;
    } else if (error.code == "request_queue_timeout" || error.code == "media_fetch_timeout" ||
               error.status == 504) {
        classification = RequestFailureClass::Timeout;
    } else if (error.code == "service_unavailable" || error.status == 503) {
        classification = RequestFailureClass::Unavailable;
    } else if (error.code == "media_fetch_failed" || error.status == 502) {
        classification = RequestFailureClass::Upstream;
    } else if (error.status >= 400 && error.status < 500) {
        classification = RequestFailureClass::ClientInput;
    }
    return RequestFailure{
        .phase          = phase,
        .classification = classification,
        .http_status    = error.status,
        .error_type     = error.type,
        .error_code     = error.code,
        .param          = error.param,
    };
}

RequestFailure make_generation_request_failure(const ApiError& error) {
    RequestFailure failure = make_request_failure(RequestFailurePhase::Generation, error);
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        failure.phase = RequestFailurePhase::Transport;
    }
    return failure;
}

RequestFailure make_internal_request_failure(RequestFailurePhase phase,
                                             const std::exception& exception) {
    return RequestFailure{
        .phase          = phase,
        .classification = RequestFailureClass::Internal,
        .http_status    = 500,
        .error_type     = "internal_error",
        .cause          = internal_failure_cause(exception),
    };
}

RequestFailure make_unknown_internal_request_failure(RequestFailurePhase phase) {
    return RequestFailure{
        .phase          = phase,
        .classification = RequestFailureClass::Internal,
        .http_status    = 500,
        .error_type     = "internal_error",
        .cause          = "unknown",
    };
}

RequestFailure make_client_disconnected_failure(RequestFailurePhase phase) {
    return RequestFailure{
        .phase          = phase,
        .classification = RequestFailureClass::ClientDisconnected,
        .http_status    = 499,
        .error_type     = "request_cancelled",
        .error_code     = "client_disconnected",
    };
}

} // namespace ninfer::serve
