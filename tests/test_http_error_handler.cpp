#include "serve/http_server.h"
#include "serve/openai_responses.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using ninfer::serve::ApiError;
using ninfer::serve::ApiException;
using ninfer::serve::ServeOptions;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse_options(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

ApiError api_error(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& exception) { return exception.error(); } catch (...) {
        return ApiError{.status = 0, .message = "wrong exception"};
    }
    return ApiError{.status = 0, .message = "no exception"};
}

// The Responses handler composes these calls: parsed options build the store, the store sets the
// request limits, and the limits drive Create parsing and prompt resolution.
int test_responses_store_wiring() {
    using namespace ninfer::serve;
    int failures = 0;

    const ServeOptions retaining =
        parse_options({"ninfer-serve", "model.ninfer", "--default-max-tokens", "77"});
    OpenAIResponsesStore retaining_store = make_openai_responses_store(retaining);
    const RequestLimits retaining_limits =
        make_openai_responses_request_limits(retaining, retaining_store);
    const RequestJson omitted = {{"model", "m"}, {"input", "hello"}};
    failures += check(retaining_store.enabled() && retaining_limits.response_store_enabled &&
                          retaining_limits.default_max_tokens == 77 &&
                          parse_openai_responses_create_request(omitted, retaining_limits).store,
                      "a default server did not retain Responses by default");

    const ServeOptions storeless =
        parse_options({"ninfer-serve", "model.ninfer", "--no-response-store"});
    OpenAIResponsesStore store = make_openai_responses_store(storeless);
    const RequestLimits limits = make_openai_responses_request_limits(storeless, store);
    failures += check(!store.enabled() && !limits.response_store_enabled &&
                          limits.default_max_tokens == storeless.default_max_tokens,
                      "--no-response-store did not reach the Responses request limits");

    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(omitted, limits);
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request.prompt, store, "resp_unstored", request.store);
    failures += check(!request.store && !resolved.session_key && store.size() == 0,
                      "an omitted store was retained by a server without a Responses store");

    RequestJson explicit_false = omitted;
    explicit_false["store"]    = false;
    failures += check(!parse_openai_responses_create_request(explicit_false, limits).store,
                      "store=false was not served without a Responses store");

    RequestJson explicit_true = omitted;
    explicit_true["store"]    = true;
    const ApiError store_error =
        api_error([&] { (void)parse_openai_responses_create_request(explicit_true, limits); });
    failures += check(store_error.status == 400 && store_error.code == "store_not_supported" &&
                          store_error.param == "store",
                      "store=true was not rejected by a server without a Responses store");

    RequestJson continuation             = omitted;
    continuation["previous_response_id"] = "resp_unstored";
    const OpenAIResponsesCreateRequest child =
        parse_openai_responses_create_request(continuation, limits);
    const ApiError chain_error = api_error([&] {
        (void)resolve_openai_responses_prompt(child.prompt, store, "resp_child", child.store);
    });
    const OpenAIResponsesPromptRequest counted =
        parse_openai_responses_input_tokens_request(continuation, limits);
    const ApiError count_error = api_error(
        [&] { (void)resolve_openai_responses_prompt(counted, store, std::nullopt, false); });
    failures +=
        check(chain_error.status == 400 && chain_error.code == "previous_response_not_supported" &&
                  count_error.code == "previous_response_not_supported",
              "previous_response_id was accepted by a server without a Responses store");
    return failures;
}

} // namespace

int main() {
    int failures = test_responses_store_wiring();
    ServeOptions options;
    options.max_request_bytes = 1234;

    const ninfer::serve::ApiError media_budget = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::MediaBudgetExceeded,
                             "vision tokens exceed processor budget"));
    failures += check(media_budget.status == 400 && media_budget.code == "media_budget_exceeded",
                      "media resource rejection did not map to HTTP 400");
    const ninfer::serve::ApiError invalid_media = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::InvalidMedia, "failed to open media"));
    failures += check(invalid_media.status == 400 && invalid_media.code == "invalid_media" &&
                          invalid_media.param == "messages",
                      "invalid media did not retain its client-input classification");
    const ninfer::serve::ApiError context_limit = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                             "prepared prompt has 200 tokens, exceeding Engine max_context 128"));
    failures +=
        check(context_limit.status == 400 && context_limit.code == "context_length_exceeded" &&
                  context_limit.message.find("200 tokens") != std::string::npos &&
                  context_limit.message.find("128") != std::string::npos,
              "context rejection lost its HTTP classification or capacity details");
    const ninfer::serve::ApiError thinking_capacity = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                             "thinking control suffix does not fit"));
    failures += check(thinking_capacity.status == 400 &&
                          thinking_capacity.code == "thinking_budget_capacity_insufficient" &&
                          thinking_capacity.param.empty(),
                      "thinking budget capacity error mapping mismatch");
    const ninfer::serve::ApiError cancelled =
        ninfer::serve::request_error_to_api_error(ninfer::RequestError(
            ninfer::RequestErrorKind::Cancelled, "request cancelled during preparation"));
    failures += check(cancelled.status == 499 && cancelled.code == "client_disconnected" &&
                          cancelled.param.empty(),
                      "preparation cancellation did not retain its HTTP classification");
    const ninfer::serve::ApiError invalid_constraint = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::InvalidOutputConstraint,
                             "response_format.json_schema.schema at #/a: keyword 'not' is not "
                             "supported"));
    failures += check(invalid_constraint.status == 400 &&
                          invalid_constraint.code == "invalid_output_constraint" &&
                          invalid_constraint.param.empty(),
                      "an invalid output constraint did not map to HTTP 400");
    const ninfer::serve::ApiError unavailable_constraint =
        ninfer::serve::request_error_to_api_error(
            ninfer::RequestError(ninfer::RequestErrorKind::OutputConstraintUnavailable,
                                 "structured output is disabled"));
    failures += check(unavailable_constraint.status == 400 &&
                          unavailable_constraint.code == "structured_outputs_unavailable" &&
                          unavailable_constraint.param.empty(),
                      "an unavailable output constraint did not map to HTTP 400");
    const ninfer::serve::ApiError violated_constraint = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::OutputConstraintViolated,
                             "the output grammar refused a generated token"));
    failures +=
        check(violated_constraint.status == 500 && violated_constraint.type == "server_error" &&
                  violated_constraint.code == "output_constraint_violated",
              "a violated output constraint did not map to a server error");

    failures +=
        check(ninfer::serve::matches_bearer_credential("Bearer secret", "secret") &&
                  ninfer::serve::matches_bearer_credential("bearer secret", "secret") &&
                  ninfer::serve::matches_bearer_credential("\tBEARER   secret\t", "secret"),
              "valid Bearer credentials were rejected because of scheme case or whitespace");
    failures += check(!ninfer::serve::matches_bearer_credential("Basic secret", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearer wrong", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearersecret", "secret") &&
                          !ninfer::serve::matches_bearer_credential("Bearer secret", ""),
                      "invalid Bearer credentials were accepted");

    httplib::Request messages_request;
    messages_request.path = "/v1/messages";
    httplib::Response messages_response;
    messages_response.status = 413;
    const auto messages_result =
        ninfer::serve::handle_unrendered_http_error(options, messages_request, messages_response);
    const Json messages_body = Json::parse(messages_response.body);
    failures += check(messages_result == httplib::Server::HandlerResponse::Handled &&
                          messages_body.at("type") == "error" &&
                          messages_body.at("error").at("type") == "request_too_large" &&
                          messages_body.at("request_id").get<std::string>().starts_with("req_") &&
                          messages_response.get_header_value("request-id") ==
                              messages_body.at("request_id").get<std::string>() &&
                          !messages_response.has_header("x-request-id") &&
                          messages_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty Anthropic 413 did not become a payload-limit error");

    httplib::Request openai_request;
    openai_request.path = "/v1/responses";
    httplib::Response openai_response;
    openai_response.status = 413;
    const auto openai_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, openai_response);
    const Json openai_body = Json::parse(openai_response.body);
    failures += check(openai_result == httplib::Server::HandlerResponse::Handled &&
                          openai_body.at("error").at("code") == "request_too_large" &&
                          openai_response.get_header_value_count("x-request-id") == 1 &&
                          openai_response.get_header_value("x-request-id").starts_with("req_") &&
                          openai_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty OpenAI 413 did not become a payload-limit error");

    httplib::Request missing_messages_request;
    missing_messages_request.path = "/v1/messages/missing";
    httplib::Response missing_messages_response;
    missing_messages_response.status = 404;
    missing_messages_response.set_header("request-id", "req_stale");
    const auto missing_messages_result = ninfer::serve::handle_unrendered_http_error(
        options, missing_messages_request, missing_messages_response);
    const Json missing_messages_body = Json::parse(missing_messages_response.body);
    failures +=
        check(missing_messages_result == httplib::Server::HandlerResponse::Handled &&
                  missing_messages_response.status == 404 &&
                  missing_messages_body.at("error").at("type") == "not_found_error" &&
                  missing_messages_body.at("request_id").get<std::string>().starts_with("req_") &&
                  missing_messages_body.at("request_id") != "req_stale" &&
                  missing_messages_response.get_header_value_count("request-id") == 1 &&
                  missing_messages_response.get_header_value("request-id") ==
                      missing_messages_body.at("request_id").get<std::string>(),
              "missing Anthropic resource did not use the protocol error envelope");

    httplib::Response authored_response;
    authored_response.status = 413;
    authored_response.set_header("x-request-id", "req_existing");
    authored_response.set_content(R"({"error":{"code":"application_error"}})", "application/json");
    const std::string authored_body = authored_response.body;
    const auto authored_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, authored_response);
    failures += check(authored_result == httplib::Server::HandlerResponse::Unhandled &&
                          authored_response.body == authored_body &&
                          authored_response.get_header_value_count("x-request-id") == 1 &&
                          authored_response.get_header_value("x-request-id") == "req_existing",
                      "application-authored 413 or its request ID was overwritten");

    httplib::Response other_response;
    other_response.status = 400;
    const auto other_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, other_response);
    failures += check(other_result == httplib::Server::HandlerResponse::Unhandled &&
                          other_response.body.empty(),
                      "non-413 response was changed by the payload-limit handler");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
