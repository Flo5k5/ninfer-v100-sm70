// Opt-in: grammar-constrained answers on a real 27B artifact. Loads the artifact twice on one GPU
// (MTP with adaptive context lookup, then no speculative backend), both with CUDA Graphs, and
// checks what the masks guarantee: every constrained answer is valid compact JSON of its schema,
// including one whose first token comes from a retained prompt frontier, the thinking budget
// control passes through the grammar, a free request sharing the batch stays free, and rejected
// formats never reach generation. With NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS set, it also loads that
// artifact with DFlash2 and checks that constrained requests are rejected before generation.
#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using Json = nlohmann::json;

constexpr const char* kCitySchema =
    R"({"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"},)"
    R"("population":{"type":"integer","minimum":0}},"required":["city","country","population"],)"
    R"("additionalProperties":false})";

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, ninfer::SpeculativeBackend backend) {
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 4096;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk   = 1024;
    options.max_concurrency = 2;
    if (backend == ninfer::SpeculativeBackend::Mtp) {
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = 4;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        options.speculative.context_lookup =
            ninfer::ContextLookupOptions{.policy       = ninfer::ContextLookupPolicy::Adaptive,
                                         .min_suffix   = 4U,
                                         .max_proposal = ninfer::kContextLookupMaximumProposal};
    }
    return options;
}

ninfer::PromptInput question(std::string text, bool thinking, ninfer::ResponseFormat format) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = thinking;
    input.options.response_format = std::move(format);
    return input;
}

ninfer::ResponseFormat city_format() {
    return ninfer::ResponseFormat{.kind            = ninfer::ResponseFormatKind::JsonSchema,
                                  .schema_json     = kCitySchema,
                                  .schema_location = "response_format.json_schema.schema"};
}

ninfer::RequestOptions greedy(std::uint32_t tokens) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens    = tokens;
    request.execution.sampling.temperature       = 0.0F;
    request.execution.sampling.presence_penalty  = 0.0F;
    request.execution.sampling.frequency_penalty = 0.0F;
    request.execution.allow_prefix_reuse         = false;
    return request;
}

// The answer is one compact JSON object of kCitySchema: ", " and ": " separators only.
void require_city(const ninfer::GenerationResult& result, const std::string& label) {
    require(result.finish_reason == ninfer::FinishReason::StopToken,
            label + " did not end at the stop token: " + result.content);
    Json value;
    try {
        value = Json::parse(result.content);
    } catch (const std::exception&) {
        throw std::runtime_error(label + " is not JSON: " + result.content);
    }
    require(value.is_object() && value.size() == 3 && value.at("city").is_string() &&
                value.at("country").is_string() && value.at("population").is_number_integer() &&
                value.at("population").get<std::int64_t>() >= 0,
            label + " does not match its schema: " + result.content);
    require(result.content.find_first_of("\n\r\t") == std::string::npos &&
                result.content.starts_with(R"({"city": ")"),
            label + " is not compact JSON in the declared property order: " + result.content);
    std::cout << label << ": " << result.content << " (" << result.generated_token_ids.size()
              << " tokens)\n";
}

void run(const char* artifact, ninfer::SpeculativeBackend backend, const char* label) {
    ninfer::Engine engine(engine_options(artifact, backend));
    const std::string prefix = std::string(label) + " ";

    // Prime the lookup state with one unconstrained pass over the reused question: without it,
    // the first retained-frontier pass can match a partial suffix instead of the full prompt
    // (observed 20 of 27 tokens) and the strict frontier assertions below flake on engine state
    // rather than on grammar behaviour.
    {
        const std::string priming_question =
            "Which city is the capital of Spain? Answer with its country and population.";
        (void)engine.generate(engine.prepare(question(priming_question, false, {})), greedy(8));
    }

    const ninfer::GenerationResult direct = engine.generate(
        engine.prepare(question("Which city is the capital of France? Answer with its country and "
                                "population.",
                                false, city_format())),
        greedy(256));
    require_city(direct, prefix + "schema answer");

    // The same prompt again starts from the frontier the first request retained: its first token
    // is sampled from the retained hidden state, through the mask as well.
    const std::string repeated_question =
        "Which city is the capital of Spain? Answer with its country and population.";
    ninfer::RequestOptions reusing       = greedy(256);
    reusing.execution.allow_prefix_reuse = true;
    const ninfer::GenerationResult seeded =
        engine.generate(engine.prepare(question(repeated_question, false, city_format())), reusing);
    require_city(seeded, prefix + "answer that retains its prompt frontier");
    ninfer::PreparedPrompt repeated =
        engine.prepare(question(repeated_question, false, city_format()));
    const std::uint32_t prompt_tokens      = repeated.summary().prompt_tokens;
    const ninfer::GenerationResult resumed = engine.generate(std::move(repeated), reusing);
    require(resumed.reused_prompt_tokens == prompt_tokens,
            prefix + "the repeated prompt did not start from its retained frontier (" +
                std::to_string(resumed.reused_prompt_tokens) + " of " +
                std::to_string(prompt_tokens) + " tokens reused)");
    require_city(resumed, prefix + "answer started from a retained prompt frontier");

    ninfer::RequestOptions capped           = greedy(512);
    capped.execution.thinking.budget        = 32;
    const ninfer::GenerationResult reasoned = engine.generate(
        engine.prepare(question("Think about the largest city of Japan, then answer with its "
                                "country and population.",
                                true, city_format())),
        capped);
    require(reasoned.thinking.applied && !reasoned.reasoning.empty(),
            prefix + "thinking control was not applied before the answer");
    require_city(reasoned, prefix + "answer after the thinking control");

    const ninfer::GenerationResult object = engine.generate(
        engine.prepare(
            question("Describe a cat in a few fields.", false,
                     ninfer::ResponseFormat{.kind = ninfer::ResponseFormatKind::JsonObject})),
        greedy(256));
    require(object.finish_reason == ninfer::FinishReason::StopToken &&
                Json::parse(object.content).is_object(),
            prefix + "json_object answer is not an object: " + object.content);

    // A constrained and a free request decode in one batch; the masks stay on their own row.
    ninfer::GenerationHandle constrained = engine.submit(
        engine.prepare(question("Which city is the capital of Italy? Answer with its country and "
                                "population.",
                                false, city_format())),
        greedy(256));
    ninfer::GenerationHandle free = engine.submit(
        engine.prepare(question("Write one sentence about the sea.", false, {})), greedy(64));
    const ninfer::GenerationResult batched       = constrained.wait();
    const ninfer::GenerationResult unconstrained = free.wait();
    require_city(batched, prefix + "batched schema answer");
    require(!unconstrained.content.empty() && unconstrained.content.front() != '{',
            prefix + "the free request of the batch was constrained: " + unconstrained.content);

    bool rejected = false;
    try {
        (void)engine.prepare(
            question("x", false,
                     ninfer::ResponseFormat{.kind        = ninfer::ResponseFormatKind::JsonSchema,
                                            .schema_json = R"({"type":"array","uniqueItems":true})",
                                            .schema_location = "schema"}));
    } catch (const ninfer::RequestError& error) {
        rejected = error.kind() == ninfer::RequestErrorKind::InvalidOutputConstraint;
    }
    require(rejected, prefix + "an unsupported schema was not rejected at preparation");
}

// DFlash2 drafts and verifies inside one CUDA Graph that no mask reaches: its Engine rejects every
// constrained request when preparing it.
void reject_under_dflash2(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 2304;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(2304);
    options.prefill_chunk             = 2304;
    options.max_concurrency           = 1;
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens  = 15;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    ninfer::Engine engine(options);
    for (ninfer::ResponseFormat format :
         {city_format(), ninfer::ResponseFormat{.kind = ninfer::ResponseFormatKind::JsonObject}}) {
        bool rejected = false;
        try {
            (void)engine.prepare(question("Name a city.", false, std::move(format)));
        } catch (const ninfer::RequestError& error) {
            rejected = error.kind() == ninfer::RequestErrorKind::OutputConstraintUnavailable;
        }
        require(rejected, "a DFlash2 Engine accepted a constrained request");
    }
    const ninfer::GenerationResult free =
        engine.generate(engine.prepare(question("Name a city.", false, {})), greedy(16));
    require(!free.generated_token_ids.empty(), "a DFlash2 Engine did not answer a free request");
    std::cout << "dflash2: constrained requests rejected, free request answered\n";
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }
    try {
        run(artifact, ninfer::SpeculativeBackend::Mtp, "mtp");
        run(artifact, ninfer::SpeculativeBackend::None, "ordinary");
        const char* dflash2 = std::getenv("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS");
        if (dflash2 != nullptr && *dflash2 != '\0') {
            reject_under_dflash2(dflash2);
        } else {
            std::cout << "skip dflash2: NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS is not set\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
