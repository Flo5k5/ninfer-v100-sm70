#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"
#include "serve/request.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;
inline constexpr std::size_t kRequestedReasoningEffortCount =
    static_cast<std::size_t>(RequestedReasoningEffort::Max) + 1;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact identity.model_id
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context          = 8192;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    // --no-response-store: keep no Responses objects, input Items, or continuation contexts, and
    // reject the requests that need them. The two limits below then have no effect. The volatile
    // Engine prefix cache and media cache still hold recent prompts, generated replies, and media
    // in memory; they have their own switches (see the zero data retention section of
    // docs/serving.md).
    bool enable_response_store             = true;
    // --no-structured-output: build no grammar compiler and reject constrained requests
    // (response_format json_object/json_schema and their Responses and Anthropic forms).
    bool enable_structured_output            = true;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    TextResidualStorage text_residual      = TextResidualStorage::BFloat16;
    PrefillAttentionKernel prefill_attention = PrefillAttentionKernel::Automatic;
    SpeculativeOptions speculative;
    ContextCacheOptions context_cache;
    bool enable_vision      = false;
    bool use_cuda_graph     = true;
    bool allow_prefix_reuse = true;
    bool enable_thinking =
        true; // default thinking mode for the generation prompt (--no-thinking opts out)
    bool preserve_thinking = false;
    std::optional<std::uint32_t> default_thinking_budget;
    // Effort for thinking-enabled requests that name none; applies only to templates that
    // expose reasoning effort.
    std::optional<RequestedReasoningEffort> default_reasoning_effort;
    bool omitted_thinking_as_summarized = false; // --omitted-thinking-as-summarized
    // Per-level substitution applied before template validation, so clients that send a level
    // the loaded template lacks (Claude Code sends "high") resolve instead of failing.
    std::array<std::optional<RequestedReasoningEffort>, kRequestedReasoningEffortCount>
        reasoning_effort_aliases{};
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    product::LogLevel log_level = product::LogLevel::Info;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
