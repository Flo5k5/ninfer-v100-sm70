#include "grammar/grammar_service.h"

#include "core/host_worker_pool.h"
#include "grammar/grammar_error.h"

#include <dlpack/dlpack.h>
#include <pthread.h>
#include <xgrammar/ninfer_hooks/warning_capture.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#ifdef __linux__
#    include <sys/resource.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

namespace ninfer::grammar {

// --- GrammarRecipe ------------------------------------------------------------------------------

struct GrammarRecipe::Node {
    enum class Kind : std::uint8_t {
        JsonSchema,
        StructuralTag,
        Sequence,
        Choice,
    };

    Kind kind = Kind::JsonSchema;
    std::string text;
    std::vector<GrammarRecipe> children;
    std::string key;
};

namespace {

// Length-prefixed, so that no schema or tag text can imitate the structure around it.
std::string leaf_key(char tag, const std::string& text) {
    return std::string(1, tag) + std::to_string(text.size()) + ':' + text;
}

GrammarRecipe::Node composite(GrammarRecipe::Node::Kind kind, char tag,
                              std::vector<GrammarRecipe> children) {
    if (children.empty()) {
        throw std::invalid_argument("a grammar sequence or choice needs at least one part");
    }
    GrammarRecipe::Node node{.kind = kind, .children = std::move(children)};
    node.key = std::string(1, tag) + std::to_string(node.children.size()) + '[';
    for (const GrammarRecipe& child : node.children) { node.key += child.key(); }
    node.key += ']';
    return node;
}

} // namespace

GrammarRecipe GrammarRecipe::json_schema(std::string admitted_schema) {
    Node node{.kind = Node::Kind::JsonSchema, .text = std::move(admitted_schema)};
    node.key = leaf_key('j', node.text);
    return GrammarRecipe(std::make_shared<const Node>(std::move(node)));
}

GrammarRecipe GrammarRecipe::structural_tag(std::string format_json) {
    Node node{.kind = Node::Kind::StructuralTag, .text = std::move(format_json)};
    node.key = leaf_key('t', node.text);
    return GrammarRecipe(std::make_shared<const Node>(std::move(node)));
}

GrammarRecipe GrammarRecipe::sequence(std::vector<GrammarRecipe> parts) {
    return GrammarRecipe(
        std::make_shared<const Node>(composite(Node::Kind::Sequence, 's', std::move(parts))));
}

GrammarRecipe GrammarRecipe::choice(std::vector<GrammarRecipe> alternatives) {
    return GrammarRecipe(
        std::make_shared<const Node>(composite(Node::Kind::Choice, 'c', std::move(alternatives))));
}

const std::string& GrammarRecipe::key() const noexcept { return node_->key; }

// --- CompiledGrammar ----------------------------------------------------------------------------

struct CompiledGrammar::Impl {
    xgrammar::CompiledGrammar compiled;
    std::vector<TokenId> stop_tokens;
    std::int32_t mask_words = 0;
};

std::size_t CompiledGrammar::memory_bytes() const noexcept {
    return impl_ ? impl_->compiled.MemorySizeBytes() : 0;
}

// --- TokenMatcher -------------------------------------------------------------------------------

struct TokenMatcher::Impl {
    explicit Impl(const CompiledGrammar& compiled_grammar)
        : grammar(compiled_grammar), shared(grammar.impl()), matcher(shared.compiled) {}

    void fill(std::int32_t* words) {
        if (matcher.IsTerminated()) {
            std::fill_n(words, shared.mask_words, 0);
            for (const TokenId token : shared.stop_tokens) {
                words[token >> 5] = static_cast<std::int32_t>(
                    static_cast<std::uint32_t>(words[token >> 5]) | (1u << (token & 31)));
            }
            return;
        }
        std::int64_t shape[1] = {shared.mask_words};
        DLTensor tensor{};
        tensor.data   = words;
        tensor.device = DLDevice{kDLCPU, 0};
        tensor.ndim   = 1;
        tensor.dtype  = DLDataType{kDLInt, 32, 1};
        tensor.shape  = shape;
        (void)matcher.FillNextTokenBitmask(&tensor, 0);
    }

    [[nodiscard]] bool is_stop(TokenId token) const {
        return std::find(shared.stop_tokens.begin(), shared.stop_tokens.end(), token) !=
               shared.stop_tokens.end();
    }

    const CompiledGrammar grammar;
    const CompiledGrammar::Impl& shared;
    xgrammar::GrammarMatcher matcher;
};

TokenMatcher::TokenMatcher(const CompiledGrammar& grammar) {
    if (!grammar) { throw std::invalid_argument("token matcher requires a compiled grammar"); }
    impl_ = std::make_unique<Impl>(grammar);
}

TokenMatcher::~TokenMatcher()                                  = default;
TokenMatcher::TokenMatcher(TokenMatcher&&) noexcept            = default;
TokenMatcher& TokenMatcher::operator=(TokenMatcher&&) noexcept = default;

std::int32_t TokenMatcher::mask_words() const noexcept { return impl_->shared.mask_words; }

void TokenMatcher::fill_mask(std::int32_t* words) { impl_->fill(words); }

std::uint32_t TokenMatcher::fill_draft_masks(std::span<const TokenId> drafts, std::int32_t* rows,
                                             std::size_t row_stride_words) {
    if (row_stride_words < static_cast<std::size_t>(impl_->shared.mask_words)) {
        throw std::invalid_argument("draft mask rows are narrower than one token mask");
    }
    impl_->fill(rows);
    std::uint32_t walked = 0;
    for (const TokenId draft : drafts) {
        const std::int32_t* row = rows + walked * row_stride_words;
        if (draft < 0 || (draft >> 5) >= impl_->shared.mask_words || impl_->is_stop(draft) ||
            ((static_cast<std::uint32_t>(row[draft >> 5]) >> (draft & 31)) & 1u) == 0u ||
            !impl_->matcher.AcceptToken(draft)) {
            break;
        }
        ++walked;
        impl_->fill(rows + walked * row_stride_words);
    }
    if (walked != 0) { impl_->matcher.Rollback(static_cast<int>(walked)); }
    return walked;
}

bool TokenMatcher::accept(TokenId token) {
    if (token < 0 || (token >> 5) >= impl_->shared.mask_words) { return false; }
    return impl_->matcher.AcceptToken(token);
}

void TokenMatcher::rollback(std::uint32_t tokens) {
    if (tokens != 0) { impl_->matcher.Rollback(static_cast<int>(tokens)); }
}

bool TokenMatcher::terminated() const { return impl_->matcher.IsTerminated(); }

// --- GrammarService -----------------------------------------------------------------------------

namespace {

using Clock = std::chrono::steady_clock;

std::string structural_tag_error(const xgrammar::StructuralTagError& error) {
    return std::visit([](const auto& value) { return std::string(value.what()); }, error);
}

xgrammar::Grammar build_grammar(const GrammarRecipe::Node& node,
                                const xgrammar::TokenizerInfo& tokenizer) {
    using Kind = GrammarRecipe::Node::Kind;
    switch (node.kind) {
    case Kind::JsonSchema:
        return xgrammar::Grammar::FromJSONSchema(node.text, /*any_whitespace=*/false,
                                                 /*indent=*/std::nullopt,
                                                 /*separators=*/std::nullopt,
                                                 /*strict_mode=*/true);
    case Kind::StructuralTag: {
        auto result = xgrammar::Grammar::FromStructuralTag(
            R"({"type":"structural_tag","format":)" + node.text + "}", tokenizer);
        if (const auto* error = std::get_if<xgrammar::StructuralTagError>(&result)) {
            throw GrammarError(GrammarErrorKind::InvalidSchema, structural_tag_error(*error));
        }
        return std::get<xgrammar::Grammar>(std::move(result));
    }
    case Kind::Sequence:
    case Kind::Choice: {
        std::vector<xgrammar::Grammar> parts;
        parts.reserve(node.children.size());
        for (const GrammarRecipe& child : node.children) {
            parts.push_back(build_grammar(child.node(), tokenizer));
        }
        if (parts.size() == 1) { return parts.front(); }
        return node.kind == Kind::Sequence ? xgrammar::Grammar::Concat(parts)
                                           : xgrammar::Grammar::Union(parts);
    }
    }
    throw std::logic_error("unknown grammar recipe node");
}

// Compile threads yield the CPU to decoding. xgrammar's own compile threads are started by the
// calling worker, so they inherit its niceness.
void lower_thread_priority(int nice) {
#ifdef __linux__
    thread_local bool lowered = false;
    if (lowered || nice <= 0) { return; }
    const auto thread = static_cast<id_t>(::syscall(SYS_gettid));
    errno             = 0;
    const int current = ::getpriority(PRIO_PROCESS, thread);
    if (errno == 0) { (void)::setpriority(PRIO_PROCESS, thread, std::min(current + nice, 19)); }
    lowered = true;
#else
    (void)nice;
#endif
}

// xgrammar's schema conversion recurses once per nested schema and per reference target it
// generates: up to (max_ref_targets + 1) x (max_schema_depth + 1) levels, more than the default
// stack of a pool thread holds (512 KiB on macOS). Each compilation runs on a thread of its own
// with this stack, whose pages are committed only as the recursion reaches them.
constexpr std::size_t kCompileStackBytes = std::size_t{64} << 20;

// Runs `body` on a new thread with a stack of `stack_bytes` and returns or rethrows its result.
// Throws std::system_error when the thread cannot start.
template <class Body>
std::invoke_result_t<Body&> run_with_stack(std::size_t stack_bytes, Body& body) {
    struct Call {
        Body& body;
        std::optional<std::invoke_result_t<Body&>> result;
        std::exception_ptr failure;
    } call{body, std::nullopt, nullptr};

    pthread_attr_t attributes;
    int error = ::pthread_attr_init(&attributes);
    if (error != 0) {
        throw std::system_error(error, std::generic_category(), "grammar compile thread");
    }
    pthread_t thread{};
    error = ::pthread_attr_setstacksize(&attributes, stack_bytes);
    if (error == 0) {
        error = ::pthread_create(
            &thread, &attributes,
            [](void* argument) -> void* {
                auto& running = *static_cast<Call*>(argument);
                try {
                    running.result.emplace(running.body());
                } catch (...) { running.failure = std::current_exception(); }
                return nullptr;
            },
            &call);
    }
    (void)::pthread_attr_destroy(&attributes);
    if (error != 0) {
        throw std::system_error(error, std::generic_category(), "grammar compile thread");
    }
    (void)::pthread_join(thread, nullptr);
    if (call.failure) { std::rethrow_exception(call.failure); }
    return std::move(*call.result);
}

struct Outcome {
    CompiledGrammar grammar;
    std::optional<GrammarError> error;
};

// Whether a grammar allows any first token. One that allows none, not even a stop token, leaves
// nothing to sample.
bool allows_a_first_token(const CompiledGrammar& grammar) {
    TokenMatcher matcher(grammar);
    std::vector<std::int32_t> words(static_cast<std::size_t>(matcher.mask_words()));
    matcher.fill_mask(words.data());
    return std::any_of(words.begin(), words.end(), [](std::int32_t word) { return word != 0; });
}

struct Flight {
    std::mutex mutex;
    std::condition_variable changed;
    std::optional<Outcome> outcome;
    std::exception_ptr failure;
    // The producer gave up before its compilation was queued: a waiter takes over.
    bool abandoned = false;

    [[nodiscard]] bool settled() const noexcept { return outcome || failure || abandoned; }
};

// Bookkeeping of one cache entry besides its key copies and payload: map and list nodes.
constexpr std::size_t kCacheEntryOverheadBytes = 256;

// The vocabulary and the cache. Compile jobs share it, so it outlives the service while a job
// still runs; the worker pool stays with the service, whose destruction drains it.
struct CompileState {
    struct Entry {
        Outcome outcome;
        std::size_t bytes = 0;
        std::list<std::string>::iterator lru;
    };

    CompileState(std::vector<std::string> token_bytes, std::vector<TokenId> stop_tokens,
                 const GrammarServiceOptions& service_options)
        : options(service_options), domain(static_cast<std::int32_t>(token_bytes.size())),
          tokenizer(token_bytes, xgrammar::VocabType::RAW, static_cast<int>(token_bytes.size()),
                    std::vector<std::int32_t>(stop_tokens.begin(), stop_tokens.end()),
                    /*add_prefix_space=*/false),
          stops(std::move(stop_tokens)), words(xgrammar::GetBitmaskSize(domain)) {}

    // Runs on a pool worker. Rejections are outcomes; anything else is a failure of the service.
    [[nodiscard]] Outcome compile(const GrammarRecipe& recipe) const {
        auto body = [&] { return compile_on_this_thread(recipe); };
        return run_with_stack(kCompileStackBytes, body);
    }

    [[nodiscard]] Outcome compile_on_this_thread(const GrammarRecipe& recipe) const {
        lower_thread_priority(options.compile_nice);
        xgrammar::ninfer::WarningCapture capture;
        Outcome outcome;
        try {
            const xgrammar::Grammar grammar = build_grammar(recipe.node(), tokenizer);
            xgrammar::GrammarCompiler compiler(tokenizer, static_cast<int>(options.compile_threads),
                                               /*cache_enabled=*/false);
            CompiledGrammar compiled(std::make_shared<const CompiledGrammar::Impl>(
                CompiledGrammar::Impl{.compiled    = compiler.CompileGrammar(grammar),
                                      .stop_tokens = stops,
                                      .mask_words  = words}));
            if (!allows_a_first_token(compiled)) {
                throw GrammarError(GrammarErrorKind::InvalidSchema, "the grammar allows no output");
            }
            outcome.grammar = std::move(compiled);
        } catch (const GrammarError& error) {
            outcome.error = error;
        } catch (const std::bad_alloc&) { throw; } catch (const std::exception& error) {
            outcome.error = GrammarError(GrammarErrorKind::InvalidSchema, error.what());
        }
        if (!outcome.error && !capture.warnings().empty()) {
            outcome.grammar = {};
            outcome.error =
                GrammarError(GrammarErrorKind::UnsupportedSchema,
                             "the grammar uses a construct that cannot be enforced exactly: " +
                                 capture.warnings().front());
        }
        return outcome;
    }

    // Keeps a finished outcome under the byte bound; called with `mutex` held. The key is stored
    // twice, in the map and in the recency list.
    void retain_locked(const std::string& key, Outcome outcome) {
        const std::size_t bytes =
            2 * key.size() + kCacheEntryOverheadBytes +
            (outcome.error ? std::strlen(outcome.error->what()) : outcome.grammar.memory_bytes());
        if (bytes > options.cache_bytes) { return; }
        while (cached_bytes > options.cache_bytes - bytes && !lru.empty()) {
            const auto victim = ready.find(lru.back());
            cached_bytes -= victim->second.bytes;
            ready.erase(victim);
            lru.pop_back();
            ++stats.evictions;
        }
        lru.push_front(key);
        try {
            ready.emplace(key, Entry{std::move(outcome), bytes, lru.begin()});
        } catch (...) {
            lru.pop_front();
            throw;
        }
        cached_bytes += bytes;
    }

    const GrammarServiceOptions options;
    const std::int32_t domain;
    const xgrammar::TokenizerInfo tokenizer;
    const std::vector<TokenId> stops;
    const std::int32_t words;

    mutable std::mutex mutex;
    std::unordered_map<std::string, Entry> ready;
    std::unordered_map<std::string, std::shared_ptr<Flight>> inflight;
    std::list<std::string> lru;
    std::size_t cached_bytes = 0;
    GrammarServiceStats stats;
};

} // namespace

struct GrammarService::Impl {
    Impl(std::vector<std::string> token_bytes, std::vector<TokenId> stop_tokens,
         const GrammarServiceOptions& options)
        : state(std::make_shared<CompileState>(std::move(token_bytes), std::move(stop_tokens),
                                               options)),
          workers(options.workers, options.queue_capacity) {}

    std::shared_ptr<CompileState> state;
    HostWorkerPool workers;
};

GrammarService::GrammarService(std::vector<std::string> token_bytes,
                               std::vector<TokenId> stop_tokens, GrammarServiceOptions options) {
    if (token_bytes.empty()) { throw std::invalid_argument("grammar vocabulary is empty"); }
    if (stop_tokens.empty()) {
        throw std::invalid_argument("a grammar vocabulary needs at least one stop token");
    }
    if (options.workers == 0 || options.compile_threads == 0 || options.queue_capacity == 0) {
        throw std::invalid_argument("grammar compile pool dimensions must be nonzero");
    }
    for (const TokenId token : stop_tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= token_bytes.size()) {
            throw std::invalid_argument("grammar stop token is outside the vocabulary");
        }
    }
    impl_ = std::make_shared<Impl>(std::move(token_bytes), std::move(stop_tokens), options);
}

GrammarService::~GrammarService() = default;

CompiledGrammar GrammarService::compile(const GrammarRecipe& recipe,
                                        const std::function<void()>& checkpoint) const {
    const std::string& key                    = recipe.key();
    const std::shared_ptr<CompileState> state = impl_->state;
    for (;;) {
        std::shared_ptr<Flight> flight;
        bool producer = false;
        {
            std::lock_guard lock(state->mutex);
            if (auto found = state->ready.find(key); found != state->ready.end()) {
                state->lru.splice(state->lru.begin(), state->lru, found->second.lru);
                ++state->stats.hits;
                if (found->second.outcome.error) { throw *found->second.outcome.error; }
                return found->second.outcome.grammar;
            }
            if (auto found = state->inflight.find(key); found != state->inflight.end()) {
                ++state->stats.singleflight_waits;
                flight = found->second;
            } else {
                flight   = std::make_shared<Flight>();
                producer = true;
                state->inflight.emplace(key, flight);
                ++state->stats.misses;
            }
        }

        if (producer) {
            try {
                (void)impl_->workers.submit(
                    [state, recipe, flight]() {
                        std::optional<Outcome> outcome;
                        std::exception_ptr failure;
                        try {
                            outcome = state->compile(recipe);
                        } catch (...) { failure = std::current_exception(); }
                        {
                            std::lock_guard lock(state->mutex);
                            state->inflight.erase(recipe.key());
                            if (outcome && outcome->error) { ++state->stats.rejected; }
                            // Keeping the outcome only saves later compilations; failing to keep
                            // it (out of memory) must not strand the requests waiting for it.
                            try {
                                if (outcome) { state->retain_locked(recipe.key(), *outcome); }
                            } catch (...) {}
                        }
                        {
                            std::lock_guard lock(flight->mutex);
                            flight->outcome = std::move(outcome);
                            flight->failure = failure;
                        }
                        flight->changed.notify_all();
                    },
                    checkpoint);
            } catch (...) {
                // The queue stayed full until this caller gave up. Its reason concerns only this
                // caller: the requests waiting for the same grammar take over its compilation.
                {
                    std::lock_guard lock(state->mutex);
                    state->inflight.erase(key);
                }
                {
                    std::lock_guard lock(flight->mutex);
                    flight->abandoned = true;
                }
                flight->changed.notify_all();
                throw;
            }
        }

        std::unique_lock lock(flight->mutex);
        while (!flight->settled()) {
            lock.unlock();
            if (checkpoint) { checkpoint(); }
            lock.lock();
            flight->changed.wait_for(lock, std::chrono::milliseconds(10),
                                     [&flight] { return flight->settled(); });
        }
        if (flight->abandoned) { continue; }
        if (flight->failure) { std::rethrow_exception(flight->failure); }
        if (flight->outcome->error) { throw *flight->outcome->error; }
        return flight->outcome->grammar;
    }
}

std::int32_t GrammarService::token_domain() const noexcept { return impl_->state->domain; }

std::int32_t GrammarService::mask_words() const noexcept { return impl_->state->words; }

GrammarServiceStats GrammarService::stats() const {
    GrammarServiceStats out;
    {
        const CompileState& state = *impl_->state;
        std::lock_guard lock(state.mutex);
        out              = state.stats;
        out.cached_bytes = state.cached_bytes;
        out.entries      = state.ready.size();
        out.inflight     = state.inflight.size();
    }
    const HostWorkerPool::Snapshot workers = impl_->workers.snapshot();
    out.queued                             = workers.queued;
    out.active                             = workers.active;
    return out;
}

} // namespace ninfer::grammar
