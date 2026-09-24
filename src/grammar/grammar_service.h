#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::grammar {

// A grammar over one complete model output, composed by a target frontend from sources it has
// admitted. Values are immutable and cheap to copy; key() is a canonical text of the composition
// and the compile-cache key.
class GrammarRecipe {
public:
    // The JSON value of an admitted schema (SchemaAdmission) in compact form: ", " and ": "
    // separators and no other whitespace. Object properties keep their declared order.
    [[nodiscard]] static GrammarRecipe json_schema(std::string admitted_schema);
    // An xgrammar structural-tag format as JSON text: the "format" member of a structural tag.
    // JSON schemas it embeds must have been admitted by the caller.
    [[nodiscard]] static GrammarRecipe structural_tag(std::string format_json);
    // Outputs made of every part in order.
    [[nodiscard]] static GrammarRecipe sequence(std::vector<GrammarRecipe> parts);
    // Outputs of any one alternative.
    [[nodiscard]] static GrammarRecipe choice(std::vector<GrammarRecipe> alternatives);

    [[nodiscard]] const std::string& key() const noexcept;

    struct Node;

    [[nodiscard]] const Node& node() const noexcept { return *node_; }

private:
    explicit GrammarRecipe(std::shared_ptr<const Node> node) noexcept : node_(std::move(node)) {}

    std::shared_ptr<const Node> node_;
};

// An immutable compiled grammar bound to one model vocabulary. Copies share it.
class CompiledGrammar {
public:
    struct Impl;

    CompiledGrammar() noexcept = default;

    explicit CompiledGrammar(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

    [[nodiscard]] explicit operator bool() const noexcept { return impl_ != nullptr; }

    [[nodiscard]] std::size_t memory_bytes() const noexcept;

    [[nodiscard]] const Impl& impl() const noexcept { return *impl_; }

private:
    std::shared_ptr<const Impl> impl_;
};

// The grammar state of one output. It advances only with committed tokens; the draft walk leaves
// it unchanged. Masks hold one bit per token of the domain: token t in bit t % 32 of word t / 32.
// Not thread-safe; one request owns it.
class TokenMatcher {
public:
    explicit TokenMatcher(const CompiledGrammar& grammar);
    ~TokenMatcher();

    TokenMatcher(TokenMatcher&&) noexcept;
    TokenMatcher& operator=(TokenMatcher&&) noexcept;
    TokenMatcher(const TokenMatcher&)            = delete;
    TokenMatcher& operator=(const TokenMatcher&) = delete;

    [[nodiscard]] std::int32_t mask_words() const noexcept;

    // Writes the tokens the grammar allows after the committed output. Once the output has ended
    // (a stop token accepted), only the stop tokens are allowed.
    void fill_mask(std::int32_t* words);

    // Writes row 0 as fill_mask(), then walks the drafts in order: draft i is walked when it is
    // not a stop token and row i allows it, and row i + 1 then holds the tokens allowed after it.
    // The walk ends at the first draft it cannot take. Returns the number of drafts walked; rows
    // past it are left untouched. The committed state is unchanged afterwards.
    [[nodiscard]] std::uint32_t fill_draft_masks(std::span<const TokenId> drafts,
                                                 std::int32_t* rows, std::size_t row_stride_words);

    // Advances the committed output by one token. Returns false, with the state unchanged, when
    // the grammar does not allow it.
    [[nodiscard]] bool accept(TokenId token);
    // Removes the last `tokens` committed tokens.
    void rollback(std::uint32_t tokens);
    [[nodiscard]] bool terminated() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct GrammarServiceOptions {
    // Compilations run at once, each on `compile_threads` xgrammar threads. Further compilations
    // wait in a queue of `queue_capacity`, beyond which their callers block.
    std::uint32_t workers         = 2;
    std::uint32_t compile_threads = 8;
    std::size_t queue_capacity    = 16;
    // Compiled grammars and rejected sources kept for reuse, by approximate memory size.
    std::size_t cache_bytes = 1ULL << 30;
    // Niceness added to the compile threads (Linux only), so that compilation yields the CPU to
    // the engine worker and the HTTP threads.
    int compile_nice = 10;
};

struct GrammarServiceStats {
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t rejected           = 0;
    std::uint64_t evictions          = 0;
    std::size_t cached_bytes         = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued               = 0;
    std::size_t active               = 0;
};

// Compiles grammars against one model vocabulary. Compilation runs on a bounded pool off the
// calling thread, identical recipes share one compilation, and results (including rejections) are
// kept in a byte-bounded LRU cache. Thread-safe.
class GrammarService {
public:
    // token_bytes[t] holds the bytes of token t for every token of the domain; special tokens are
    // their literal text. stop_tokens are the ids that end an output.
    GrammarService(std::vector<std::string> token_bytes, std::vector<TokenId> stop_tokens,
                   GrammarServiceOptions options);
    ~GrammarService();

    GrammarService(const GrammarService&)            = delete;
    GrammarService& operator=(const GrammarService&) = delete;

    // Returns the compiled grammar of the recipe, compiling it if needed. The caller blocks until
    // it is ready; `checkpoint` runs periodically while it waits and may throw to stop waiting,
    // in which case the compilation still completes and is cached. Throws GrammarError when
    // xgrammar rejects the recipe or relaxes a construct it cannot enforce exactly.
    [[nodiscard]] CompiledGrammar compile(const GrammarRecipe& recipe,
                                          const std::function<void()>& checkpoint) const;

    [[nodiscard]] std::int32_t token_domain() const noexcept;
    [[nodiscard]] std::int32_t mask_words() const noexcept;
    [[nodiscard]] GrammarServiceStats stats() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer::grammar
