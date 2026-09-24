#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::grammar {

enum class GrammarErrorKind : std::uint8_t {
    // Malformed JSON, or a schema the JSON Schema rules make invalid or unsatisfiable.
    InvalidSchema,
    // A keyword, keyword combination or grammar construct that xgrammar cannot enforce exactly.
    UnsupportedSchema,
    // A schema or grammar beyond the configured compile-cost limits.
    LimitExceeded,
};

// A problem with a request's grammar source. The message names the cause and may quote the
// client's schema; it is meant for the client that sent it, never for the process logs.
class GrammarError final : public std::invalid_argument {
public:
    GrammarError(GrammarErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] GrammarErrorKind kind() const noexcept { return kind_; }

private:
    GrammarErrorKind kind_;
};

} // namespace ninfer::grammar
