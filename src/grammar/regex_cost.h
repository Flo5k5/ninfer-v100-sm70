#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::grammar {

// Largest bound of a regex repetition. xgrammar's automaton holds a bounded repetition as that many
// copies of its body up to this count. Above it, xgrammar compiles the body as a separate rule
// without the JSON string escaping, which lets a raw quote or control character through, and its
// trial automaton still unrolls every copy at a cost quadratic in the count: larger bounds are
// rejected.
inline constexpr std::uint64_t kRegexUnrollLimit = 128;

// The atoms of a regex pattern, counted once per copy xgrammar's automaton holds: the product, over
// the repetitions that enclose the atom, of their copy counts (the upper bound, or the lower bound
// plus one when there is none). xgrammar precomputes a token mask for every automaton state. A
// state after a character class, a dot or a class escape (\d, \w, \s, \p and their negations) can
// take most of the vocabulary; a state after a literal character only the tokens that start with
// it.
struct PatternCost {
    std::uint64_t classes  = 0;
    std::uint64_t literals = 0;
};

// Throws GrammarError: InvalidSchema on a pattern xgrammar cannot parse (unbalanced groups or
// classes, a trailing backslash, a quantifier with nothing to repeat, a malformed brace
// repetition), UnsupportedSchema on a repetition bound above kRegexUnrollLimit.
[[nodiscard]] PatternCost pattern_cost(std::string_view pattern);

} // namespace ninfer::grammar
