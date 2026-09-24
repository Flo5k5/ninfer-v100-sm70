#include "grammar/regex_cost.h"

#include "grammar/grammar_error.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>

namespace ninfer::grammar {
namespace {

// Far above any limit; keeps products and sums of atom counts from overflowing.
constexpr std::uint64_t kSaturatedCount = std::uint64_t{1} << 40;

std::uint64_t add_counts(std::uint64_t a, std::uint64_t b) {
    return std::min(a + b, kSaturatedCount);
}

std::uint64_t multiply_counts(std::uint64_t a, std::uint64_t b) {
    if (a == 0 || b == 0) { return 0; }
    return a > kSaturatedCount / b ? kSaturatedCount : std::min(a * b, kSaturatedCount);
}

PatternCost operator+(PatternCost a, PatternCost b) {
    return PatternCost{.classes  = add_counts(a.classes, b.classes),
                       .literals = add_counts(a.literals, b.literals)};
}

PatternCost operator*(PatternCost cost, std::uint64_t copies) {
    return PatternCost{.classes  = multiply_counts(cost.classes, copies),
                       .literals = multiply_counts(cost.literals, copies)};
}

constexpr PatternCost kClass{.classes = 1, .literals = 0};
constexpr PatternCost kLiteral{.classes = 0, .literals = 1};

bool is_hex_digit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

bool is_flag_character(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-';
}

// Reads a pattern the way xgrammar's regex parser does, keeping only what the cost needs.
class RegexCost {
public:
    explicit RegexCost(std::string_view pattern) : pattern_(pattern) {}

    PatternCost cost() { return alternatives(false); }

private:
    struct Repetition {
        std::uint64_t lower = 0;
        std::optional<std::uint64_t> upper;
    };

    [[noreturn]] static void invalid(const std::string& what) {
        throw GrammarError(GrammarErrorKind::InvalidSchema, "invalid regex pattern: " + what);
    }

    [[noreturn]] static void unsupported(const std::string& what) {
        throw GrammarError(GrammarErrorKind::UnsupportedSchema, what);
    }

    [[nodiscard]] bool at_end() const noexcept { return at_ >= pattern_.size(); }

    [[nodiscard]] char peek() const noexcept { return pattern_[at_]; }

    // The alternatives of a group up to its closing parenthesis, or of the whole pattern.
    PatternCost alternatives(bool nested) {
        PatternCost total;
        while (!at_end()) {
            const char c = peek();
            if (c == ')') {
                if (!nested) { invalid("unbalanced ')'"); }
                ++at_;
                return total;
            }
            if (c == '|' || c == '^' || c == '$') {
                ++at_;
                continue;
            }
            PatternCost atom;
            if (c == '(') {
                ++at_;
                group_prefix();
                atom = alternatives(true);
            } else if (c == '[') {
                character_class();
                atom = kClass;
            } else if (c == '.') {
                ++at_;
                atom = kClass;
            } else if (c == '\\') {
                atom = escape() ? kClass : kLiteral;
            } else if (c == '*' || c == '+' || c == '?' || c == '{') {
                invalid("a quantifier has nothing to repeat");
            } else {
                ++at_;
                atom = kLiteral;
            }
            total = total + atom * copies();
        }
        if (nested) { invalid("unbalanced '('"); }
        return total;
    }

    // Skips "?:", "?=", "?!", "?<=", "?<!", "?<name>" and inline flags after an opening
    // parenthesis.
    void group_prefix() {
        if (at_end() || peek() != '?') { return; }
        ++at_;
        if (!at_end() && peek() == '<') {
            if (at_ + 1 < pattern_.size() &&
                (pattern_[at_ + 1] == '=' || pattern_[at_ + 1] == '!')) {
                at_ += 2;
                return;
            }
            const std::size_t close = pattern_.find('>', at_);
            if (close == std::string_view::npos) { invalid("unterminated group name"); }
            at_ = close + 1;
            return;
        }
        while (!at_end() && is_flag_character(peek())) { ++at_; }
        if (!at_end() && (peek() == ':' || peek() == '=' || peek() == '!')) { ++at_; }
    }

    void character_class() {
        ++at_;
        if (!at_end() && peek() == '^') { ++at_; }
        if (!at_end() && peek() == ']') { ++at_; }
        while (!at_end() && peek() != ']') {
            if (peek() == '\\') {
                (void)escape();
            } else {
                ++at_;
            }
        }
        if (at_end()) { invalid("unterminated character class"); }
        ++at_;
    }

    void braced() {
        const std::size_t close = pattern_.find('}', at_);
        if (close == std::string_view::npos) { invalid("unterminated escape"); }
        at_ = close + 1;
    }

    // Whether the escape is a class; otherwise it denotes one character.
    bool escape() {
        ++at_;
        if (at_end()) { invalid("trailing '\\'"); }
        const char escaped = pattern_[at_++];
        switch (escaped) {
        case 'd':
        case 'D':
        case 'w':
        case 'W':
        case 's':
        case 'S':
            return true;
        case 'p':
        case 'P':
            if (!at_end() && peek() == '{') {
                braced();
            } else if (!at_end()) {
                ++at_;
            }
            return true;
        case 'x':
        case 'u': {
            if (!at_end() && peek() == '{') {
                braced();
                return false;
            }
            const std::size_t digits = escaped == 'x' ? 2 : 4;
            for (std::size_t i = 0; i < digits && !at_end() && is_hex_digit(peek()); ++i) { ++at_; }
            return false;
        }
        case 'c':
            if (!at_end()) { ++at_; }
            return false;
        default:
            return false;
        }
    }

    // Copies of the preceding atom that the automaton holds, read from the quantifier after it.
    std::uint64_t copies() {
        if (at_end()) { return 1; }
        std::uint64_t count = 1;
        const char c        = peek();
        if (c == '*' || c == '+' || c == '?') {
            ++at_;
        } else if (c == '{') {
            const Repetition bounds = repetition();
            if (bounds.upper.value_or(bounds.lower) > kRegexUnrollLimit) {
                unsupported("pattern repetition bounds above " + std::to_string(kRegexUnrollLimit) +
                            " are not supported");
            }
            count = bounds.upper ? *bounds.upper : bounds.lower + 1;
        } else {
            return 1;
        }
        if (!at_end() && (peek() == '?' || peek() == '+')) { ++at_; }
        return count;
    }

    // "{n}", "{n,}" or "{n,m}", with the spaces xgrammar skips around the counts.
    Repetition repetition() {
        ++at_;
        skip_spaces();
        Repetition bounds{.lower = count()};
        skip_spaces();
        if (!at_end() && peek() == '}') {
            ++at_;
            bounds.upper = bounds.lower;
            return bounds;
        }
        if (at_end() || peek() != ',') { invalid("malformed repetition"); }
        ++at_;
        skip_spaces();
        if (!at_end() && peek() == '}') {
            ++at_;
            return bounds;
        }
        bounds.upper = count();
        if (*bounds.upper < bounds.lower) {
            invalid("a repetition's upper bound is below its lower bound");
        }
        skip_spaces();
        if (at_end() || peek() != '}') { invalid("malformed repetition"); }
        ++at_;
        return bounds;
    }

    void skip_spaces() {
        while (!at_end() && peek() == ' ') { ++at_; }
    }

    std::uint64_t count() {
        std::uint64_t value = 0;
        std::size_t digits  = 0;
        while (!at_end() && peek() >= '0' && peek() <= '9') {
            if (++digits > 9) { invalid("a repetition count is too large"); }
            value = value * 10 + static_cast<std::uint64_t>(peek() - '0');
            ++at_;
        }
        if (digits == 0) { invalid("malformed repetition"); }
        return value;
    }

    std::string_view pattern_;
    std::size_t at_ = 0;
};

} // namespace

PatternCost pattern_cost(std::string_view pattern) { return RegexCost(pattern).cost(); }

} // namespace ninfer::grammar
