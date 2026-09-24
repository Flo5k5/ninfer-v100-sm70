#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::grammar {

// Limits of one grammar, checked on its sources before compiling: xgrammar can neither cancel a
// compilation nor bound its time. Compile time grows with the token-mask precomputation of the
// grammar's states, which max_compile_units bounds; the other limits bound the documents
// themselves and the depth of xgrammar's recursion over them.
struct GrammarLimits {
    // Serialized size of one JSON schema.
    std::size_t max_schema_bytes = 256 * 1024;
    // Nesting of arrays and objects anywhere in one schema document, annotations and enum values
    // included: parsing, serializing and hashing it recurse on this depth.
    std::uint32_t max_json_depth = 128;
    // Nesting of schema objects: properties, items, combinators and definitions.
    std::uint32_t max_schema_depth = 32;
    // Schema objects and enum values over every schema of one grammar.
    std::uint32_t max_schema_nodes = 10000;
    // Distinct schemas that the '$ref's of one schema name. xgrammar generates a referenced schema
    // where a reference to it first appears, so one recursion can pass through every target, each
    // up to max_schema_depth deep. The compile thread's stack holds the deepest schema these two
    // limits admit.
    std::uint32_t max_ref_targets = 128;
    // Compile units over every schema of one grammar (CompileCost::units). One unit is about 8 ms
    // on eight threads of a Xeon Gold 6240 with the Qwen3.8 vocabulary.
    std::uint32_t max_compile_units = 512;
    // Source length of one regex pattern (pattern and patternProperties keys).
    std::uint32_t max_pattern_bytes = 1024;
    // Unrolled atoms of one regex pattern (PatternCost), well below the 100,000 states at which
    // xgrammar's automaton builder gives up.
    std::uint32_t max_pattern_atoms = 16384;
    // Declared properties of an object with minProperties or maxProperties, whose rules grow with
    // the product of the property count and the count range.
    std::uint32_t max_counted_properties = 64;
};

// A string's minLength and maxLength cost their larger bound in units, up to this cap: xgrammar's
// compile time stops growing near it.
inline constexpr std::uint32_t kLengthBoundUnitCap = 256;
// Unrolled literal characters of regex patterns per unit.
inline constexpr std::uint64_t kPatternLiteralsPerUnit = 1024;
// Squared optional property counts per unit.
inline constexpr std::uint64_t kOptionalPairsPerUnit = 11000;

// What the schemas of one grammar cost to compile, as the quantities xgrammar's compile time grows
// with. Measured with the Qwen3.8 vocabulary, each term's unit takes about the same time.
struct CompileCost {
    // Unrolled class-like regex atoms (PatternCost::classes): one unit each.
    std::uint64_t pattern_classes = 0;
    // Unrolled literal regex characters: one unit per kPatternLiteralsPerUnit.
    std::uint64_t pattern_literals = 0;
    // Each object's optional property count squared, summed: the rules that let every optional
    // property be skipped grow quadratically. One unit per kOptionalPairsPerUnit.
    std::uint64_t optional_pairs = 0;
    // String length bounds, in units: the larger bound up to kLengthBoundUnitCap. xgrammar expands
    // identical bounds once, so each distinct pair of bounds counts once per schema.
    std::uint64_t length_bounds = 0;

    [[nodiscard]] std::uint64_t units() const noexcept;
};

// Admits the JSON schemas of one grammar. Each schema must stay inside the subset that xgrammar
// 0.2.7, with NInfer's local changes, enforces exactly in strict mode, and all of them together
// inside the limits:
//
// - a schema is `true` or an object; annotations are allowed anywhere and ignored (title,
//   description, default, examples, deprecated, readOnly, writeOnly, $comment, $schema,
//   nullable, discriminator, contentEncoding, contentMediaType, and names starting with "x-"),
//   and so are the definition containers $defs and definitions, whose schemas are admitted too;
// - besides those, an object holds exactly one form: a local $ref; const or enum, with an optional
//   type that every value matches; one of anyOf, oneOf and allOf; or a typed schema;
// - a typed schema has `type` (one name or an array of names) with keywords of those types, or no
//   type with object or array keywords only: properties, required (names of properties),
//   additionalProperties, patternProperties, propertyNames, minProperties, maxProperties,
//   unevaluatedProperties; items, prefixItems, minItems, maxItems, unevaluatedItems; minimum,
//   maximum, exclusiveMinimum, exclusiveMaximum, multipleOf; and for strings exactly one of a
//   supported format (date, date-time, duration, email, hostname, ipv4, ipv6, time, uri, uuid), a
//   pattern, or length bounds;
// - an object uses propertyNames or patternProperties (a single pattern) only without declared
//   properties, propertyNames only with additional properties allowed, and patternProperties only
//   without additional properties; minProperties and maxProperties need declared properties and
//   no undeclared keys, which could repeat; an object that allows undeclared keys declares no name
//   holding a quote, a backslash or a control character;
// - a $ref is "#" or "#/" followed by non-empty keys, naming a schema admitted here, and a schema
//   names at most max_ref_targets distinct targets;
// - every schema accepts some finite value: required properties, items and references that only
//   ever lead back to themselves are rejected;
// - a pattern is printable ASCII (other characters can be written as \uXXXX escapes), repeats
//   nothing more than kRegexUnrollLimit times, and unrolls to at most max_pattern_atoms atoms.
//
// xgrammar silently ignores or weakens what lies outside the subset (not, if/then/else, dependent
// schemas, uniqueItems, contains, keywords next to a $ref or a combinator, unknown formats, a
// pattern next to length bounds, required names that are not properties, overlapping property
// patterns), so admitting it would constrain output with a weaker grammar. Constructs it can only
// approximate (an overlapping oneOf, some multipleOf ranges, a pattern that needs a character a
// JSON string must escape) are rejected when the grammar is compiled.
class SchemaAdmission {
public:
    explicit SchemaAdmission(const GrammarLimits& limits) : limits_(limits) {}

    // Checks one schema, given as JSON text, and charges it to this grammar. `location` names the
    // schema in error messages (for example "response_format.json_schema.schema"). Returns the
    // schema as compact JSON text with its key order preserved. Throws GrammarError.
    [[nodiscard]] std::string admit(std::string_view schema_json, std::string_view location);

    [[nodiscard]] std::uint32_t nodes() const noexcept { return nodes_; }

    [[nodiscard]] const CompileCost& compile_cost() const noexcept { return cost_; }

    [[nodiscard]] std::uint64_t compile_units() const noexcept { return cost_.units(); }

private:
    GrammarLimits limits_;
    std::uint32_t nodes_ = 0;
    CompileCost cost_;
};

} // namespace ninfer::grammar
