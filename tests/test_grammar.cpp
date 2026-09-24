// Grammar service behavior on a byte-level toy vocabulary: schema admission (the enforced JSON
// Schema subset and the compile-cost limits), compilation and its cache, and the token matcher
// that masks, walks drafts and commits tokens. No model or GPU is involved.
#include "grammar/grammar_error.h"
#include "grammar/grammar_service.h"
#include "grammar/regex_cost.h"
#include "grammar/schema_admission.h"

#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <unistd.h>

using namespace ninfer;
using namespace ninfer::grammar;

namespace {

int failures = 0;

void check(bool condition, std::string_view what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// Tokens 0..255 are single bytes; the added tokens below are matched as text.
constexpr TokenId kThink      = 256;
constexpr TokenId kCloseThink = 257;
constexpr TokenId kObjectKey  = 258; // {"
constexpr TokenId kStop       = 259;
constexpr TokenId kEndOfText  = 260;
constexpr TokenId kDomain     = 261;

std::vector<std::string> toy_vocabulary() {
    std::vector<std::string> tokens;
    for (int byte = 0; byte < 256; ++byte) { tokens.emplace_back(1, static_cast<char>(byte)); }
    tokens.emplace_back("<think>");
    tokens.emplace_back("</think>");
    tokens.emplace_back("{\"");
    tokens.emplace_back("<|im_end|>");
    tokens.emplace_back("<|endoftext|>");
    return tokens;
}

std::vector<TokenId> bytes(std::string_view text) {
    std::vector<TokenId> ids;
    for (const char c : text) { ids.push_back(static_cast<unsigned char>(c)); }
    return ids;
}

std::vector<TokenId> join(std::initializer_list<std::vector<TokenId>> parts) {
    std::vector<TokenId> out;
    for (const auto& part : parts) { out.insert(out.end(), part.begin(), part.end()); }
    return out;
}

GrammarService make_service(GrammarServiceOptions options = {}) {
    options.compile_threads = 2;
    return GrammarService(toy_vocabulary(), {kStop, kEndOfText}, options);
}

// The reasoning part of a thinking output: any text without <think> or </think>, then </think>
// and a blank line.
GrammarRecipe reasoning_prefix() {
    return GrammarRecipe::structural_tag(
        R"({"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text",)"
        R"("excludes":["<think>","</think>"]},"end":"</think>"},{"type":"const_string",)"
        R"("value":"\n\n"}]})");
}

bool accepts(const CompiledGrammar& grammar, const std::vector<TokenId>& tokens) {
    TokenMatcher matcher(grammar);
    for (const TokenId token : tokens) {
        if (!matcher.accept(token)) { return false; }
    }
    return matcher.terminated();
}

std::optional<GrammarErrorKind> admission_error(std::string_view schema,
                                                GrammarLimits limits = {}) {
    try {
        SchemaAdmission admission(limits);
        (void)admission.admit(schema, "schema");
    } catch (const GrammarError& error) { return error.kind(); }
    return std::nullopt;
}

std::string admission_message(std::string_view schema, GrammarLimits limits = {}) {
    try {
        SchemaAdmission admission(limits);
        (void)admission.admit(schema, "response_format.json_schema.schema");
    } catch (const GrammarError& error) { return error.what(); }
    return {};
}

// Schemas inside the enforced subset; each must also compile without an xgrammar warning.
const char* const kAdmitted[] = {
    R"({"type":"object","properties":{"name":{"type":"string"},"n":{"type":"integer",)"
    R"("minimum":0}},"required":["name"],"additionalProperties":false})",
    R"({"type":"object","properties":{"a":{"type":"integer"}},)"
    R"("additionalProperties":{"type":"string"}})",
    R"({"type":"object","properties":{"a":{"type":"integer"},"b":{}},"required":["a"],)"
    R"("minProperties":1,"maxProperties":2})",
    R"({"type":"array","items":{"type":"number"},"minItems":1,"maxItems":4})",
    R"({"type":"array","prefixItems":[{"type":"string"},{"type":"boolean"}]})",
    R"({"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"title":"Opt"})",
    R"({"enum":["a","b"],"type":"string","description":"choice"})",
    R"({"enum":["a",null],"type":["string","null"]})",
    R"({"const":1.0,"type":"integer"})",
    R"({"const":3})",
    R"({"$defs":{"n":{"type":"object","properties":{"next":{"$ref":"#/$defs/n"}}}},)"
    R"("$ref":"#/$defs/n"})",
    R"({"type":"array","items":{"$ref":"#"}})",
    R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"$ref":"#/properties/a"}}})",
    R"({"type":"string","format":"date-time"})",
    R"({"type":"string","pattern":"^[a-z]+$"})",
    R"({"type":"string","minLength":1,"maxLength":8})",
    R"({"type":["string","null"],"maxLength":4})",
    R"({"type":"integer","format":"int64","x-order":1})",
    R"({"properties":{"a":{}},"required":["a"]})",
    R"({"type":"object","propertyNames":{"pattern":"^[a-z]+$"},"additionalProperties":true})",
    R"({"type":"object","propertyNames":{"pattern":"^[a-z]+$"}})",
    R"({"type":"object","patternProperties":{"^x_":{"type":"integer"}}})",
    R"({"type":"string","format":"email"})",
    R"({"type":"string","pattern":"^a{0,128}$"})",
    R"({"type":"object","properties":{"a\"b":{"type":"integer"}},"additionalProperties":false})",
    R"({"$defs":{"node":{"type":"object","properties":{"children":{"type":"array",)"
    R"("items":{"$ref":"#/$defs/node"}}},"required":["children"]}},"$ref":"#/$defs/node"})",
    R"({"type":"object","properties":{"a":{"$ref":"#"}}})",
    R"({"type":"array","prefixItems":[{"type":"string"},{"$ref":"#"}],"minItems":1})",
    R"(true)",
    R"({})",
};

// Schemas xgrammar would silently weaken or cannot compile safely, with the error each gets.
const std::pair<const char*, GrammarErrorKind> kRejected[] = {
    {R"({"type":"array","uniqueItems":true})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"array","contains":{"type":"integer"}})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"array","minContains":1})", GrammarErrorKind::UnsupportedSchema},
    {R"({"not":{"type":"string"}})", GrammarErrorKind::UnsupportedSchema},
    {R"({"if":{"type":"string"},"then":{"minLength":1}})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","dependentRequired":{"a":["b"]}})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"array","items":{},"additionalItems":false})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","foo":1})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","pattern":"^a+$","maxLength":4})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","format":"date","pattern":"^a"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","format":"uri-reference"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"$ref":"#/$defs/a","type":"object","$defs":{"a":{}}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"anyOf":[{"type":"string"}],"type":"string"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"enum":["a"],"maxLength":1})", GrammarErrorKind::UnsupportedSchema},
    {R"({"const":1,"enum":[1]})", GrammarErrorKind::UnsupportedSchema},
    {R"({"enum":["a",1],"type":"string"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"const":1,"type":"string"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"$ref":"https://example.com/schema"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"$ref":"#/$defs/missing"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"$ref":"#/x-s","x-s":{"type":"string","not":{"const":"a"}}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","properties":{"p":{"$ref":"#/properties/q/default"},)"
     R"("q":{"type":"integer","default":{"type":"string"}}}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"$defs":{"a~b":{}},"$ref":"#/$defs/a~0b"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"anyOf":[{"type":"string"}],"$defs":{"x":{"$ref":"#/anyOf/0"}}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","propertyNames":{"maxLength":3},)"
     R"("additionalProperties":{"$ref":"#/propertyNames"}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","properties":{"a":{}},"required":["a","b"]})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","required":["b"],"additionalProperties":{"type":"integer"}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","patternProperties":{"^x":{}},"required":["x1"]})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","patternProperties":{"^x":{}},"propertyNames":{"maxLength":2}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","properties":{"a":{}},"propertyNames":{"pattern":"^[a-z]+$"},)"
     R"("additionalProperties":true})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","properties":{"ab":{}},"patternProperties":{"^a":{}}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","patternProperties":{"^a":{},"b$":{}}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","patternProperties":{"^x-":{}},"additionalProperties":{}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","additionalProperties":{"type":"integer"},"minProperties":2})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","properties":{"a":{}},"additionalProperties":true,)"
     R"("maxProperties":1})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"minLength":2})", GrammarErrorKind::UnsupportedSchema},
    {R"({"required":["a"]})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"integer","$id":"x"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object",)", GrammarErrorKind::InvalidSchema},
    {R"(false)", GrammarErrorKind::InvalidSchema},
    {R"({"type":"text"})", GrammarErrorKind::InvalidSchema},
    {R"({"enum":["a"],"type":"text"})", GrammarErrorKind::InvalidSchema},
    {R"({"enum":[]})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"object","required":"a"})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"string","pattern":"(a"})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"string","pattern":"a{2"})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"string","maxLength":-1})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"object","properties":{"a":{}},"minProperties":-1})",
     GrammarErrorKind::InvalidSchema},
    // Repetitions past the unroll limit, which xgrammar compiles without JSON string escaping.
    {R"({"type":"string","pattern":"^.{200}$"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","pattern":"^[^a]{0,300}$"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","pattern":"^\\S{0,500}$"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","pattern":"^x(.{0,129})$"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","pattern":"^a{0,10000}$"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","pattern":"^a{129,}$"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","propertyNames":{"pattern":"^[^a]{0,200}$"}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","patternProperties":{"^[^a]{1,200}$":{"type":"integer"}}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"string","pattern":"^((a{128}){128}){2}$"})", GrammarErrorKind::LimitExceeded},
    {R"({"type":"string","pattern":"^[A-Za-z\u00c0-\u00ff ]+$"})",
     GrammarErrorKind::UnsupportedSchema},
    // Keys xgrammar would emit although the object admits none, or spell without escapes.
    {R"({"type":"object","propertyNames":{"pattern":"^a"},"additionalProperties":false})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","propertyNames":{"pattern":"^a"},"unevaluatedProperties":false})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","properties":{"a\"b":{}},"additionalProperties":true})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","properties":{"a\\b":{}},"additionalProperties":{}})",
     GrammarErrorKind::UnsupportedSchema},
    {R"({"type":"object","properties":{"a\nb":{}},"unevaluatedProperties":true})",
     GrammarErrorKind::UnsupportedSchema},
    // Schemas no finite value satisfies.
    {R"({"$ref":"#"})", GrammarErrorKind::InvalidSchema},
    {R"({"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"})",
     GrammarErrorKind::InvalidSchema},
    {R"({"type":"object","properties":{"a":{"$ref":"#"}},"required":["a"]})",
     GrammarErrorKind::InvalidSchema},
    {R"({"type":"array","items":{"$ref":"#"},"minItems":1})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"array","minItems":1})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"object","properties":{"a":{}},"minProperties":2,"additionalProperties":false})",
     GrammarErrorKind::InvalidSchema},
    {R"({"type":"object","properties":{"a":{},"b":{}},"required":["a","b"],"maxProperties":1})",
     GrammarErrorKind::InvalidSchema},
    {R"({"type":"string","minLength":3,"maxLength":2})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"array","items":{},"minItems":3,"maxItems":2})", GrammarErrorKind::InvalidSchema},
    {R"({"type":"array","items":{},"maxItems":-1})", GrammarErrorKind::InvalidSchema},
    // References spelled so that xgrammar would compile their target once per spelling.
    {R"({"$ref":"#/"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"$defs":{"a":{}},"$ref":"#//$defs/a"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"$defs":{"a":{}},"$ref":"#/$defs//a"})", GrammarErrorKind::UnsupportedSchema},
    {R"({"$defs":{"a":{}},"$ref":"#/$defs/a/"})", GrammarErrorKind::UnsupportedSchema},
};

std::string nested_annotation(std::size_t levels) {
    return R"({"type":"string","x-a":)" + std::string(levels, '[') + std::string(levels, ']') + "}";
}

std::string object_of(std::size_t optional) {
    std::string schema = R"({"type":"object","properties":{)";
    for (std::size_t i = 0; i < optional; ++i) {
        schema += (i == 0 ? "\"p" : ",\"p") + std::to_string(i) + "\":{\"type\":\"integer\"}";
    }
    return schema + "}}";
}

// `count` required properties, each an object with one optional property.
std::string small_objects(std::size_t count) {
    std::string properties;
    std::string required;
    for (std::size_t i = 0; i < count; ++i) {
        const std::string name = "o" + std::to_string(i);
        properties += (i == 0 ? "\"" : ",\"") + name +
                      R"(":{"type":"object","properties":{"x":{"type":"integer"}}})";
        required += (i == 0 ? "\"" : ",\"") + name + "\"";
    }
    return R"({"type":"object","properties":{)" + properties + R"(},"required":[)" + required +
           "]}";
}

// `count` required string properties with the same maxLength.
std::string repeated_bound(std::size_t count, std::size_t max_length) {
    std::string properties;
    std::string required;
    for (std::size_t i = 0; i < count; ++i) {
        const std::string name = "s" + std::to_string(i);
        properties += (i == 0 ? "\"" : ",\"") + name + R"(":{"type":"string","maxLength":)" +
                      std::to_string(max_length) + "}";
        required += (i == 0 ? "\"" : ",\"") + name + "\"";
    }
    return R"({"type":"object","properties":{)" + properties + R"(},"required":[)" + required +
           "]}";
}

// An object whose optional properties refer to `targets` distinct definitions.
std::string referencing(std::size_t targets) {
    std::string definitions;
    std::string properties;
    for (std::size_t i = 0; i < targets; ++i) {
        const std::string name = "d" + std::to_string(i);
        definitions += (i == 0 ? "\"" : ",\"") + name + R"(":{"type":"integer"})";
        properties += (i == 0 ? "\"p" : ",\"p") + std::to_string(i) + R"(":{"$ref":"#/$defs/)" +
                      name + R"("})";
    }
    return R"({"$defs":{)" + definitions + R"(},"type":"object","properties":{)" + properties +
           "}}";
}

// A chain of max_ref_targets definitions, each nesting objects as deep as max_schema_depth allows
// before referring to the next one.
std::string deepest_reference_chain(const GrammarLimits& limits) {
    std::string definitions;
    for (std::uint32_t target = 0; target < limits.max_ref_targets; ++target) {
        std::string inner = target + 1 < limits.max_ref_targets
                                ? R"({"$ref":"#/$defs/d)" + std::to_string(target + 1) + R"("})"
                                : std::string(R"({"type":"integer"})");
        // A definition sits one level below the root: max_schema_depth - 1 objects put the
        // innermost schema at max_schema_depth.
        for (std::uint32_t level = 1; level < limits.max_schema_depth; ++level) {
            inner = R"({"type":"object","properties":{"x":)" + inner + "}}";
        }
        definitions += (target == 0 ? "\"d" : ",\"d") + std::to_string(target) + "\":" + inner;
    }
    return R"({"$defs":{)" + definitions + R"(},"$ref":"#/$defs/d0"})";
}

std::uint64_t units_of(std::string_view schema) {
    SchemaAdmission admission(GrammarLimits{});
    try {
        (void)admission.admit(schema, "schema");
    } catch (const GrammarError& error) {
        std::cerr << "unexpected rejection: " << error.what() << '\n';
        return 0;
    }
    return admission.compile_units();
}

void admission_contract() {
    for (const char* schema : kAdmitted) {
        const auto error = admission_error(schema);
        check(!error, std::string("admits ") + schema);
    }
    for (const auto& [schema, kind] : kRejected) {
        const auto error = admission_error(schema);
        check(error && *error == kind, std::string("rejects ") + schema);
    }

    const std::string message =
        admission_message(R"({"type":"object","properties":{"a/b":{"type":"array",)"
                          R"("uniqueItems":true}}})");
    check(message == "response_format.json_schema.schema at #/properties/a~1b: keyword "
                     "'uniqueItems' is not supported",
          "names the schema location and the JSON pointer: " + message);

    GrammarLimits limits;
    limits.max_schema_bytes = 32;
    check(admission_error(R"({"type":"object","description":"a long description"})", limits) ==
              GrammarErrorKind::LimitExceeded,
          "limits the schema size");
    // Nesting inside annotations is never walked, yet parsing and hashing recurse on it.
    check(admission_error(nested_annotation(129)) == GrammarErrorKind::LimitExceeded &&
              admission_error(nested_annotation(100000)) == GrammarErrorKind::LimitExceeded &&
              !admission_error(nested_annotation(100)),
          "limits the nesting of the whole document before parsing it");
    limits                  = {};
    limits.max_schema_depth = 2;
    check(admission_error(R"({"type":"array","items":{"type":"array","items":{"type":"array",)"
                          R"("items":{}}}})",
                          limits) == GrammarErrorKind::LimitExceeded,
          "limits the schema depth");
    limits                  = {};
    limits.max_schema_nodes = 3;
    check(admission_error(R"({"enum":[1,2,3]})", limits) == GrammarErrorKind::LimitExceeded,
          "counts enum values as schema nodes");
    limits = {};
    check(admission_error(R"({"type":"string","maxLength":100000})", limits) == std::nullopt,
          "one length bound saturates at the unit cap");
    check(admission_error(R"({"type":"object","properties":{"a":{"type":"string","maxLength":)"
                          R"(300},"b":{"type":"string","minLength":300},"c":{"type":"string",)"
                          R"("maxLength":1}}})",
                          limits) == GrammarErrorKind::LimitExceeded,
          "limits the total of string length bounds");
    check(units_of(repeated_bound(10, 255)) == 255,
          "identical length bounds of one schema count once");
    check(admission_error(R"({"type":"string","pattern":"^([a-z]{1,32}){1,32}$"})") ==
              GrammarErrorKind::LimitExceeded,
          "charges unrolled regex copies as a product");
    check(units_of(R"({"type":"string","pattern":"^(abcdefgh){128}$"})") == 1,
          "a unit pays for 1024 unrolled literal characters");
    check(admission_error(object_of(2373)) == std::nullopt &&
              admission_error(object_of(2374)) == GrammarErrorKind::LimitExceeded,
          "charges optional properties quadratically");
    check(units_of(small_objects(600)) == 1,
          "the optional properties of small objects add up before rounding");
    // A target that stands in the document, unlike a definition, is compiled again for its
    // references.
    check(units_of(R"({"type":"object","properties":{"a":{"type":"string","pattern":)"
                   R"("^[a-z]{1,100}$"},"b":{"$ref":"#/properties/a"},"c":{"$ref":)"
                   R"("#/properties/a"}}})") == 200 + 1,
          "a referenced schema outside the definitions is charged again, once per target");
    check(units_of(R"({"$defs":{"a":{"type":"string","pattern":"^[a-z]{1,100}$"}},)"
                   R"("type":"object","properties":{"b":{"$ref":"#/$defs/a"},)"
                   R"("c":{"$ref":"#/$defs/a"}}})") == 100 + 1,
          "a referenced definition is charged once");
    check(!admission_error(referencing(128)) &&
              admission_error(referencing(129)) == GrammarErrorKind::LimitExceeded,
          "limits the distinct targets of references");
    GrammarLimits counted;
    counted.max_counted_properties = 2;
    check(admission_error(R"({"type":"object","properties":{"a":{},"b":{},"c":{}},)"
                          R"("maxProperties":2})",
                          counted) == GrammarErrorKind::LimitExceeded,
          "limits the properties an object count applies to");
    limits.max_compile_units = 100;
    check(admission_error(R"({"type":"string","pattern":"^[a-z]{1,128}$"})", limits) ==
              GrammarErrorKind::LimitExceeded,
          "charges every unrolled copy of a class");
    limits.max_pattern_bytes = 4;
    check(admission_error(R"({"type":"string","pattern":"^abcdef$"})", limits) ==
              GrammarErrorKind::LimitExceeded,
          "limits the pattern length");
    limits                   = {};
    limits.max_pattern_atoms = 10;
    check(admission_error(R"({"type":"string","pattern":"^abcdefghijk$"})", limits) ==
              GrammarErrorKind::LimitExceeded,
          "limits the unrolled atoms of a pattern");
    check(admission_message(R"({"type":"object","properties":{"a":{"type":"string",)"
                            R"("pattern":"^.{200}$"}}})") ==
              "response_format.json_schema.schema at #/properties/a/pattern: pattern repetition "
              "bounds above 128 are not supported",
          "names the pattern of a repetition past the unroll limit");

    // One admission charges every schema of a grammar; a rejected schema charges nothing.
    SchemaAdmission shared(GrammarLimits{});
    (void)shared.admit(R"({"type":"string","maxLength":256})", "first");
    (void)shared.admit(R"({"type":"string","maxLength":256})", "second");
    check(shared.compile_units() == 512, "admissions of one grammar share the budget");
    bool over = false;
    try {
        (void)shared.admit(R"({"type":"string","maxLength":1})", "third");
    } catch (const GrammarError& error) { over = error.kind() == GrammarErrorKind::LimitExceeded; }
    check(over && shared.compile_units() == 512,
          "the shared budget rejects the schema that exceeds it, which charges nothing");
    check(CompileCost{.pattern_classes  = 2,
                      .pattern_literals = kPatternLiteralsPerUnit + 1,
                      .optional_pairs   = kOptionalPairsPerUnit + 1,
                      .length_bounds    = 3}
                  .units() == 2 + 2 + 2 + 3,
          "each cost term rounds up to whole units");
}

void regex_cost_contract() {
    const auto costs = [](const char* pattern, std::uint64_t classes, std::uint64_t literals) {
        const PatternCost cost = pattern_cost(pattern);
        return cost.classes == classes && cost.literals == literals;
    };
    check(costs("[a-z]{1,128}", 128, 0), "a class counts each unrolled copy");
    check(costs("(ab){1,64}", 0, 128), "a literal counts each unrolled copy");
    check(costs("^a{128}$|\\x41\\u0042\\n", 0, 131),
          "escapes that denote one character are literals");
    check(costs("([a-z]{1,16}){1,16}", 256, 0), "nested copies multiply");
    check(costs("((.{2}){3}){4}", 24, 0), "every nested level multiplies");
    check(costs("\\d{3,}", 4, 0), "an open repetition unrolls its minimum");
    check(costs("[a-z ]{ 1 , 32 }", 32, 0), "counts may have spaces around them");
    check(costs("\\w+\\s?.*[^a]", 4, 0), "stars and options are one copy");
    check(costs("\\p{L}{3}|[\\]]{2}", 5, 0),
          "property escapes and classes with escapes are classes");
    check(costs("(?:[a-z]{2})(?i)x{3}", 2, 3), "group prefixes are skipped");
    for (const char* invalid : {"(a", "a)", "[a-", "a\\", "*a", "a{2,1}", "a{x}", "x{"}) {
        std::optional<GrammarErrorKind> kind;
        try {
            (void)pattern_cost(invalid);
        } catch (const GrammarError& error) { kind = error.kind(); }
        check(kind == GrammarErrorKind::InvalidSchema,
              std::string("rejects the pattern ") + invalid);
    }
    for (const char* unrolled : {"[a-z]{200}", "a{129,}", "(ab){0,129}", "x{ 1 , 1000 }"}) {
        std::optional<GrammarErrorKind> kind;
        try {
            (void)pattern_cost(unrolled);
        } catch (const GrammarError& error) { kind = error.kind(); }
        check(kind == GrammarErrorKind::UnsupportedSchema,
              std::string("rejects the repetition past the unroll limit in ") + unrolled);
    }
}

void compile_contract() {
    GrammarService service = make_service();
    check(service.token_domain() == kDomain && service.mask_words() == 9,
          "one mask word per 32 tokens of the domain");

    SchemaAdmission admission(GrammarLimits{});
    const GrammarRecipe object    = GrammarRecipe::json_schema(admission.admit(
        R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},)"
        R"("required":["a","b"]})",
        "schema"));
    const CompiledGrammar compact = service.compile(object, {});
    check(accepts(compact, join({bytes(R"({"a": 1, "b": true})"), {kStop}})),
          "compact JSON uses ', ' and ': ' separators");
    check(accepts(compact, join({{kObjectKey}, bytes(R"(a": 1, "b": true})"), {kStop}})),
          "added tokens match as text");
    check(!accepts(compact, join({bytes(R"({"a":1,"b":true})"), {kStop}})),
          "compact JSON has a space after separators");
    check(!accepts(compact, join({bytes("{\n  \"a\": 1, \"b\": true}"), {kStop}})),
          "compact JSON has no other whitespace");
    check(!accepts(compact, join({bytes(R"({"b": true, "a": 1})"), {kStop}})),
          "properties keep their declared order");

    const GrammarRecipe thinking   = GrammarRecipe::sequence({reasoning_prefix(), object});
    const CompiledGrammar reasoned = service.compile(thinking, {});
    check(accepts(reasoned, join({bytes("plan it\n"),
                                  {kCloseThink},
                                  bytes("\n\n{\"a\": 2, \"b\": false}"),
                                  {kStop}})),
          "the reasoning part precedes the JSON value");
    check(!accepts(reasoned, join({bytes("x"),
                                   {kThink},
                                   bytes("y"),
                                   {kCloseThink},
                                   bytes("\n\n{\"a\": 2, \"b\": false}"),
                                   {kStop}})),
          "the reasoning part excludes <think>");
    check(!accepts(reasoned, join({bytes("{\"a\": 2, \"b\": false}"), {kStop}})),
          "the JSON value needs the reasoning part first");

    const GrammarRecipe either = GrammarRecipe::choice(
        {GrammarRecipe::json_schema(admission.admit(R"({"type":"integer"})", "a")),
         GrammarRecipe::json_schema(admission.admit(R"({"type":"null"})", "b"))});
    const CompiledGrammar choice = service.compile(either, {});
    check(accepts(choice, join({bytes("42"), {kStop}})) &&
              accepts(choice, join({bytes("null"), {kStop}})) &&
              !accepts(choice, join({bytes("true"), {kStop}})),
          "a choice accepts either alternative only");

    for (const char* schema : kAdmitted) {
        std::optional<std::string> error;
        try {
            SchemaAdmission each(GrammarLimits{});
            (void)service.compile(GrammarRecipe::json_schema(each.admit(schema, "schema")), {});
        } catch (const GrammarError& rejected) { error = rejected.what(); }
        check(!error, std::string("compiles the admitted schema ") + schema);
    }

    GrammarServiceStats stats   = service.stats();
    const CompiledGrammar again = service.compile(object, {});
    check(&again.impl() == &compact.impl(), "an identical recipe reuses the cached grammar");
    check(service.stats().hits == stats.hits + 1, "the reuse counts as a hit");

    // Constructs xgrammar can only approximate are rejected, and the rejection is cached.
    const GrammarRecipe overlapping = GrammarRecipe::json_schema(
        admission.admit(R"({"oneOf":[{"type":"integer"},{"type":"number"}]})", "overlap"));
    std::optional<GrammarErrorKind> kind;
    try {
        (void)service.compile(overlapping, {});
    } catch (const GrammarError& error) { kind = error.kind(); }
    check(kind == GrammarErrorKind::UnsupportedSchema, "an overlapping oneOf is rejected");
    stats = service.stats();
    try {
        (void)service.compile(overlapping, {});
    } catch (const GrammarError&) {}
    check(service.stats().hits == stats.hits + 1 && service.stats().rejected == stats.rejected,
          "a rejected recipe is not compiled again");

    kind.reset();
    try {
        (void)service.compile(GrammarRecipe::structural_tag(R"({"type":"nope"})"), {});
    } catch (const GrammarError& error) { kind = error.kind(); }
    check(kind == GrammarErrorKind::InvalidSchema, "an invalid structural tag is rejected");

    // Where xgrammar 0.2.7 would accept invalid JSON, the vendored changes refuse or reject.
    const auto compile_error = [&](const char* schema) {
        std::optional<GrammarErrorKind> error;
        try {
            SchemaAdmission each(GrammarLimits{});
            (void)service.compile(GrammarRecipe::json_schema(each.admit(schema, "schema")), {});
        } catch (const GrammarError& rejected) { error = rejected.kind(); }
        return error;
    };
    check(compile_error(R"({"type":"string","pattern":"^\"a\"$"})") ==
                  GrammarErrorKind::InvalidSchema &&
              compile_error(R"({"type":"string","pattern":"^a\\nb$"})") ==
                  GrammarErrorKind::InvalidSchema &&
              compile_error(R"({"type":"string","pattern":"^\\u0022$"})") ==
                  GrammarErrorKind::InvalidSchema,
          "a pattern that needs a character a JSON string must escape is rejected");
    check(compile_error(R"({"type":"string","pattern":"^é.*$"})") ==
              GrammarErrorKind::UnsupportedSchema,
          "a pattern outside printable ASCII is rejected before compiling");
    const CompiledGrammar accented =
        service.compile(GrammarRecipe::json_schema(admission.admit(
                            R"({"type":"string","pattern":"^\\u00e9[a-z]*$"})", "u")),
                        {});
    check(accepts(accented, join({bytes("\"\xc3\xa9te\""), {kStop}})) &&
              !accepts(accented, join({bytes("\"\xc3\xa9t\xc3\xa9\""), {kStop}})) &&
              !accepts(accented, join({bytes("\"\xc3\xa9\"\""), {kStop}})),
          "a \\uXXXX escape stands for a character outside ASCII");
    const CompiledGrammar bounded = service.compile(
        GrammarRecipe::json_schema(admission.admit(R"({"type":"string","maxLength":8})", "b")), {});
    check(accepts(bounded, join({bytes("\"a b\""), {kStop}})) &&
              !accepts(bounded, join({bytes("\"a\tb\""), {kStop}})) &&
              !accepts(bounded, join({bytes("\"a\x01z\""), {kStop}})),
          "a length-bounded string refuses raw control characters");
    const CompiledGrammar open = service.compile(
        GrammarRecipe::json_schema(admission.admit(
            R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],)"
            R"("additionalProperties":{"type":"string"}})",
            "o")),
        {});
    check(accepts(open, join({bytes(R"({"a": 1, "b": "x"})"), {kStop}})) &&
              !accepts(open, join({bytes(R"({"a": 1, "a": "x"})"), {kStop}})) &&
              !accepts(open, join({bytes(R"({"a": 1, "\u0061": "x"})"), {kStop}})),
          "an additional key cannot repeat a declared property, escaped or not");

    const CompiledGrammar email = service.compile(
        GrammarRecipe::json_schema(admission.admit(R"({"type":"string","format":"email"})", "e")),
        {});
    check(accepts(email, join({bytes("\"a.b@c.de\""), {kStop}})) &&
              !accepts(email, join({bytes("\"\\\"x\\\"@c.de\""), {kStop}})) &&
              !accepts(email, join({bytes("\"\\a@c.de\""), {kStop}})),
          "an email has a dot-atom local part, never a quoted one");
    const CompiledGrammar quoted_name = service.compile(
        GrammarRecipe::json_schema(admission.admit(
            R"({"type":"object","properties":{"a\"b":{"type":"integer"}},"required":["a\"b"],)"
            R"("additionalProperties":false})",
            "q")),
        {});
    check(accepts(quoted_name, join({bytes(R"({"a\"b": 1})"), {kStop}})) &&
              !accepts(quoted_name, join({bytes(R"({"a"b": 1})"), {kStop}})),
          "a declared name is escaped in a closed object");

    // The deepest schema the limits admit: every reference target in one chain, each as deep as
    // allowed. xgrammar recurses through all of it on the compile thread.
    SchemaAdmission deepest_admission(GrammarLimits{});
    const std::string deepest = deepest_reference_chain(GrammarLimits{});
    std::optional<std::string> deep_error;
    try {
        (void)service.compile(
            GrammarRecipe::json_schema(deepest_admission.admit(deepest, "deepest")), {});
    } catch (const GrammarError& error) { deep_error = error.what(); }
    check(!deep_error,
          "the deepest admissible reference chain compiles: " + deep_error.value_or(std::string()));

    // A grammar that allows no first token is rejected when compiled, admission aside.
    kind.reset();
    try {
        (void)service.compile(GrammarRecipe::json_schema(R"({"$ref":"#"})"), {});
    } catch (const GrammarError& error) { kind = error.kind(); }
    check(kind == GrammarErrorKind::InvalidSchema, "a grammar that allows no output is rejected");

    bool no_stop = false;
    try {
        GrammarService unusable(toy_vocabulary(), {}, GrammarServiceOptions{});
    } catch (const std::invalid_argument&) { no_stop = true; }
    check(no_stop, "a vocabulary without stop tokens is rejected");
}

void flight_contract() {
    GrammarService service = make_service();
    SchemaAdmission admission(GrammarLimits{});
    const GrammarRecipe recipe = GrammarRecipe::json_schema(
        admission.admit(R"({"type":"array","items":{"type":"string","maxLength":24}})", "s"));

    constexpr int kCallers = 6;
    std::vector<std::thread> callers;
    std::atomic<int> compiled{0};
    for (int i = 0; i < kCallers; ++i) {
        callers.emplace_back([&] {
            if (service.compile(recipe, {})) { ++compiled; }
        });
    }
    for (std::thread& caller : callers) { caller.join(); }
    const GrammarServiceStats stats = service.stats();
    check(compiled == kCallers, "every caller gets the grammar");
    check(stats.misses == 1 && stats.hits + stats.singleflight_waits == kCallers - 1,
          "identical concurrent recipes share one compilation");

    // A caller that gives up before its compilation is queued leaves nothing behind.
    const GrammarRecipe other =
        GrammarRecipe::json_schema(admission.admit(R"({"type":"boolean"})", "b"));
    bool stopped = false;
    try {
        (void)service.compile(other, [] { throw std::runtime_error("deadline"); });
    } catch (const std::runtime_error&) { stopped = true; }
    const GrammarServiceStats after = service.stats();
    check(stopped && after.inflight == 0 && after.entries == stats.entries,
          "an abandoned compilation leaves no flight or entry");
    check(static_cast<bool>(service.compile(other, {})), "the recipe compiles on a later request");

    // Its reason is its own: the callers waiting for the same grammar take the compilation over.
    GrammarService handover = make_service();
    const GrammarRecipe shared =
        GrammarRecipe::json_schema(admission.admit(R"({"type":"null"})", "n"));
    std::thread waiter;
    std::optional<std::string> waiter_error;
    bool waiter_compiled  = false;
    bool producer_stopped = false;
    try {
        (void)handover.compile(shared, [&] {
            if (!waiter.joinable()) {
                waiter = std::thread([&] {
                    try {
                        waiter_compiled = static_cast<bool>(handover.compile(shared, {}));
                    } catch (const std::exception& error) { waiter_error = error.what(); }
                });
            }
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (handover.stats().singleflight_waits == 0 &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            throw std::runtime_error("the first caller gave up");
        });
    } catch (const std::runtime_error&) { producer_stopped = true; }
    if (waiter.joinable()) { waiter.join(); }
    const GrammarServiceStats handed = handover.stats();
    check(producer_stopped && waiter_compiled && !waiter_error && handed.misses == 2 &&
              handed.singleflight_waits == 1 && handed.entries == 1,
          "a waiter takes over the compilation its first caller gave up before queueing");

    // The cache stays under its byte bound by evicting the least recently used grammar.
    const GrammarRecipe first =
        GrammarRecipe::json_schema(admission.admit(R"({"type":"integer"})", "1"));
    const GrammarRecipe second =
        GrammarRecipe::json_schema(admission.admit(R"({"type":"string"})", "2"));
    GrammarService probe = make_service();
    (void)probe.compile(first, {});
    (void)probe.compile(second, {});
    GrammarServiceOptions small;
    small.cache_bytes      = probe.stats().cached_bytes - 1;
    GrammarService bounded = make_service(small);
    (void)bounded.compile(first, {});
    (void)bounded.compile(second, {});
    const GrammarServiceStats evicted = bounded.stats();
    check(evicted.evictions == 1 && evicted.entries == 1 &&
              evicted.cached_bytes <= small.cache_bytes,
          "the cache evicts the least recently used grammar to stay under its bound");
    (void)bounded.compile(first, {});
    check(bounded.stats().misses == 3, "an evicted grammar is compiled again");
}

std::vector<std::int32_t> mask_of(TokenMatcher& matcher) {
    std::vector<std::int32_t> mask(static_cast<std::size_t>(matcher.mask_words()));
    matcher.fill_mask(mask.data());
    return mask;
}

bool allows(const std::vector<std::int32_t>& mask, TokenId token) {
    return ((static_cast<std::uint32_t>(mask[token >> 5]) >> (token & 31)) & 1u) != 0u;
}

void matcher_contract() {
    GrammarService service = make_service();
    SchemaAdmission admission(GrammarLimits{});
    const CompiledGrammar grammar = service.compile(
        GrammarRecipe::json_schema(admission.admit(
            R"({"type":"object","properties":{"k":{"type":"integer"}},"required":["k"]})", "s")),
        {});
    TokenMatcher matcher(grammar);
    const std::vector<std::int32_t> initial = mask_of(matcher);
    check(allows(initial, '{') && allows(initial, kObjectKey) && !allows(initial, 'x') &&
              !allows(initial, kStop),
          "the first mask allows only the start of the object");

    // Every walked draft row equals the mask a manual accept loop computes.
    const std::vector<TokenId> drafts = join({{kObjectKey}, bytes("k\": 7}")});
    const std::size_t stride          = static_cast<std::size_t>(matcher.mask_words()) + 3;
    std::vector<std::int32_t> rows((drafts.size() + 1) * stride, -1);
    const std::uint32_t walked = matcher.fill_draft_masks(drafts, rows.data(), stride);
    check(walked == drafts.size(), "a valid chain is walked to its end");
    TokenMatcher manual(grammar);
    for (std::size_t i = 0; i <= drafts.size(); ++i) {
        const std::vector<std::int32_t> expected = mask_of(manual);
        check(std::equal(expected.begin(), expected.end(), rows.begin() + i * stride),
              "draft row " + std::to_string(i) + " equals the manual mask");
        if (i < drafts.size()) { check(manual.accept(drafts[i]), "the manual loop accepts"); }
    }
    check(mask_of(matcher) == initial, "the walk leaves the committed state unchanged");

    // The walk ends at the first draft its row excludes, and before a stop token.
    const std::vector<TokenId> invalid = join({{kObjectKey}, bytes("x")});
    std::vector<std::int32_t> sentinel((invalid.size() + 1) * stride, -1);
    check(matcher.fill_draft_masks(invalid, sentinel.data(), stride) == 1,
          "the walk ends at an excluded draft");
    check(sentinel[2 * stride] == -1, "rows past the walk are untouched");
    for (const TokenId token : join({{kObjectKey}, bytes("k\": 7}")})) {
        check(matcher.accept(token), "the committed output advances");
    }
    std::vector<std::int32_t> stop_rows(2 * stride, -1);
    check(
        matcher.fill_draft_masks(std::vector<TokenId>{kStop}, stop_rows.data(), stride) == 0 &&
            allows(std::vector<std::int32_t>(stop_rows.begin(), stop_rows.begin() + stride), kStop),
        "a stop token is allowed but never walked");

    check(!matcher.accept('x'), "a token the grammar excludes is refused");
    check(matcher.accept(kStop) && matcher.terminated(), "the stop token ends the output");
    const std::vector<std::int32_t> ended = mask_of(matcher);
    check(allows(ended, kStop) && allows(ended, kEndOfText) && !allows(ended, '}'),
          "an ended output allows only the stop tokens");
    matcher.rollback(1);
    check(!matcher.terminated() && allows(mask_of(matcher), kStop), "rollback reopens the output");
}

// Nothing xgrammar reports may reach the process output: its messages can quote request schemas.
void logging_contract() {
    std::fflush(stderr);
    const int saved = ::dup(STDERR_FILENO);
    std::FILE* sink = std::tmpfile();
    ::dup2(::fileno(sink), STDERR_FILENO);

    GrammarService service = make_service();
    SchemaAdmission admission(GrammarLimits{});
    try {
        (void)service.compile(GrammarRecipe::json_schema(admission.admit(
                                  R"({"oneOf":[{"type":"integer"},{"type":"number"}]})", "w")),
                              {});
    } catch (const GrammarError&) {}
    try {
        (void)xgrammar::Grammar::FromRegex("(?=secret)abc");
    } catch (const std::exception&) {}

    std::fflush(stderr);
    ::dup2(saved, STDERR_FILENO);
    ::close(saved);
    const long written = std::ftell(sink);
    std::fclose(sink);
    check(written == 0, "xgrammar writes nothing to stderr");
}

} // namespace

int main() {
    admission_contract();
    regex_cost_contract();
    compile_contract();
    flight_contract();
    matcher_contract();
    logging_contract();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " grammar service\n";
    return failures == 0 ? 0 : 1;
}
