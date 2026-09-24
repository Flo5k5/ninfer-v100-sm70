#include "grammar/schema_admission.h"

#include "grammar/grammar_error.h"
#include "grammar/regex_cost.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::grammar {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::array<std::string_view, 13> kAnnotations = {
    "title",         "description",     "default",         "examples", "deprecated",
    "readOnly",      "writeOnly",       "$comment",        "$schema",  "nullable",
    "discriminator", "contentEncoding", "contentMediaType"};
constexpr std::array<std::string_view, 7> kTypes = {"object", "array",   "string", "integer",
                                                    "number", "boolean", "null"};
constexpr std::array<std::string_view, 8> kObjectKeywords = {
    "properties",    "required",      "additionalProperties", "patternProperties",
    "propertyNames", "minProperties", "maxProperties",        "unevaluatedProperties"};
constexpr std::array<std::string_view, 5> kArrayKeywords  = {"items", "prefixItems", "minItems",
                                                             "maxItems", "unevaluatedItems"};
constexpr std::array<std::string_view, 3> kStringKeywords = {"minLength", "maxLength", "pattern"};
constexpr std::array<std::string_view, 5> kNumberKeywords = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"};
constexpr std::array<std::string_view, 10> kFormats = {
    "date", "date-time", "duration", "email", "hostname", "ipv4", "ipv6", "time", "uri", "uuid"};
constexpr std::array<std::string_view, 3> kCombinators = {"anyOf", "oneOf", "allOf"};

template <std::size_t N>
bool contains(const std::array<std::string_view, N>& set, std::string_view value) {
    return std::find(set.begin(), set.end(), value) != set.end();
}

bool is_annotation(std::string_view key) {
    return contains(kAnnotations, key) || key.starts_with("x-");
}

std::string escape_pointer_token(std::string_view token) {
    std::string out;
    out.reserve(token.size());
    for (const char c : token) {
        if (c == '~') {
            out += "~0";
        } else if (c == '/') {
            out += "~1";
        } else {
            out += c;
        }
    }
    return out;
}

std::uint32_t saturating_add(std::uint32_t a, std::uint64_t b) {
    const std::uint64_t sum =
        static_cast<std::uint64_t>(a) + std::min<std::uint64_t>(b, UINT32_MAX);
    return sum > UINT32_MAX ? UINT32_MAX : static_cast<std::uint32_t>(sum);
}

// Far above any limit; keeps sums of costs from overflowing.
constexpr std::uint64_t kSaturatedCost = std::uint64_t{1} << 62;

std::uint64_t saturating_sum(std::uint64_t a, std::uint64_t b) {
    return std::min(std::min(a, kSaturatedCost) + std::min(b, kSaturatedCost), kSaturatedCost);
}

void add_cost(CompileCost& into, const CompileCost& cost) {
    into.pattern_classes  = saturating_sum(into.pattern_classes, cost.pattern_classes);
    into.pattern_literals = saturating_sum(into.pattern_literals, cost.pattern_literals);
    into.optional_pairs   = saturating_sum(into.optional_pairs, cost.optional_pairs);
    into.length_bounds    = saturating_sum(into.length_bounds, cost.length_bounds);
}

std::uint64_t divide_rounding_up(std::uint64_t value, std::uint64_t divisor) {
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

// A property name that xgrammar's rule for undeclared keys would spell raw inside a JSON string.
bool needs_escaping(std::string_view name) {
    return std::any_of(name.begin(), name.end(), [](char c) {
        return c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20;
    });
}

// Nesting of arrays and objects in JSON text, outside strings.
std::uint32_t json_depth(std::string_view text) {
    std::uint32_t depth   = 0;
    std::uint32_t deepest = 0;
    bool in_string        = false;
    bool escaped          = false;
    for (const char c : text) {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else if (c == '"') {
            in_string = true;
        } else if (c == '[' || c == '{') {
            deepest = std::max(deepest, ++depth);
        } else if ((c == ']' || c == '}') && depth > 0) {
            --depth;
        }
    }
    return deepest;
}

bool matches_type(const Json& value, std::string_view type) {
    if (type == "null") { return value.is_null(); }
    if (type == "boolean") { return value.is_boolean(); }
    if (type == "string") { return value.is_string(); }
    if (type == "object") { return value.is_object(); }
    if (type == "array") { return value.is_array(); }
    if (type == "number") { return value.is_number(); }
    if (type == "integer") {
        if (value.is_number_integer()) { return true; }
        if (!value.is_number_float()) { return false; }
        const double number = value.get<double>();
        return std::isfinite(number) && number == std::floor(number);
    }
    return false;
}

// Where a schema sits, as xgrammar resolves a local reference: object keys from the root. A schema
// under an array, or under a key that is empty or holds '/', cannot be referenced.
struct SchemaPath {
    std::string keys;
    bool addressable = true;

    [[nodiscard]] SchemaPath child(std::string_view key) const {
        return SchemaPath{keys + "/" + std::string(key),
                          addressable && !key.empty() && key.find('/') == std::string_view::npos};
    }

    [[nodiscard]] static SchemaPath element() {
        return SchemaPath{.keys = {}, .addressable = false};
    }
};

// A schema as xgrammar compiles it, for the checks over the whole document.
struct Node {
    // Accepts a finite value once `needed` of its `children` do: every required property or item,
    // one branch of a choice, the target of a reference.
    std::vector<std::uint32_t> children;
    std::size_t needed = 0;
    // What compiling the schema and the schemas nested in it costs, definitions and reference
    // targets aside. A reference to a schema other than a definition compiles it once more.
    CompileCost cost;
    std::string pointer;
    bool definition = false;
};

constexpr std::uint32_t kRoot = 0;

class SchemaWalker {
public:
    SchemaWalker(const GrammarLimits& limits, std::string_view location, std::uint32_t& nodes,
                 CompileCost& cost)
        : limits_(limits), location_(location), nodes_(nodes), cost_(cost) {}

    // Walks one schema and returns its node. `default_type` is the type xgrammar assumes when the
    // schema names none (string for propertyNames, none elsewhere).
    std::uint32_t schema(const Json& value, const std::string& pointer, const SchemaPath& path,
                         std::uint32_t depth, std::string_view default_type = {}) {
        if (depth > limits_.max_schema_depth) {
            fail(GrammarErrorKind::LimitExceeded, pointer,
                 "schema nesting exceeds " + std::to_string(limits_.max_schema_depth) + " levels");
        }
        count_nodes(pointer, 1);
        const std::uint32_t id = add_node(pointer);
        if (value.is_boolean()) {
            if (!value.get<bool>()) {
                fail(GrammarErrorKind::InvalidSchema, pointer, "the schema false accepts no value");
            }
            admit_path(path, id);
            return id;
        }
        if (!value.is_object()) {
            fail(GrammarErrorKind::InvalidSchema, pointer, "a schema must be an object or true");
        }
        // Referenced elsewhere, a schema is read without the type its position implies.
        if (default_type.empty() || value.contains("type")) { admit_path(path, id); }

        std::vector<std::string_view> forms;
        std::vector<std::string_view> keywords;
        for (const auto& [key, child] : value.items()) {
            if (is_annotation(key)) { continue; }
            if (key == "$defs" || key == "definitions") {
                definitions(child, pointer + "/" + escape_pointer_token(key), path.child(key),
                            depth);
            } else if (key == "$ref" || key == "const" || key == "enum" ||
                       contains(kCombinators, key)) {
                forms.push_back(key);
            } else {
                keywords.push_back(key);
            }
        }
        if (forms.size() > 1) {
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "'" + std::string(forms[0]) + "' and '" + std::string(forms[1]) +
                     "' cannot be combined in one schema");
        }
        if (forms.empty()) {
            typed(value, keywords, pointer, path, depth, default_type, id);
            return id;
        }

        const std::string_view form = forms.front();
        const bool literal          = form == "const" || form == "enum";
        for (const std::string_view key : keywords) {
            if (literal && key == "type") { continue; }
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "keyword '" + std::string(key) + "' cannot be combined with '" +
                     std::string(form) + "'");
        }
        const Json& body = value.at(std::string(form));
        if (form == "$ref") {
            reference(body, pointer, id);
        } else if (literal) {
            if (form == "enum" && (!body.is_array() || body.empty())) {
                fail(GrammarErrorKind::InvalidSchema, pointer, "'enum' must be a non-empty array");
            }
            count_nodes(pointer, form == "enum" ? body.size() : 1);
            if (value.contains("type")) {
                literal_types(value.at("type"), form == "enum" ? body : Json::array({body}),
                              pointer);
            }
        } else {
            if (!body.is_array() || body.empty()) {
                fail(GrammarErrorKind::InvalidSchema, pointer,
                     "'" + std::string(form) + "' must be a non-empty array of schemas");
            }
            std::vector<std::uint32_t> branches;
            for (std::size_t i = 0; i < body.size(); ++i) {
                branches.push_back(
                    nested(id, body[i], pointer + "/" + std::string(form) + "/" + std::to_string(i),
                           SchemaPath::element(), depth + 1));
            }
            const std::size_t needed = form == "allOf" ? branches.size() : 1;
            require(id, std::move(branches), needed);
        }
        return id;
    }

    // Resolves every '$ref' as xgrammar does and charges what it compiles for them. xgrammar
    // compiles each spelling of a path separately, so a target has one spelling: "#" or "#/"
    // followed by non-empty keys, which name object members from the root.
    void check_references() {
        std::unordered_set<std::uint32_t> targets;
        for (const Reference& reference : references_) {
            const std::uint32_t target = resolve(reference);
            require(reference.node, {target}, 1);
            if (target == kRoot || !targets.insert(target).second) { continue; }
            if (targets.size() > limits_.max_ref_targets) {
                fail(GrammarErrorKind::LimitExceeded, reference.pointer,
                     "the schema's references name more than " +
                         std::to_string(limits_.max_ref_targets) + " distinct schemas");
            }
            // A definition is compiled only for its references; any other target is compiled
            // where it stands and once more for its reference.
            if (!graph_[target].definition) { charge(reference.pointer, graph_[target].cost); }
        }
    }

    // Rejects a schema that no finite value satisfies. xgrammar would compile it into a grammar
    // that allows nothing, or one whose output can never end.
    void check_finite() const {
        std::vector<std::size_t> missing(graph_.size());
        std::vector<std::vector<std::uint32_t>> parents(graph_.size());
        std::vector<std::uint32_t> ready;
        for (std::uint32_t id = 0; id < graph_.size(); ++id) {
            missing[id] = graph_[id].needed;
            for (const std::uint32_t child : graph_[id].children) { parents[child].push_back(id); }
            if (missing[id] == 0) { ready.push_back(id); }
        }
        std::vector<bool> finite(graph_.size(), false);
        while (!ready.empty()) {
            const std::uint32_t id = ready.back();
            ready.pop_back();
            finite[id] = true;
            for (const std::uint32_t parent : parents[id]) {
                if (missing[parent] > 0 && --missing[parent] == 0) { ready.push_back(parent); }
            }
        }
        for (std::uint32_t id = 0; id < graph_.size(); ++id) {
            if (!finite[id]) {
                fail(GrammarErrorKind::InvalidSchema, graph_[id].pointer,
                     "no finite value satisfies this schema: what it requires is unsatisfiable or "
                     "only leads back to itself");
            }
        }
    }

private:
    struct Reference {
        std::string uri;
        std::string pointer;
        std::uint32_t node = 0;
    };

    [[noreturn]] void fail(GrammarErrorKind kind, const std::string& pointer,
                           const std::string& what) const {
        throw GrammarError(kind, std::string(location_) +
                                     (pointer.empty() ? "" : " at #" + pointer) + ": " + what);
    }

    std::uint32_t add_node(const std::string& pointer) {
        graph_.push_back(Node{.children = {}, .needed = 0, .cost = {}, .pointer = pointer});
        return static_cast<std::uint32_t>(graph_.size() - 1);
    }

    void require(std::uint32_t id, std::vector<std::uint32_t> children, std::size_t needed) {
        graph_[id].children = std::move(children);
        graph_[id].needed   = needed;
    }

    // A node satisfied when every one of `parts` is.
    std::uint32_t all_of(const std::string& pointer, std::vector<std::uint32_t> parts) {
        const std::uint32_t id   = add_node(pointer);
        const std::size_t needed = parts.size();
        require(id, std::move(parts), needed);
        return id;
    }

    // Walks a schema nested in `parent`, which compiles it too.
    std::uint32_t nested(std::uint32_t parent, const Json& value, const std::string& pointer,
                         const SchemaPath& path, std::uint32_t depth,
                         std::string_view default_type = {}) {
        const std::uint32_t id = schema(value, pointer, path, depth, default_type);
        add_cost(graph_[parent].cost, graph_[id].cost);
        return id;
    }

    void admit_path(const SchemaPath& path, std::uint32_t id) {
        if (path.addressable) { admitted_paths_.emplace(path.keys, id); }
    }

    void count_nodes(const std::string& pointer, std::size_t count) {
        nodes_ = saturating_add(nodes_, count);
        if (nodes_ > limits_.max_schema_nodes) {
            fail(GrammarErrorKind::LimitExceeded, pointer,
                 "the grammar's schemas hold more than " +
                     std::to_string(limits_.max_schema_nodes) + " schema objects and enum values");
        }
    }

    // Adds to the grammar's cost, which only grows: exceeding the limit here already fails.
    void charge(const std::string& pointer, const CompileCost& cost) {
        add_cost(cost_, cost);
        if (cost_.units() > limits_.max_compile_units) {
            fail(GrammarErrorKind::LimitExceeded, pointer,
                 "the grammar would take too long to compile: string length bounds, regex "
                 "patterns and optional properties cost " +
                     std::to_string(cost_.units()) + " units, above the limit of " +
                     std::to_string(limits_.max_compile_units));
        }
    }

    // Charges a cost of schema `id` itself, which every compilation of it pays.
    void charge_schema(std::uint32_t id, const CompileCost& cost) {
        add_cost(graph_[id].cost, cost);
        charge(graph_[id].pointer, cost);
    }

    void literal_types(const Json& type, const Json& values, const std::string& pointer) const {
        std::vector<std::string> names;
        for (const Json& name : type.is_array() ? type : Json::array({type})) {
            if (!name.is_string() || !contains(kTypes, name.get<std::string>())) {
                fail(GrammarErrorKind::InvalidSchema, pointer,
                     "'type' must name JSON types: object, array, string, integer, number, "
                     "boolean, null");
            }
            names.push_back(name.get<std::string>());
        }
        if (names.empty()) {
            fail(GrammarErrorKind::InvalidSchema, pointer, "'type' must not be empty");
        }
        // xgrammar ignores a type next to const or enum: every value must already satisfy it.
        for (const Json& value : values) {
            const bool matches =
                std::any_of(names.begin(), names.end(),
                            [&](const std::string& name) { return matches_type(value, name); });
            if (!matches) {
                fail(GrammarErrorKind::UnsupportedSchema, pointer,
                     "a 'const' or 'enum' value does not have the schema's 'type'");
            }
        }
    }

    // A pattern of schema `owner`: its value or one of its property patterns.
    void pattern(const Json& value, const std::string& pointer, std::uint32_t owner) {
        if (!value.is_string()) {
            fail(GrammarErrorKind::InvalidSchema, pointer, "a pattern must be a string");
        }
        const auto& text = value.get_ref<const std::string&>();
        if (text.size() > limits_.max_pattern_bytes) {
            fail(GrammarErrorKind::LimitExceeded, pointer,
                 "a pattern is longer than " + std::to_string(limits_.max_pattern_bytes) +
                     " bytes");
        }
        // Past this range xgrammar falls back to a grammar without JSON string escaping.
        if (!std::all_of(text.begin(), text.end(), [](char c) { return c >= 0x20 && c <= 0x7e; })) {
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "a pattern must be printable ASCII: write other characters as \\uXXXX escapes");
        }
        PatternCost atoms;
        try {
            atoms = pattern_cost(text);
        } catch (const GrammarError& error) { fail(error.kind(), pointer, error.what()); }
        if (atoms.classes + atoms.literals > limits_.max_pattern_atoms) {
            fail(GrammarErrorKind::LimitExceeded, pointer,
                 "a pattern unrolls to more than " + std::to_string(limits_.max_pattern_atoms) +
                     " characters and classes");
        }
        add_cost(graph_[owner].cost,
                 CompileCost{.pattern_classes = atoms.classes, .pattern_literals = atoms.literals});
        charge(pointer,
               CompileCost{.pattern_classes = atoms.classes, .pattern_literals = atoms.literals});
    }

    void definitions(const Json& value, const std::string& pointer, const SchemaPath& path,
                     std::uint32_t depth) {
        if (!value.is_object()) {
            fail(GrammarErrorKind::InvalidSchema, pointer, "definitions must be an object");
        }
        for (const auto& [name, child] : value.items()) {
            const std::uint32_t id = schema(child, pointer + "/" + escape_pointer_token(name),
                                            path.child(name), depth + 1);
            graph_[id].definition  = true;
        }
    }

    void reference(const Json& value, const std::string& pointer, std::uint32_t id) {
        if (!value.is_string()) {
            fail(GrammarErrorKind::InvalidSchema, pointer, "'$ref' must be a string");
        }
        const auto& uri = value.get_ref<const std::string&>();
        if (uri != "#" && !uri.starts_with("#/")) {
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "only local references ('#' or '#/...') are supported");
        }
        // Resolved once the whole document is walked.
        references_.push_back(Reference{.uri = uri, .pointer = pointer, .node = id});
        graph_[id].needed = 1;
    }

    std::uint32_t resolve(const Reference& reference) const {
        const std::string& uri = reference.uri;
        if (uri == "#") { return kRoot; }
        std::string keys;
        for (std::size_t begin = 2;;) {
            const std::size_t end = std::min(uri.find('/', begin), uri.size());
            const std::string_view key(uri.data() + begin, end - begin);
            if (key.empty()) {
                fail(GrammarErrorKind::UnsupportedSchema, reference.pointer,
                     "a '$ref' path cannot have empty keys");
            }
            if (key.find('~') != std::string_view::npos ||
                key.find('%') != std::string_view::npos) {
                fail(GrammarErrorKind::UnsupportedSchema, reference.pointer,
                     "a '$ref' path cannot use '~' escapes or percent-encoding");
            }
            keys += '/';
            keys += key;
            if (end == uri.size()) { break; }
            begin = end + 1;
        }
        const auto found = admitted_paths_.find(keys);
        if (found == admitted_paths_.end()) {
            fail(GrammarErrorKind::UnsupportedSchema, reference.pointer,
                 "'$ref' must name a schema of this document, such as one in '$defs'");
        }
        return found->second;
    }

    static std::uint64_t non_negative_integer(const Json& value) {
        if (value.is_number_unsigned()) { return value.get<std::uint64_t>(); }
        if (value.is_number_integer() && value.get<std::int64_t>() >= 0) {
            return static_cast<std::uint64_t>(value.get<std::int64_t>());
        }
        return UINT64_MAX;
    }

    // Reads the optional `lower` and `upper` bounds of one schema: non-negative integers, the
    // lower one not above the upper one. An absent bound reads as nullopt.
    std::pair<std::optional<std::uint64_t>, std::optional<std::uint64_t>>
    bounds(const Json& value, const std::string& pointer, const char* lower,
           const char* upper) const {
        std::pair<std::optional<std::uint64_t>, std::optional<std::uint64_t>> read;
        for (const char* key : {lower, upper}) {
            if (!value.contains(key)) { continue; }
            const std::uint64_t n = non_negative_integer(value.at(key));
            if (n == UINT64_MAX) {
                fail(GrammarErrorKind::InvalidSchema, pointer,
                     "'" + std::string(key) + "' must be a non-negative integer");
            }
            (key == lower ? read.first : read.second) = n;
        }
        if (read.first && read.second && *read.first > *read.second) {
            fail(GrammarErrorKind::InvalidSchema, pointer,
                 "'" + std::string(lower) + "' is above '" + std::string(upper) + "'");
        }
        return read;
    }

    void typed(const Json& value, const std::vector<std::string_view>& keywords,
               const std::string& pointer, const SchemaPath& path, std::uint32_t depth,
               std::string_view default_type, std::uint32_t id) {
        std::vector<std::string_view> types;
        if (!value.contains("type") && !default_type.empty()) {
            types.push_back(default_type);
        } else if (value.contains("type")) {
            const Json& type = value.at("type");
            const auto add   = [&](const Json& name) {
                if (!name.is_string() || !contains(kTypes, name.get<std::string>())) {
                    fail(GrammarErrorKind::InvalidSchema, pointer,
                         "'type' must name JSON types: object, array, string, integer, number, "
                         "boolean, null");
                }
                types.push_back(*std::find(kTypes.begin(), kTypes.end(), name.get<std::string>()));
            };
            if (type.is_array()) {
                if (type.empty()) {
                    fail(GrammarErrorKind::InvalidSchema, pointer, "'type' must not be empty");
                }
                for (const Json& name : type) { add(name); }
            } else {
                add(type);
            }
        }
        const auto has_type = [&](std::string_view name) {
            return std::find(types.begin(), types.end(), name) != types.end();
        };
        // Without a type, xgrammar reads an object only from properties, additionalProperties or
        // unevaluatedProperties, an array only from items, prefixItems or unevaluatedItems, and
        // any other schema as any value, which would drop every other keyword.
        const auto has_any = [&](std::initializer_list<const char*> keys) {
            return std::any_of(keys.begin(), keys.end(),
                               [&](const char* key) { return value.contains(key); });
        };
        const bool untyped = types.empty();
        const bool object_shape =
            untyped && has_any({"properties", "additionalProperties", "unevaluatedProperties"});
        const bool array_shape =
            untyped && !object_shape && has_any({"items", "prefixItems", "unevaluatedItems"});
        const bool object = has_type("object") || object_shape;
        const bool array  = has_type("array") || array_shape;
        const bool string = has_type("string");
        const bool number = has_type("integer") || has_type("number");

        for (const std::string_view key : keywords) {
            if (key == "type") { continue; }
            if (key == "format") {
                // format is an annotation for every type but string.
                continue;
            }
            const bool applies = (object && contains(kObjectKeywords, key)) ||
                                 (array && contains(kArrayKeywords, key)) ||
                                 (string && contains(kStringKeywords, key)) ||
                                 (number && contains(kNumberKeywords, key));
            if (applies) { continue; }
            const bool known = contains(kObjectKeywords, key) || contains(kArrayKeywords, key) ||
                               contains(kStringKeywords, key) || contains(kNumberKeywords, key);
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 known ? "keyword '" + std::string(key) + "' needs a 'type' it applies to"
                       : "keyword '" + std::string(key) + "' is not supported");
        }

        if (string) { string_rule(value, pointer, id); }
        const std::unordered_set<std::string> required =
            object ? object_rule(value, pointer, id) : std::unordered_set<std::string>{};
        const std::uint64_t min_items =
            array ? bounds(value, pointer, "minItems", "maxItems").first.value_or(0) : 0;

        std::vector<std::uint32_t> declared;
        std::vector<std::uint32_t> required_members;
        for (const char* key : {"properties", "patternProperties"}) {
            if (!object || !value.contains(key)) { continue; }
            const bool properties = std::string_view(key) == "properties";
            const std::string at  = pointer + "/" + key;
            for (const auto& [name, child] : value.at(key).items()) {
                if (!properties) { pattern(Json(name), at, id); }
                const std::uint32_t member =
                    nested(id, child, at + "/" + escape_pointer_token(name),
                           path.child(key).child(name), depth + 1);
                if (properties) {
                    declared.push_back(member);
                    if (required.contains(name)) { required_members.push_back(member); }
                }
            }
        }
        // xgrammar reads unevaluatedItems only without items, and in strict mode allows no item
        // past prefixItems when neither is present or the one it reads is false.
        std::optional<std::uint32_t> extra_items;
        for (const char* key : {"additionalProperties", "unevaluatedProperties", "propertyNames",
                                "items", "unevaluatedItems"}) {
            const bool applies = (object && contains(kObjectKeywords, key)) ||
                                 (array && contains(kArrayKeywords, key));
            if (!applies || !value.contains(key)) { continue; }
            const Json& child = value.at(key);
            if (child.is_boolean() && !child.get<bool>()) { continue; }
            const std::string_view name = key;
            const std::uint32_t member  = nested(id, child, pointer + "/" + key, path.child(key),
                                                 depth + 1, name == "propertyNames" ? "string" : "");
            if (name == "items" || (name == "unevaluatedItems" && !value.contains("items"))) {
                extra_items = member;
            }
        }
        std::vector<std::uint32_t> prefix;
        if (array && value.contains("prefixItems")) {
            const Json& items = value.at("prefixItems");
            if (!items.is_array()) {
                fail(GrammarErrorKind::InvalidSchema, pointer,
                     "'prefixItems' must be an array of schemas");
            }
            for (std::size_t i = 0; i < items.size(); ++i) {
                prefix.push_back(nested(id, items[i], pointer + "/prefixItems/" + std::to_string(i),
                                        SchemaPath::element(), depth + 1));
            }
        }

        // A string, number, boolean or null type, or no type at all, always has a finite value;
        // an object needs its required properties and enough of the others for minProperties; an
        // array needs its first minItems items, positional ones first.
        const bool any_value = untyped && !object && !array;
        if (any_value || string || number || has_type("boolean") || has_type("null")) { return; }
        std::vector<std::uint32_t> branches;
        if (object) {
            const std::uint64_t min_properties =
                value.contains("minProperties") ? non_negative_integer(value.at("minProperties"))
                                                : 0;
            std::vector<std::uint32_t> parts = required_members;
            if (min_properties > required_members.size()) {
                const std::uint32_t enough = add_node(pointer);
                require(enough, declared, min_properties);
                parts.push_back(enough);
            }
            branches.push_back(all_of(pointer, std::move(parts)));
        }
        if (array) {
            const std::size_t positional =
                static_cast<std::size_t>(std::min<std::uint64_t>(min_items, prefix.size()));
            std::vector<std::uint32_t> parts(prefix.begin(), prefix.begin() + positional);
            if (min_items > prefix.size()) {
                // Without a schema for further items, a node that needs a missing child.
                const std::uint32_t further = extra_items ? *extra_items : add_node(pointer);
                if (!extra_items) { graph_[further].needed = 1; }
                parts.push_back(further);
            }
            branches.push_back(all_of(pointer, std::move(parts)));
        }
        require(id, std::move(branches), 1);
    }

    // The object keywords xgrammar enforces exactly: it ignores required names that are not
    // properties, applies propertyNames and property patterns only to some keys and one pattern
    // per key, spells undeclared keys without escapes, and counts repeated undeclared keys against
    // minProperties and maxProperties. Returns the required names.
    std::unordered_set<std::string> object_rule(const Json& value, const std::string& pointer,
                                                std::uint32_t id) {
        for (const char* key : {"properties", "patternProperties"}) {
            if (value.contains(key) && !value.at(key).is_object()) {
                fail(GrammarErrorKind::InvalidSchema, pointer,
                     "'" + std::string(key) + "' must be an object of schemas");
            }
        }
        const bool properties = value.contains("properties");
        const auto closed     = [&](const char* key) {
            return value.contains(key) && value.at(key).is_boolean() && !value.at(key).get<bool>();
        };
        const auto open = [&](const char* key) { return value.contains(key) && !closed(key); };
        const bool additional      = open("additionalProperties") || open("unevaluatedProperties");
        const bool patterns        = value.contains("patternProperties");
        const bool names           = value.contains("propertyNames");
        const std::size_t declared = properties ? value.at("properties").size() : 0;

        std::unordered_set<std::string> required;
        if (value.contains("required")) {
            const Json& list = value.at("required");
            if (!list.is_array() || !std::all_of(list.begin(), list.end(), [](const Json& name) {
                    return name.is_string();
                })) {
                fail(GrammarErrorKind::InvalidSchema, pointer,
                     "'required' must be an array of property names");
            }
            for (const Json& name : list) {
                const auto& text = name.get_ref<const std::string&>();
                if (!properties || !value.at("properties").contains(text)) {
                    fail(GrammarErrorKind::UnsupportedSchema, pointer,
                         "required property '" + text + "' must be declared in 'properties'");
                }
                required.insert(text);
            }
        }
        if (names && (properties || patterns)) {
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "'propertyNames' cannot be combined with 'properties' or 'patternProperties'");
        }
        if (names && (closed("additionalProperties") || closed("unevaluatedProperties"))) {
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "'propertyNames' needs additional properties to be allowed");
        }
        if (patterns && (properties || additional)) {
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "'patternProperties' cannot be combined with 'properties' or additional "
                 "properties");
        }
        if (patterns && value.at("patternProperties").size() != 1) {
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "'patternProperties' must hold exactly one pattern");
        }
        if (properties && additional) {
            for (const auto& [name, child] : value.at("properties").items()) {
                if (needs_escaping(name)) {
                    fail(GrammarErrorKind::UnsupportedSchema, pointer,
                         "a property name with a quote, a backslash or a control character "
                         "needs an object without additional properties");
                }
            }
        }
        if (value.contains("minProperties") || value.contains("maxProperties")) {
            if (!properties || additional || patterns || names) {
                fail(GrammarErrorKind::UnsupportedSchema, pointer,
                     "'minProperties' and 'maxProperties' need declared properties and no other "
                     "keys");
            }
            if (declared > limits_.max_counted_properties) {
                fail(GrammarErrorKind::LimitExceeded, pointer,
                     "'minProperties' and 'maxProperties' apply to at most " +
                         std::to_string(limits_.max_counted_properties) + " properties");
            }
            const auto [lower, upper] = bounds(value, pointer, "minProperties", "maxProperties");
            (void)lower;
            if (upper && *upper < required.size()) {
                fail(GrammarErrorKind::InvalidSchema, pointer,
                     "'maxProperties' is below the number of required properties");
            }
        }
        const std::uint64_t optional = declared - required.size();
        charge_schema(id, CompileCost{.optional_pairs = optional * optional});
        return required;
    }

    // xgrammar's string rule takes a supported format first, then a pattern, then length
    // bounds, and ignores whatever follows: only one of them may be present.
    void string_rule(const Json& value, const std::string& pointer, std::uint32_t id) {
        const bool format  = value.contains("format");
        const bool regex   = value.contains("pattern");
        const bool lengths = value.contains("minLength") || value.contains("maxLength");
        if (static_cast<int>(format) + static_cast<int>(regex) + static_cast<int>(lengths) > 1) {
            fail(GrammarErrorKind::UnsupportedSchema, pointer,
                 "a string schema may use only one of 'format', 'pattern', and "
                 "'minLength'/'maxLength'");
        }
        if (format) {
            const Json& name = value.at("format");
            if (!name.is_string() || !contains(kFormats, name.get<std::string>())) {
                fail(GrammarErrorKind::UnsupportedSchema, pointer,
                     "string format must be one of date, date-time, duration, email, hostname, "
                     "ipv4, ipv6, time, uri, uuid");
            }
        }
        if (regex) { pattern(value.at("pattern"), pointer + "/pattern", id); }
        if (lengths) {
            const auto [lower, upper] = bounds(value, pointer, "minLength", "maxLength");
            // xgrammar expands identical bounds once: a pair repeated in this schema is free.
            if (!length_bounds_.emplace(lower, upper).second) { return; }
            const std::uint64_t bound = std::max(lower.value_or(0), upper.value_or(0));
            charge(pointer, CompileCost{.length_bounds =
                                            std::min<std::uint64_t>(bound, kLengthBoundUnitCap)});
        }
    }

    const GrammarLimits& limits_;
    std::string_view location_;
    std::uint32_t& nodes_;
    CompileCost& cost_;
    std::vector<Node> graph_;
    std::unordered_map<std::string, std::uint32_t> admitted_paths_;
    std::vector<Reference> references_;
    std::set<std::pair<std::optional<std::uint64_t>, std::optional<std::uint64_t>>> length_bounds_;
};

} // namespace

std::uint64_t CompileCost::units() const noexcept {
    return saturating_sum(
        saturating_sum(pattern_classes,
                       divide_rounding_up(pattern_literals, kPatternLiteralsPerUnit)),
        saturating_sum(divide_rounding_up(optional_pairs, kOptionalPairsPerUnit), length_bounds));
}

std::string SchemaAdmission::admit(std::string_view schema_json, std::string_view location) {
    if (schema_json.size() > limits_.max_schema_bytes) {
        throw GrammarError(GrammarErrorKind::LimitExceeded,
                           std::string(location) + ": the schema is larger than " +
                               std::to_string(limits_.max_schema_bytes) + " bytes");
    }
    // Checked on the text: parsing, serializing and xgrammar's hashing all recurse on the depth.
    if (json_depth(schema_json) > limits_.max_json_depth) {
        throw GrammarError(GrammarErrorKind::LimitExceeded,
                           std::string(location) +
                               ": the schema nests arrays and objects more "
                               "than " +
                               std::to_string(limits_.max_json_depth) + " levels deep");
    }
    Json schema;
    try {
        schema = Json::parse(schema_json);
    } catch (const Json::exception&) {
        throw GrammarError(GrammarErrorKind::InvalidSchema,
                           std::string(location) + ": the schema is not valid JSON");
    }
    // Charged to a copy, so that a rejected schema leaves the grammar's cost unchanged.
    CompileCost cost    = cost_;
    std::uint32_t nodes = nodes_;
    SchemaWalker walker(limits_, location, nodes, cost);
    (void)walker.schema(schema, "", SchemaPath{}, 0);
    walker.check_references();
    walker.check_finite();
    cost_  = cost;
    nodes_ = nodes;
    return schema.dump();
}

} // namespace ninfer::grammar
