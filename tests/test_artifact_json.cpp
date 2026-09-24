#include "artifact/schema.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using ninfer::artifact::ArtifactError;
using ninfer::artifact::Json;
using ninfer::artifact::parse_json;

constexpr std::string_view kLabel = "fixture directory";

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string rejection(std::string_view text) {
    try {
        (void)parse_json(text, kLabel);
    } catch (const ArtifactError& error) { return error.what(); }
    throw std::runtime_error("accepted " + std::string(text.substr(0, 80)));
}

template <typename Function>
double seconds(Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// A directory-shaped text with count bindings and count uses: an object and an array of small
// sibling objects. A nonempty repeat is appended as one more binding name.
std::string sibling_objects(std::size_t count, std::string_view repeat = {}) {
    std::string text = R"({"bindings":{)";
    for (std::size_t i = 0; i < count; ++i) {
        const auto id = std::to_string(i);
        text += (i == 0 ? R"("b)" : R"(,"b)") + id + R"(":{"object":"o)" + id + R"(","begin":0})";
    }
    if (!repeat.empty()) { text += R"(,")" + std::string(repeat) + R"(":{})"; }
    text += R"(},"uses":[)";
    for (std::size_t i = 0; i < count; ++i) {
        text += (i == 0 ? R"({"parameter":"p)" : R"(,{"parameter":"p)") + std::to_string(i) +
                R"(","input":"x"})";
    }
    return text + "]}";
}

// A repeated member name is rejected in every object, at any depth, and the error names it.
void test_duplicate_members() {
    struct Case {
        std::string_view text;
        std::string_view member;
    };
    constexpr Case cases[] = {
        {R"({"objects":[],"bindings":{},"objects":[]})", "objects"},
        {R"({"bindings":{"a":{"object":"x","begin":0,"object":"y"}}})", "object"},
        {R"({"objects":[{"id":"a"},{"id":"b","kind":"tensor","id":"c"}]})", "id"},
        {R"([[{"u":{"a":1,"b":{"c":[0,{"d":1,"e":2,"d":3}]}}}]])", "d"},
        {R"({"id":1,"\u0069d":2})", "id"},
        {R"({"":1,"":2})", ""},
    };
    for (const auto& [text, member] : cases) {
        const auto message = rejection(text);
        require(message == std::string(kLabel) + ": duplicate JSON member " + std::string(member),
                "unexpected rejection of " + std::string(text) + ": " + message);
    }
}

// An accepted text yields the value nlohmann::json::parse returns, number types included, and
// the same name in different objects, nested or sibling, is not a repeat.
void test_accepted_value() {
    constexpr std::string_view text =
        R"({"id":{"id":{"id":1},"x":{"id":2}},"list":[{"id":3},{"id":4,"list":[{"id":5}]}],)"
        R"("u":18446744073709551615,"i":-9223372036854775808,"z":-0,"f":0.5,"g":1e2,"w":-0.0,)"
        R"("s":"a\"\u00e9\ud83d\ude00","t":true,"n":null,"a":[1,-1,1.0,[],{}],"o":{}})";
    const Json value    = parse_json(text, kLabel);
    const Json expected = Json::parse(text);
    require(value == expected, "accepted value differs from nlohmann::json::parse");
    const Json leaves          = value.flatten();
    const Json expected_leaves = expected.flatten();
    for (const auto& [pointer, leaf] : expected_leaves.items()) {
        require(leaves.at(pointer).type() == leaf.type(), "value type differs at " + pointer);
    }
    require(parse_json(R"( "scalar" )", kLabel) == "scalar", "top-level scalar differs");
}

// Every other parse error is nlohmann's own description behind the label.
void test_parse_errors() {
    constexpr std::string_view texts[] = {
        "", R"({"a":1,})", R"({"a":1} {})", R"({"a":1e999})", R"({"a":[1,tru]})", R"({"a" 1})",
    };
    for (const auto text : texts) {
        std::string expected;
        try {
            [[maybe_unused]] const Json value = Json::parse(text);
        } catch (const Json::exception& error) { expected = error.what(); }
        require(!expected.empty(), "nlohmann::json::parse accepted " + std::string(text));
        const auto message = rejection(text);
        require(message == std::string(kLabel) + ": " + expected,
                "unexpected rejection of " + std::string(text) + ": " + message);
    }
}

// Parsing stays linear in the number of sibling objects, and a repeat is still found among them.
// 100,000 sibling objects, more than the 35B-A3B directory's 65,000: nlohmann's callback parser,
// which rescans the enclosing container after each nested object, took hundreds of times as long
// as Json::parse on this text.
void test_sibling_objects() {
    constexpr std::size_t kCount = 50'000;
    const std::string text       = sibling_objects(kCount);
    Json value;
    Json expected;
    double checked = std::numeric_limits<double>::infinity();
    double plain   = std::numeric_limits<double>::infinity();
    for (int run = 0; run < 3; ++run) {
        checked = std::min(checked, seconds([&] { value = parse_json(text, kLabel); }));
        plain   = std::min(plain, seconds([&] { expected = Json::parse(text); }));
    }
    require(value == expected, "sibling objects: value differs from nlohmann::json::parse");
    require(checked < 10 * plain, "sibling objects: parse_json took " + std::to_string(checked) +
                                      " s, nlohmann::json::parse " + std::to_string(plain) + " s");

    const auto message = rejection(sibling_objects(kCount, "b0"));
    require(message == std::string(kLabel) + ": duplicate JSON member b0",
            "sibling objects: unexpected rejection of a repeated first binding: " + message);
}

} // namespace

int main() {
    try {
        test_duplicate_members();
        test_accepted_value();
        test_parse_errors();
        test_sibling_objects();
        std::cout << "artifact JSON checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
