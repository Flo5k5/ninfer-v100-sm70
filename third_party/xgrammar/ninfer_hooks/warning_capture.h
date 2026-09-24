#pragma once

// NInfer extension of the xgrammar logging hooks: collects the warnings xgrammar emits on the
// calling thread while a capture is active. xgrammar warns when it relaxes a construct it cannot
// represent exactly (an overlapping oneOf read as anyOf, an ignored multipleOf, an ignored regex
// assertion); callers reject such grammars instead of constraining output with a weaker one.
// Warnings emitted on other threads, or with no active capture, are dropped.

#include <string>
#include <vector>

namespace xgrammar::ninfer {

class WarningCapture {
public:
    WarningCapture();
    ~WarningCapture();

    WarningCapture(const WarningCapture&)            = delete;
    WarningCapture& operator=(const WarningCapture&) = delete;

    [[nodiscard]] const std::vector<std::string>& warnings() const noexcept { return warnings_; }

private:
    friend void record_warning(const std::string& message);

    WarningCapture* previous_ = nullptr;
    std::vector<std::string> warnings_;
};

} // namespace xgrammar::ninfer
