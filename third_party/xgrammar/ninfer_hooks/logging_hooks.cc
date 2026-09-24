// NInfer hooks for xgrammar's customizable logging (XGRAMMAR_LOG_CUSTOMIZE=1).
//
// xgrammar formats diagnostics from grammar and schema content: regex patterns it ignores, rule
// names, rejected strings. Nothing is printed, so request content never reaches the process logs:
// warnings go to the calling thread's active WarningCapture, if any, and are dropped otherwise. A
// fatal message becomes a std::runtime_error carrying only the message; callers return it to the
// client and never log it.
#include "ninfer_hooks/warning_capture.h"
#include "support/logging.h"

#include <stdexcept>
#include <string>

namespace xgrammar {
namespace ninfer {
namespace {

thread_local WarningCapture* active_capture = nullptr;

} // namespace

WarningCapture::WarningCapture() : previous_(active_capture) { active_capture = this; }

WarningCapture::~WarningCapture() { active_capture = previous_; }

void record_warning(const std::string& message) {
    if (active_capture != nullptr) { active_capture->warnings_.push_back(message); }
}

} // namespace ninfer

[[noreturn]] void LogFatalImpl(const std::string&, int, const std::string& message) {
    throw std::runtime_error(message);
}

void LogMessageImpl(const std::string&, int, int level, const std::string& message) {
    if (level >= XGRAMMAR_LOG_LEVEL_WARNING) { ninfer::record_warning(message); }
}

} // namespace xgrammar
