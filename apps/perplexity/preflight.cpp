#include "preflight.h"

#include <stdexcept>

namespace ninfer::perplexity {

std::filesystem::path prepare_report_directory(const std::filesystem::path& directory) {
    if (std::filesystem::exists(directory)) {
        if (!std::filesystem::is_directory(directory) ||
            std::filesystem::directory_iterator(directory) !=
                std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     directory.string());
        }
    } else if (!std::filesystem::create_directories(directory)) {
        throw std::runtime_error("cannot create output directory: " + directory.string());
    }
    return std::filesystem::absolute(directory).lexically_normal();
}

Preflight run_preflight(const Options& options) {
    Preflight out;
    if (options.output) { out.report_directory = prepare_report_directory(*options.output); }
    if (options.logits_out) {
        check_logits_destination(*options.logits_out);
        if (options.logits_reference) {
            out.reference = read_kld_base_header(*options.logits_reference);
            check_reference_context(*out.reference, options.context);
        }
    }
    return out;
}

} // namespace ninfer::perplexity
