#pragma once
// Conformance suite runner (OW-009). Case format: docs/language/WRL_SYNTAX.md ("Conformance case format").

#include <iosfwd>
#include <string>

#include "occamworm/core/json.hpp"

namespace occamworm {

struct ConformanceSummary {
    int passed = 0;
    int failed = 0;
};

// Runs one case; returns true on success and otherwise fills `detail` with a human-readable reason.
bool run_conformance_case(const Json& testcase, const std::string& base_directory, std::string& detail);

// Runs every *.json file in the directory (sorted by file name), printing one line per case.
ConformanceSummary run_conformance_suite(const std::string& directory, std::ostream& out);

}  // namespace occamworm
