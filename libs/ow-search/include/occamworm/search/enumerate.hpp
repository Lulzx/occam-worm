#pragma once
// Program enumerator and deduplicator (OW-010, §7.2 S0 and §7.3 gates 1-2 and 4). Grammar and ordering
// are specified in docs/language/WRL_SYNTAX.md ("Enumeration").

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "occamworm/core/json.hpp"
#include "occamworm/ir/ir.hpp"

namespace occamworm {

struct ParameterTemplate {
    std::string unit = "1";
    double lower = 0.0;
    double upper = 1.0;
    int bits = 8;
};

struct LutSpec {
    std::vector<int> sizes;      // table lengths
    std::vector<double> values;  // entry values; every table over these values is enumerated
};

struct EnumerationConfig {
    std::string name;
    Tier tier = Tier::G1;
    int max_nodes = 5;       // total expression nodes over all register updates
    int max_depth = 3;       // depth of each update expression (a leaf has depth 1)
    int max_registers = 1;
    int max_parameters = 1;  // distinct parameters a program may use
    std::vector<std::string> ops;  // per-tier operator allowlist (operators and leaf classes)
    std::vector<double> constants;      // dimensionless constant leaves (needs "const" in ops)
    std::vector<double> tau_constants;  // constant time-constant operands, in seconds
    std::vector<std::string> sum_in_selects;  // "exc", "inh", "all" (needs "sum_in")
    std::vector<int> count_values;            // compared states (needs "count_in")
    std::vector<int> delays;                  // ticks (needs "delay")
    std::vector<std::string> type_names;      // (needs "type_mask")
    LutSpec lut;                              // (needs "lut")
    std::vector<ParameterTemplate> parameters;  // trainable parameter pool (needs "param")
    std::optional<double> dt_max;
    std::vector<double> register_init;  // per register; missing entries are 0
};

struct EnumerationSummary {
    long long generated = 0;           // candidate sources produced by the grammar
    long long rejected_type = 0;       // unit, type or name errors
    long long rejected_stability = 0;  // explicit-integrator stability bound violated
    long long rejected_budget = 0;     // exceeds max_registers / max_parameters after canonicalisation
    long long duplicates = 0;          // canonical hash already seen
    long long unique = 0;              // distinct canonical programs emitted
};

struct EnumeratedProgram {
    long long index = 0;  // 0-based position in the emission order
    Ir ir;                // canonical IR with hash, canonical source and bit counts
};

EnumerationConfig enumeration_config_from_json(const Json& json);  // throws Error(Errc::Input)

// Visits every distinct canonical program in the grammar budget exactly once, in a deterministic order, and
// returns the accounting of every attempted candidate.
EnumerationSummary enumerate_programs(const EnumerationConfig& config, const std::function<void(const EnumeratedProgram&)>& visit);

// WRL source of one candidate: `registers` registers r0.. updated by the given expressions, observing r0.
// Shared by the enumerator and by the independent brute-force check in the tests.
std::string assemble_program_source(const EnumerationConfig& config, int registers, const std::vector<std::string>& updates);

Json program_record_json(const EnumeratedProgram& program);
Json summary_record_json(const EnumerationConfig& config, const EnumerationSummary& summary);

}  // namespace occamworm
