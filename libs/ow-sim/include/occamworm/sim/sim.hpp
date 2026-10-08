#pragma once
// Scalar reference interpreter (OW-009). Exact semantics: docs/language/WRL_SYNTAX.md ("Execution
// semantics") and docs/runtime/SIM_SEMANTICS.md ("Implemented (OW-009)").

#include <map>
#include <string>
#include <vector>

#include "occamworm/core/graph.hpp"
#include "occamworm/core/json.hpp"
#include "occamworm/ir/ir.hpp"

namespace occamworm {

constexpr long long kMaxSteps = 10000000;

struct StimulusEvent {
    std::string neuron;
    long long start = 0;  // first active tick (inclusive)
    long long end = 0;    // first inactive tick (exclusive)
    double amplitude = 0.0;
};

struct SimInput {
    GraphSpec graph;                    // after edge deletions have been applied
    double dt = 0.0;
    long long n_steps = 0;
    std::vector<long long> sample_ticks;  // strictly increasing, within [0, n_steps]
    std::vector<StimulusEvent> stimulus;
    std::map<std::string, double> params;  // by source name; overrides declared defaults
    // register source name -> neuron id -> initial value (sparse; unspecified neurons keep the declared init)
    std::map<std::string, std::map<std::string, double>> initial_state;
    std::vector<std::string> observed;     // neuron ids to report; empty = all, in graph order
};

struct SimResult {
    std::vector<std::string> neurons;                    // reported neurons, in report order
    std::vector<long long> sample_ticks;
    std::vector<std::string> register_names;             // source names, canonical register order
    std::vector<std::vector<std::vector<double>>> registers;  // [register][sample][reported neuron]
    std::string observation_operator;
    std::string observation_register;
    std::vector<std::vector<double>> observation;        // [sample][reported neuron]
};

// Parses the sim-input JSON documented in WRL_SYNTAX.md and applies its "edits". Throws Error(Errc::Input).
SimInput sim_input_from_json(const Json& json);

// Runs the program. Throws Error(Errc::Input) for invalid inputs and Error(Errc::Runtime) for non-finite values.
SimResult simulate(const Ir& ir, const SimInput& input);

Json sim_result_to_json(const Ir& ir, const SimInput& input, const SimResult& result);

// Returns a copy of the input with the neuron list reordered: new position j holds old neuron permutation[j].
SimInput permute_neurons(const SimInput& input, const std::vector<std::size_t>& permutation);

}  // namespace occamworm
