// ow-sim: input validation, determinism, error handling and the §16.2 property tests on random graphs.
// Exact numerical expectations live in tests/conformance (derived analytically, not from this code).

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "occamworm/ir/compile.hpp"
#include "occamworm/sim/sim.hpp"
#include "ow_test.hpp"

using namespace occamworm;

namespace {

// Small deterministic generator for random test graphs (not part of the library).
class Lcg {
public:
    explicit Lcg(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return state_ >> 33U;
    }
    std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
    double unit() { return static_cast<double>(next() % 1000000) / 1000000.0; }

private:
    std::uint64_t state_;
};

const std::string kMixedProgram = R"(wrl 0.1
tier G1
state v : 1 = 0
state h : 1 = 0
param tau : s = 0.4 in [0.05, 5] trainable bits 8
param g : 1 = 1 in [0, 4] trainable bits 8
input u = stimulus
input exc = sum_in(v, exc)
input inh = sum_in(v, inh)
let drive = u + 0.8 * exc - 0.5 * inh + 0.3 * type_mask(motor) + 0.2 * delay(v, 2)
next v = leaky_integrate(v, tanh(drive - h), tau)
next h = leaky_integrate(h, relu(v), 2[s])
gap v scale g
observe calcium_linear_v1(v, 0.5[s])
)";

SimInput random_input(Lcg& rng, std::size_t neurons) {
    SimInput input;
    for (std::size_t i = 0; i < neurons; ++i) {
        input.graph.neurons.push_back({"n" + std::to_string(i), rng.below(3) == 0 ? "motor" : "inter"});
    }
    const std::size_t chemical = 2 * neurons;
    for (std::size_t e = 0; e < chemical; ++e) {
        input.graph.chemical.push_back({"n" + std::to_string(rng.below(neurons)), "n" + std::to_string(rng.below(neurons)),
                                        0.2 + rng.unit(), rng.below(3) == 0 ? -1 : 1, static_cast<int>(rng.below(4))});
    }
    for (std::size_t e = 0; e < neurons; ++e) {
        const std::size_t a = rng.below(neurons);
        const std::size_t b = (a + 1 + rng.below(neurons - 1)) % neurons;
        input.graph.gap.push_back({"n" + std::to_string(a), "n" + std::to_string(b), 0.5 + rng.unit()});
    }
    input.dt = 0.1;
    input.n_steps = 25;
    for (long long tick = 0; tick <= input.n_steps; tick += 5) {
        input.sample_ticks.push_back(tick);
    }
    input.stimulus.push_back({"n" + std::to_string(rng.below(neurons)), 0, 8, 1.0});
    input.stimulus.push_back({"n" + std::to_string(rng.below(neurons)), 3, 12, -0.7});
    return input;
}

SimInput two_neuron_input() {
    SimInput input;
    input.graph.neurons = {{"A", "generic"}, {"B", "generic"}};
    input.graph.chemical = {{"A", "B", 1.0, 1, 0}};
    input.dt = 0.1;
    input.n_steps = 3;
    input.sample_ticks = {0, 3};
    return input;
}

const std::string kTwoRegisterProgram = "wrl 0.1\ntier G1\ndt_max 0.2\nstate v : 1 = 0\nparam tau : s = 1 in [0.5, 2] trainable bits 8\n"
                                        "next v = leaky_integrate(v, stimulus + sum_in(v, all), tau)\nobserve identity_v1(v)\n";

}  // namespace

OW_TEST(sim_rejects_invalid_inputs) {
    const Ir ir = compile(kTwoRegisterProgram);
    SimInput input = two_neuron_input();
    OW_CHECK_NOTHROW(simulate(ir, input));
    input.dt = 0.0;
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.dt = 0.5;  // above dt_max
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.sample_ticks = {2, 1};
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.sample_ticks = {0, 4};  // beyond n_steps
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.params["tau"] = 100.0;  // outside bounds
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.params["nope"] = 1.0;
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.stimulus.push_back({"Z", 0, 1, 1.0});
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.observed = {"Z"};
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.initial_state["nope"]["A"] = 1.0;
    OW_CHECK_THROWS(simulate(ir, input), Errc::Input);
    input = two_neuron_input();
    input.graph.chemical.push_back({"A", "Z", 1.0, 1, 0});
    OW_CHECK_THROWS(simulate(ir, input), Errc::Graph);
}

OW_TEST(sim_reports_non_finite_values) {
    const Ir ir = compile("wrl 0.1\ntier G1\nstate v : 1 = 1\nnext v = v * 1e200\nobserve identity_v1(v)\n");
    SimInput input = two_neuron_input();
    input.n_steps = 5;
    input.sample_ticks = {0};
    OW_CHECK_THROWS(simulate(ir, input), Errc::Runtime);
}

OW_TEST(sim_samples_only_requested_ticks_and_neurons) {
    const Ir ir = compile(kTwoRegisterProgram);
    SimInput input = two_neuron_input();
    input.observed = {"B"};
    input.stimulus.push_back({"A", 0, 3, 1.0});
    const SimResult result = simulate(ir, input);
    OW_CHECK_EQ(result.neurons.size(), std::size_t{1});
    OW_CHECK_EQ(result.registers[0].size(), std::size_t{2});
    OW_CHECK_EQ(result.registers[0][0].size(), std::size_t{1});
    OW_CHECK_EQ(result.registers[0][0][0], 0.0);
    OW_CHECK(result.registers[0][1][0] > 0.0);
}

OW_TEST(sim_is_bitwise_deterministic) {
    const Ir ir = compile(kMixedProgram);
    Lcg rng(7);
    const SimInput input = random_input(rng, 6);
    const std::string first = sim_result_to_json(ir, input, simulate(ir, input)).dump();
    const std::string second = sim_result_to_json(ir, input, simulate(ir, input)).dump();
    OW_CHECK_EQ(first, second);
}

OW_TEST(sim_result_survives_ir_json_round_trip) {
    const Ir ir = compile(kMixedProgram);
    const Ir again = ir_from_json(Json::parse(ir_to_json(ir).dump()));
    Lcg rng(11);
    const SimInput input = random_input(rng, 5);
    OW_CHECK_EQ(sim_result_to_json(ir, input, simulate(ir, input)).dump(), sim_result_to_json(again, input, simulate(again, input)).dump());
}

OW_TEST(property_relabelling_neurons_permutes_outputs) {
    const Ir ir = compile(kMixedProgram);
    Lcg rng(2026);
    for (int trial = 0; trial < 25; ++trial) {
        const std::size_t n = 3 + rng.below(5);
        const SimInput input = random_input(rng, n);
        std::vector<std::size_t> permutation(n);
        for (std::size_t i = 0; i < n; ++i) {
            permutation[i] = i;
        }
        for (std::size_t i = n - 1; i > 0; --i) {  // Fisher-Yates
            std::swap(permutation[i], permutation[rng.below(i + 1)]);
        }
        const SimResult base = simulate(ir, input);
        const SimResult permuted = simulate(ir, permute_neurons(input, permutation));
        for (std::size_t r = 0; r < base.registers.size(); ++r) {
            for (std::size_t s = 0; s < base.registers[r].size(); ++s) {
                for (std::size_t j = 0; j < n; ++j) {
                    OW_CHECK_NEAR(permuted.registers[r][s][j], base.registers[r][s][permutation[j]], 1e-12);
                }
            }
        }
        for (std::size_t s = 0; s < base.observation.size(); ++s) {
            for (std::size_t j = 0; j < n; ++j) {
                OW_CHECK_NEAR(permuted.observation[s][j], base.observation[s][permutation[j]], 1e-12);
            }
        }
    }
}

OW_TEST(property_disconnected_subnetworks_are_independent) {
    const Ir ir = compile(kMixedProgram);
    SimInput both;
    both.graph.neurons = {{"a0", "inter"}, {"a1", "motor"}, {"b0", "inter"}, {"b1", "inter"}};
    both.graph.chemical = {{"a0", "a1", 1.0, 1, 1}, {"a1", "a0", 0.5, -1, 0}, {"b0", "b1", 1.0, 1, 2}};
    both.graph.gap = {{"a0", "a1", 1.0}, {"b0", "b1", 2.0}};
    both.dt = 0.1;
    both.n_steps = 20;
    both.sample_ticks = {0, 10, 20};
    both.stimulus = {{"a0", 0, 10, 1.0}};
    const SimResult with_b_quiet = simulate(ir, both);
    // Driving the other component must not change this one, bit for bit.
    SimInput driven = both;
    driven.stimulus.push_back({"b0", 0, 10, 2.0});
    const SimResult with_b_driven = simulate(ir, driven);
    for (std::size_t s = 0; s < 3; ++s) {
        for (std::size_t j = 0; j < 2; ++j) {
            OW_CHECK_EQ(with_b_quiet.registers[0][s][j], with_b_driven.registers[0][s][j]);
        }
    }
    // And component b alone behaves exactly as inside the larger graph.
    SimInput alone;
    alone.graph.neurons = {{"b0", "inter"}, {"b1", "inter"}};
    alone.graph.chemical = {{"b0", "b1", 1.0, 1, 2}};
    alone.graph.gap = {{"b0", "b1", 2.0}};
    alone.dt = both.dt;
    alone.n_steps = both.n_steps;
    alone.sample_ticks = both.sample_ticks;
    alone.stimulus = {{"b0", 0, 10, 2.0}};
    const SimResult alone_result = simulate(ir, alone);
    for (std::size_t s = 0; s < 3; ++s) {
        for (std::size_t j = 0; j < 2; ++j) {
            OW_CHECK_EQ(alone_result.registers[0][s][j], with_b_driven.registers[0][s][j + 2]);
        }
    }
}

OW_TEST(property_zero_edge_graph_has_no_propagation) {
    const Ir ir = compile(kMixedProgram);
    SimInput input;
    input.graph.neurons = {{"x", "inter"}, {"y", "inter"}, {"z", "inter"}};
    input.dt = 0.1;
    input.n_steps = 30;
    input.sample_ticks = {30};
    input.stimulus = {{"x", 0, 30, 1.0}};
    const SimResult result = simulate(ir, input);
    OW_CHECK(result.registers[0][0][0] > 0.1);
    OW_CHECK_EQ(result.registers[0][0][1], 0.0);
    OW_CHECK_EQ(result.registers[0][0][2], 0.0);
}

OW_TEST(property_symmetric_subgraphs_evolve_identically) {
    const Ir ir = compile(kMixedProgram);
    SimInput input;
    input.graph.neurons = {{"s", "inter"}, {"l", "inter"}, {"r", "inter"}};
    input.graph.chemical = {{"s", "l", 0.7, 1, 1}, {"s", "r", 0.7, 1, 1}};
    input.graph.gap = {{"s", "l", 1.0}, {"s", "r", 1.0}};
    input.dt = 0.1;
    input.n_steps = 20;
    input.sample_ticks = {20};
    input.stimulus = {{"s", 0, 10, 1.0}};
    const SimResult result = simulate(ir, input);
    OW_CHECK_NEAR(result.registers[0][0][1], result.registers[0][0][2], 1e-15);
    OW_CHECK(result.registers[0][0][1] != 0.0);
}

OW_TEST(sim_input_json_parses_edits_and_dense_initial_state) {
    const Json json = Json::parse(R"({
      "dt": 0.5, "n_steps": 2,
      "graph": { "neurons": [{"id":"A"},{"id":"B"}], "chemical": [{"pre":"A","post":"B"}], "gap": [{"a":"A","b":"B","g":1}] },
      "edits": { "delete_chemical": [["A","B"]], "delete_gap": [["B","A"]] },
      "initial_state": { "v": [1.0, 2.0] }
    })");
    const SimInput input = sim_input_from_json(json);
    OW_CHECK_EQ(input.graph.chemical.size(), std::size_t{0});
    OW_CHECK_EQ(input.graph.gap.size(), std::size_t{0});
    OW_CHECK_EQ(input.initial_state.at("v").at("B"), 2.0);
    OW_CHECK_EQ(input.sample_ticks.size(), std::size_t{3});  // default: every tick
    OW_CHECK_THROWS(sim_input_from_json(Json::parse(R"({"dt":1})")), Errc::Input);
}
