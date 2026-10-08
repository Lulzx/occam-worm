#include "occamworm/sim/sim.hpp"

#include <algorithm>
#include <cmath>

#include "occamworm/core/build_info.hpp"
#include "occamworm/core/error.hpp"
#include "occamworm/ir/scalar_ops.hpp"

namespace occamworm {
namespace {

[[noreturn]] void input_error(const std::string& message) { throw Error(Errc::Input, message); }

class Simulator {
public:
    Simulator(const Ir& ir, const SimInput& input) : ir_(ir), input_(input), graph_(Graph::build(input.graph)) {}

    SimResult run() {
        validate();
        prepare();
        SimResult result;
        result.sample_ticks = input_.sample_ticks;
        for (const std::size_t index : reported_) {
            result.neurons.push_back(graph_.ids[index]);
        }
        result.register_names.reserve(ir_.registers.size());
        for (const IrRegister& reg : ir_.registers) {
            result.register_names.push_back(reg.source_name);
        }
        result.registers.assign(ir_.registers.size(), {});
        result.observation_operator = std::string(obs_operator_name(ir_.observation.op));
        result.observation_register = ir_.registers[static_cast<std::size_t>(ir_.observation.reg)].source_name;

        std::size_t next_sample = 0;
        const auto record = [&](long long tick) {
            if (next_sample < input_.sample_ticks.size() && input_.sample_ticks[next_sample] == tick) {
                for (std::size_t r = 0; r < state_.size(); ++r) {
                    result.registers[r].push_back(pick(state_[r]));
                }
                result.observation.push_back(pick(observation_state_));
                ++next_sample;
            }
        };
        record(0);
        for (long long tick = 0; tick < input_.n_steps; ++tick) {
            step(tick);
            record(tick + 1);
        }
        return result;
    }

private:
    void validate() const {
        if (!std::isfinite(input_.dt) || !(input_.dt > 0.0)) {
            input_error("dt must be a positive finite number");
        }
        if (ir_.dt_max && input_.dt > *ir_.dt_max) {
            input_error("dt exceeds the program's declared dt_max");
        }
        if (input_.n_steps < 0 || input_.n_steps > kMaxSteps) {
            input_error("n_steps must be in [0, " + std::to_string(kMaxSteps) + "]");
        }
        long long previous = -1;
        for (const long long tick : input_.sample_ticks) {
            if (tick <= previous || tick > input_.n_steps) {
                input_error("sample_ticks must be strictly increasing and within [0, n_steps]");
            }
            previous = tick;
        }
    }

    void prepare() {
        const std::size_t n = graph_.size();
        // Parameter values: declared defaults, then overrides (bounds-checked).
        param_values_.clear();
        for (const IrParam& param : ir_.params) {
            param_values_.push_back(param.value);
        }
        for (const auto& [name, value] : input_.params) {
            const auto it = std::ranges::find_if(ir_.params, [&](const IrParam& p) { return p.source_name == name; });
            if (it == ir_.params.end()) {
                if (std::ranges::find(ir_.eliminated_params, name) != ir_.eliminated_params.end()) {
                    continue;  // the parameter was removed as dead code
                }
                input_error("unknown parameter '" + name + "'");
            }
            if (!std::isfinite(value) || value < it->lower || value > it->upper) {
                input_error("parameter '" + name + "' lies outside its declared bounds");
            }
            param_values_[static_cast<std::size_t>(it - ir_.params.begin())] = value;
        }

        // Initial registers.
        state_.assign(ir_.registers.size(), {});
        for (std::size_t r = 0; r < ir_.registers.size(); ++r) {
            state_[r].assign(n, ir_.registers[r].init);
        }
        for (const auto& [name, values] : input_.initial_state) {
            const auto it = std::ranges::find_if(ir_.registers, [&](const IrRegister& reg) { return reg.source_name == name; });
            if (it == ir_.registers.end()) {
                if (std::ranges::find(ir_.eliminated_registers, name) != ir_.eliminated_registers.end()) {
                    continue;
                }
                input_error("initial_state names unknown register '" + name + "'");
            }
            for (const auto& [neuron, value] : values) {
                const auto index = graph_.index_of(neuron);
                if (!index) {
                    input_error("initial_state names unknown neuron '" + neuron + "'");
                }
                if (!std::isfinite(value)) {
                    input_error("initial_state values must be finite");
                }
                state_[static_cast<std::size_t>(it - ir_.registers.begin())][*index] = value;
            }
        }

        // Stimulus events.
        for (const StimulusEvent& event : input_.stimulus) {
            if (!graph_.index_of(event.neuron)) {
                input_error("stimulus names unknown neuron '" + event.neuron + "'");
            }
            if (event.start < 0 || event.end < event.start || !std::isfinite(event.amplitude)) {
                input_error("stimulus needs 0 <= start <= end and a finite amplitude");
            }
        }

        // History ring: D+1 slots per register, slot(tick) = tick mod (D+1), pre-filled with the initial
        // state so that reads before tick 0 return the initial state (constant history extension).
        max_delay_ = graph_.max_edge_delay;
        for (const Instr& instr : ir_.instrs) {
            if (instr.op == Op::Delay) {
                max_delay_ = std::max(max_delay_, instr.k);
            }
        }
        ring_size_ = static_cast<std::size_t>(max_delay_) + 1;
        ring_.assign(ir_.registers.size(), std::vector<std::vector<double>>(ring_size_));
        for (std::size_t r = 0; r < ir_.registers.size(); ++r) {
            for (auto& slot : ring_[r]) {
                slot = state_[r];
            }
        }

        // Calcium / observation state starts at the observed register's initial value.
        observation_state_ = state_[static_cast<std::size_t>(ir_.observation.reg)];
        calcium_tau_ = 0.0;
        if (ir_.observation.op == ObsOperator::CalciumLinearV1) {
            calcium_tau_ = ir_.observation.tau.kind == IrTauOperand::Kind::Param
                               ? param_values_[static_cast<std::size_t>(ir_.observation.tau.param)]
                               : ir_.observation.tau.value;
        }
        gap_scale_ = 1.0;
        if (ir_.gap && ir_.gap->scale_param >= 0) {
            gap_scale_ = param_values_[static_cast<std::size_t>(ir_.gap->scale_param)];
        }

        // Reported neurons.
        reported_.clear();
        if (input_.observed.empty()) {
            for (std::size_t i = 0; i < n; ++i) {
                reported_.push_back(i);
            }
        } else {
            for (const std::string& id : input_.observed) {
                const auto index = graph_.index_of(id);
                if (!index) {
                    input_error("observed names unknown neuron '" + id + "'");
                }
                reported_.push_back(*index);
            }
        }
        stimulus_.assign(n, 0.0);
        values_.assign(ir_.instrs.size(), 0.0);
    }

    std::vector<double> pick(const std::vector<double>& full) const {
        std::vector<double> out;
        out.reserve(reported_.size());
        for (const std::size_t index : reported_) {
            out.push_back(full[index]);
        }
        return out;
    }

    // Value of register r at neuron i, d ticks before tick t (d = 0 is the old state at tick t).
    double history(std::size_t r, std::size_t i, int d, long long t) const {
        if (d == 0) {
            return state_[r][i];
        }
        const long long size = static_cast<long long>(ring_size_);
        const long long slot = (((t - d) % size) + size) % size;  // floor-mod: negative ticks hit pre-filled slots
        return ring_[r][static_cast<std::size_t>(slot)][i];
    }

    void load_stimulus(long long tick) {
        std::ranges::fill(stimulus_, 0.0);
        for (const StimulusEvent& event : input_.stimulus) {
            if (event.start <= tick && tick < event.end) {
                stimulus_[*graph_.index_of(event.neuron)] += event.amplitude;
            }
        }
    }

    double evaluate(const Instr& instr, std::size_t i, long long tick) const {
        const auto arg = [&](std::size_t k) { return values_[static_cast<std::size_t>(instr.args[k])]; };
        const auto reg = static_cast<std::size_t>(instr.index);
        switch (instr.op) {
            case Op::Const: return instr.value;
            case Op::Param: return param_values_[reg];
            case Op::State: return state_[reg][i];
            case Op::Stimulus: return stimulus_[i];
            case Op::TypeMask: return graph_.types[i] == instr.type_name ? 1.0 : 0.0;
            case Op::SumIn: {
                double sum = 0.0;
                for (std::size_t e = graph_.in_offsets[i]; e < graph_.in_offsets[i + 1]; ++e) {
                    const bool excitatory = graph_.sign[e] > 0;
                    if ((instr.select == SumSelect::Exc && !excitatory) || (instr.select == SumSelect::Inh && excitatory)) {
                        continue;
                    }
                    const double term = graph_.weight[e] * history(reg, graph_.pre[e], graph_.delay[e], tick);
                    sum += (instr.select == SumSelect::All && !excitatory) ? -term : term;
                }
                return sum;
            }
            case Op::CountIn: {
                double count = 0.0;
                for (std::size_t e = graph_.in_offsets[i]; e < graph_.in_offsets[i + 1]; ++e) {
                    if (history(reg, graph_.pre[e], graph_.delay[e], tick) == static_cast<double>(instr.k)) {
                        count += 1.0;
                    }
                }
                return count;
            }
            case Op::Delay: return history(reg, i, instr.k, tick);
            case Op::Add: {
                double acc = arg(0);
                for (std::size_t k = 1; k < instr.args.size(); ++k) {
                    acc += arg(k);
                }
                return acc;
            }
            case Op::Mul: {
                double acc = arg(0);
                for (std::size_t k = 1; k < instr.args.size(); ++k) {
                    acc *= arg(k);
                }
                return acc;
            }
            case Op::Neg: return -arg(0);
            case Op::Abs: return std::fabs(arg(0));
            case Op::Min: {
                double acc = arg(0);
                for (std::size_t k = 1; k < instr.args.size(); ++k) {
                    acc = scalar::min2(acc, arg(k));
                }
                return acc;
            }
            case Op::Max: {
                double acc = arg(0);
                for (std::size_t k = 1; k < instr.args.size(); ++k) {
                    acc = scalar::max2(acc, arg(k));
                }
                return acc;
            }
            case Op::Clamp: return scalar::clamp(arg(0), arg(1), arg(2));
            case Op::Relu: return scalar::relu(arg(0));
            case Op::Tanh: return std::tanh(arg(0));
            case Op::Sigmoid: return scalar::sigmoid(arg(0));
            case Op::Threshold: return scalar::threshold(arg(0), arg(1));
            case Op::Select: return scalar::select(arg(0), arg(1), arg(2));
            case Op::Lut: return instr.table[scalar::lut_index(arg(0), instr.table.size())];
            case Op::LeakyIntegrate: return scalar::leaky_integrate(arg(0), arg(1), arg(2), input_.dt);
            case Op::EulerLeak: return scalar::euler_leak(arg(0), arg(1), arg(2), input_.dt);
        }
        throw Error(Errc::Runtime, "unhandled operator");
    }

    // One synchronous step from tick t to t+1 (§6.1 steps 1-7; step 8, sampling, is done by the caller).
    void step(long long tick) {
        const std::size_t n = graph_.size();
        load_stimulus(tick);                                           // step 1
        std::vector<std::vector<double>> next(ir_.registers.size(), std::vector<double>(n));
        for (std::size_t i = 0; i < n; ++i) {                          // steps 2, 3, 5: reads see only old state
            for (std::size_t k = 0; k < ir_.instrs.size(); ++k) {
                const double value = evaluate(ir_.instrs[k], i, tick);
                if (!std::isfinite(value)) {
                    throw Error(Errc::Runtime, "non-finite value at tick " + std::to_string(tick) + ", neuron '" +
                                                   graph_.ids[i] + "', instruction n" + std::to_string(k));
                }
                values_[k] = value;
            }
            for (std::size_t r = 0; r < next.size(); ++r) {
                next[r][i] = values_[static_cast<std::size_t>(ir_.writes[r])];
            }
        }
        if (ir_.gap) {                                                 // step 4 combined with step 5, see docs
            const auto g = static_cast<std::size_t>(ir_.gap->reg);
            for (std::size_t i = 0; i < n; ++i) {
                double total = 0.0;
                double weighted = 0.0;
                for (std::size_t e = graph_.gap_offsets[i]; e < graph_.gap_offsets[i + 1]; ++e) {
                    const double c = gap_scale_ * graph_.gap_conductance[e];
                    total += c;
                    weighted += c * state_[g][graph_.gap_neighbor[e]];
                }
                next[g][i] = (next[g][i] + input_.dt * weighted) / (1.0 + input_.dt * total);
                if (!std::isfinite(next[g][i])) {
                    throw Error(Errc::Runtime, "non-finite value after gap coupling at tick " + std::to_string(tick));
                }
            }
        }
        const auto& observed_next = next[static_cast<std::size_t>(ir_.observation.reg)];   // step 6
        for (std::size_t i = 0; i < n; ++i) {
            observation_state_[i] = ir_.observation.op == ObsOperator::IdentityV1
                                        ? observed_next[i]
                                        : scalar::leaky_integrate(observation_state_[i], observed_next[i], calcium_tau_, input_.dt);
        }
        const std::size_t slot = static_cast<std::size_t>((tick + 1) % static_cast<long long>(ring_size_));  // step 7
        for (std::size_t r = 0; r < next.size(); ++r) {
            state_[r] = std::move(next[r]);
            ring_[r][slot] = state_[r];
        }
    }

    const Ir& ir_;
    const SimInput& input_;
    Graph graph_;
    std::vector<double> param_values_;
    std::vector<std::vector<double>> state_;
    std::vector<std::vector<std::vector<double>>> ring_;
    std::vector<double> observation_state_;
    std::vector<double> stimulus_;
    std::vector<double> values_;
    std::vector<std::size_t> reported_;
    std::size_t ring_size_ = 1;
    int max_delay_ = 0;
    double calcium_tau_ = 0.0;
    double gap_scale_ = 1.0;
};

}  // namespace

SimInput sim_input_from_json(const Json& json) {
    SimInput input;
    try {
        GraphSpec spec = graph_spec_from_json(json.at("graph"));
        if (const Json* edits = json.find("edits")) {
            if (const Json* chemical = edits->find("delete_chemical")) {
                for (const Json& pair : chemical->as_array()) {
                    delete_chemical_edges(spec, pair.as_array().at(0).as_string(), pair.as_array().at(1).as_string());
                }
            }
            if (const Json* gap = edits->find("delete_gap")) {
                for (const Json& pair : gap->as_array()) {
                    delete_gap_edges(spec, pair.as_array().at(0).as_string(), pair.as_array().at(1).as_string());
                }
            }
        }
        input.graph = std::move(spec);
        input.dt = json.at("dt").as_double();
        input.n_steps = json.at("n_steps").as_int();
        if (const Json* ticks = json.find("sample_ticks")) {
            for (const Json& tick : ticks->as_array()) {
                input.sample_ticks.push_back(tick.as_int());
            }
        } else if (input.n_steps >= 0 && input.n_steps <= kMaxSteps) {
            for (long long tick = 0; tick <= input.n_steps; ++tick) {
                input.sample_ticks.push_back(tick);
            }
        }
        if (const Json* stimulus = json.find("stimulus")) {
            for (const Json& item : stimulus->as_array()) {
                StimulusEvent event;
                event.neuron = item.at("neuron").as_string();
                event.start = item.at("start").as_int();
                event.end = item.at("end").as_int();
                event.amplitude = item.at("amplitude").as_double();
                input.stimulus.push_back(std::move(event));
            }
        }
        if (const Json* params = json.find("params")) {
            for (const auto& [name, value] : params->as_object()) {
                input.params[name] = value.as_double();
            }
        }
        if (const Json* initial = json.find("initial_state")) {
            for (const auto& [reg, values] : initial->as_object()) {
                auto& target = input.initial_state[reg];
                if (values.is_object()) {
                    for (const auto& [neuron, value] : values.as_object()) {
                        target[neuron] = value.as_double();
                    }
                } else {  // dense list in neuron order
                    const auto& list = values.as_array();
                    if (list.size() != input.graph.neurons.size()) {
                        input_error("dense initial_state for '" + reg + "' must list every neuron");
                    }
                    for (std::size_t i = 0; i < list.size(); ++i) {
                        target[input.graph.neurons[i].id] = list[i].as_double();
                    }
                }
            }
        }
        if (const Json* observed = json.find("observed")) {
            for (const Json& id : observed->as_array()) {
                input.observed.push_back(id.as_string());
            }
        }
    } catch (const Error& error) {
        if (error.code() == Errc::Json) {
            throw Error(Errc::Input, "malformed simulation input: " + error.message());
        }
        throw;
    } catch (const std::out_of_range&) {
        input_error("malformed edit pair (expected [id, id])");
    }
    return input;
}

SimResult simulate(const Ir& ir, const SimInput& input) { return Simulator(ir, input).run(); }

Json sim_result_to_json(const Ir& ir, const SimInput& input, const SimResult& result) {
    Json out = Json::object();
    out.set("schema", "occamworm.sim.result/0.1");
    out.set("program_hash", ir.program_hash);
    out.set("grammar_version", std::string(kGrammarVersion));
    out.set("compiler_build", OW_BUILD_STRING);
    out.set("dt", input.dt);
    out.set("n_steps", input.n_steps);
    Json ticks = Json::array();
    for (const long long tick : result.sample_ticks) {
        ticks.push(tick);
    }
    out.set("sample_ticks", std::move(ticks));
    Json neurons = Json::array();
    for (const std::string& id : result.neurons) {
        neurons.push(id);
    }
    out.set("neurons", std::move(neurons));
    const auto matrix = [](const std::vector<std::vector<double>>& rows) {
        Json json = Json::array();
        for (const auto& row : rows) {
            Json values = Json::array();
            for (const double value : row) {
                values.push(value);
            }
            json.push(std::move(values));
        }
        return json;
    };
    Json registers = Json::object();
    for (std::size_t r = 0; r < result.register_names.size(); ++r) {
        registers.set(result.register_names[r], matrix(result.registers[r]));
    }
    out.set("registers", std::move(registers));
    Json observation = Json::object();
    observation.set("operator", result.observation_operator);
    observation.set("register", result.observation_register);
    observation.set("values", matrix(result.observation));
    out.set("observation", std::move(observation));
    return out;
}

SimInput permute_neurons(const SimInput& input, const std::vector<std::size_t>& permutation) {
    SimInput out = input;
    out.graph.neurons.clear();
    for (const std::size_t old_index : permutation) {
        out.graph.neurons.push_back(input.graph.neurons.at(old_index));
    }
    return out;
}

}  // namespace occamworm
