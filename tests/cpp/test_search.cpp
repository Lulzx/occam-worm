// ow-search: OW-010 done-when. Every canonical program inside a small grammar budget is visited exactly once.
//
// The check compares the enumerator with an INDEPENDENT brute force. The enumerator builds expression trees
// by dynamic programming over node counts; the brute force below enumerates raw prefix-notation token
// sequences with a depth-first search over open argument slots, renders each to WRL source, and runs it
// through the real parser, checker and canonicaliser. The two sets of canonical hashes must be equal and
// the enumerator must emit no hash twice.

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "occamworm/core/numfmt.hpp"
#include "occamworm/ir/compile.hpp"
#include "occamworm/search/enumerate.hpp"
#include "ow_test.hpp"

using namespace occamworm;

namespace {

struct Token {
    std::string text;       // leaf text, or operator name
    int arity = 0;
    bool last_is_tau = false;  // the last argument must be a time-constant operand
    std::string table;      // lut: appended as the final argument
};

struct RawTree {
    std::string text;
    int nodes = 0;
    int depth = 0;
};

bool has_op(const EnumerationConfig& config, const std::string& name) {
    return std::ranges::find(config.ops, name) != config.ops.end();
}

std::vector<Token> generic_leaves(const EnumerationConfig& config, int registers) {
    std::vector<Token> leaves;
    for (int r = 0; r < registers; ++r) {
        leaves.push_back({"r" + std::to_string(r), 0, false, {}});
    }
    if (has_op(config, "stimulus")) {
        leaves.push_back({"stimulus", 0, false, {}});
    }
    for (int r = 0; r < registers; ++r) {
        if (has_op(config, "sum_in")) {
            for (const std::string& s : config.sum_in_selects) {
                leaves.push_back({"sum_in(r" + std::to_string(r) + ", " + s + ")", 0, false, {}});
            }
        }
        if (has_op(config, "count_in")) {
            for (const int k : config.count_values) {
                leaves.push_back({"count_in(r" + std::to_string(r) + ", " + std::to_string(k) + ")", 0, false, {}});
            }
        }
        if (has_op(config, "delay")) {
            for (const int n : config.delays) {
                leaves.push_back({"delay(r" + std::to_string(r) + ", " + std::to_string(n) + ")", 0, false, {}});
            }
        }
    }
    if (has_op(config, "type_mask")) {
        for (const std::string& type : config.type_names) {
            leaves.push_back({"type_mask(" + type + ")", 0, false, {}});
        }
    }
    if (has_op(config, "const")) {
        for (const double c : config.constants) {
            leaves.push_back({canonical_number(c) + "[1]", 0, false, {}});
        }
    }
    if (has_op(config, "param")) {
        for (std::size_t p = 0; p < config.parameters.size(); ++p) {
            if (config.parameters[p].unit != "s") {
                leaves.push_back({"p" + std::to_string(p), 0, false, {}});
            }
        }
    }
    return leaves;
}

std::vector<Token> tau_leaves(const EnumerationConfig& config) {
    std::vector<Token> leaves;
    if (has_op(config, "param")) {
        for (std::size_t p = 0; p < config.parameters.size(); ++p) {
            if (config.parameters[p].unit == "s") {
                leaves.push_back({"p" + std::to_string(p), 0, false, {}});
            }
        }
    }
    for (const double c : config.tau_constants) {
        leaves.push_back({canonical_number(c) + "[s]", 0, false, {}});
    }
    return leaves;
}

std::vector<Token> operator_tokens(const EnumerationConfig& config) {
    static const std::map<std::string, std::pair<int, bool>> arity = {
        {"neg", {1, false}}, {"abs", {1, false}}, {"relu", {1, false}}, {"tanh", {1, false}}, {"sigmoid", {1, false}},
        {"add", {2, false}}, {"mul", {2, false}}, {"min", {2, false}}, {"max", {2, false}}, {"threshold", {2, false}},
        {"clamp", {3, false}}, {"select", {3, false}}, {"leaky_integrate", {3, true}}, {"euler_leak", {3, true}}};
    std::vector<Token> tokens;
    for (const std::string& name : config.ops) {
        const auto it = arity.find(name);
        if (it != arity.end()) {
            tokens.push_back({name, it->second.first, it->second.second, {}});
        }
    }
    if (has_op(config, "lut")) {
        for (const int size : config.lut.sizes) {
            // Independent table enumeration: counting in base |values| with the first entry most significant.
            std::vector<std::size_t> digits(static_cast<std::size_t>(size), 0);
            const std::size_t base = config.lut.values.size();
            std::size_t total = 1;
            for (int i = 0; i < size; ++i) {
                total *= base;
            }
            for (std::size_t code = 0; code < total; ++code) {
                std::size_t rest = code;
                for (int i = size - 1; i >= 0; --i) {
                    digits[static_cast<std::size_t>(i)] = rest % base;
                    rest /= base;
                }
                std::string table = "[";
                for (int i = 0; i < size; ++i) {
                    table += (i > 0 ? ", " : "") + canonical_number(config.lut.values[digits[static_cast<std::size_t>(i)]]);
                }
                tokens.push_back({"lut", 1, false, table + "]"});
            }
        }
    }
    return tokens;
}

// Slot kinds on the pending stack: 'n' = any expression, 't' = time-constant operand.
void extend(const EnumerationConfig& config, const std::vector<Token>& generic, const std::vector<Token>& taus,
            const std::vector<Token>& operators, std::vector<Token>& sequence, std::vector<char>& slots,
            std::vector<RawTree>& completed) {
    if (slots.empty()) {
        // Render the prefix sequence and measure its depth.
        std::size_t position = 0;
        const auto render = [&](auto&& self, int& depth) -> std::string {
            const Token& token = sequence[position++];
            if (token.arity == 0) {
                depth = 1;
                return token.text;
            }
            std::string text = token.text + "(";
            int deepest = 0;
            for (int i = 0; i < token.arity; ++i) {
                int child_depth = 0;
                text += (i > 0 ? ", " : "") + self(self, child_depth);
                deepest = std::max(deepest, child_depth);
            }
            if (!token.table.empty()) {
                text += ", " + token.table;
            }
            depth = deepest + 1;
            return text + ")";
        };
        int depth = 0;
        const std::string text = render(render, depth);
        completed.push_back({text, static_cast<int>(sequence.size()), depth});
        return;
    }
    const int remaining_budget = config.max_nodes - static_cast<int>(sequence.size());
    if (remaining_budget < static_cast<int>(slots.size())) {
        return;  // not enough room to fill every open slot
    }
    const char kind = slots.back();
    const auto choose = [&](const Token& token) {
        slots.pop_back();
        sequence.push_back(token);
        // Children are consumed first-argument-first, so push their slots in reverse.
        if (token.arity > 0) {
            for (int i = token.arity - 1; i >= 0; --i) {
                slots.push_back((token.last_is_tau && i == token.arity - 1) ? 't' : 'n');
            }
        }
        extend(config, generic, taus, operators, sequence, slots, completed);
        if (token.arity > 0) {
            slots.resize(slots.size() - static_cast<std::size_t>(token.arity));
        }
        sequence.pop_back();
        slots.push_back(kind);
    };
    if (kind == 't') {
        for (const Token& token : taus) {
            choose(token);
        }
        return;
    }
    for (const Token& token : generic) {
        choose(token);
    }
    for (const Token& token : operators) {
        choose(token);
    }
}

// Canonical hashes of every program the brute force can build inside the budget, plus the raw count.
std::set<std::string> brute_force_hashes(const EnumerationConfig& config, long long& raw_count) {
    std::set<std::string> hashes;
    raw_count = 0;
    for (int registers = 1; registers <= config.max_registers; ++registers) {
        const std::vector<Token> generic = generic_leaves(config, registers);
        const std::vector<Token> taus = tau_leaves(config);
        const std::vector<Token> operators = operator_tokens(config);
        std::vector<RawTree> trees;
        std::vector<Token> sequence;
        std::vector<char> slots = {'n'};
        extend(config, generic, taus, operators, sequence, slots, trees);
        trees.erase(std::remove_if(trees.begin(), trees.end(), [&](const RawTree& t) { return t.depth > config.max_depth; }), trees.end());

        // Assign one tree to every register, keeping the total node count inside the budget.
        std::vector<std::size_t> choice(static_cast<std::size_t>(registers), 0);
        const auto visit_all = [&](auto&& self, int position, int nodes_used, std::vector<std::string>& updates) -> void {
            if (position == registers) {
                ++raw_count;
                const auto compiled = try_compile(assemble_program_source(config, registers, updates));
                if (compiled && static_cast<int>(compiled->registers.size()) <= config.max_registers &&
                    static_cast<int>(compiled->params.size()) <= config.max_parameters) {
                    hashes.insert(compiled->program_hash);
                }
                return;
            }
            for (const RawTree& tree : trees) {
                if (nodes_used + tree.nodes + (registers - position - 1) > config.max_nodes) {
                    continue;
                }
                updates[static_cast<std::size_t>(position)] = tree.text;
                self(self, position + 1, nodes_used + tree.nodes, updates);
            }
        };
        std::vector<std::string> updates(static_cast<std::size_t>(registers));
        visit_all(visit_all, 0, 0, updates);
    }
    return hashes;
}

EnumerationConfig load_config(const std::string& relative_path) {
    return enumeration_config_from_json(Json::parse(read_text_file(std::string(OW_SOURCE_DIR) + "/" + relative_path)));
}

struct EnumerationRun {
    EnumerationSummary summary;
    std::vector<std::string> hashes;
    std::vector<std::string> sources;
};

EnumerationRun run_enumeration(const EnumerationConfig& config) {
    EnumerationRun run;
    run.summary = enumerate_programs(config, [&](const EnumeratedProgram& program) {
        OW_CHECK_EQ(program.index, static_cast<long long>(run.hashes.size()));
        run.hashes.push_back(program.ir.program_hash);
        run.sources.push_back(program.ir.canonical_source);
    });
    return run;
}

// The done-when check for one configuration.
void check_exactly_once(const EnumerationConfig& config) {
    const EnumerationRun run = run_enumeration(config);
    const std::set<std::string> visited(run.hashes.begin(), run.hashes.end());
    OW_CHECK_EQ(visited.size(), run.hashes.size());  // no canonical program is emitted twice
    long long raw = 0;
    const std::set<std::string> expected = brute_force_hashes(config, raw);
    OW_CHECK_EQ(visited.size(), expected.size());
    OW_CHECK(visited == expected);  // and none is missed
    OW_CHECK_EQ(run.summary.unique, static_cast<long long>(run.hashes.size()));
    OW_CHECK(raw >= run.summary.generated);  // the enumerator prunes mirrored commutative operands, never more
    OW_CHECK(static_cast<long long>(expected.size()) < raw);  // brute force really contains duplicates
    std::cout << "    " << config.name << ": brute-force candidates " << raw << ", enumerator candidates " << run.summary.generated
              << ", canonical programs " << run.hashes.size() << " (duplicates " << run.summary.duplicates << ", stability rejects "
              << run.summary.rejected_stability << ", type/unit rejects " << run.summary.rejected_type << ")\n";
}

}  // namespace

OW_TEST(enumeration_g0_tiny_visits_every_canonical_program_once) { check_exactly_once(load_config("configs/searches/g0-tiny.json")); }

OW_TEST(enumeration_g1_tiny_visits_every_canonical_program_once) {
    const EnumerationConfig config = load_config("configs/searches/g1-tiny.json");
    check_exactly_once(config);
    const EnumerationRun run = run_enumeration(config);
    OW_CHECK(run.summary.rejected_stability > 0);  // the unstable tau bound is filtered, not enumerated
}

OW_TEST(enumeration_two_registers_and_unit_rejections) {
    const Json json = Json::parse(R"({
      "name": "g1-two-registers", "tier": "G1", "max_nodes": 5, "max_depth": 3, "max_registers": 2, "max_parameters": 2,
      "ops": ["add", "mul", "tanh", "neg", "delay", "stimulus", "param", "leaky_integrate"],
      "delays": [1],
      "parameters": [ {"unit": "V", "lower": 0, "upper": 1, "bits": 4}, {"unit": "1", "lower": 0, "upper": 1, "bits": 4},
                      {"unit": "s", "lower": 0.1, "upper": 2, "bits": 4} ],
      "register_init": [0, 1]
    })");
    const EnumerationConfig config = enumeration_config_from_json(json);
    check_exactly_once(config);
    const EnumerationRun run = run_enumeration(config);
    OW_CHECK(run.summary.rejected_type > 0);  // adding a volt-valued parameter to a dimensionless register is a unit error
}

OW_TEST(enumeration_is_deterministic_and_emits_reparseable_canonical_source) {
    const EnumerationConfig config = load_config("configs/searches/g1-tiny.json");
    const EnumerationRun first = run_enumeration(config);
    const EnumerationRun second = run_enumeration(config);
    OW_CHECK(first.hashes == second.hashes);
    OW_CHECK(first.sources == second.sources);
    for (std::size_t i = 0; i < first.sources.size(); i += 97) {  // a spread of programs
        OW_CHECK_EQ(compile(first.sources[i]).program_hash, first.hashes[i]);
    }
}

OW_TEST(enumeration_budget_and_allowlist_are_respected) {
    EnumerationConfig config = load_config("configs/searches/g1-tiny.json");
    config.max_nodes = 3;
    config.ops = {"add", "stimulus"};
    const EnumerationRun run = run_enumeration(config);
    // Trees over {r0, stimulus} with add: 2 leaves; add(a,b) with a<=b: 3 pairs; sizes 1 and 3 only.
    // Canonical classes: r0, stimulus, r0+stimulus, r0+r0, stimulus+stimulus.
    OW_CHECK_EQ(run.hashes.size(), std::size_t{5});
    for (const std::string& source : run.sources) {
        OW_CHECK(source.find("tanh") == std::string::npos);
    }
}

OW_TEST(enumeration_config_validation) {
    OW_CHECK_THROWS(enumeration_config_from_json(Json::parse(R"({"name":"x","tier":"G7","max_nodes":3,"max_depth":2,"max_registers":1,"max_parameters":0,"ops":[]})")), Errc::Input);
    OW_CHECK_THROWS(enumeration_config_from_json(Json::parse(R"({"name":"x","tier":"G1","max_nodes":3,"max_depth":2,"max_registers":1,"max_parameters":0,"ops":["lut"]})")), Errc::Input);
    OW_CHECK_THROWS(enumeration_config_from_json(Json::parse(R"({"name":"x","tier":"G1","max_nodes":99,"max_depth":2,"max_registers":1,"max_parameters":0,"ops":[]})")), Errc::Input);
    OW_CHECK_THROWS(enumeration_config_from_json(Json::parse(R"({"name":"x","tier":"G1"})")), Errc::Input);
}
