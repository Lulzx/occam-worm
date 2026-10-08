#include "occamworm/search/enumerate.hpp"

#include <algorithm>
#include <set>
#include <unordered_set>

#include "occamworm/core/build_info.hpp"
#include "occamworm/core/error.hpp"
#include "occamworm/core/numfmt.hpp"
#include "occamworm/ir/compile.hpp"

namespace occamworm {
namespace {

[[noreturn]] void config_error(const std::string& message) { throw Error(Errc::Input, "enumeration config: " + message); }

struct Tree {
    std::string text;
    int depth = 1;
};

enum class Shape { Unary, Binary, Ternary, Leaky, Lut };

struct OpShape {
    Op op;
    Shape shape;
};

constexpr OpShape kInnerOps[] = {
    {Op::Neg, Shape::Unary},       {Op::Abs, Shape::Unary},        {Op::Relu, Shape::Unary},
    {Op::Tanh, Shape::Unary},      {Op::Sigmoid, Shape::Unary},    {Op::Add, Shape::Binary},
    {Op::Mul, Shape::Binary},      {Op::Min, Shape::Binary},       {Op::Max, Shape::Binary},
    {Op::Threshold, Shape::Binary}, {Op::Clamp, Shape::Ternary},   {Op::Select, Shape::Ternary},
    {Op::LeakyIntegrate, Shape::Leaky}, {Op::EulerLeak, Shape::Leaky}, {Op::Lut, Shape::Lut},
};

bool is_leaf_class(const std::string& name) {
    return name == "const" || name == "param" || name == "stimulus" || name == "sum_in" || name == "count_in" ||
           name == "delay" || name == "type_mask";
}

std::string register_name(int index) { return "r" + std::to_string(index); }

std::string constant_text(double value, const char* unit) { return canonical_number(value) + "[" + unit + "]"; }

// Every table of `size` entries over `values`, in lexicographic order of value indices.
std::vector<std::vector<double>> all_tables(int size, const std::vector<double>& values) {
    std::vector<std::vector<double>> tables;
    std::vector<std::size_t> digits(static_cast<std::size_t>(size), 0);
    while (true) {
        std::vector<double> table;
        for (const std::size_t d : digits) {
            table.push_back(values[d]);
        }
        tables.push_back(std::move(table));
        std::size_t position = digits.size();
        while (position > 0) {
            --position;
            if (++digits[position] < values.size()) {
                break;
            }
            digits[position] = 0;
            if (position == 0) {
                return tables;
            }
        }
    }
}

std::string table_text(const std::vector<double>& table) {
    std::string text = "[";
    for (std::size_t i = 0; i < table.size(); ++i) {
        text += (i > 0 ? ", " : "") + canonical_number(table[i]);
    }
    return text + "]";
}

class Enumerator {
public:
    explicit Enumerator(const EnumerationConfig& config) : config_(config) {
        for (const std::string& name : config.ops) {
            ops_.insert(name);
        }
    }

    EnumerationSummary run(const std::function<void(const EnumeratedProgram&)>& visit) {
        for (int registers = 1; registers <= config_.max_registers; ++registers) {
            build_trees(registers);
            std::vector<std::size_t> sizes(static_cast<std::size_t>(registers), 1);
            for (int total = registers; total <= config_.max_nodes; ++total) {
                visit_size_tuples(registers, total, 0, sizes, visit);
            }
        }
        return summary_;
    }

private:
    bool has(const char* name) const { return ops_.contains(name); }

    std::vector<Tree> leaf_trees(int registers) const {
        std::vector<Tree> leaves;
        for (int r = 0; r < registers; ++r) {
            leaves.push_back({register_name(r), 1});
        }
        if (has("stimulus")) {
            leaves.push_back({"stimulus", 1});
        }
        if (has("sum_in")) {
            for (int r = 0; r < registers; ++r) {
                for (const std::string& select : config_.sum_in_selects) {
                    leaves.push_back({"sum_in(" + register_name(r) + ", " + select + ")", 1});
                }
            }
        }
        if (has("count_in")) {
            for (int r = 0; r < registers; ++r) {
                for (const int k : config_.count_values) {
                    leaves.push_back({"count_in(" + register_name(r) + ", " + std::to_string(k) + ")", 1});
                }
            }
        }
        if (has("delay")) {
            for (int r = 0; r < registers; ++r) {
                for (const int ticks : config_.delays) {
                    leaves.push_back({"delay(" + register_name(r) + ", " + std::to_string(ticks) + ")", 1});
                }
            }
        }
        if (has("type_mask")) {
            for (const std::string& type : config_.type_names) {
                leaves.push_back({"type_mask(" + type + ")", 1});
            }
        }
        if (has("const")) {
            for (const double c : config_.constants) {
                leaves.push_back({constant_text(c, "1"), 1});
            }
        }
        if (has("param")) {
            for (std::size_t p = 0; p < config_.parameters.size(); ++p) {
                if (parse_unit(config_.parameters[p].unit) != Unit::time()) {  // time-unit parameters only serve as tau
                    leaves.push_back({"p" + std::to_string(p), 1});
                }
            }
        }
        return leaves;
    }

    std::vector<Tree> tau_trees() const {
        std::vector<Tree> taus;
        if (has("param")) {
            for (std::size_t p = 0; p < config_.parameters.size(); ++p) {
                if (parse_unit(config_.parameters[p].unit) == Unit::time()) {
                    taus.push_back({"p" + std::to_string(p), 1});
                }
            }
        }
        for (const double c : config_.tau_constants) {
            taus.push_back({constant_text(c, "s"), 1});
        }
        return taus;
    }

    // by_size_[n] = all expression trees with exactly n nodes and depth <= max_depth, in a fixed order.
    void build_trees(int registers) {
        by_size_.assign(static_cast<std::size_t>(config_.max_nodes) + 1, {});
        by_size_[1] = leaf_trees(registers);
        const std::vector<Tree> taus = tau_trees();
        std::vector<std::vector<double>> tables;
        if (has("lut")) {
            for (const int size : config_.lut.sizes) {
                for (auto& table : all_tables(size, config_.lut.values)) {
                    tables.push_back(std::move(table));
                }
            }
        }
        for (int n = 2; n <= config_.max_nodes; ++n) {
            auto& out = by_size_[static_cast<std::size_t>(n)];
            for (const OpShape& entry : kInnerOps) {
                if (!has(std::string(op_name(entry.op)).c_str())) {
                    continue;
                }
                const std::string name(op_name(entry.op));
                switch (entry.shape) {
                    case Shape::Unary:
                        for (const Tree& a : trees(n - 1)) {
                            push(out, name + "(" + a.text + ")", a.depth + 1);
                        }
                        break;
                    case Shape::Binary:
                        for (int na = 1; na <= n - 2; ++na) {
                            for (const Tree& a : trees(na)) {
                                for (const Tree& b : trees(n - 1 - na)) {
                                    // Commutative operators: one operand order is enough (the other canonicalises
                                    // to the same program), so skip the mirrored pair.
                                    if (op_is_commutative_nary(entry.op) && a.text > b.text) {
                                        continue;
                                    }
                                    push(out, name + "(" + a.text + ", " + b.text + ")", std::max(a.depth, b.depth) + 1);
                                }
                            }
                        }
                        break;
                    case Shape::Ternary:
                        for (int na = 1; na <= n - 3; ++na) {
                            for (int nb = 1; na + nb <= n - 2; ++nb) {
                                for (const Tree& a : trees(na)) {
                                    for (const Tree& b : trees(nb)) {
                                        for (const Tree& c : trees(n - 1 - na - nb)) {
                                            push(out, name + "(" + a.text + ", " + b.text + ", " + c.text + ")",
                                                 std::max({a.depth, b.depth, c.depth}) + 1);
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    case Shape::Leaky:
                        for (int na = 1; na <= n - 3; ++na) {
                            for (const Tree& a : trees(na)) {
                                for (const Tree& b : trees(n - 2 - na)) {
                                    for (const Tree& tau : taus) {
                                        push(out, name + "(" + a.text + ", " + b.text + ", " + tau.text + ")",
                                             std::max(a.depth, b.depth) + 1);
                                    }
                                }
                            }
                        }
                        break;
                    case Shape::Lut:
                        for (const Tree& a : trees(n - 1)) {
                            for (const auto& table : tables) {
                                push(out, "lut(" + a.text + ", " + table_text(table) + ")", a.depth + 1);
                            }
                        }
                        break;
                }
            }
        }
    }

    const std::vector<Tree>& trees(int n) const { return by_size_[static_cast<std::size_t>(n)]; }

    void push(std::vector<Tree>& out, std::string text, int depth) const {
        if (depth <= config_.max_depth) {
            out.push_back({std::move(text), depth});
        }
    }

    // Enumerates size tuples (n_0, ..., n_{R-1}) with the given total, then the cartesian product of trees.
    void visit_size_tuples(int registers, int remaining, int position, std::vector<std::size_t>& sizes,
                           const std::function<void(const EnumeratedProgram&)>& visit) {
        if (position == registers - 1) {
            sizes[static_cast<std::size_t>(position)] = static_cast<std::size_t>(remaining);
            if (remaining >= 1) {
                std::vector<std::string> updates(static_cast<std::size_t>(registers));
                visit_products(registers, 0, sizes, updates, visit);
            }
            return;
        }
        for (int n = 1; n <= remaining - (registers - 1 - position); ++n) {
            sizes[static_cast<std::size_t>(position)] = static_cast<std::size_t>(n);
            visit_size_tuples(registers, remaining - n, position + 1, sizes, visit);
        }
    }

    void visit_products(int registers, int position, const std::vector<std::size_t>& sizes, std::vector<std::string>& updates,
                        const std::function<void(const EnumeratedProgram&)>& visit) {
        if (position == registers) {
            attempt(registers, updates, visit);
            return;
        }
        for (const Tree& tree : trees(static_cast<int>(sizes[static_cast<std::size_t>(position)]))) {
            updates[static_cast<std::size_t>(position)] = tree.text;
            visit_products(registers, position + 1, sizes, updates, visit);
        }
    }

    void attempt(int registers, const std::vector<std::string>& updates, const std::function<void(const EnumeratedProgram&)>& visit) {
        ++summary_.generated;
        const auto compiled = try_compile(assemble_program_source(config_, registers, updates));
        if (!compiled) {
            if (compiled.error().code() == Errc::Stability) {
                ++summary_.rejected_stability;
            } else {
                ++summary_.rejected_type;
            }
            return;
        }
        const Ir& ir = compiled.value();
        if (static_cast<int>(ir.registers.size()) > config_.max_registers || static_cast<int>(ir.params.size()) > config_.max_parameters) {
            ++summary_.rejected_budget;
            return;
        }
        if (!seen_.insert(ir.program_hash).second) {
            ++summary_.duplicates;
            return;
        }
        EnumeratedProgram program;
        program.index = summary_.unique++;
        program.ir = ir;
        visit(program);
    }

    const EnumerationConfig& config_;
    std::set<std::string> ops_;
    std::vector<std::vector<Tree>> by_size_;
    std::unordered_set<std::string> seen_;  // membership only; never iterated, so order cannot leak into output
    EnumerationSummary summary_;
};

}  // namespace

std::string assemble_program_source(const EnumerationConfig& config, int registers, const std::vector<std::string>& updates) {
    std::string source = "wrl 0.1\ntier " + std::string(tier_name(config.tier)) + "\n";
    if (config.dt_max) {
        source += "dt_max " + canonical_number(*config.dt_max) + "\n";
    }
    for (int r = 0; r < registers; ++r) {
        const double init = static_cast<std::size_t>(r) < config.register_init.size() ? config.register_init[static_cast<std::size_t>(r)] : 0.0;
        source += "state " + register_name(r) + " : 1 = " + canonical_number(init) + "\n";
    }
    for (std::size_t p = 0; p < config.parameters.size(); ++p) {
        const ParameterTemplate& t = config.parameters[p];
        source += "param p" + std::to_string(p) + " : " + t.unit + " in [" + canonical_number(t.lower) + ", " + canonical_number(t.upper) +
                  "] trainable bits " + std::to_string(t.bits) + "\n";
    }
    for (int r = 0; r < registers; ++r) {
        source += "next " + register_name(r) + " = " + updates[static_cast<std::size_t>(r)] + "\n";
    }
    source += "observe identity_v1(r0)\n";
    return source;
}

EnumerationConfig enumeration_config_from_json(const Json& json) {
    EnumerationConfig config;
    try {
        config.name = json.at("name").as_string();
        const auto tier = parse_tier(json.at("tier").as_string());
        if (!tier) {
            config_error("unknown tier");
        }
        config.tier = *tier;
        config.max_nodes = static_cast<int>(json.at("max_nodes").as_int());
        config.max_depth = static_cast<int>(json.at("max_depth").as_int());
        config.max_registers = static_cast<int>(json.at("max_registers").as_int());
        config.max_parameters = static_cast<int>(json.at("max_parameters").as_int());
        if (config.max_nodes < 1 || config.max_nodes > 12 || config.max_depth < 1 || config.max_registers < 1 || config.max_registers > 3 ||
            config.max_parameters < 0) {
            config_error("budgets out of range (max_nodes 1..12, max_depth >= 1, max_registers 1..3, max_parameters >= 0)");
        }
        for (const Json& item : json.at("ops").as_array()) {
            const std::string name = item.as_string();
            const auto op = parse_op_name(name);
            const bool inner = std::ranges::any_of(kInnerOps, [&](const OpShape& entry) { return entry.op == op; });
            if (!is_leaf_class(name) && !inner) {
                config_error("unknown or non-enumerable op '" + name + "'");
            }
            if (op && !op_allowed_in_tier(config.tier, *op)) {
                config_error("op '" + name + "' is not allowed in tier " + std::string(tier_name(config.tier)));
            }
            if (name == "const" || name == "param") {
                if (name == "param" && config.tier == Tier::G0) {
                    config_error("G0 has no parameters");
                }
            }
            config.ops.push_back(name);
        }
        const auto numbers = [&](const char* key, std::vector<double>& out) {
            if (const Json* list = json.find(key)) {
                for (const Json& item : list->as_array()) {
                    out.push_back(item.as_double());
                }
            }
        };
        numbers("constants", config.constants);
        numbers("tau_constants", config.tau_constants);
        numbers("register_init", config.register_init);
        if (const Json* list = json.find("sum_in_selects")) {
            for (const Json& item : list->as_array()) {
                if (!parse_select(item.as_string())) {
                    config_error("unknown sum_in selector '" + item.as_string() + "'");
                }
                config.sum_in_selects.push_back(item.as_string());
            }
        }
        if (const Json* list = json.find("count_values")) {
            for (const Json& item : list->as_array()) {
                config.count_values.push_back(static_cast<int>(item.as_int()));
            }
        }
        if (const Json* list = json.find("delays")) {
            for (const Json& item : list->as_array()) {
                config.delays.push_back(static_cast<int>(item.as_int()));
            }
        }
        if (const Json* list = json.find("type_names")) {
            for (const Json& item : list->as_array()) {
                config.type_names.push_back(item.as_string());
            }
        }
        if (const Json* lut = json.find("lut")) {
            for (const Json& item : lut->at("sizes").as_array()) {
                const long long size = item.as_int();
                if (size < 1 || size > 8) {
                    config_error("lut sizes must be in 1..8");
                }
                config.lut.sizes.push_back(static_cast<int>(size));
            }
            for (const Json& item : lut->at("values").as_array()) {
                config.lut.values.push_back(item.as_double());
            }
            if (config.lut.values.empty() || config.lut.values.size() > 4) {
                config_error("lut values must list 1..4 entries");
            }
        }
        if (const Json* list = json.find("parameters")) {
            for (const Json& item : list->as_array()) {
                ParameterTemplate t;
                t.unit = item.at("unit").as_string();
                t.lower = item.at("lower").as_double();
                t.upper = item.at("upper").as_double();
                t.bits = static_cast<int>(item.at("bits").as_int());
                config.parameters.push_back(std::move(t));
            }
        }
        if (const Json* dt = json.find("dt_max")) {
            config.dt_max = dt->as_double();
        }
    } catch (const Error& error) {
        if (error.code() == Errc::Json) {
            throw Error(Errc::Input, "enumeration config: " + error.message());
        }
        throw;
    }
    return config;
}

EnumerationSummary enumerate_programs(const EnumerationConfig& config, const std::function<void(const EnumeratedProgram&)>& visit) {
    return Enumerator(config).run(visit);
}

Json program_record_json(const EnumeratedProgram& program) {
    Json out = Json::object();
    out.set("type", "program");
    out.set("index", program.index);
    out.set("hash", program.ir.program_hash);
    out.set("l_struct", program.ir.bits.l_struct);
    out.set("l_params", program.ir.bits.l_params);
    out.set("source", program.ir.canonical_source);
    return out;
}

Json summary_record_json(const EnumerationConfig& config, const EnumerationSummary& summary) {
    Json out = Json::object();
    out.set("type", "summary");
    out.set("config", config.name);
    out.set("grammar_version", std::string(kGrammarVersion));
    out.set("compiler_build", OW_BUILD_STRING);
    out.set("generated", summary.generated);
    Json rejected = Json::object();
    rejected.set("type_or_unit", summary.rejected_type);
    rejected.set("stability", summary.rejected_stability);
    rejected.set("budget", summary.rejected_budget);
    out.set("rejected", std::move(rejected));
    out.set("duplicates", summary.duplicates);
    out.set("unique", summary.unique);
    return out;
}

}  // namespace occamworm
