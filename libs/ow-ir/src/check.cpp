#include "occamworm/ir/check.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include "occamworm/core/error.hpp"
#include "occamworm/core/numfmt.hpp"
#include "occamworm/ir/parser.hpp"

namespace occamworm {
namespace {

[[noreturn]] void fail(Errc code, const Expr& at, const std::string& message) {
    throw Error(code, message, at.line, at.column);
}

[[noreturn]] void fail(Errc code, const Statement& at, const std::string& message) {
    throw Error(code, message, at.line, at.column);
}

const std::set<std::string, std::less<>>& reserved_names() {
    static const std::set<std::string, std::less<>> names = {
        "wrl", "rule", "tier", "dt_max", "stimulus_unit", "state", "param", "input", "let", "next", "gap",
        "observe", "in", "scale", "bits", "trainable", "fixed", "sub", "decay", "dimensionless"};
    return names;
}

bool is_reserved(const std::string& name) { return reserved_names().contains(name) || parse_op_name(name).has_value(); }

class Checker {
public:
    explicit Checker(const ProgramAst& program) : program_(program) {
        out_.rule_name = program.rule_name;
        out_.tier = program.tier;
        out_.dt_max = program.dt_max;
        out_.stimulus_unit = program.stimulus_unit;
        if (program.tier == Tier::G0 && !program.stimulus_unit.dimensionless()) {
            throw Error(Errc::Tier, "G0 programs are dimensionless: stimulus_unit must be 1");
        }
    }

    Resolved run() {
        for (const Statement& statement : program_.statements) {
            switch (statement.kind) {
                case Statement::Kind::State: declare_register(statement); break;
                case Statement::Kind::Param: declare_param(statement); break;
                case Statement::Kind::Input:
                case Statement::Kind::Let: declare_binding(statement); break;
                case Statement::Kind::Next: declare_write(statement); break;
                case Statement::Kind::Gap: declare_gap(statement); break;
                case Statement::Kind::Observe: declare_observation(statement); break;
            }
        }
        if (!observed_) {
            throw Error(Errc::Name, "missing 'observe' declaration");
        }
        out_.writes.assign(out_.registers.size(), -1);
        for (std::size_t r = 0; r < out_.registers.size(); ++r) {
            out_.writes[r] = write_node_[r] >= 0 ? write_node_[r] : state_node(static_cast<int>(r));
        }
        return std::move(out_);
    }

private:
    enum class EntityKind { Register, Param, Binding };
    struct Entity {
        EntityKind kind;
        int index;
    };

    // ---- declarations ----------------------------------------------------------------------------

    void define(const Statement& at, const std::string& name, Entity entity) {
        if (is_reserved(name) || name == "stimulus") {
            fail(Errc::Name, at, "'" + name + "' is a reserved word");
        }
        if (!scope_.emplace(name, entity).second) {
            fail(Errc::Name, at, "'" + name + "' is already defined");
        }
    }

    void declare_register(const Statement& statement) {
        const RegisterDecl& decl = statement.reg;
        if (program_.tier == Tier::G0) {
            if (!decl.unit.dimensionless()) {
                fail(Errc::Tier, statement, "G0 registers are dimensionless");
            }
            if (decl.init != std::floor(decl.init)) {
                fail(Errc::Tier, statement, "G0 registers have integer initial values");
            }
        }
        define(statement, statement.name, {EntityKind::Register, static_cast<int>(out_.registers.size())});
        out_.registers.push_back(decl);
        write_node_.push_back(-1);
        state_nodes_.push_back(-1);
    }

    void declare_param(const Statement& statement) {
        if (program_.tier == Tier::G0) {
            fail(Errc::Tier, statement, "G0 programs have no parameters");
        }
        const ParamDecl& decl = statement.param;
        IrParam param;
        param.name = decl.name;
        param.source_name = decl.name;
        param.unit = decl.unit;
        param.trainable = decl.trainable;
        param.bits = decl.bits;
        if (decl.trainable) {
            if (!decl.lower || !decl.upper) {
                fail(Errc::Type, statement, "a trainable parameter needs bounds: 'in [lower, upper]'");
            }
            param.lower = *decl.lower;
            param.upper = *decl.upper;
            if (!(param.lower < param.upper)) {
                fail(Errc::Type, statement, "parameter bounds must satisfy lower < upper");
            }
            param.value = decl.value ? *decl.value : 0.5 * (param.lower + param.upper);
        } else {
            if (!decl.value) {
                fail(Errc::Type, statement, "a fixed parameter needs a value: '= <number>'");
            }
            if (decl.lower || decl.upper) {
                fail(Errc::Type, statement, "a fixed parameter has no bounds");
            }
            param.value = *decl.value;
            param.lower = param.upper = param.value;
        }
        if (param.value < param.lower || param.value > param.upper) {
            fail(Errc::Type, statement, "parameter value lies outside its bounds");
        }
        define(statement, statement.name, {EntityKind::Param, static_cast<int>(out_.params.size())});
        out_.params.push_back(std::move(param));
        param_nodes_.push_back(-1);
    }

    void declare_binding(const Statement& statement) {
        if (statement.kind == Statement::Kind::Input && !is_input_form(*statement.expr)) {
            fail(Errc::Type, statement,
                 "an 'input' must be stimulus, sum_in, count_in, type_mask or delay; use 'let' for other expressions");
        }
        const int node = eval(*statement.expr);
        define(statement, statement.name, {EntityKind::Binding, node});
    }

    static bool is_input_form(const Expr& expr) {
        if (expr.kind == Expr::Kind::Name) {
            return expr.name == "stimulus";
        }
        if (expr.kind != Expr::Kind::Call) {
            return false;
        }
        return expr.name == "stimulus" || expr.name == "sum_in" || expr.name == "count_in" || expr.name == "type_mask" ||
               expr.name == "delay";
    }

    void declare_write(const Statement& statement) {
        const auto it = scope_.find(statement.name);
        if (it == scope_.end() || it->second.kind != EntityKind::Register) {
            fail(Errc::Name, statement, "'next " + statement.name + "' does not name a state register");
        }
        const int reg = it->second.index;
        if (write_node_[static_cast<std::size_t>(reg)] >= 0) {
            fail(Errc::Name, statement, "register '" + statement.name + "' is written twice");
        }
        int node = eval(*statement.expr);
        node = coerce(node, out_.registers[static_cast<std::size_t>(reg)].unit, *statement.expr, "next " + statement.name);
        write_node_[static_cast<std::size_t>(reg)] = node;
    }

    void declare_gap(const Statement& statement) {
        if (program_.tier != Tier::G1) {
            fail(Errc::Tier, statement, "gap coupling is only available in tier G1");
        }
        if (out_.gap) {
            fail(Errc::Name, statement, "only one 'gap' declaration is allowed");
        }
        IrGap gap;
        gap.reg = lookup_register(statement, statement.name);
        if (!statement.gap_scale.empty()) {
            const auto it = scope_.find(statement.gap_scale);
            if (it == scope_.end() || it->second.kind != EntityKind::Param) {
                fail(Errc::Name, statement, "gap scale '" + statement.gap_scale + "' is not a parameter");
            }
            const IrParam& scale = out_.params[static_cast<std::size_t>(it->second.index)];
            if (!scale.unit.dimensionless()) {
                fail(Errc::Unit, statement, "gap scale must be dimensionless");
            }
            if (scale.lower < 0.0) {
                fail(Errc::Stability, statement, "gap scale must be non-negative over its bounds (conductance magnitudes, §5.8)");
            }
            gap.scale_param = it->second.index;
        }
        out_.gap = gap;
    }

    void declare_observation(const Statement& statement) {
        if (observed_) {
            fail(Errc::Name, statement, "only one 'observe' declaration is allowed");
        }
        const ObserveDecl& decl = statement.observe;
        const auto op = parse_obs_operator(decl.operator_name);
        if (!op) {
            fail(Errc::Name, statement, "unknown observation operator '" + decl.operator_name + "'");
        }
        ResolvedObservation obs;
        obs.op = *op;
        obs.reg = lookup_register(statement, decl.register_name);
        if (*op == ObsOperator::CalciumLinearV1) {
            if (program_.tier != Tier::G1) {
                fail(Errc::Tier, statement, "calcium_linear_v1 is only available in tier G1");
            }
            if (!decl.tau) {
                fail(Errc::Type, statement, "calcium_linear_v1 needs a time constant: calcium_linear_v1(reg, tau)");
            }
            obs.tau = resolve_tau_operand(*decl.tau, /*euler=*/false, "calcium time constant");
        } else if (decl.tau) {
            fail(Errc::Type, *decl.tau, "identity_v1 takes no time constant");
        }
        out_.observation = obs;
        observed_ = true;
    }

    int lookup_register(const Statement& at, const std::string& name) {
        const auto it = scope_.find(name);
        if (it == scope_.end() || it->second.kind != EntityKind::Register) {
            fail(Errc::Name, at, "'" + name + "' is not a state register");
        }
        return it->second.index;
    }

    // ---- node construction -----------------------------------------------------------------------

    int add_node(Instr instr, const Expr& at) {
        if (!op_allowed_in_tier(program_.tier, instr.op)) {
            fail(Errc::Tier, at,
                 "operator '" + std::string(op_name(instr.op)) + "' is not allowed in tier " + std::string(tier_name(program_.tier)));
        }
        if (out_.raw.size() >= kMaxRawNodes) {
            throw Error(Errc::Limit, "program exceeds " + std::to_string(kMaxRawNodes) + " expression nodes");
        }
        out_.raw.push_back(std::move(instr));
        return static_cast<int>(out_.raw.size()) - 1;
    }

    const Instr& node(int id) const { return out_.raw[static_cast<std::size_t>(id)]; }

    int state_node(int reg) {
        int& cached = state_nodes_[static_cast<std::size_t>(reg)];
        if (cached < 0) {
            Instr instr;
            instr.op = Op::State;
            instr.index = reg;
            instr.unit = out_.registers[static_cast<std::size_t>(reg)].unit;
            if (out_.raw.size() >= kMaxRawNodes) {
                throw Error(Errc::Limit, "program exceeds the expression node limit");
            }
            out_.raw.push_back(std::move(instr));
            cached = static_cast<int>(out_.raw.size()) - 1;
        }
        return cached;
    }

    int param_node(int param, const Expr& at) {
        int& cached = param_nodes_[static_cast<std::size_t>(param)];
        if (cached < 0) {
            Instr instr;
            instr.op = Op::Param;
            instr.index = param;
            instr.unit = out_.params[static_cast<std::size_t>(param)].unit;
            cached = add_node(std::move(instr), at);
        }
        return cached;
    }

    // ---- unit helpers ----------------------------------------------------------------------------

    std::string unit_text(Unit unit) const { return "[" + to_string(unit) + "]"; }

    // Returns a node of exactly `unit`: weak zeros adopt it; anything else must already match.
    int coerce(int id, Unit unit, const Expr& at, const std::string& context) {
        const Instr& source = node(id);
        if (source.weak_zero) {
            Instr zero;
            zero.op = Op::Const;
            zero.value = 0.0;
            zero.unit = unit;
            return add_node(std::move(zero), at);
        }
        if (source.unit != unit) {
            fail(Errc::Unit, at,
                 "unit mismatch in " + context + ": expected " + unit_text(unit) + " but found " + unit_text(source.unit));
        }
        return id;
    }

    // Makes every argument carry one common unit and returns it.
    Unit unify(std::vector<int>& ids, const Expr& at, const std::string& context) {
        Unit common;
        bool found = false;
        for (const int id : ids) {
            if (!node(id).weak_zero) {
                common = node(id).unit;
                found = true;
                break;
            }
        }
        if (!found) {
            common = Unit::none();
        }
        for (int& id : ids) {
            id = coerce(id, common, at, context);
        }
        return common;
    }

    // ---- expressions -----------------------------------------------------------------------------

    int eval(const Expr& expr) {
        switch (expr.kind) {
            case Expr::Kind::Number: return eval_number(expr);
            case Expr::Kind::Name: return eval_name(expr);
            case Expr::Kind::Call: return eval_call(expr);
            case Expr::Kind::List: fail(Errc::Type, expr, "a [list] is only valid as the table argument of lut");
        }
        fail(Errc::Type, expr, "invalid expression");
    }

    int eval_number(const Expr& expr) {
        Instr instr;
        instr.op = Op::Const;
        instr.value = expr.number == 0.0 ? 0.0 : expr.number;  // normalise -0.0
        instr.unit = expr.unit.value_or(Unit::none());
        instr.weak_zero = !expr.unit.has_value() && expr.number == 0.0;
        if (program_.tier == Tier::G0) {
            if (expr.number != std::floor(expr.number)) {
                fail(Errc::Tier, expr, "G0 programs may only use integer constants");
            }
            if (!instr.unit.dimensionless()) {
                fail(Errc::Tier, expr, "G0 programs are dimensionless");
            }
        }
        return add_node(std::move(instr), expr);
    }

    int eval_name(const Expr& expr) {
        if (expr.name == "stimulus") {
            return stimulus_node(expr);
        }
        const auto it = scope_.find(expr.name);
        if (it == scope_.end()) {
            fail(Errc::Name, expr, "unknown name '" + expr.name + "'");
        }
        switch (it->second.kind) {
            case EntityKind::Register: return state_node(it->second.index);
            case EntityKind::Param: return param_node(it->second.index, expr);
            case EntityKind::Binding: return it->second.index;
        }
        fail(Errc::Name, expr, "unknown name '" + expr.name + "'");
    }

    int stimulus_node(const Expr& at) {
        if (stimulus_node_ < 0) {
            Instr instr;
            instr.op = Op::Stimulus;
            instr.unit = out_.stimulus_unit;
            stimulus_node_ = add_node(std::move(instr), at);
        }
        return stimulus_node_;
    }

    static void expect_arity(const Expr& call, std::size_t count) {
        if (call.args.size() != count) {
            fail(Errc::Type, call,
                 call.name + " takes " + std::to_string(count) + " argument" + (count == 1 ? "" : "s") + ", got " +
                     std::to_string(call.args.size()));
        }
    }

    std::vector<int> eval_args(const Expr& call) {
        std::vector<int> ids;
        for (const ExprPtr& arg : call.args) {
            ids.push_back(eval(*arg));
        }
        return ids;
    }

    // Argument that must be a bare word (register name, selector, type name).
    static const std::string& word_arg(const Expr& call, std::size_t position, const char* what) {
        const Expr& arg = *call.args[position];
        if (arg.kind != Expr::Kind::Name) {
            fail(Errc::Type, arg, std::string(call.name) + ": expected " + what);
        }
        return arg.name;
    }

    // Argument that must be a non-negative integer literal.
    static int count_arg(const Expr& call, std::size_t position, const char* what, int maximum) {
        const Expr& arg = *call.args[position];
        if (arg.kind != Expr::Kind::Number || arg.unit || arg.number != std::floor(arg.number) || arg.number < 0 ||
            arg.number > maximum) {
            fail(Errc::Type, arg, std::string(call.name) + ": " + what + " must be an integer literal in [0, " + std::to_string(maximum) + "]");
        }
        return static_cast<int>(arg.number);
    }

    int register_arg(const Expr& call, std::size_t position) {
        const std::string& name = word_arg(call, position, "a register name");
        const auto it = scope_.find(name);
        if (it == scope_.end() || it->second.kind != EntityKind::Register) {
            fail(Errc::Name, *call.args[position], "'" + name + "' is not a state register");
        }
        return it->second.index;
    }

    int eval_call(const Expr& call) {
        const std::string& f = call.name;
        if (f == "sub") {
            expect_arity(call, 2);
            std::vector<int> ids = eval_args(call);
            Instr negated;
            negated.op = Op::Neg;
            negated.args = {ids[1]};
            negated.unit = node(ids[1]).unit;
            negated.weak_zero = node(ids[1]).weak_zero;
            const int neg = add_node(std::move(negated), call);
            return build_nary(Op::Add, {ids[0], neg}, call);
        }
        if (f == "decay") {
            expect_arity(call, 2);
            const int x = eval(*call.args[0]);
            const int tau = resolve_tau(*call.args[1], /*euler=*/false);
            Instr zero;
            zero.op = Op::Const;
            zero.unit = node(x).unit;
            const int target = add_node(std::move(zero), call);
            return build_leak(Op::LeakyIntegrate, x, target, tau, call);
        }
        const auto op = parse_op_name(f);
        if (!op) {
            fail(Errc::Name, call, "unknown function '" + f + "'");
        }
        switch (*op) {
            case Op::Const: fail(Errc::Type, call, "write constants as literals, e.g. 0.5 or 0.5[s]");
            case Op::Param: {
                expect_arity(call, 1);
                const std::string& name = word_arg(call, 0, "a parameter name");
                const auto it = scope_.find(name);
                if (it == scope_.end() || it->second.kind != EntityKind::Param) {
                    fail(Errc::Name, call, "'" + name + "' is not a parameter");
                }
                return param_node(it->second.index, call);
            }
            case Op::State: {
                expect_arity(call, 1);
                return state_node(register_arg(call, 0));
            }
            case Op::Stimulus: {
                expect_arity(call, 0);
                return stimulus_node(call);
            }
            case Op::TypeMask: {
                expect_arity(call, 1);
                Instr instr;
                instr.op = Op::TypeMask;
                instr.type_name = word_arg(call, 0, "a neuron type name");
                return add_node(std::move(instr), call);
            }
            case Op::SumIn: {
                expect_arity(call, 2);
                Instr instr;
                instr.op = Op::SumIn;
                instr.index = register_arg(call, 0);
                const std::string& selector = word_arg(call, 1, "exc, inh or all");
                const auto select = parse_select(selector);
                if (!select) {
                    fail(Errc::Type, *call.args[1], "sum_in selector must be exc, inh or all, got '" + selector + "'");
                }
                instr.select = *select;
                instr.unit = out_.registers[static_cast<std::size_t>(instr.index)].unit;
                return add_node(std::move(instr), call);
            }
            case Op::CountIn: {
                expect_arity(call, 2);
                Instr instr;
                instr.op = Op::CountIn;
                instr.index = register_arg(call, 0);
                instr.k = count_arg(call, 1, "the compared state", 1000000);
                if (!out_.registers[static_cast<std::size_t>(instr.index)].unit.dimensionless()) {
                    fail(Errc::Unit, call, "count_in needs a dimensionless (discrete state) register");
                }
                return add_node(std::move(instr), call);
            }
            case Op::Delay: {
                expect_arity(call, 2);
                Instr instr;
                instr.op = Op::Delay;
                instr.index = register_arg(call, 0);
                instr.k = count_arg(call, 1, "the delay in ticks", kMaxEdgeDelayTicksForPrograms);
                instr.unit = out_.registers[static_cast<std::size_t>(instr.index)].unit;
                return add_node(std::move(instr), call);
            }
            case Op::Add:
            case Op::Mul:
            case Op::Min:
            case Op::Max: {
                if (call.args.size() < 2) {
                    fail(Errc::Type, call, f + " takes at least 2 arguments");
                }
                return build_nary(*op, eval_args(call), call);
            }
            case Op::Neg:
            case Op::Abs:
            case Op::Relu: {
                expect_arity(call, 1);
                Instr instr;
                instr.op = *op;
                instr.args = eval_args(call);
                instr.unit = node(instr.args[0]).unit;
                instr.weak_zero = node(instr.args[0]).weak_zero;
                return add_node(std::move(instr), call);
            }
            case Op::Tanh:
            case Op::Sigmoid: {
                expect_arity(call, 1);
                Instr instr;
                instr.op = *op;
                const int x = eval(*call.args[0]);
                instr.args = {coerce(x, Unit::none(), *call.args[0], f + " argument")};
                return add_node(std::move(instr), call);
            }
            case Op::Clamp: {
                expect_arity(call, 3);
                std::vector<int> ids = eval_args(call);
                Instr instr;
                instr.op = Op::Clamp;
                instr.unit = unify(ids, call, "clamp");
                instr.args = ids;
                if (node(ids[1]).op == Op::Const && node(ids[2]).op == Op::Const && node(ids[1]).value > node(ids[2]).value) {
                    fail(Errc::Type, call, "clamp bounds must satisfy lo <= hi");
                }
                return add_node(std::move(instr), call);
            }
            case Op::Threshold: {
                expect_arity(call, 2);
                std::vector<int> ids = eval_args(call);
                Instr instr;
                instr.op = Op::Threshold;
                unify(ids, call, "threshold");
                instr.args = ids;
                return add_node(std::move(instr), call);
            }
            case Op::Select: {
                expect_arity(call, 3);
                std::vector<int> ids = eval_args(call);
                ids[0] = coerce(ids[0], Unit::none(), *call.args[0], "select condition");
                std::vector<int> branches = {ids[1], ids[2]};
                Instr instr;
                instr.op = Op::Select;
                instr.unit = unify(branches, call, "select branches");
                instr.args = {ids[0], branches[0], branches[1]};
                return add_node(std::move(instr), call);
            }
            case Op::Lut: {
                expect_arity(call, 2);
                const Expr& list = *call.args[1];
                if (list.kind != Expr::Kind::List || list.args.empty() || list.args.size() > kMaxLutEntries) {
                    fail(Errc::Type, list, "lut needs a table: lut(x, [v0, v1, ...]) with 1.." + std::to_string(kMaxLutEntries) + " entries");
                }
                Instr instr;
                instr.op = Op::Lut;
                const int x = eval(*call.args[0]);
                instr.args = {coerce(x, Unit::none(), *call.args[0], "lut index")};
                for (const ExprPtr& entry : list.args) {
                    if (program_.tier == Tier::G0 && entry->number != std::floor(entry->number)) {
                        fail(Errc::Tier, *entry, "G0 tables hold integers");
                    }
                    instr.table.push_back(entry->number == 0.0 ? 0.0 : entry->number);
                }
                return add_node(std::move(instr), call);
            }
            case Op::LeakyIntegrate:
            case Op::EulerLeak: {
                expect_arity(call, 3);
                std::vector<int> ids = {eval(*call.args[0]), eval(*call.args[1])};
                const int tau = resolve_tau(*call.args[2], *op == Op::EulerLeak);
                const Unit unit = unify(ids, call, f + " value and target");
                (void)unit;
                return build_leak(*op, ids[0], ids[1], tau, call, /*unified=*/true);
            }
        }
        fail(Errc::Name, call, "unknown function '" + f + "'");
    }

    int build_nary(Op op, std::vector<int> ids, const Expr& at) {
        Instr instr;
        instr.op = op;
        if (op == Op::Mul) {
            Unit product;
            for (const int id : ids) {
                product = product * node(id).unit;
            }
            instr.unit = product;
        } else {
            instr.unit = unify(ids, at, std::string(op_name(op)));
            // A sum or extremum of weak zeros stays a (weak) zero only when every operand is one.
        }
        instr.args = std::move(ids);
        return add_node(std::move(instr), at);
    }

    int build_leak(Op op, int x, int target, int tau, const Expr& at, bool unified = false) {
        std::vector<int> ids = {x, target};
        Unit unit;
        if (unified) {
            unit = node(x).unit;
        } else {
            unit = unify(ids, at, "leaky_integrate value and target");
        }
        Instr instr;
        instr.op = op;
        instr.args = {ids[0], ids[1], tau};
        instr.unit = unit;
        return add_node(std::move(instr), at);
    }

    // Resolves a time-constant argument: a parameter or a positive constant, in seconds. Applies the §5.5
    // stability rules. Returns the node id of the parameter or constant.
    int resolve_tau(const Expr& expr, bool euler) {
        const int id = eval(expr);
        const Instr& tau = node(id);
        double lower = 0.0;
        if (tau.op == Op::Param) {
            lower = out_.params[static_cast<std::size_t>(tau.index)].lower;
        } else if (tau.op == Op::Const && !tau.weak_zero) {
            lower = tau.value;
        } else {
            fail(Errc::Type, expr, "a time constant must be a parameter or a constant with a [s] unit");
        }
        if (tau.unit != Unit::time()) {
            fail(Errc::Unit, expr, "time constant must have unit [s], found " + unit_text(tau.unit));
        }
        if (!(lower > 0.0)) {
            fail(Errc::Stability, expr, "time constant must be strictly positive over its declared bounds");
        }
        if (euler) {
            if (!out_.dt_max) {
                fail(Errc::Stability, expr, "euler_leak requires a 'dt_max' declaration to bound dt/tau");
            }
            if (*out_.dt_max / lower > 1.0) {
                fail(Errc::Stability,
                     expr,
                     "explicit Euler is unstable here: dt_max/tau_lower = " + canonical_number(*out_.dt_max) + "/" +
                         canonical_number(lower) + " exceeds 1; use leaky_integrate or tighten the bounds");
            }
        }
        return id;
    }

    IrTauOperand resolve_tau_operand(const Expr& expr, bool euler, const char* what) {
        const int id = resolve_tau(expr, euler);
        const Instr& tau = node(id);
        IrTauOperand operand;
        if (tau.op == Op::Param) {
            operand.kind = IrTauOperand::Kind::Param;
            operand.param = tau.index;
        } else {
            operand.kind = IrTauOperand::Kind::Const;
            operand.value = tau.value;
        }
        (void)what;
        return operand;
    }

    static constexpr int kMaxEdgeDelayTicksForPrograms = 100000;

    const ProgramAst& program_;
    Resolved out_;
    std::map<std::string, Entity, std::less<>> scope_;
    std::vector<int> write_node_;
    std::vector<int> state_nodes_;
    std::vector<int> param_nodes_;
    int stimulus_node_ = -1;
    bool observed_ = false;
};

}  // namespace

Resolved check_program(const ProgramAst& program) { return Checker(program).run(); }

}  // namespace occamworm
