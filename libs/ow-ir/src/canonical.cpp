#include "occamworm/ir/canonical.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <map>
#include <numeric>

#include "occamworm/core/error.hpp"
#include "occamworm/core/sha256.hpp"
#include "occamworm/ir/bits.hpp"
#include "occamworm/ir/scalar_ops.hpp"
#include "occamworm/ir/source.hpp"

namespace occamworm {
namespace {

// ---- hash-consed pool of canonical nodes -----------------------------------------------------------

class Pool {
public:
    const Instr& node(int id) const { return nodes_[static_cast<std::size_t>(id)]; }
    const std::string& digest(int id) const { return digests_[static_cast<std::size_t>(id)]; }
    std::size_t size() const { return nodes_.size(); }

    // Structurally equal nodes (same op, attributes, unit and argument nodes) get the same id: CSE.
    int intern(Instr instr) {
        instr.weak_zero = false;
        const std::string key = compute_digest(instr);
        const auto it = index_.find(key);
        if (it != index_.end()) {
            return it->second;
        }
        const int id = static_cast<int>(nodes_.size());
        nodes_.push_back(std::move(instr));
        digests_.push_back(key);
        index_.emplace(key, id);
        return id;
    }

private:
    static void put_u32(Sha256& hasher, std::uint32_t value) {
        const std::array<std::uint8_t, 4> bytes = {static_cast<std::uint8_t>(value >> 24), static_cast<std::uint8_t>(value >> 16),
                                                   static_cast<std::uint8_t>(value >> 8), static_cast<std::uint8_t>(value)};
        hasher.update(std::span<const std::uint8_t>(bytes));
    }
    static void put_double(Sha256& hasher, double value) {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
        put_u32(hasher, static_cast<std::uint32_t>(bits >> 32));
        put_u32(hasher, static_cast<std::uint32_t>(bits));
    }

    std::string compute_digest(const Instr& n) const {
        Sha256 hasher;
        put_u32(hasher, static_cast<std::uint32_t>(n.op));
        put_u32(hasher, static_cast<std::uint32_t>(n.index));
        put_u32(hasher, static_cast<std::uint32_t>(n.k));
        put_u32(hasher, static_cast<std::uint32_t>(n.select));
        put_double(hasher, n.value);
        put_u32(hasher, static_cast<std::uint32_t>(n.unit.seconds));
        put_u32(hasher, static_cast<std::uint32_t>(n.unit.volts));
        put_u32(hasher, static_cast<std::uint32_t>(n.type_name.size()));
        hasher.update(n.type_name);
        put_u32(hasher, static_cast<std::uint32_t>(n.table.size()));
        for (const double entry : n.table) {
            put_double(hasher, entry);
        }
        put_u32(hasher, static_cast<std::uint32_t>(n.args.size()));
        for (const int arg : n.args) {
            hasher.update(digest(arg));
        }
        const Sha256Digest result = hasher.finalize();
        return std::string(reinterpret_cast<const char*>(result.data()), result.size());
    }

    std::vector<Instr> nodes_;
    std::vector<std::string> digests_;
    std::map<std::string, int> index_;
};

// ---- normaliser ------------------------------------------------------------------------------------

class Normaliser {
public:
    Normaliser(const Resolved& resolved, std::vector<int> reg_map, std::vector<int> param_map)
        : resolved_(resolved), reg_map_(std::move(reg_map)), param_map_(std::move(param_map)) {}

    Pool& pool() { return pool_; }

    // Normalises everything reachable from the given raw roots and returns the root pool ids.
    std::vector<int> run(const std::vector<int>& raw_roots) {
        const std::size_t count = resolved_.raw.size();
        std::vector<char> reachable(count, 0);
        std::vector<int> stack;
        for (const int root : raw_roots) {
            if (!reachable[static_cast<std::size_t>(root)]) {
                reachable[static_cast<std::size_t>(root)] = 1;
                stack.push_back(root);
            }
        }
        while (!stack.empty()) {
            const int id = stack.back();
            stack.pop_back();
            for (const int arg : resolved_.raw[static_cast<std::size_t>(id)].args) {
                if (!reachable[static_cast<std::size_t>(arg)]) {
                    reachable[static_cast<std::size_t>(arg)] = 1;
                    stack.push_back(arg);
                }
            }
        }
        memo_.assign(count, -1);
        // Raw ids are topologically ordered (arguments precede users), so a forward sweep suffices.
        for (std::size_t id = 0; id < count; ++id) {
            if (reachable[id]) {
                memo_[id] = convert(resolved_.raw[id]);
            }
        }
        std::vector<int> roots;
        roots.reserve(raw_roots.size());
        for (const int root : raw_roots) {
            roots.push_back(memo_[static_cast<std::size_t>(root)]);
        }
        return roots;
    }

private:
    const Instr& node(int id) const { return pool_.node(id); }
    bool is_const(int id) const { return node(id).op == Op::Const; }

    int rank(int id) const {
        switch (node(id).op) {
            case Op::Const: return 0;
            case Op::Param: return 1;
            case Op::State: return 2;
            case Op::Stimulus:
            case Op::TypeMask:
            case Op::SumIn:
            case Op::CountIn:
            case Op::Delay: return 3;
            default: return 4;
        }
    }

    void sort_operands(std::vector<int>& ids) const {
        std::ranges::sort(ids, [&](int a, int b) {
            if (rank(a) != rank(b)) {
                return rank(a) < rank(b);
            }
            return pool_.digest(a) < pool_.digest(b);
        });
    }

    // Leaves that name an eliminated register or parameter get index -1. Such nodes can only be reached from
    // sub-expressions that simplification discards (for example `b` in `a + b - b`); build_candidate verifies
    // that none survives into the emitted instructions.
    int map_register(int old_index) const { return reg_map_[static_cast<std::size_t>(old_index)]; }
    int map_param(int old_index) const { return param_map_[static_cast<std::size_t>(old_index)]; }

    int make_const(double value, Unit unit) {
        Instr instr;
        instr.op = Op::Const;
        instr.value = value == 0.0 ? 0.0 : value;  // normalise -0.0
        instr.unit = unit;
        return pool_.intern(std::move(instr));
    }

    int make_leaf(Op op, int index, Unit unit) {
        Instr instr;
        instr.op = op;
        instr.index = index;
        instr.unit = unit;
        return pool_.intern(std::move(instr));
    }

    int make_op(Op op, std::vector<int> args, Unit unit) {
        Instr instr;
        instr.op = op;
        instr.args = std::move(args);
        instr.unit = unit;
        return pool_.intern(std::move(instr));
    }

    // Splices nested nodes of the same associative operator into one operand list.
    std::vector<int> flatten(Op op, const std::vector<int>& args) const {
        std::vector<int> flat;
        for (const int arg : args) {
            if (node(arg).op == op) {
                flat.insert(flat.end(), node(arg).args.begin(), node(arg).args.end());
            } else {
                flat.push_back(arg);
            }
        }
        return flat;
    }

    int make_add(const std::vector<int>& args, Unit unit) {
        std::vector<int> terms;
        std::vector<double> constants;
        for (const int id : flatten(Op::Add, args)) {
            if (is_const(id)) {
                constants.push_back(node(id).value);
            } else {
                terms.push_back(id);
            }
        }
        // x + (-x) cancels exactly.
        bool cancelled = true;
        while (cancelled) {
            cancelled = false;
            for (std::size_t i = 0; i < terms.size() && !cancelled; ++i) {
                for (std::size_t j = 0; j < terms.size() && !cancelled; ++j) {
                    if (i != j && node(terms[j]).op == Op::Neg && node(terms[j]).args[0] == terms[i]) {
                        terms.erase(terms.begin() + static_cast<std::ptrdiff_t>(std::max(i, j)));
                        terms.erase(terms.begin() + static_cast<std::ptrdiff_t>(std::min(i, j)));
                        cancelled = true;
                    }
                }
            }
        }
        double constant = 0.0;
        if (!constants.empty()) {
            std::ranges::sort(constants);
            constant = constants[0];
            for (std::size_t i = 1; i < constants.size(); ++i) {
                constant += constants[i];
            }
        }
        sort_operands(terms);
        std::vector<int> operands;
        if (constant != 0.0) {
            operands.push_back(make_const(constant, unit));
        }
        operands.insert(operands.end(), terms.begin(), terms.end());
        if (operands.empty()) {
            return make_const(0.0, unit);
        }
        if (operands.size() == 1) {
            return operands[0];
        }
        return make_op(Op::Add, std::move(operands), unit);
    }

    int make_mul(const std::vector<int>& args, Unit unit) {
        std::vector<int> factors;
        std::vector<double> constants;
        Unit constant_unit;
        for (const int id : flatten(Op::Mul, args)) {
            if (is_const(id)) {
                constants.push_back(node(id).value);
                constant_unit = constant_unit * node(id).unit;
            } else {
                factors.push_back(id);
            }
        }
        double constant = 1.0;
        const bool has_constant = !constants.empty();
        if (has_constant) {
            std::ranges::sort(constants);
            constant = constants[0];
            for (std::size_t i = 1; i < constants.size(); ++i) {
                constant *= constants[i];
            }
            if (constant == 0.0) {
                return make_const(0.0, unit);
            }
        }
        sort_operands(factors);
        if (factors.empty()) {
            return make_const(constant, unit);
        }
        if (has_constant && constant == -1.0 && factors.size() == 1 && constant_unit.dimensionless()) {
            return make_neg(factors[0]);
        }
        std::vector<int> operands;
        if (has_constant && !(constant == 1.0 && constant_unit.dimensionless())) {
            operands.push_back(make_const(constant, constant_unit));
        }
        operands.insert(operands.end(), factors.begin(), factors.end());
        if (operands.size() == 1) {
            return operands[0];
        }
        return make_op(Op::Mul, std::move(operands), unit);
    }

    int make_neg(int x) {
        const Instr& n = node(x);
        if (n.op == Op::Const) {
            return make_const(-n.value, n.unit);
        }
        if (n.op == Op::Neg) {
            return n.args[0];
        }
        return make_op(Op::Neg, {x}, n.unit);
    }

    int make_abs(int x) {
        const Instr& n = node(x);
        if (n.op == Op::Const) {
            return make_const(n.value < 0.0 ? -n.value : n.value, n.unit);
        }
        if (n.op == Op::Abs || n.op == Op::Relu) {
            return x;  // |abs(y)| = abs(y); |relu(y)| = relu(y)
        }
        if (n.op == Op::Neg) {
            return make_abs(n.args[0]);
        }
        return make_op(Op::Abs, {x}, n.unit);
    }

    int make_relu(int x) {
        const Instr& n = node(x);
        if (n.op == Op::Const) {
            return make_const(scalar::relu(n.value), n.unit);
        }
        if (n.op == Op::Relu || n.op == Op::Abs) {
            return x;
        }
        return make_op(Op::Relu, {x}, n.unit);
    }

    int make_extremum(Op op, const std::vector<int>& args, Unit unit) {
        std::vector<int> terms;
        bool have_constant = false;
        double constant = 0.0;
        for (const int id : flatten(op, args)) {
            if (is_const(id)) {
                const double v = node(id).value;
                constant = !have_constant ? v : (op == Op::Min ? scalar::min2(constant, v) : scalar::max2(constant, v));
                have_constant = true;
            } else if (std::ranges::find(terms, id) == terms.end()) {  // min/max are idempotent
                terms.push_back(id);
            }
        }
        sort_operands(terms);
        if (terms.empty()) {
            return make_const(constant, unit);
        }
        if (have_constant && op == Op::Max && constant == 0.0 && terms.size() == 1) {
            return make_relu(terms[0]);  // max(0, x) is relu(x)
        }
        std::vector<int> operands;
        if (have_constant) {
            operands.push_back(make_const(constant, unit));
        }
        operands.insert(operands.end(), terms.begin(), terms.end());
        if (operands.size() == 1) {
            return operands[0];
        }
        return make_op(op, std::move(operands), unit);
    }

    int convert(const Instr& raw) {
        std::vector<int> args;
        args.reserve(raw.args.size());
        for (const int arg : raw.args) {
            args.push_back(memo_[static_cast<std::size_t>(arg)]);
        }
        switch (raw.op) {
            case Op::Const: return make_const(raw.value, raw.unit);
            case Op::Param: return make_leaf(Op::Param, map_param(raw.index), raw.unit);
            case Op::State: return make_leaf(Op::State, map_register(raw.index), raw.unit);
            case Op::Stimulus: return make_leaf(Op::Stimulus, 0, raw.unit);
            case Op::TypeMask: {
                Instr instr;
                instr.op = Op::TypeMask;
                instr.type_name = raw.type_name;
                return pool_.intern(std::move(instr));
            }
            case Op::SumIn: {
                Instr instr;
                instr.op = Op::SumIn;
                instr.index = map_register(raw.index);
                instr.select = raw.select;
                instr.unit = raw.unit;
                return pool_.intern(std::move(instr));
            }
            case Op::CountIn: {
                Instr instr;
                instr.op = Op::CountIn;
                instr.index = map_register(raw.index);
                instr.k = raw.k;
                return pool_.intern(std::move(instr));
            }
            case Op::Delay: {
                if (raw.k == 0) {
                    return make_leaf(Op::State, map_register(raw.index), raw.unit);
                }
                Instr instr;
                instr.op = Op::Delay;
                instr.index = map_register(raw.index);
                instr.k = raw.k;
                instr.unit = raw.unit;
                return pool_.intern(std::move(instr));
            }
            case Op::Add: return make_add(args, raw.unit);
            case Op::Mul: return make_mul(args, raw.unit);
            case Op::Neg: return make_neg(args[0]);
            case Op::Abs: return make_abs(args[0]);
            case Op::Relu: return make_relu(args[0]);
            case Op::Min:
            case Op::Max: return make_extremum(raw.op, args, raw.unit);
            case Op::Clamp: {
                if (is_const(args[0]) && is_const(args[1]) && is_const(args[2])) {
                    return make_const(scalar::clamp(node(args[0]).value, node(args[1]).value, node(args[2]).value), raw.unit);
                }
                return make_op(Op::Clamp, args, raw.unit);
            }
            case Op::Tanh:
            case Op::Sigmoid:  // not folded: libm results are not bit-identical across platforms
                return make_op(raw.op, args, raw.unit);
            case Op::Threshold: {
                if (is_const(args[0]) && is_const(args[1])) {
                    return make_const(scalar::threshold(node(args[0]).value, node(args[1]).value), Unit::none());
                }
                return make_op(Op::Threshold, args, Unit::none());
            }
            case Op::Select: {
                if (is_const(args[0])) {
                    return node(args[0]).value != 0.0 ? args[1] : args[2];
                }
                if (args[1] == args[2]) {
                    return args[1];
                }
                return make_op(Op::Select, args, raw.unit);
            }
            case Op::Lut: {
                if (is_const(args[0])) {
                    return make_const(raw.table[scalar::lut_index(node(args[0]).value, raw.table.size())], Unit::none());
                }
                const bool uniform = std::ranges::all_of(raw.table, [&](double v) { return v == raw.table[0]; });
                if (uniform) {
                    return make_const(raw.table[0], Unit::none());
                }
                Instr instr;
                instr.op = Op::Lut;
                instr.args = args;
                instr.table = raw.table;
                for (double& v : instr.table) {
                    v = v == 0.0 ? 0.0 : v;
                }
                return pool_.intern(std::move(instr));
            }
            case Op::LeakyIntegrate:
            case Op::EulerLeak: {
                if (args[0] == args[1]) {
                    return args[0];  // x + (x - x) * a == x exactly
                }
                return make_op(raw.op, args, raw.unit);
            }
        }
        throw Error(Errc::Runtime, "internal error: unhandled operator in canonicalisation");
    }

    const Resolved& resolved_;
    std::vector<int> reg_map_;
    std::vector<int> param_map_;
    Pool pool_;
    std::vector<int> memo_;
};

// ---- liveness ----------------------------------------------------------------------------------------

struct Liveness {
    std::vector<bool> registers;
    std::vector<bool> params;
};

// Collects register and parameter references reachable from a pool node.
void collect_refs(const Pool& pool, int root, std::vector<bool>& seen, std::vector<bool>& registers, std::vector<bool>& params,
                  std::vector<int>& new_registers) {
    std::vector<int> stack = {root};
    while (!stack.empty()) {
        const int id = stack.back();
        stack.pop_back();
        if (seen[static_cast<std::size_t>(id)]) {
            continue;
        }
        seen[static_cast<std::size_t>(id)] = true;
        const Instr& n = pool.node(id);
        switch (n.op) {
            case Op::State:
            case Op::SumIn:
            case Op::CountIn:
            case Op::Delay:
                if (!registers[static_cast<std::size_t>(n.index)]) {
                    registers[static_cast<std::size_t>(n.index)] = true;
                    new_registers.push_back(n.index);
                }
                break;
            case Op::Param: params[static_cast<std::size_t>(n.index)] = true; break;
            default: break;
        }
        for (const int arg : n.args) {
            stack.push_back(arg);
        }
    }
}

Liveness compute_liveness(const Resolved& resolved) {
    const std::size_t reg_count = resolved.registers.size();
    const std::size_t param_count = resolved.params.size();
    std::vector<int> identity_regs(reg_count);
    std::iota(identity_regs.begin(), identity_regs.end(), 0);
    std::vector<int> identity_params(param_count);
    std::iota(identity_params.begin(), identity_params.end(), 0);
    Normaliser normaliser(resolved, identity_regs, identity_params);
    const std::vector<int> roots = normaliser.run(resolved.writes);

    Liveness live{std::vector<bool>(reg_count, false), std::vector<bool>(param_count, false)};
    std::vector<int> worklist = {resolved.observation.reg};
    live.registers[static_cast<std::size_t>(resolved.observation.reg)] = true;
    while (!worklist.empty()) {
        const int reg = worklist.back();
        worklist.pop_back();
        std::vector<bool> seen(normaliser.pool().size(), false);
        collect_refs(normaliser.pool(), roots[static_cast<std::size_t>(reg)], seen, live.registers, live.params, worklist);
    }
    if (resolved.gap && live.registers[static_cast<std::size_t>(resolved.gap->reg)] && resolved.gap->scale_param >= 0) {
        live.params[static_cast<std::size_t>(resolved.gap->scale_param)] = true;
    }
    if (resolved.observation.tau.kind == IrTauOperand::Kind::Param) {
        live.params[static_cast<std::size_t>(resolved.observation.tau.param)] = true;
    }
    return live;
}

// ---- candidate construction --------------------------------------------------------------------------

// Builds the canonical Ir for one relabelling. inv_reg[new] = old declaration index; same for params.
Ir build_candidate(const Resolved& resolved, const std::vector<int>& inv_reg, const std::vector<int>& inv_param) {
    std::vector<int> reg_map(resolved.registers.size(), -1);
    for (std::size_t i = 0; i < inv_reg.size(); ++i) {
        reg_map[static_cast<std::size_t>(inv_reg[i])] = static_cast<int>(i);
    }
    std::vector<int> param_map(resolved.params.size(), -1);
    for (std::size_t i = 0; i < inv_param.size(); ++i) {
        param_map[static_cast<std::size_t>(inv_param[i])] = static_cast<int>(i);
    }

    Normaliser normaliser(resolved, reg_map, param_map);
    std::vector<int> raw_roots;
    for (const int old : inv_reg) {
        raw_roots.push_back(resolved.writes[static_cast<std::size_t>(old)]);
    }
    const std::vector<int> roots = normaliser.run(raw_roots);
    const Pool& pool = normaliser.pool();

    Ir ir;
    ir.rule_name = resolved.rule_name;
    ir.tier = resolved.tier;
    ir.dt_max = resolved.dt_max;
    ir.stimulus_unit = resolved.stimulus_unit;
    for (std::size_t i = 0; i < inv_reg.size(); ++i) {
        const RegisterDecl& decl = resolved.registers[static_cast<std::size_t>(inv_reg[i])];
        ir.registers.push_back({"s" + std::to_string(i), decl.name, decl.unit, decl.init == 0.0 ? 0.0 : decl.init});
    }
    for (std::size_t i = 0; i < inv_param.size(); ++i) {
        IrParam param = resolved.params[static_cast<std::size_t>(inv_param[i])];
        param.name = "p" + std::to_string(i);
        param.source_name = resolved.params[static_cast<std::size_t>(inv_param[i])].source_name;
        ir.params.push_back(std::move(param));
    }
    for (std::size_t r = 0; r < resolved.registers.size(); ++r) {
        if (reg_map[r] < 0) {
            ir.eliminated_registers.push_back(resolved.registers[r].name);
        }
    }
    for (std::size_t p = 0; p < resolved.params.size(); ++p) {
        if (param_map[p] < 0) {
            ir.eliminated_params.push_back(resolved.params[p].source_name);
        }
    }

    // Canonical numbering: depth-first post-order from the writes, arguments in canonical order.
    std::vector<int> emitted(pool.size(), -1);
    for (const int root : roots) {
        std::vector<std::pair<int, std::size_t>> stack = {{root, 0}};
        while (!stack.empty()) {
            auto& [id, next_arg] = stack.back();
            if (emitted[static_cast<std::size_t>(id)] >= 0) {
                stack.pop_back();
                continue;
            }
            const Instr& n = pool.node(id);
            if (next_arg < n.args.size()) {
                const int child = n.args[next_arg++];
                if (emitted[static_cast<std::size_t>(child)] < 0) {
                    stack.emplace_back(child, 0);
                }
                continue;
            }
            const bool leaf_with_index = n.op == Op::State || n.op == Op::SumIn || n.op == Op::CountIn || n.op == Op::Delay || n.op == Op::Param;
            if (leaf_with_index && n.index < 0) {
                throw Error(Errc::Runtime, "internal error: live code references an eliminated register or parameter");
            }
            Instr out = n;
            for (int& arg : out.args) {
                arg = emitted[static_cast<std::size_t>(arg)];
            }
            emitted[static_cast<std::size_t>(id)] = static_cast<int>(ir.instrs.size());
            ir.instrs.push_back(std::move(out));
            stack.pop_back();
        }
    }
    for (const int root : roots) {
        ir.writes.push_back(emitted[static_cast<std::size_t>(root)]);
    }

    if (resolved.gap && reg_map[static_cast<std::size_t>(resolved.gap->reg)] >= 0) {
        IrGap gap;
        gap.reg = reg_map[static_cast<std::size_t>(resolved.gap->reg)];
        gap.scale_param = resolved.gap->scale_param >= 0 ? param_map[static_cast<std::size_t>(resolved.gap->scale_param)] : -1;
        ir.gap = gap;
    }
    ir.observation.op = resolved.observation.op;
    ir.observation.reg = reg_map[static_cast<std::size_t>(resolved.observation.reg)];
    ir.observation.tau = resolved.observation.tau;
    if (ir.observation.tau.kind == IrTauOperand::Kind::Param) {
        ir.observation.tau.param = param_map[static_cast<std::size_t>(ir.observation.tau.param)];
    }
    ir.canonical_source = print_source(ir, {});
    return ir;
}

std::size_t factorial(std::size_t n) {
    std::size_t result = 1;
    for (std::size_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

}  // namespace

Ir canonicalise(const Resolved& resolved) {
    const Liveness live = compute_liveness(resolved);
    std::vector<int> live_regs;
    for (std::size_t r = 0; r < live.registers.size(); ++r) {
        if (live.registers[r]) {
            live_regs.push_back(static_cast<int>(r));
        }
    }
    std::vector<int> used_params;
    for (std::size_t p = 0; p < live.params.size(); ++p) {
        if (live.params[p]) {
            used_params.push_back(static_cast<int>(p));
        }
    }

    const bool relabel_all = live_regs.size() <= 6 && used_params.size() <= 6 &&
                             factorial(live_regs.size()) * factorial(used_params.size()) <= kMaxRelabelings;
    Ir best;
    bool have_best = false;
    // inv_reg[new] = old; enumerate every permutation of the live registers and parameters.
    std::vector<int> reg_order = live_regs;
    do {
        std::vector<int> param_order = used_params;
        do {
            Ir candidate = build_candidate(resolved, reg_order, param_order);
            if (!have_best || candidate.canonical_source < best.canonical_source) {
                best = std::move(candidate);
                have_best = true;
            }
            if (!relabel_all) {
                break;
            }
        } while (std::ranges::next_permutation(param_order).found);
        if (!relabel_all) {
            break;
        }
    } while (std::ranges::next_permutation(reg_order).found);

    best.program_hash = sha256_hex(best.canonical_source);
    best.bits = compute_bits(best);
    return best;
}

}  // namespace occamworm
