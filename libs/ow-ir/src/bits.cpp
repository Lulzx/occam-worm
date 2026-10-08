#include "occamworm/ir/bits.hpp"

#include <algorithm>
#include <map>

#include "occamworm/core/numfmt.hpp"

namespace occamworm {
namespace {

int unit_bits(Unit unit) { return zigzag_gamma0_bits(unit.seconds) + zigzag_gamma0_bits(unit.volts); }

// Instruction kind codewords (prefix-free): state 00, const 01, param 100, add 101, mul 110, and every
// other kind 111 followed by a 5-bit index (op_code_index).
int kind_bits(Op op) {
    switch (op) {
        case Op::State:
        case Op::Const: return 2;
        case Op::Param:
        case Op::Add:
        case Op::Mul: return 3;
        default: return 8;
    }
}

}  // namespace

BitBreakdown compute_bits(const Ir& ir) {
    BitBreakdown bits;

    // Header: tier (3 bits fixed), dt_max flag (+ constant), stimulus unit.
    bits.header = 3 + 1 + (ir.dt_max ? constant_bits(*ir.dt_max) : 0) + unit_bits(ir.stimulus_unit);

    // Registers: count, then unit and initial value of each.
    bits.registers = elias_gamma_bits(ir.registers.size());
    for (const IrRegister& reg : ir.registers) {
        bits.registers += unit_bits(reg.unit) + constant_bits(reg.init);
    }

    // Parameters: count, then unit, trainable flag and either the fixed value or the two bounds. The declared
    // precision of a trainable parameter is charged in L_params, not here.
    bits.parameters = elias_gamma0_bits(ir.params.size());
    for (const IrParam& param : ir.params) {
        bits.parameters += unit_bits(param.unit) + 1;
        bits.parameters += param.trainable ? constant_bits(param.lower) + constant_bits(param.upper) : constant_bits(param.value);
    }

    // Instructions: count, then per instruction its kind codeword, attributes and argument references.
    // An argument reference is Elias-gamma of the distance back to the producing instruction (>= 1).
    bits.instructions = elias_gamma0_bits(ir.instrs.size());
    std::map<std::string, int> by_op;
    for (std::size_t i = 0; i < ir.instrs.size(); ++i) {
        const Instr& instr = ir.instrs[i];
        int cost = kind_bits(instr.op);
        switch (instr.op) {
            case Op::Const: cost += constant_bits(instr.value) + unit_bits(instr.unit); break;
            case Op::Param:
            case Op::State: cost += elias_gamma0_bits(static_cast<unsigned long long>(instr.index)); break;
            case Op::TypeMask: {
                const int name_cost = elias_gamma_bits(instr.type_name.size()) + 8 * static_cast<int>(instr.type_name.size());
                cost += name_cost;
                bits.type_dispatch += name_cost;
                break;
            }
            case Op::SumIn: cost += elias_gamma0_bits(static_cast<unsigned long long>(instr.index)) + 2; break;
            case Op::CountIn:
                cost += elias_gamma0_bits(static_cast<unsigned long long>(instr.index)) +
                        elias_gamma0_bits(static_cast<unsigned long long>(instr.k));
                break;
            case Op::Delay:
                cost += elias_gamma0_bits(static_cast<unsigned long long>(instr.index)) +
                        elias_gamma_bits(static_cast<unsigned long long>(instr.k));
                break;
            case Op::Lut: {
                int table_cost = elias_gamma_bits(instr.table.size());
                for (const double entry : instr.table) {
                    table_cost += constant_bits(entry);
                }
                cost += table_cost;
                bits.type_dispatch += table_cost;
                break;
            }
            case Op::Add:
            case Op::Mul:
            case Op::Min:
            case Op::Max: cost += elias_gamma_bits(instr.args.size() - 1); break;  // arity - 1 >= 1
            default: break;
        }
        for (const int arg : instr.args) {
            cost += elias_gamma_bits(static_cast<unsigned long long>(static_cast<int>(i) - arg));
        }
        bits.instructions += cost;
        by_op[std::string(op_name(instr.op))] += cost;
    }
    for (const auto& [name, cost] : by_op) {
        bits.instruction_bits_by_op.emplace_back(name, cost);
    }

    // Writes: one reference per register, as a distance back from the end of the instruction list.
    for (const int write : ir.writes) {
        bits.writes += elias_gamma_bits(static_cast<unsigned long long>(static_cast<int>(ir.instrs.size()) - write));
    }

    // Gap coupling: presence flag, register, optional scale parameter.
    bits.gap = 1;
    if (ir.gap) {
        bits.gap += elias_gamma0_bits(static_cast<unsigned long long>(ir.gap->reg)) + 1;
        if (ir.gap->scale_param >= 0) {
            bits.gap += elias_gamma0_bits(static_cast<unsigned long long>(ir.gap->scale_param));
        }
    }

    // Observation: operator (1 bit), register, and for calcium the time-constant operand (1 bit + reference).
    bits.observation = 1 + elias_gamma0_bits(static_cast<unsigned long long>(ir.observation.reg));
    if (ir.observation.op == ObsOperator::CalciumLinearV1) {
        bits.observation += 1;
        bits.observation += ir.observation.tau.kind == IrTauOperand::Kind::Param
                                ? elias_gamma0_bits(static_cast<unsigned long long>(ir.observation.tau.param))
                                : constant_bits(ir.observation.tau.value);
    }

    bits.l_ast = bits.header + bits.registers + bits.parameters + (bits.instructions - bits.type_dispatch) + bits.writes +
                 bits.gap + bits.observation;
    bits.l_topology_overrides = 0;
    bits.l_struct = bits.l_ast + bits.l_topology_overrides + bits.type_dispatch;

    for (const IrParam& param : ir.params) {
        if (param.trainable) {
            bits.l_params += param.bits;
        }
    }
    return bits;
}

}  // namespace occamworm
