#include "occamworm/ir/source.hpp"

#include "occamworm/core/numfmt.hpp"

namespace occamworm {

std::string print_instruction(const Instr& instr, const std::vector<IrRegister>& registers,
                              const std::vector<IrParam>& params) {
    const auto arg_list = [&]() {
        std::string text;
        for (std::size_t i = 0; i < instr.args.size(); ++i) {
            if (i > 0) {
                text += ", ";
            }
            text += "n" + std::to_string(instr.args[i]);
        }
        return text;
    };
    const std::string name(op_name(instr.op));
    switch (instr.op) {
        case Op::Const: return canonical_number(instr.value) + "[" + to_string(instr.unit) + "]";
        case Op::Param: return "param(" + params[static_cast<std::size_t>(instr.index)].name + ")";
        case Op::State: return "state(" + registers[static_cast<std::size_t>(instr.index)].name + ")";
        case Op::Stimulus: return "stimulus()";
        case Op::TypeMask: return "type_mask(" + instr.type_name + ")";
        case Op::SumIn:
            return "sum_in(" + registers[static_cast<std::size_t>(instr.index)].name + ", " +
                   std::string(select_name(instr.select)) + ")";
        case Op::CountIn:
            return "count_in(" + registers[static_cast<std::size_t>(instr.index)].name + ", " + std::to_string(instr.k) + ")";
        case Op::Delay:
            return "delay(" + registers[static_cast<std::size_t>(instr.index)].name + ", " + std::to_string(instr.k) + ")";
        case Op::Lut: {
            std::string table;
            for (std::size_t i = 0; i < instr.table.size(); ++i) {
                if (i > 0) {
                    table += ", ";
                }
                table += canonical_number(instr.table[i]);
            }
            return "lut(" + arg_list() + ", [" + table + "])";
        }
        default: return name + "(" + arg_list() + ")";
    }
}

std::string print_source(const Ir& ir, const PrintOptions& options) {
    std::string out;
    out += "wrl " + std::string(kGrammarVersion) + "\n";
    if (!options.identity) {
        out += "rule " + ir.rule_name + "\n";
    }
    out += "tier " + std::string(tier_name(ir.tier)) + "\n";
    if (ir.dt_max) {
        out += "dt_max " + canonical_number(*ir.dt_max) + "\n";
    }
    out += "stimulus_unit " + to_string(ir.stimulus_unit) + "\n";
    for (const IrRegister& reg : ir.registers) {
        out += "state " + reg.name + " : " + to_string(reg.unit) + " = " + canonical_number(reg.init) + "\n";
    }
    for (const IrParam& param : ir.params) {
        out += "param " + param.name + " : " + to_string(param.unit);
        if (param.trainable) {
            if (!options.identity) {
                out += " = " + canonical_number(param.value);
            }
            out += " in [" + canonical_number(param.lower) + ", " + canonical_number(param.upper) + "] trainable bits " +
                   std::to_string(param.bits);
        } else {
            out += " = " + canonical_number(param.value) + " fixed";
        }
        out += "\n";
    }
    for (std::size_t i = 0; i < ir.instrs.size(); ++i) {
        out += "let n" + std::to_string(i) + " = " + print_instruction(ir.instrs[i], ir.registers, ir.params) + "\n";
    }
    for (std::size_t r = 0; r < ir.registers.size(); ++r) {
        out += "next " + ir.registers[r].name + " = n" + std::to_string(ir.writes[r]) + "\n";
    }
    if (ir.gap) {
        out += "gap " + ir.registers[static_cast<std::size_t>(ir.gap->reg)].name;
        if (ir.gap->scale_param >= 0) {
            out += " scale " + ir.params[static_cast<std::size_t>(ir.gap->scale_param)].name;
        }
        out += "\n";
    }
    out += "observe " + std::string(obs_operator_name(ir.observation.op)) + "(" +
           ir.registers[static_cast<std::size_t>(ir.observation.reg)].name;
    if (ir.observation.op == ObsOperator::CalciumLinearV1) {
        if (ir.observation.tau.kind == IrTauOperand::Kind::Param) {
            out += ", " + ir.params[static_cast<std::size_t>(ir.observation.tau.param)].name;
        } else {
            out += ", " + canonical_number(ir.observation.tau.value) + "[s]";
        }
    }
    out += ")\n";
    return out;
}

}  // namespace occamworm
