#include "occamworm/ir/compile.hpp"

#include <fstream>
#include <sstream>

#include "occamworm/core/build_info.hpp"
#include "occamworm/ir/bits.hpp"
#include "occamworm/ir/canonical.hpp"
#include "occamworm/ir/check.hpp"
#include "occamworm/ir/parser.hpp"
#include "occamworm/ir/source.hpp"

namespace occamworm {

Ir compile(std::string_view source) { return canonicalise(check_program(parse_program(source))); }

std::expected<Ir, Error> try_compile(std::string_view source) {
    try {
        return compile(source);
    } catch (const Error& error) {
        return std::unexpected(error);
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw Error(Errc::Io, "cannot open '" + path + "'");
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

namespace {

Json unit_json(Unit unit) { return Json(to_string(unit)); }

Json instr_attrs(const Instr& instr) {
    Json attrs = Json::object();
    switch (instr.op) {
        case Op::Const: attrs.set("value", instr.value); break;
        case Op::Param: attrs.set("param", instr.index); break;
        case Op::State: attrs.set("register", instr.index); break;
        case Op::TypeMask: attrs.set("type", instr.type_name); break;
        case Op::SumIn:
            attrs.set("register", instr.index);
            attrs.set("select", std::string(select_name(instr.select)));
            break;
        case Op::CountIn:
            attrs.set("register", instr.index);
            attrs.set("k", instr.k);
            break;
        case Op::Delay:
            attrs.set("register", instr.index);
            attrs.set("ticks", instr.k);
            break;
        case Op::Lut: {
            Json table = Json::array();
            for (const double entry : instr.table) {
                table.push(entry);
            }
            attrs.set("table", std::move(table));
            break;
        }
        default: break;
    }
    return attrs;
}

}  // namespace

Json ir_to_json(const Ir& ir) {
    Json out = Json::object();
    out.set("schema", std::string(kIrSchema));
    out.set("grammar_version", std::string(kGrammarVersion));
    out.set("bit_code_version", kBitCodeVersion);
    out.set("compiler_build", OW_BUILD_STRING);
    out.set("program_hash", ir.program_hash);
    out.set("rule", ir.rule_name);
    out.set("tier", std::string(tier_name(ir.tier)));
    out.set("dt_max", ir.dt_max ? Json(*ir.dt_max) : Json());
    out.set("stimulus_unit", unit_json(ir.stimulus_unit));

    Json registers = Json::array();
    for (std::size_t i = 0; i < ir.registers.size(); ++i) {
        const IrRegister& reg = ir.registers[i];
        Json item = Json::object();
        item.set("index", i);
        item.set("name", reg.name);
        item.set("source_name", reg.source_name);
        item.set("unit", unit_json(reg.unit));
        item.set("init", reg.init);
        registers.push(std::move(item));
    }
    out.set("registers", std::move(registers));

    Json params = Json::array();
    for (std::size_t i = 0; i < ir.params.size(); ++i) {
        const IrParam& param = ir.params[i];
        Json item = Json::object();
        item.set("index", i);
        item.set("name", param.name);
        item.set("source_name", param.source_name);
        item.set("unit", unit_json(param.unit));
        item.set("value", param.value);
        item.set("lower", param.lower);
        item.set("upper", param.upper);
        item.set("trainable", param.trainable);
        item.set("bits", param.bits);
        params.push(std::move(item));
    }
    out.set("parameters", std::move(params));

    Json instrs = Json::array();
    for (std::size_t i = 0; i < ir.instrs.size(); ++i) {
        const Instr& instr = ir.instrs[i];
        Json item = Json::object();
        item.set("id", i);
        item.set("op", std::string(op_name(instr.op)));
        Json args = Json::array();
        for (const int arg : instr.args) {
            args.push(arg);
        }
        item.set("args", std::move(args));
        item.set("attrs", instr_attrs(instr));
        item.set("unit", unit_json(instr.unit));
        instrs.push(std::move(item));
    }
    out.set("instructions", std::move(instrs));

    Json writes = Json::array();
    for (std::size_t r = 0; r < ir.writes.size(); ++r) {
        Json item = Json::object();
        item.set("register", r);
        item.set("value", ir.writes[r]);
        writes.push(std::move(item));
    }
    out.set("writes", std::move(writes));

    if (ir.gap) {
        Json gap = Json::object();
        gap.set("register", ir.gap->reg);
        gap.set("scale_param", ir.gap->scale_param >= 0 ? Json(ir.gap->scale_param) : Json());
        out.set("gap", std::move(gap));
    } else {
        out.set("gap", Json());
    }

    Json observation = Json::object();
    observation.set("operator", std::string(obs_operator_name(ir.observation.op)));
    observation.set("register", ir.observation.reg);
    if (ir.observation.op == ObsOperator::CalciumLinearV1) {
        Json tau = Json::object();
        if (ir.observation.tau.kind == IrTauOperand::Kind::Param) {
            tau.set("param", ir.observation.tau.param);
        } else {
            tau.set("const", ir.observation.tau.value);
        }
        observation.set("tau", std::move(tau));
    } else {
        observation.set("tau", Json());
    }
    out.set("observation", std::move(observation));

    Json eliminated = Json::object();
    Json eliminated_registers = Json::array();
    for (const std::string& name : ir.eliminated_registers) {
        eliminated_registers.push(name);
    }
    Json eliminated_params = Json::array();
    for (const std::string& name : ir.eliminated_params) {
        eliminated_params.push(name);
    }
    eliminated.set("registers", std::move(eliminated_registers));
    eliminated.set("parameters", std::move(eliminated_params));
    out.set("eliminated", std::move(eliminated));

    const BitBreakdown& bits = ir.bits;
    Json l_struct = Json::object();
    l_struct.set("total_bits", bits.l_struct);
    l_struct.set("l_ast", bits.l_ast);
    l_struct.set("l_topology_overrides", bits.l_topology_overrides);
    l_struct.set("l_type_dispatch", bits.type_dispatch);
    Json components = Json::object();
    components.set("header", bits.header);
    components.set("registers", bits.registers);
    components.set("parameters", bits.parameters);
    components.set("instructions", bits.instructions);
    components.set("writes", bits.writes);
    components.set("gap", bits.gap);
    components.set("observation", bits.observation);
    l_struct.set("components", std::move(components));
    Json by_op = Json::object();
    for (const auto& [name, cost] : bits.instruction_bits_by_op) {
        by_op.set(name, cost);
    }
    l_struct.set("instruction_bits_by_op", std::move(by_op));
    out.set("l_struct", std::move(l_struct));

    Json l_params = Json::object();
    l_params.set("total_bits", bits.l_params);
    Json per_parameter = Json::array();
    for (const IrParam& param : ir.params) {
        if (param.trainable) {
            Json item = Json::object();
            item.set("name", param.name);
            item.set("bits", param.bits);
            per_parameter.push(std::move(item));
        }
    }
    l_params.set("per_parameter", std::move(per_parameter));
    out.set("l_params", std::move(l_params));

    out.set("canonical_source", ir.canonical_source);
    return out;
}

namespace {

Unit json_unit(const Json& json) { return parse_unit(json.as_string()); }

Op json_op(const std::string& name) {
    const auto op = parse_op_name(name);
    if (!op) {
        throw Error(Errc::Json, "unknown op '" + name + "' in IR JSON");
    }
    return *op;
}

int json_int(const Json& json) { return static_cast<int>(json.as_int()); }

}  // namespace

Ir ir_from_json(const Json& json) {
    Ir ir;
    if (json.at("schema").as_string() != kIrSchema) {
        throw Error(Errc::Json, "unsupported IR schema '" + json.at("schema").as_string() + "'");
    }
    ir.rule_name = json.at("rule").as_string();
    const auto tier = parse_tier(json.at("tier").as_string());
    if (!tier) {
        throw Error(Errc::Json, "unknown tier in IR JSON");
    }
    ir.tier = *tier;
    if (!json.at("dt_max").is_null()) {
        ir.dt_max = json.at("dt_max").as_double();
    }
    ir.stimulus_unit = json_unit(json.at("stimulus_unit"));
    for (const Json& item : json.at("registers").as_array()) {
        ir.registers.push_back({item.at("name").as_string(), item.at("source_name").as_string(), json_unit(item.at("unit")),
                                item.at("init").as_double()});
    }
    for (const Json& item : json.at("parameters").as_array()) {
        IrParam param;
        param.name = item.at("name").as_string();
        param.source_name = item.at("source_name").as_string();
        param.unit = json_unit(item.at("unit"));
        param.value = item.at("value").as_double();
        param.lower = item.at("lower").as_double();
        param.upper = item.at("upper").as_double();
        param.trainable = item.at("trainable").as_bool();
        param.bits = json_int(item.at("bits"));
        ir.params.push_back(std::move(param));
    }
    for (const Json& item : json.at("instructions").as_array()) {
        Instr instr;
        instr.op = json_op(item.at("op").as_string());
        for (const Json& arg : item.at("args").as_array()) {
            instr.args.push_back(json_int(arg));
        }
        instr.unit = json_unit(item.at("unit"));
        const Json& attrs = item.at("attrs");
        switch (instr.op) {
            case Op::Const: instr.value = attrs.at("value").as_double(); break;
            case Op::Param: instr.index = json_int(attrs.at("param")); break;
            case Op::State: instr.index = json_int(attrs.at("register")); break;
            case Op::TypeMask: instr.type_name = attrs.at("type").as_string(); break;
            case Op::SumIn: {
                instr.index = json_int(attrs.at("register"));
                const auto select = parse_select(attrs.at("select").as_string());
                if (!select) {
                    throw Error(Errc::Json, "unknown sum_in selector in IR JSON");
                }
                instr.select = *select;
                break;
            }
            case Op::CountIn:
                instr.index = json_int(attrs.at("register"));
                instr.k = json_int(attrs.at("k"));
                break;
            case Op::Delay:
                instr.index = json_int(attrs.at("register"));
                instr.k = json_int(attrs.at("ticks"));
                break;
            case Op::Lut:
                for (const Json& entry : attrs.at("table").as_array()) {
                    instr.table.push_back(entry.as_double());
                }
                break;
            default: break;
        }
        ir.instrs.push_back(std::move(instr));
    }
    for (const Json& item : json.at("writes").as_array()) {
        ir.writes.push_back(json_int(item.at("value")));
    }
    if (!json.at("gap").is_null()) {
        IrGap gap;
        gap.reg = json_int(json.at("gap").at("register"));
        gap.scale_param = json.at("gap").at("scale_param").is_null() ? -1 : json_int(json.at("gap").at("scale_param"));
        ir.gap = gap;
    }
    const Json& observation = json.at("observation");
    const auto op = parse_obs_operator(observation.at("operator").as_string());
    if (!op) {
        throw Error(Errc::Json, "unknown observation operator in IR JSON");
    }
    ir.observation.op = *op;
    ir.observation.reg = json_int(observation.at("register"));
    if (*op == ObsOperator::CalciumLinearV1) {
        const Json& tau = observation.at("tau");
        if (const Json* param = tau.find("param")) {
            ir.observation.tau.kind = IrTauOperand::Kind::Param;
            ir.observation.tau.param = json_int(*param);
        } else {
            ir.observation.tau.kind = IrTauOperand::Kind::Const;
            ir.observation.tau.value = tau.at("const").as_double();
        }
    }
    for (const Json& name : json.at("eliminated").at("registers").as_array()) {
        ir.eliminated_registers.push_back(name.as_string());
    }
    for (const Json& name : json.at("eliminated").at("parameters").as_array()) {
        ir.eliminated_params.push_back(name.as_string());
    }
    ir.canonical_source = json.at("canonical_source").as_string();
    ir.program_hash = json.at("program_hash").as_string();
    ir.bits = compute_bits(ir);
    return ir;
}

}  // namespace occamworm
