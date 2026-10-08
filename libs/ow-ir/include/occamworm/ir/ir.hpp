#pragma once
// Canonical SSA-like intermediate representation (§5.6). `Instr` is used both for the unnormalised
// resolved graph (checker output) and for the canonical instruction list.

#include <optional>
#include <string>
#include <vector>

#include "occamworm/core/units.hpp"
#include "occamworm/ir/op.hpp"

namespace occamworm {

struct Instr {
    Op op = Op::Const;
    std::vector<int> args;        // earlier instruction ids, in evaluation order
    double value = 0.0;           // Const
    int index = 0;                // Param: parameter index; State/SumIn/CountIn/Delay: register index
    int k = 0;                    // CountIn: compared integer; Delay: ticks
    SumSelect select = SumSelect::All;  // SumIn
    std::string type_name;        // TypeMask
    std::vector<double> table;    // Lut
    Unit unit;                    // unit of the produced value
    bool weak_zero = false;       // checker-internal: an unannotated literal 0 that adopts the unit it is used with
};

struct IrRegister {
    std::string name;         // canonical name s0, s1, ...
    std::string source_name;  // name in the compiled source (metadata, not hashed)
    Unit unit;
    double init = 0.0;
};

struct IrParam {
    std::string name;         // canonical name p0, p1, ...
    std::string source_name;  // metadata, not hashed
    Unit unit;
    double value = 0.0;       // default value; part of the hash only for fixed parameters
    double lower = 0.0;
    double upper = 0.0;
    bool trainable = false;
    int bits = 0;             // declared precision (trainable only); charged in L_params
};

struct IrTauOperand {
    enum class Kind { None, Param, Const };
    Kind kind = Kind::None;
    int param = -1;      // Param: parameter index
    double value = 0.0;  // Const: seconds
};

struct IrObservation {
    ObsOperator op = ObsOperator::IdentityV1;
    int reg = 0;
    IrTauOperand tau;  // CalciumLinearV1 only
};

struct IrGap {
    int reg = 0;
    int scale_param = -1;  // -1: scale 1
};

struct BitBreakdown {
    // Per-component bits, in the order they are reported.
    int header = 0;
    int registers = 0;
    int parameters = 0;
    int instructions = 0;  // all instruction codewords and arguments
    int writes = 0;
    int gap = 0;
    int observation = 0;
    int type_dispatch = 0;   // type_mask names and lut tables; included in `instructions`, listed separately
    int l_ast = 0;           // header + registers + parameters + (instructions - type_dispatch) + writes + gap + observation
    int l_topology_overrides = 0;  // always 0 in G0/G1: no topology overrides exist yet
    int l_struct = 0;        // l_ast + l_topology_overrides + type_dispatch
    int l_params = 0;        // sum of declared bits of trainable parameters
    std::vector<std::pair<std::string, int>> instruction_bits_by_op;  // sorted by op name
};

struct Ir {
    std::string rule_name;  // metadata, not hashed
    Tier tier = Tier::G1;
    std::optional<double> dt_max;
    Unit stimulus_unit;
    std::vector<IrRegister> registers;
    std::vector<IrParam> params;
    std::vector<Instr> instrs;
    std::vector<int> writes;  // writes[r] = instruction id of the new value of register r
    std::optional<IrGap> gap;
    IrObservation observation;
    std::vector<std::string> eliminated_registers;  // source names removed as dead
    std::vector<std::string> eliminated_params;

    // Derived by canonicalise().
    std::string canonical_source;  // identity text; program_hash = SHA-256 of these bytes
    std::string program_hash;
    BitBreakdown bits;
};

}  // namespace occamworm
