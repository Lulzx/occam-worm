#pragma once
// Name resolution, unit and type checking, tier and stability checks. The result is the unnormalised
// "resolved graph": one Instr per distinct sub-expression, in creation order, with registers and
// parameters numbered in declaration order.

#include <optional>
#include <string>
#include <vector>

#include "occamworm/ir/ast.hpp"
#include "occamworm/ir/ir.hpp"

namespace occamworm {

constexpr std::size_t kMaxRawNodes = 50000;
constexpr std::size_t kMaxLutEntries = 4096;

struct ResolvedObservation {
    ObsOperator op = ObsOperator::IdentityV1;
    int reg = 0;
    IrTauOperand tau;
};

struct Resolved {
    std::string rule_name;
    Tier tier = Tier::G1;
    std::optional<double> dt_max;
    Unit stimulus_unit;
    std::vector<RegisterDecl> registers;  // declaration order
    std::vector<IrParam> params;          // declaration order; name == source name
    std::vector<Instr> raw;               // unnormalised nodes; args refer to earlier entries
    std::vector<int> writes;              // per register: raw node id of its new value (hold -> state node)
    std::optional<IrGap> gap;
    ResolvedObservation observation;
};

// Throws Error with Errc::Name, Unit, Type, Tier, Stability or Limit.
Resolved check_program(const ProgramAst& program);

}  // namespace occamworm
