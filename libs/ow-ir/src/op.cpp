#include "occamworm/ir/op.hpp"

#include <array>

namespace occamworm {
namespace {

struct OpName {
    Op op;
    std::string_view name;
};

constexpr std::array kOpNames = {
    OpName{Op::Const, "const"},
    OpName{Op::Param, "param"},
    OpName{Op::State, "state"},
    OpName{Op::Stimulus, "stimulus"},
    OpName{Op::TypeMask, "type_mask"},
    OpName{Op::SumIn, "sum_in"},
    OpName{Op::CountIn, "count_in"},
    OpName{Op::Delay, "delay"},
    OpName{Op::Add, "add"},
    OpName{Op::Mul, "mul"},
    OpName{Op::Neg, "neg"},
    OpName{Op::Abs, "abs"},
    OpName{Op::Min, "min"},
    OpName{Op::Max, "max"},
    OpName{Op::Clamp, "clamp"},
    OpName{Op::Relu, "relu"},
    OpName{Op::Tanh, "tanh"},
    OpName{Op::Sigmoid, "sigmoid"},
    OpName{Op::Threshold, "threshold"},
    OpName{Op::Select, "select"},
    OpName{Op::Lut, "lut"},
    OpName{Op::LeakyIntegrate, "leaky_integrate"},
    OpName{Op::EulerLeak, "euler_leak"},
};

}  // namespace

std::string_view op_name(Op op) {
    for (const OpName& entry : kOpNames) {
        if (entry.op == op) {
            return entry.name;
        }
    }
    return "?";
}

std::optional<Op> parse_op_name(std::string_view name) {
    for (const OpName& entry : kOpNames) {
        if (entry.name == name) {
            return entry.op;
        }
    }
    return std::nullopt;
}

std::string_view tier_name(Tier tier) { return tier == Tier::G0 ? "G0" : "G1"; }

std::optional<Tier> parse_tier(std::string_view name) {
    if (name == "G0") {
        return Tier::G0;
    }
    if (name == "G1") {
        return Tier::G1;
    }
    return std::nullopt;
}

std::string_view select_name(SumSelect select) {
    switch (select) {
        case SumSelect::Exc: return "exc";
        case SumSelect::Inh: return "inh";
        case SumSelect::All: return "all";
        case SumSelect::Mod: return "mod";
    }
    return "?";
}

std::optional<SumSelect> parse_select(std::string_view name) {
    if (name == "exc") {
        return SumSelect::Exc;
    }
    if (name == "inh") {
        return SumSelect::Inh;
    }
    if (name == "all") {
        return SumSelect::All;
    }
    if (name == "mod") {
        return SumSelect::Mod;
    }
    return std::nullopt;
}

std::string_view obs_operator_name(ObsOperator op) {
    return op == ObsOperator::IdentityV1 ? "identity_v1" : "calcium_linear_v1";
}

std::optional<ObsOperator> parse_obs_operator(std::string_view name) {
    if (name == "identity_v1") {
        return ObsOperator::IdentityV1;
    }
    if (name == "calcium_linear_v1") {
        return ObsOperator::CalciumLinearV1;
    }
    return std::nullopt;
}

bool op_allowed_in_tier(Tier tier, Op op) {
    switch (op) {
        // Allowed in both tiers.
        case Op::Const:
        case Op::State:
        case Op::Stimulus:
        case Op::TypeMask:
        case Op::Delay:
        case Op::Add:
        case Op::Mul:
        case Op::Neg:
        case Op::Abs:
        case Op::Min:
        case Op::Max:
        case Op::Clamp:
        case Op::Threshold:
        case Op::Select:
            return true;
        // G0 only: discrete truth-table machinery.
        case Op::CountIn:
        case Op::Lut:
            return tier == Tier::G0;
        // G1 only: continuous local rules.
        case Op::Param:
        case Op::SumIn:
        case Op::Relu:
        case Op::Tanh:
        case Op::Sigmoid:
        case Op::LeakyIntegrate:
        case Op::EulerLeak:
            return tier == Tier::G1;
    }
    return false;
}

bool op_is_commutative_nary(Op op) { return op == Op::Add || op == Op::Mul || op == Op::Min || op == Op::Max; }

int op_code_index(Op op) {
    switch (op) {
        case Op::Stimulus: return 0;
        case Op::TypeMask: return 1;
        case Op::SumIn: return 2;
        case Op::CountIn: return 3;
        case Op::Delay: return 4;
        case Op::Neg: return 5;
        case Op::Abs: return 6;
        case Op::Min: return 7;
        case Op::Max: return 8;
        case Op::Clamp: return 9;
        case Op::Relu: return 10;
        case Op::Tanh: return 11;
        case Op::Sigmoid: return 12;
        case Op::Threshold: return 13;
        case Op::Select: return 14;
        case Op::Lut: return 15;
        case Op::LeakyIntegrate: return 16;
        case Op::EulerLeak: return 17;
        default: return -1;  // State, Const, Param, Add, Mul have dedicated short codewords
    }
}

}  // namespace occamworm
