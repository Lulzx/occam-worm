#pragma once
// WRL operator set, grammar tiers and observation operators (docs/language/WRL_SYNTAX.md).

#include <optional>
#include <string_view>

namespace occamworm {

inline constexpr std::string_view kGrammarVersion = "0.1";

enum class Tier { G0, G1 };

enum class Op {
    // Leaves.
    Const,       // attrs: value, unit
    Param,       // attrs: index (parameter)
    State,       // attrs: index (register); the OLD value of the register
    Stimulus,    // exogenous drive of this neuron at this tick
    TypeMask,    // attrs: type_name; 1 if the neuron has that type, else 0
    // Neighbourhood and history reads (reads of old state, §6.1).
    SumIn,       // attrs: index (register), select; weighted chemical input sum
    CountIn,     // attrs: index (register), k; number of chemical inputs whose register value equals k
    Delay,       // attrs: index (register), k = ticks; own register k ticks ago
    // Arithmetic.
    Add,         // n-ary, left fold in listed order
    Mul,         // n-ary, left fold in listed order
    Neg,
    Abs,
    Min,         // n-ary
    Max,         // n-ary
    Clamp,       // clamp(x, lo, hi) = min(max(x, lo), hi)
    // Nonlinearities.
    Relu,
    Tanh,
    Sigmoid,
    Threshold,   // threshold(x, theta) = 1 if x >= theta else 0
    // Branching and tables.
    Select,      // select(c, a, b) = a if c != 0 else b
    Lut,         // attrs: table; lut(x) = table[clamp(floor(x), 0, n-1)]
    // Temporal.
    LeakyIntegrate,  // exact exponential discretisation (unconditionally stable)
    EulerLeak,       // explicit-Euler discretisation (stability bound checked at compile time)
};

enum class SumSelect { Exc, Inh, All };

enum class ObsOperator { IdentityV1, CalciumLinearV1 };

std::string_view op_name(Op op);
std::optional<Op> parse_op_name(std::string_view name);  // canonical IR ops only (no sugar)
std::string_view tier_name(Tier tier);
std::optional<Tier> parse_tier(std::string_view name);
std::string_view select_name(SumSelect select);
std::optional<SumSelect> parse_select(std::string_view name);
std::string_view obs_operator_name(ObsOperator op);
std::optional<ObsOperator> parse_obs_operator(std::string_view name);

bool op_allowed_in_tier(Tier tier, Op op);
bool op_is_commutative_nary(Op op);  // Add, Mul, Min, Max

// Index of an operator among the "rare" instruction kinds (L_struct code `111` + 5 bits); see bits.cpp.
int op_code_index(Op op);

}  // namespace occamworm
