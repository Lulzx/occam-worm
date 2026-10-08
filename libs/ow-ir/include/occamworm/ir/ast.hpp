#pragma once
// Typed WRL abstract syntax tree (§5.6 step 1). The parser produces this tree; names are not yet resolved.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "occamworm/core/units.hpp"
#include "occamworm/ir/op.hpp"

namespace occamworm {

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

// Expression node. Infix operators are desugared by the parser: a+b -> add(a,b), a-b -> add(a,neg(b)),
// a*b -> mul(a,b), -a -> neg(a).
struct Expr {
    enum class Kind {
        Number,  // literal; `unit` is set when written with a [unit] suffix
        Name,    // identifier (variable, register, parameter, or a selector/type word depending on position)
        Call,    // name(args...)
        List,    // [number, number, ...] (only valid as the table argument of lut)
    };

    Kind kind = Kind::Number;
    int line = 0;
    int column = 0;
    double number = 0.0;
    std::optional<Unit> unit;  // explicit unit annotation on a Number
    std::string name;          // Name or Call function name
    std::vector<ExprPtr> args;  // Call arguments or List elements
};

struct RegisterDecl {
    std::string name;
    Unit unit;
    double init = 0.0;
};

struct ParamDecl {
    std::string name;
    Unit unit;
    std::optional<double> value;  // omitted for trainable parameters -> midpoint of the bounds
    std::optional<double> lower;
    std::optional<double> upper;
    bool trainable = false;
    int bits = 0;
};

struct ObserveDecl {
    std::string operator_name;
    std::string register_name;
    ExprPtr tau;  // calcium_linear_v1 only: a parameter name or a number with [s]
};

// One statement, in source order. Name resolution happens in statement order (definitions before use).
struct Statement {
    enum class Kind { State, Param, Input, Let, Next, Gap, Observe };

    Kind kind = Kind::Let;
    int line = 0;
    int column = 0;
    std::string name;          // declared or target name
    RegisterDecl reg;          // State
    ParamDecl param;           // Param
    ExprPtr expr;              // Input, Let, Next
    std::string gap_scale;     // Gap: optional parameter name
    ObserveDecl observe;       // Observe
};

struct ProgramAst {
    std::string wrl_version;
    std::string rule_name = "unnamed";
    Tier tier = Tier::G1;
    bool has_tier = false;
    std::optional<double> dt_max;
    Unit stimulus_unit;
    std::vector<Statement> statements;
};

}  // namespace occamworm
