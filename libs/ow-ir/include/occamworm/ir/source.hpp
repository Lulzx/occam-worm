#pragma once
// Source serializer: prints an Ir as WRL source. The default "identity" text is the canonical IR byte
// string that is hashed (docs/language/WRL_SYNTAX.md, "Canonical text").

#include <string>

#include "occamworm/ir/ir.hpp"

namespace occamworm {

struct PrintOptions {
    // identity = true: canonical names only, no rule name, trainable values omitted (hash input).
    // identity = false: also prints `rule <name>` and trainable parameter values (executable source that
    // reproduces the program's default parameter values when compiled).
    bool identity = true;
};

std::string print_source(const Ir& ir, const PrintOptions& options = {});

// One instruction as the right-hand side of a `let`, e.g. "add(n1, n2)" or "5e-3[s]".
std::string print_instruction(const Instr& instr, const std::vector<IrRegister>& registers,
                              const std::vector<IrParam>& params);

}  // namespace occamworm
