#pragma once
// Structural description length L_struct with a frozen prefix-free code (§5.7), and L_params.
// The code is specified in docs/language/WRL_SYNTAX.md ("Bit cost"); changing it changes the meaning of
// every stored bit count, so it is versioned with the grammar version.

#include "occamworm/ir/ir.hpp"

namespace occamworm {

inline constexpr int kBitCodeVersion = 1;

BitBreakdown compute_bits(const Ir& ir);

}  // namespace occamworm
