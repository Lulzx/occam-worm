#pragma once
// Canonicalisation (§5.6 steps 3-6): commutative/associative normalisation, constant folding, algebraic
// simplification, common-subexpression elimination, dead-register elimination, alpha-renaming of
// registers, parameters and locals, canonical numbering, program hash and bit accounting.

#include "occamworm/ir/check.hpp"
#include "occamworm/ir/ir.hpp"

namespace occamworm {

// Registers and parameters are relabelled to the lexicographically smallest canonical text over all
// relabellings when registers! * parameters! <= kMaxRelabelings; larger programs keep declaration order.
constexpr std::size_t kMaxRelabelings = 720;

Ir canonicalise(const Resolved& resolved);

}  // namespace occamworm
