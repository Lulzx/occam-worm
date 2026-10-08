#pragma once
// Canonical number formatting. Canonical IR bytes (and therefore program hashes) contain doubles in
// shortest round-trip decimal scientific form, so they are identical on every platform.

#include <optional>
#include <string>
#include <string_view>

namespace occamworm {

// value = (-1)^negative * d1.d2d3... * 10^exponent, with the fewest significant digits that round-trip.
// Zero (including -0.0) is {false, "0", 0}. Input must be finite.
struct Decimal {
    bool negative = false;
    std::string digits;  // 1..17 characters, no leading zero except for the value zero, no trailing zeros
    int exponent = 0;
};

Decimal shortest_decimal(double value);

// "d[.ddd]e<exp>" with a leading '-' for negatives, no '+' and no exponent padding: 0.25 -> "2.5e-1",
// 1 -> "1e0", 12 -> "1.2e1", 0 -> "0e0".
std::string canonical_number(double value);

// Parses a JSON-style or WRL number (digits, optional fraction, optional exponent). Returns nullopt on
// syntax error, overflow to infinity, or a trailing remainder.
std::optional<double> parse_number(std::string_view text);

// Prefix-free integer codes used by the L_struct accounting (docs/language/WRL_SYNTAX.md, "Bit cost").
int elias_gamma_bits(unsigned long long n);   // n >= 1: 2*floor(log2 n)+1
int elias_gamma0_bits(unsigned long long n);  // n >= 0: elias_gamma_bits(n+1)
int zigzag_gamma0_bits(long long n);          // signed integers via zigzag then gamma0
int constant_bits(double value);              // typed constant: sign + digit count + digits + exponent

}  // namespace occamworm
