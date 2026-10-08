#pragma once
// Physical units for WRL (§5.2). A unit is a product of integer powers of two base units:
// seconds (s) and volts (V). Dimensionless is the identity element.

#include <compare>
#include <string>
#include <string_view>

namespace occamworm {

struct Unit {
    int seconds = 0;  // exponent of s
    int volts = 0;    // exponent of V

    constexpr bool dimensionless() const { return seconds == 0 && volts == 0; }
    constexpr auto operator<=>(const Unit&) const = default;

    friend constexpr Unit operator*(Unit a, Unit b) { return {a.seconds + b.seconds, a.volts + b.volts}; }
    constexpr Unit inverse() const { return {-seconds, -volts}; }

    static constexpr Unit none() { return {}; }
    static constexpr Unit time() { return {1, 0}; }
};

constexpr int kMaxUnitExponent = 8;

// Grammar: unit := term (('*' | '/') term)* ; term := 'dimensionless' | '1' | 's' | 'V' , optionally '^' int.
// Operators associate left to right. Throws Error(Errc::Unit) on malformed input.
Unit parse_unit(std::string_view text);

// Canonical text: "1" for dimensionless, otherwise V then s, e.g. "V", "s^-1", "V*s^-1".
std::string to_string(const Unit& unit);

}  // namespace occamworm
