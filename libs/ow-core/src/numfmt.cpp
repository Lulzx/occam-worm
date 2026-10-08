#include "occamworm/core/numfmt.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>

namespace occamworm {

Decimal shortest_decimal(double value) {
    Decimal out;
    if (value == 0.0) {
        out.digits = "0";
        return out;
    }
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
    const std::string_view text(buffer, static_cast<std::size_t>(result.ptr - buffer));
    // Format: [-]d[.ddd]e[+-]XX
    std::size_t pos = 0;
    if (text[pos] == '-') {
        out.negative = true;
        ++pos;
    }
    const std::size_t e_pos = text.find('e');
    for (std::size_t i = pos; i < e_pos; ++i) {
        if (text[i] != '.') {
            out.digits.push_back(text[i]);
        }
    }
    while (out.digits.size() > 1 && out.digits.back() == '0') {
        out.digits.pop_back();
    }
    std::size_t exp_pos = e_pos + 1;
    if (text[exp_pos] == '+') {
        ++exp_pos;
    }
    std::from_chars(text.data() + exp_pos, text.data() + text.size(), out.exponent);
    return out;
}

std::string canonical_number(double value) {
    const Decimal d = shortest_decimal(value);
    std::string out;
    if (d.negative) {
        out.push_back('-');
    }
    out.push_back(d.digits[0]);
    if (d.digits.size() > 1) {
        out.push_back('.');
        out.append(d.digits, 1, std::string::npos);
    }
    out.push_back('e');
    out += std::to_string(d.exponent);
    return out;
}

std::optional<double> parse_number(std::string_view text) {
    // Validate the grammar ourselves (strtod accepts hex, inf, nan and leading spaces).
    std::size_t i = 0;
    const auto is_digit = [&](std::size_t k) { return k < text.size() && text[k] >= '0' && text[k] <= '9'; };
    if (i < text.size() && (text[i] == '-' || text[i] == '+')) {
        ++i;
    }
    if (!is_digit(i)) {
        return std::nullopt;
    }
    while (is_digit(i)) {
        ++i;
    }
    if (i < text.size() && text[i] == '.') {
        ++i;
        if (!is_digit(i)) {
            return std::nullopt;
        }
        while (is_digit(i)) {
            ++i;
        }
    }
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        if (i < text.size() && (text[i] == '-' || text[i] == '+')) {
            ++i;
        }
        if (!is_digit(i)) {
            return std::nullopt;
        }
        while (is_digit(i)) {
            ++i;
        }
    }
    if (i != text.size()) {
        return std::nullopt;
    }
    const std::string copy(text);
    const double value = std::strtod(copy.c_str(), nullptr);
    if (!std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

int elias_gamma_bits(unsigned long long n) {
    int log2_floor = 0;
    while (n > 1) {
        n >>= 1U;
        ++log2_floor;
    }
    return 2 * log2_floor + 1;
}

int elias_gamma0_bits(unsigned long long n) { return elias_gamma_bits(n + 1); }

int zigzag_gamma0_bits(long long n) {
    const unsigned long long folded = n >= 0 ? 2ULL * static_cast<unsigned long long>(n)
                                             : 2ULL * static_cast<unsigned long long>(-(n + 1)) + 1ULL;
    return elias_gamma0_bits(folded);
}

int constant_bits(double value) {
    const Decimal d = shortest_decimal(value);
    const int digit_count = static_cast<int>(d.digits.size());
    return 1 + elias_gamma_bits(static_cast<unsigned long long>(digit_count)) + 4 * digit_count +
           zigzag_gamma0_bits(d.exponent);
}

}  // namespace occamworm
