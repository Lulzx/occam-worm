#include "occamworm/core/units.hpp"

#include <cstdlib>

#include "occamworm/core/error.hpp"

namespace occamworm {
namespace {

[[noreturn]] void fail(const std::string& message) { throw Error(Errc::Unit, message); }

class UnitParser {
public:
    explicit UnitParser(std::string_view text) : text_(text) {}

    Unit parse() {
        skip_space();
        Unit result = parse_term();
        skip_space();
        while (pos_ < text_.size()) {
            const char op = text_[pos_];
            if (op != '*' && op != '/') {
                fail("unexpected character in unit expression '" + std::string(text_) + "'");
            }
            ++pos_;
            skip_space();
            const Unit term = parse_term();
            result = (op == '*') ? result * term : result * term.inverse();
            check_range(result);
            skip_space();
        }
        return result;
    }

private:
    void skip_space() {
        while (pos_ < text_.size() && text_[pos_] == ' ') {
            ++pos_;
        }
    }

    static void check_range(const Unit& unit) {
        if (std::abs(unit.seconds) > kMaxUnitExponent || std::abs(unit.volts) > kMaxUnitExponent) {
            fail("unit exponent out of range");
        }
    }

    std::string_view read_word() {
        const std::size_t start = pos_;
        while (pos_ < text_.size() && ((text_[pos_] >= 'a' && text_[pos_] <= 'z') || (text_[pos_] >= 'A' && text_[pos_] <= 'Z') ||
                                       (text_[pos_] >= '0' && text_[pos_] <= '9') || text_[pos_] == '_')) {
            ++pos_;
        }
        return text_.substr(start, pos_ - start);
    }

    Unit parse_term() {
        const std::string_view word = read_word();
        Unit base;
        if (word == "1" || word == "dimensionless") {
            base = Unit::none();
        } else if (word == "s") {
            base = {1, 0};
        } else if (word == "V") {
            base = {0, 1};
        } else {
            fail("unknown unit '" + std::string(word) + "' (known: s, V, 1, dimensionless)");
        }
        if (pos_ < text_.size() && text_[pos_] == '^') {
            ++pos_;
            bool negative = false;
            if (pos_ < text_.size() && text_[pos_] == '-') {
                negative = true;
                ++pos_;
            }
            const std::size_t start = pos_;
            int exponent = 0;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                exponent = exponent * 10 + (text_[pos_] - '0');
                if (exponent > kMaxUnitExponent) {
                    fail("unit exponent out of range");
                }
                ++pos_;
            }
            if (pos_ == start) {
                fail("missing exponent after '^'");
            }
            if (negative) {
                exponent = -exponent;
            }
            base = {base.seconds * exponent, base.volts * exponent};
        }
        check_range(base);
        return base;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

void append_factor(std::string& out, const char* name, int exponent) {
    if (exponent == 0) {
        return;
    }
    if (!out.empty()) {
        out.push_back('*');
    }
    out += name;
    if (exponent != 1) {
        out += "^" + std::to_string(exponent);
    }
}

}  // namespace

Unit parse_unit(std::string_view text) { return UnitParser(text).parse(); }

std::string to_string(const Unit& unit) {
    if (unit.dimensionless()) {
        return "1";
    }
    std::string out;
    append_factor(out, "V", unit.volts);
    append_factor(out, "s", unit.seconds);
    return out;
}

}  // namespace occamworm
