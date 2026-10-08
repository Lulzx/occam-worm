#pragma once
// Error type shared by every library. Codes are stable strings so that diagnostics are
// deterministic and testable (docs/language/WRL_SYNTAX.md, "Diagnostics").

#include <stdexcept>
#include <string>
#include <string_view>

namespace occamworm {

enum class Errc {
    Syntax,     // lexical or grammatical error
    Name,       // unknown, duplicate or misplaced identifier
    Unit,       // unit mismatch or malformed unit expression
    Type,       // operator arity, argument kind or domain error
    Tier,       // operator not permitted in the declared grammar tier
    Stability,  // explicit-integrator stability bound violated
    Limit,      // input exceeds a hard resource limit
    Graph,      // invalid graph
    Input,      // invalid simulation input or configuration
    Runtime,    // non-finite value or other failure while simulating
    Json,       // malformed JSON
    Io,         // file access
};

constexpr std::string_view errc_name(Errc code) {
    switch (code) {
        case Errc::Syntax: return "E_SYNTAX";
        case Errc::Name: return "E_NAME";
        case Errc::Unit: return "E_UNIT";
        case Errc::Type: return "E_TYPE";
        case Errc::Tier: return "E_TIER";
        case Errc::Stability: return "E_STABILITY";
        case Errc::Limit: return "E_LIMIT";
        case Errc::Graph: return "E_GRAPH";
        case Errc::Input: return "E_INPUT";
        case Errc::Runtime: return "E_RUNTIME";
        case Errc::Json: return "E_JSON";
        case Errc::Io: return "E_IO";
    }
    return "E_UNKNOWN";
}

class Error : public std::runtime_error {
public:
    Error(Errc code, std::string message, int line = 0, int column = 0)
        : std::runtime_error(format(code, message, line, column)),
          code_(code),
          message_(std::move(message)),
          line_(line),
          column_(column) {}

    Errc code() const noexcept { return code_; }
    const std::string& message() const noexcept { return message_; }
    int line() const noexcept { return line_; }      // 1-based; 0 when unknown
    int column() const noexcept { return column_; }  // 1-based; 0 when unknown

private:
    static std::string format(Errc code, const std::string& message, int line, int column) {
        std::string out(errc_name(code));
        if (line > 0) {
            out += " at " + std::to_string(line) + ":" + std::to_string(column);
        }
        out += ": " + message;
        return out;
    }

    Errc code_;
    std::string message_;
    int line_;
    int column_;
};

}  // namespace occamworm
