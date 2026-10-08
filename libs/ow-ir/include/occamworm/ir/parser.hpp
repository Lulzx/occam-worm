#pragma once
// Lexer and parser for the line-oriented WRL concrete syntax (docs/language/WRL_SYNTAX.md).

#include <string>
#include <string_view>
#include <vector>

#include "occamworm/ir/ast.hpp"

namespace occamworm {

constexpr std::size_t kMaxSourceBytes = 1U << 20;
constexpr int kMaxExprDepth = 32;
constexpr int kMaxChainLength = 64;
constexpr std::size_t kMaxTokens = 200000;

struct Token {
    enum class Kind { Ident, Number, Punct, Newline, End };

    Kind kind = Kind::End;
    std::string text;  // identifier text, number text, or the punctuation character
    double number = 0.0;
    int line = 0;
    int column = 0;
};

// Throws Error(Errc::Syntax) or Error(Errc::Limit).
std::vector<Token> tokenize(std::string_view source);
ProgramAst parse_program(std::string_view source);

}  // namespace occamworm
