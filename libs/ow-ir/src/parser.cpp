#include "occamworm/ir/parser.hpp"

#include <algorithm>
#include <cmath>

#include "occamworm/core/error.hpp"
#include "occamworm/core/numfmt.hpp"

namespace occamworm {
namespace {

bool is_ident_start(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_ident_char(char c) { return is_ident_start(c) || is_digit(c); }

constexpr std::size_t kMaxIdentLength = 64;

}  // namespace

std::vector<Token> tokenize(std::string_view source) {
    if (source.size() > kMaxSourceBytes) {
        throw Error(Errc::Limit, "source exceeds " + std::to_string(kMaxSourceBytes) + " bytes");
    }
    std::vector<Token> tokens;
    int line = 1;
    int column = 1;
    int depth = 0;  // open ( and [ ; newlines are insignificant inside brackets
    std::size_t i = 0;
    const auto advance = [&](std::size_t count) {
        i += count;
        column += static_cast<int>(count);
    };
    const auto push = [&](Token token) {
        if (tokens.size() >= kMaxTokens) {
            throw Error(Errc::Limit, "too many tokens", line, column);
        }
        tokens.push_back(std::move(token));
    };
    while (i < source.size()) {
        const char c = source[i];
        if (c == ' ' || c == '\t' || c == '\r') {
            advance(1);
        } else if (c == '\n') {
            if (depth == 0 && !tokens.empty() && tokens.back().kind != Token::Kind::Newline) {
                push({Token::Kind::Newline, "\n", 0.0, line, column});
            }
            ++i;
            ++line;
            column = 1;
        } else if (c == '#') {
            while (i < source.size() && source[i] != '\n') {
                ++i;
                ++column;
            }
        } else if (is_ident_start(c)) {
            const std::size_t start = i;
            const int start_column = column;
            while (i < source.size() && is_ident_char(source[i])) {
                advance(1);
            }
            if (i - start > kMaxIdentLength) {
                throw Error(Errc::Limit, "identifier longer than " + std::to_string(kMaxIdentLength) + " characters", line,
                            start_column);
            }
            push({Token::Kind::Ident, std::string(source.substr(start, i - start)), 0.0, line, start_column});
        } else if (is_digit(c)) {
            const std::size_t start = i;
            const int start_column = column;
            while (i < source.size() && is_digit(source[i])) {
                advance(1);
            }
            if (i < source.size() && source[i] == '.') {
                advance(1);
                while (i < source.size() && is_digit(source[i])) {
                    advance(1);
                }
            }
            if (i < source.size() && (source[i] == 'e' || source[i] == 'E')) {
                advance(1);
                if (i < source.size() && (source[i] == '+' || source[i] == '-')) {
                    advance(1);
                }
                while (i < source.size() && is_digit(source[i])) {
                    advance(1);
                }
            }
            if (i < source.size() && is_ident_char(source[i])) {
                throw Error(Errc::Syntax, "malformed number", line, start_column);
            }
            const std::string text(source.substr(start, i - start));
            const auto value = parse_number(text);
            if (!value) {
                throw Error(Errc::Syntax, "malformed or out-of-range number '" + text + "'", line, start_column);
            }
            push({Token::Kind::Number, text, *value, line, start_column});
        } else if (std::string_view("()[],:=+-*/^").find(c) != std::string_view::npos) {
            if (c == '(' || c == '[') {
                ++depth;
            } else if (c == ')' || c == ']') {
                if (--depth < 0) {
                    throw Error(Errc::Syntax, "unbalanced closing bracket", line, column);
                }
            }
            push({Token::Kind::Punct, std::string(1, c), 0.0, line, column});
            advance(1);
        } else {
            throw Error(Errc::Syntax, std::string("unexpected character '") + c + "'", line, column);
        }
    }
    if (depth != 0) {
        throw Error(Errc::Syntax, "unbalanced opening bracket at end of input", line, column);
    }
    if (!tokens.empty() && tokens.back().kind != Token::Kind::Newline) {
        push({Token::Kind::Newline, "\n", 0.0, line, column});
    }
    push({Token::Kind::End, "", 0.0, line, column});
    return tokens;
}

namespace {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    ProgramAst parse() {
        ProgramAst program;
        skip_newlines();
        const Token& first = peek();
        if (!is_word("wrl")) {
            fail(first, "a WRL source must start with 'wrl <version>'");
        }
        advance();
        const Token& version = peek();
        if (version.kind != Token::Kind::Number || version.text != "0.1") {
            fail(version, "unsupported WRL version (this compiler implements 0.1)");
        }
        program.wrl_version = version.text;
        advance();
        end_statement();

        bool has_rule = false;
        bool has_dt_max = false;
        bool has_stimulus_unit = false;
        while (true) {
            skip_newlines();
            const Token& head = peek();
            if (head.kind == Token::Kind::End) {
                break;
            }
            if (head.kind != Token::Kind::Ident) {
                fail(head, "expected a statement keyword");
            }
            const std::string keyword = head.text;
            if (keyword == "rule") {
                reject_duplicate(has_rule, head, "rule");
                advance();
                program.rule_name = expect_ident("rule name");
                end_statement();
            } else if (keyword == "tier") {
                reject_duplicate(program.has_tier, head, "tier");
                advance();
                const Token& name = peek();
                const std::string text = expect_ident("tier name");
                const auto tier = parse_tier(text);
                if (!tier) {
                    fail(name, "unknown tier '" + text + "' (known: G0, G1)");
                }
                program.tier = *tier;
                program.has_tier = true;
                end_statement();
            } else if (keyword == "dt_max") {
                reject_duplicate(has_dt_max, head, "dt_max");
                advance();
                program.dt_max = parse_dt_max();
                has_dt_max = true;
                end_statement();
            } else if (keyword == "stimulus_unit") {
                reject_duplicate(has_stimulus_unit, head, "stimulus_unit");
                advance();
                program.stimulus_unit = parse_unit_tokens();
                has_stimulus_unit = true;
                end_statement();
            } else if (keyword == "state") {
                program.statements.push_back(parse_state());
            } else if (keyword == "param") {
                program.statements.push_back(parse_param());
            } else if (keyword == "input" || keyword == "let" || keyword == "next") {
                program.statements.push_back(parse_binding(keyword));
            } else if (keyword == "gap") {
                program.statements.push_back(parse_gap());
            } else if (keyword == "observe") {
                program.statements.push_back(parse_observe());
            } else {
                fail(head, "unknown statement '" + keyword + "'");
            }
        }
        if (!program.has_tier) {
            throw Error(Errc::Syntax, "missing 'tier' declaration");
        }
        return program;
    }

private:
    [[noreturn]] static void fail(const Token& at, const std::string& message) {
        throw Error(Errc::Syntax, message, at.line, at.column);
    }

    static void reject_duplicate(bool already, const Token& at, const char* what) {
        if (already) {
            fail(at, std::string("duplicate '") + what + "' declaration");
        }
    }

    const Token& peek(std::size_t ahead = 0) const {
        const std::size_t index = std::min(pos_ + ahead, tokens_.size() - 1);
        return tokens_[index];
    }
    void advance() {
        if (pos_ + 1 < tokens_.size()) {
            ++pos_;
        }
    }
    bool is_word(std::string_view word) const { return peek().kind == Token::Kind::Ident && peek().text == word; }
    bool is_punct(char c) const { return peek().kind == Token::Kind::Punct && peek().text[0] == c; }

    void skip_newlines() {
        while (peek().kind == Token::Kind::Newline) {
            advance();
        }
    }
    void end_statement() {
        const Token& token = peek();
        if (token.kind != Token::Kind::Newline && token.kind != Token::Kind::End) {
            fail(token, "unexpected '" + token.text + "' at end of statement");
        }
    }
    void expect_punct(char c) {
        if (!is_punct(c)) {
            fail(peek(), std::string("expected '") + c + "'");
        }
        advance();
    }
    void expect_word(std::string_view word) {
        if (!is_word(word)) {
            fail(peek(), "expected '" + std::string(word) + "'");
        }
        advance();
    }
    std::string expect_ident(const char* what) {
        const Token& token = peek();
        if (token.kind != Token::Kind::Ident) {
            fail(token, std::string("expected ") + what);
        }
        std::string text = token.text;
        advance();
        return text;
    }

    double parse_signed_number() {
        bool negative = false;
        if (is_punct('-')) {
            negative = true;
            advance();
        } else if (is_punct('+')) {
            advance();
        }
        const Token& token = peek();
        if (token.kind != Token::Kind::Number) {
            fail(token, "expected a number");
        }
        const double value = token.number;
        advance();
        return negative ? -value : value;
    }

    // Collects tokens that spell a unit expression, stopping at the first token that cannot belong to one.
    Unit parse_unit_tokens() {
        const Token& first = peek();
        std::string text;
        while (true) {
            const Token& token = peek();
            const bool unit_word = token.kind == Token::Kind::Ident && (token.text == "s" || token.text == "V" || token.text == "dimensionless");
            const bool unit_number = token.kind == Token::Kind::Number && text_is_unit_number(token.text);
            const bool unit_punct = token.kind == Token::Kind::Punct && (token.text == "*" || token.text == "/" || token.text == "^" || token.text == "-");
            if (!unit_word && !unit_number && !unit_punct) {
                break;
            }
            text += token.text;
            advance();
        }
        if (text.empty()) {
            fail(first, "expected a unit");
        }
        try {
            return parse_unit(text);
        } catch (const Error& error) {
            throw Error(Errc::Unit, error.message(), first.line, first.column);
        }
    }
    static bool text_is_unit_number(const std::string& text) {
        return !text.empty() && std::ranges::all_of(text, [](char c) { return c >= '0' && c <= '9'; });
    }

    // "[unit]" suffix directly after a number inside an expression.
    Unit parse_bracket_unit() {
        expect_punct('[');
        const Unit unit = parse_unit_tokens();
        expect_punct(']');
        return unit;
    }

    double parse_dt_max() {
        const Token& at = peek();
        const double value = parse_signed_number();
        if (is_punct('[')) {
            const Unit unit = parse_bracket_unit();
            if (unit != Unit::time()) {
                fail(at, "dt_max must be a time in seconds");
            }
        }
        if (!(value > 0.0)) {
            throw Error(Errc::Type, "dt_max must be positive", at.line, at.column);
        }
        return value;
    }

    Statement parse_state() {
        Statement statement;
        statement.kind = Statement::Kind::State;
        statement.line = peek().line;
        statement.column = peek().column;
        advance();  // 'state'
        statement.name = expect_ident("register name");
        statement.reg.name = statement.name;
        expect_punct(':');
        statement.reg.unit = parse_unit_tokens();
        expect_punct('=');
        statement.reg.init = parse_signed_number();
        end_statement();
        return statement;
    }

    Statement parse_param() {
        Statement statement;
        statement.kind = Statement::Kind::Param;
        statement.line = peek().line;
        statement.column = peek().column;
        advance();  // 'param'
        statement.name = expect_ident("parameter name");
        ParamDecl& param = statement.param;
        param.name = statement.name;
        expect_punct(':');
        param.unit = parse_unit_tokens();
        if (is_punct('=')) {
            advance();
            param.value = parse_signed_number();
        }
        if (is_word("in")) {
            advance();
            expect_punct('[');
            param.lower = parse_signed_number();
            expect_punct(',');
            param.upper = parse_signed_number();
            expect_punct(']');
        }
        if (is_word("trainable")) {
            advance();
            param.trainable = true;
            expect_word("bits");
            const Token& bits = peek();
            const double value = parse_signed_number();
            if (value != std::floor(value) || value < 1 || value > 32) {
                fail(bits, "bits must be an integer in [1, 32]");
            }
            param.bits = static_cast<int>(value);
        } else if (is_word("fixed")) {
            advance();
        } else {
            fail(peek(), "expected 'trainable bits <n>' or 'fixed'");
        }
        end_statement();
        return statement;
    }

    Statement parse_binding(const std::string& keyword) {
        Statement statement;
        statement.line = peek().line;
        statement.column = peek().column;
        statement.kind = keyword == "input" ? Statement::Kind::Input : keyword == "let" ? Statement::Kind::Let : Statement::Kind::Next;
        advance();
        statement.name = expect_ident("a name");
        expect_punct('=');
        statement.expr = parse_expr(0);
        end_statement();
        return statement;
    }

    Statement parse_gap() {
        Statement statement;
        statement.kind = Statement::Kind::Gap;
        statement.line = peek().line;
        statement.column = peek().column;
        advance();  // 'gap'
        statement.name = expect_ident("register name");
        if (is_word("scale")) {
            advance();
            statement.gap_scale = expect_ident("parameter name");
        }
        end_statement();
        return statement;
    }

    Statement parse_observe() {
        Statement statement;
        statement.kind = Statement::Kind::Observe;
        statement.line = peek().line;
        statement.column = peek().column;
        advance();  // 'observe'
        statement.observe.operator_name = expect_ident("observation operator");
        expect_punct('(');
        statement.observe.register_name = expect_ident("register name");
        if (is_punct(',')) {
            advance();
            statement.observe.tau = parse_expr(0);
        }
        expect_punct(')');
        end_statement();
        return statement;
    }

    // expr := term (('+' | '-') term)*
    ExprPtr parse_expr(int depth) {
        check_depth(depth);
        ExprPtr left = parse_term(depth + 1);
        int chain = 0;
        while (is_punct('+') || is_punct('-')) {
            check_chain(++chain);
            const Token& op = peek();
            const bool subtract = op.text[0] == '-';
            advance();
            ExprPtr right = parse_term(depth + 1);
            if (subtract) {
                right = make_call("neg", op, {}, std::move(right));
            }
            left = make_call("add", op, std::move(left), std::move(right));
        }
        return left;
    }

    // term := unary ('*' unary)*
    ExprPtr parse_term(int depth) {
        check_depth(depth);
        ExprPtr left = parse_unary(depth + 1);
        int chain = 0;
        while (is_punct('*') || is_punct('/')) {
            check_chain(++chain);
            const Token& op = peek();
            if (op.text[0] == '/') {
                fail(op, "division is not part of WRL v0.1; use leaky_integrate/decay or multiply by a parameter");
            }
            advance();
            ExprPtr right = parse_unary(depth + 1);
            left = make_call("mul", op, std::move(left), std::move(right));
        }
        return left;
    }

    ExprPtr parse_unary(int depth) {
        check_depth(depth);
        if (is_punct('-')) {
            const Token& op = peek();
            advance();
            ExprPtr operand = parse_unary(depth + 1);
            return make_call("neg", op, {}, std::move(operand));
        }
        return parse_primary(depth + 1);
    }

    ExprPtr parse_primary(int depth) {
        check_depth(depth);
        const Token& token = peek();
        auto node = std::make_unique<Expr>();
        node->line = token.line;
        node->column = token.column;
        if (token.kind == Token::Kind::Number) {
            node->kind = Expr::Kind::Number;
            node->number = token.number;
            advance();
            if (is_punct('[')) {
                node->unit = parse_bracket_unit();
            }
            return node;
        }
        if (token.kind == Token::Kind::Ident) {
            node->name = token.text;
            advance();
            if (is_punct('(')) {
                node->kind = Expr::Kind::Call;
                advance();
                if (!is_punct(')')) {
                    while (true) {
                        node->args.push_back(parse_argument(depth + 1));
                        if (is_punct(',')) {
                            advance();
                            continue;
                        }
                        break;
                    }
                }
                expect_punct(')');
            } else {
                node->kind = Expr::Kind::Name;
            }
            return node;
        }
        if (is_punct('(')) {
            advance();
            ExprPtr inner = parse_expr(depth + 1);
            expect_punct(')');
            return inner;
        }
        fail(token, "expected an expression");
    }

    ExprPtr parse_argument(int depth) {
        if (!is_punct('[')) {
            return parse_expr(depth);
        }
        auto list = std::make_unique<Expr>();
        list->kind = Expr::Kind::List;
        list->line = peek().line;
        list->column = peek().column;
        advance();
        while (true) {
            auto element = std::make_unique<Expr>();
            element->kind = Expr::Kind::Number;
            element->line = peek().line;
            element->column = peek().column;
            element->number = parse_signed_number();
            list->args.push_back(std::move(element));
            if (is_punct(',')) {
                advance();
                continue;
            }
            break;
        }
        expect_punct(']');
        return list;
    }

    // Left-associative chains build a left-nested tree; bound its length so recursion stays shallow.
    static void check_chain(int length) {
        if (length > kMaxChainLength) {
            throw Error(Errc::Limit, "more than " + std::to_string(kMaxChainLength) + " operands in one sum or product");
        }
    }

    static void check_depth(int depth) {
        if (depth > kMaxExprDepth * 4) {  // each nesting level uses several parser frames
            throw Error(Errc::Limit, "expression nested deeper than " + std::to_string(kMaxExprDepth) + " levels");
        }
    }

    static ExprPtr make_call(const char* name, const Token& at, ExprPtr first, ExprPtr second) {
        auto call = std::make_unique<Expr>();
        call->kind = Expr::Kind::Call;
        call->name = name;
        call->line = at.line;
        call->column = at.column;
        if (first) {
            call->args.push_back(std::move(first));
        }
        call->args.push_back(std::move(second));
        return call;
    }

    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
};

}  // namespace

ProgramAst parse_program(std::string_view source) { return Parser(tokenize(source)).parse(); }

}  // namespace occamworm
