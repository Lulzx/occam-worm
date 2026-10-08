#include "occamworm/core/json.hpp"

#include <charconv>
#include <cmath>

#include "occamworm/core/error.hpp"
#include "occamworm/core/numfmt.hpp"

namespace occamworm {
namespace {

constexpr int kMaxDepth = 200;

[[noreturn]] void fail(const std::string& message) { throw Error(Errc::Json, message); }

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Json parse_document() {
        skip_space();
        Json value = parse_value(0);
        skip_space();
        if (pos_ != text_.size()) {
            error("trailing characters after JSON value");
        }
        return value;
    }

private:
    [[noreturn]] void error(const std::string& message) const {
        // Report line/column of the current position.
        int line = 1;
        int column = 1;
        for (std::size_t i = 0; i < pos_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        throw Error(Errc::Json, message, line, column);
    }

    bool at_end() const { return pos_ >= text_.size(); }
    char peek() const { return at_end() ? '\0' : text_[pos_]; }

    void skip_space() {
        while (!at_end() && (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' || text_[pos_] == '\r')) {
            ++pos_;
        }
    }

    void expect(char c) {
        if (peek() != c) {
            error(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    bool consume_literal(std::string_view literal) {
        if (text_.substr(pos_, literal.size()) == literal) {
            pos_ += literal.size();
            return true;
        }
        return false;
    }

    Json parse_value(int depth) {
        if (depth > kMaxDepth) {
            error("nesting too deep");
        }
        if (at_end()) {
            error("unexpected end of input");
        }
        const char c = peek();
        if (c == '{') {
            return parse_object(depth);
        }
        if (c == '[') {
            return parse_array(depth);
        }
        if (c == '"') {
            return Json(parse_string());
        }
        if (c == 't' && consume_literal("true")) {
            return Json(true);
        }
        if (c == 'f' && consume_literal("false")) {
            return Json(false);
        }
        if (c == 'n' && consume_literal("null")) {
            return Json();
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number_value();
        }
        error("unexpected character");
    }

    Json parse_object(int depth) {
        expect('{');
        Json::Object members;
        skip_space();
        if (peek() == '}') {
            ++pos_;
            return Json(std::move(members));
        }
        while (true) {
            skip_space();
            if (peek() != '"') {
                error("expected object key");
            }
            std::string key = parse_string();
            skip_space();
            expect(':');
            skip_space();
            Json value = parse_value(depth + 1);
            for (const auto& member : members) {
                if (member.first == key) {
                    error("duplicate object key '" + key + "'");
                }
            }
            members.emplace_back(std::move(key), std::move(value));
            skip_space();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            return Json(std::move(members));
        }
    }

    Json parse_array(int depth) {
        expect('[');
        Json::Array items;
        skip_space();
        if (peek() == ']') {
            ++pos_;
            return Json(std::move(items));
        }
        while (true) {
            skip_space();
            items.push_back(parse_value(depth + 1));
            skip_space();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            return Json(std::move(items));
        }
    }

    unsigned parse_hex4() {
        unsigned code = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = peek();
            unsigned digit = 0;
            if (c >= '0' && c <= '9') {
                digit = static_cast<unsigned>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                digit = static_cast<unsigned>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                digit = static_cast<unsigned>(c - 'A' + 10);
            } else {
                error("invalid \\u escape");
            }
            code = code * 16U + digit;
            ++pos_;
        }
        return code;
    }

    static void append_utf8(std::string& out, unsigned code) {
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (at_end()) {
                error("unterminated string");
            }
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                --pos_;
                error("control character in string");
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (at_end()) {
                error("unterminated escape");
            }
            const char escape = text_[pos_++];
            switch (escape) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    unsigned code = parse_hex4();
                    if (code >= 0xD800 && code <= 0xDBFF) {
                        if (!consume_literal("\\u")) {
                            error("unpaired surrogate");
                        }
                        const unsigned low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            error("invalid surrogate pair");
                        }
                        code = 0x10000 + ((code - 0xD800) << 10U) + (low - 0xDC00);
                    } else if (code >= 0xDC00 && code <= 0xDFFF) {
                        error("unpaired surrogate");
                    }
                    append_utf8(out, code);
                    break;
                }
                default:
                    --pos_;
                    error("invalid escape");
            }
        }
    }

    Json parse_number_value() {
        const std::size_t start = pos_;
        if (peek() == '-') {
            ++pos_;
        }
        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        } else {
            error("invalid number");
        }
        if (peek() == '.') {
            ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) {
                error("invalid number");
            }
            while (peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') {
                ++pos_;
            }
            if (!(peek() >= '0' && peek() <= '9')) {
                error("invalid number");
            }
            while (peek() >= '0' && peek() <= '9') {
                ++pos_;
            }
        }
        const auto value = parse_number(text_.substr(start, pos_ - start));
        if (!value) {
            pos_ = start;
            error("number out of range");
        }
        return Json(*value);
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

void write_string(std::string& out, const std::string& text) {
    static constexpr char kHex[] = "0123456789abcdef";
    out.push_back('"');
    for (const char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += "\\u00";
                    out.push_back(kHex[(static_cast<unsigned char>(c) >> 4) & 0xF]);
                    out.push_back(kHex[static_cast<unsigned char>(c) & 0xF]);
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

void write_number(std::string& out, double value) {
    if (!std::isfinite(value)) {
        fail("cannot serialise a non-finite number");
    }
    if (value == std::floor(value) && std::fabs(value) < 9.007199254740992e15) {
        if (value == 0.0) {
            out.push_back('0');  // also normalises -0.0
            return;
        }
        out += std::to_string(static_cast<long long>(value));
        return;
    }
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    out.append(buffer, result.ptr);
}

void newline(std::string& out, int indent, int level) {
    out.push_back('\n');
    out.append(static_cast<std::size_t>(indent * level), ' ');
}

}  // namespace

bool Json::as_bool() const {
    if (!is_bool()) {
        fail("expected a boolean");
    }
    return std::get<bool>(value_);
}

double Json::as_double() const {
    if (!is_number()) {
        fail("expected a number");
    }
    return std::get<double>(value_);
}

long long Json::as_int() const {
    const double value = as_double();
    if (value != std::floor(value) || std::fabs(value) > 9.007199254740992e15) {
        fail("expected an integer");
    }
    return static_cast<long long>(value);
}

const std::string& Json::as_string() const {
    if (!is_string()) {
        fail("expected a string");
    }
    return std::get<std::string>(value_);
}

const Json::Array& Json::as_array() const {
    if (!is_array()) {
        fail("expected an array");
    }
    return std::get<Array>(value_);
}

Json::Array& Json::as_array() {
    if (!is_array()) {
        fail("expected an array");
    }
    return std::get<Array>(value_);
}

const Json::Object& Json::as_object() const {
    if (!is_object()) {
        fail("expected an object");
    }
    return std::get<Object>(value_);
}

Json::Object& Json::as_object() {
    if (!is_object()) {
        fail("expected an object");
    }
    return std::get<Object>(value_);
}

const Json* Json::find(std::string_view key) const {
    if (!is_object()) {
        return nullptr;
    }
    for (const auto& member : std::get<Object>(value_)) {
        if (member.first == key) {
            return &member.second;
        }
    }
    return nullptr;
}

const Json& Json::at(std::string_view key) const {
    const Json* found = find(key);
    if (found == nullptr) {
        fail("missing key '" + std::string(key) + "'");
    }
    return *found;
}

Json& Json::set(std::string key, Json value) {
    auto& members = as_object();
    for (auto& member : members) {
        if (member.first == key) {
            member.second = std::move(value);
            return member.second;
        }
    }
    members.emplace_back(std::move(key), std::move(value));
    return members.back().second;
}

Json& Json::push(Json value) {
    auto& items = as_array();
    items.push_back(std::move(value));
    return items.back();
}

Json Json::parse(std::string_view text) { return Parser(text).parse_document(); }

void Json::dump_to(std::string& out, int indent, int level) const {
    if (is_null()) {
        out += "null";
    } else if (is_bool()) {
        out += std::get<bool>(value_) ? "true" : "false";
    } else if (is_number()) {
        write_number(out, std::get<double>(value_));
    } else if (is_string()) {
        write_string(out, std::get<std::string>(value_));
    } else if (is_array()) {
        const auto& items = std::get<Array>(value_);
        if (items.empty()) {
            out += "[]";
            return;
        }
        // Pretty mode keeps arrays of scalars on one line so numeric traces stay readable.
        bool all_scalar = true;
        for (const auto& item : items) {
            all_scalar = all_scalar && !item.is_array() && !item.is_object();
        }
        const bool inline_items = indent < 0 || all_scalar;
        out.push_back('[');
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (i > 0) {
                out.push_back(',');
                if (indent >= 0 && inline_items) {
                    out.push_back(' ');
                }
            }
            if (!inline_items) {
                newline(out, indent, level + 1);
            }
            items[i].dump_to(out, indent, level + 1);
        }
        if (!inline_items) {
            newline(out, indent, level);
        }
        out.push_back(']');
    } else {
        const auto& members = std::get<Object>(value_);
        if (members.empty()) {
            out += "{}";
            return;
        }
        out.push_back('{');
        for (std::size_t i = 0; i < members.size(); ++i) {
            if (i > 0) {
                out.push_back(',');
            }
            if (indent >= 0) {
                newline(out, indent, level + 1);
            }
            write_string(out, members[i].first);
            out.push_back(':');
            if (indent >= 0) {
                out.push_back(' ');
            }
            members[i].second.dump_to(out, indent, level + 1);
        }
        if (indent >= 0) {
            newline(out, indent, level);
        }
        out.push_back('}');
    }
}

std::string Json::dump(int indent) const {
    std::string out;
    dump_to(out, indent, 0);
    return out;
}

}  // namespace occamworm
