#pragma once
// Minimal JSON value, parser and writer. Objects keep insertion order so output is deterministic.
// Numbers are doubles; integral values print without a fraction. Non-finite numbers cannot be written.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace occamworm {

class Json {
public:
    using Array = std::vector<Json>;
    using Member = std::pair<std::string, Json>;
    using Object = std::vector<Member>;

    Json() = default;  // null
    Json(std::nullptr_t) {}
    Json(bool value) : value_(value) {}
    Json(double value) : value_(value) {}
    Json(int value) : value_(static_cast<double>(value)) {}
    Json(long value) : value_(static_cast<double>(value)) {}
    Json(long long value) : value_(static_cast<double>(value)) {}
    Json(unsigned value) : value_(static_cast<double>(value)) {}
    Json(unsigned long value) : value_(static_cast<double>(value)) {}
    Json(unsigned long long value) : value_(static_cast<double>(value)) {}
    Json(const char* value) : value_(std::string(value)) {}
    Json(std::string value) : value_(std::move(value)) {}
    Json(std::string_view value) : value_(std::string(value)) {}
    Json(Array value) : value_(std::move(value)) {}
    Json(Object value) : value_(std::move(value)) {}

    static Json array() { return Json(Array{}); }
    static Json object() { return Json(Object{}); }

    bool is_null() const { return std::holds_alternative<std::monostate>(value_); }
    bool is_bool() const { return std::holds_alternative<bool>(value_); }
    bool is_number() const { return std::holds_alternative<double>(value_); }
    bool is_string() const { return std::holds_alternative<std::string>(value_); }
    bool is_array() const { return std::holds_alternative<Array>(value_); }
    bool is_object() const { return std::holds_alternative<Object>(value_); }

    // Typed accessors throw Error(Errc::Json) on a type mismatch.
    bool as_bool() const;
    double as_double() const;
    long long as_int() const;  // requires an integral value
    const std::string& as_string() const;
    const Array& as_array() const;
    Array& as_array();
    const Object& as_object() const;
    Object& as_object();

    // Object helpers.
    const Json* find(std::string_view key) const;  // nullptr when absent or not an object
    const Json& at(std::string_view key) const;    // throws when absent
    Json& set(std::string key, Json value);        // replaces an existing key, else appends
    // Array helper.
    Json& push(Json value);

    // Parsing throws Error(Errc::Json). Strict RFC 8259; nesting depth is limited to 200.
    static Json parse(std::string_view text);

    // indent < 0: compact; otherwise pretty-printed with `indent` spaces per level.
    std::string dump(int indent = -1) const;

private:
    void dump_to(std::string& out, int indent, int level) const;

    std::variant<std::monostate, bool, double, std::string, Array, Object> value_;
};

}  // namespace occamworm
