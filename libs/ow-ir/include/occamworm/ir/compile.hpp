#pragma once
// Front-end driver and IR JSON.

#include <expected>
#include <string>
#include <string_view>

#include "occamworm/core/error.hpp"
#include "occamworm/core/json.hpp"
#include "occamworm/ir/ir.hpp"

namespace occamworm {

// parse -> check -> canonicalise. Throws Error.
Ir compile(std::string_view source);
std::expected<Ir, Error> try_compile(std::string_view source);

// Reads a UTF-8 text file; throws Error(Errc::Io).
std::string read_text_file(const std::string& path);

inline constexpr std::string_view kIrSchema = "occamworm.wrl.ir/0.1";

// The documented IR JSON (docs/language/WRL_SYNTAX.md, "IR JSON schema").
Json ir_to_json(const Ir& ir);
// Parses the executable parts of an IR JSON document back into an Ir (used to run IR directly).
Ir ir_from_json(const Json& json);

}  // namespace occamworm
