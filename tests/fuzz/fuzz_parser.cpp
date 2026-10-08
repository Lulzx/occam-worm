// libFuzzer target for the WRL front end: parser, checker, canonicaliser, bit accounting and IR JSON.
// Invariants checked for every accepted program: the canonical source recompiles to the same text and hash,
// and the IR JSON round-trips. Rejected programs must fail with an occamworm::Error and nothing else.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "occamworm/core/error.hpp"
#include "occamworm/ir/compile.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    using namespace occamworm;
    if (size > 8192) {
        return 0;
    }
    const std::string source(reinterpret_cast<const char*>(data), size);
    try {
        const Ir ir = compile(source);
        const Ir again = compile(ir.canonical_source);
        if (again.canonical_source != ir.canonical_source || again.program_hash != ir.program_hash) {
            std::abort();  // canonicalisation is not idempotent
        }
        const Ir from_json = ir_from_json(Json::parse(ir_to_json(ir).dump()));
        if (from_json.canonical_source != ir.canonical_source) {
            std::abort();
        }
    } catch (const Error&) {
        // expected for malformed or invalid programs
    }
    return 0;
}
