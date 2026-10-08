// Deterministic mutation test (the always-on companion of the libFuzzer target in tests/fuzz): corrupt valid
// WRL programs and check that the front end either accepts them consistently or fails with a clean Error.

#include <cstdint>
#include <string>
#include <vector>

#include "occamworm/core/json.hpp"
#include "occamworm/ir/compile.hpp"
#include "ow_test.hpp"

using namespace occamworm;

namespace {

std::vector<std::string> seed_programs() {
    std::vector<std::string> seeds;
    for (const char* name : {"leak-adapt", "leak-adapt-euler", "gap-leak", "rule90"}) {
        seeds.push_back(read_text_file(std::string(OW_SOURCE_DIR) + "/configs/rules/" + name + ".wrl"));
    }
    seeds.push_back("wrl 0.1\ntier G1\nstate v : 1 = 0\nnext v = clamp(select(threshold(v, 0.5), v + 1, -v), -1, 1)\nobserve identity_v1(v)\n");
    return seeds;
}

}  // namespace

OW_TEST(mutated_programs_never_crash_and_canonicalise_idempotently) {
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto next = [&]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::size_t>(state >> 33U);
    };
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_()[],:=+-*/^# \n.";
    int accepted = 0;
    int rejected = 0;
    for (const std::string& seed : seed_programs()) {
        for (int iteration = 0; iteration < 600; ++iteration) {
            std::string mutated = seed;
            const int edits = 1 + static_cast<int>(next() % 4);
            for (int e = 0; e < edits && !mutated.empty(); ++e) {
                const std::size_t position = next() % mutated.size();
                switch (next() % 4) {
                    case 0: mutated[position] = alphabet[next() % alphabet.size()]; break;
                    case 1: mutated.insert(position, 1, alphabet[next() % alphabet.size()]); break;
                    case 2: mutated.erase(position, 1 + next() % 8); break;
                    default: mutated.insert(position, mutated.substr(next() % mutated.size(), next() % 12)); break;
                }
            }
            try {
                const Ir ir = compile(mutated);
                const Ir again = compile(ir.canonical_source);
                OW_CHECK_EQ(again.canonical_source, ir.canonical_source);
                OW_CHECK_EQ(again.program_hash, ir.program_hash);
                ++accepted;
            } catch (const Error&) {
                ++rejected;
            }
        }
    }
    OW_CHECK(accepted > 0);
    OW_CHECK(rejected > 0);
}

OW_TEST(garbage_bytes_are_rejected_cleanly) {
    std::uint64_t state = 12345;
    for (int iteration = 0; iteration < 500; ++iteration) {
        std::string junk;
        const std::size_t length = state % 200;
        for (std::size_t i = 0; i < length; ++i) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            junk.push_back(static_cast<char>(state >> 56U));
        }
        OW_CHECK(!try_compile(junk).has_value());
    }
}
