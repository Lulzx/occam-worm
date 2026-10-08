// SHA-256 against the FIPS 180-2 (appendix B) vectors and streaming/boundary behaviour.

#include <string>

#include "occamworm/core/sha256.hpp"
#include "ow_test.hpp"

using namespace occamworm;

OW_TEST(sha256_fips_abc) {
    OW_CHECK_EQ(sha256_hex("abc"), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
}

OW_TEST(sha256_fips_two_block) {
    OW_CHECK_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
                std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
}

OW_TEST(sha256_fips_million_a) {
    Sha256 hasher;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) {
        hasher.update(chunk);
    }
    OW_CHECK_EQ(to_hex(hasher.finalize()), std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
}

OW_TEST(sha256_empty) {
    OW_CHECK_EQ(sha256_hex(""), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
}

OW_TEST(sha256_896_bit_message) {
    // FIPS 180-2 extended vector (112 bytes).
    OW_CHECK_EQ(sha256_hex("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
                std::string("cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"));
}

OW_TEST(sha256_padding_boundaries_match_chunked_updates) {
    // Messages around the 55/56/64-byte padding boundaries hash identically whether fed whole or byte-by-byte.
    for (std::size_t length = 0; length < 200; ++length) {
        const std::string message(length, static_cast<char>('a' + static_cast<int>(length % 26)));
        Sha256 incremental;
        for (const char c : message) {
            incremental.update(std::string_view(&c, 1));
        }
        OW_CHECK_EQ(to_hex(incremental.finalize()), sha256_hex(message));
    }
}
