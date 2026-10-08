#pragma once
// SHA-256 per FIPS 180-4 (identical to FIPS 180-2 for SHA-256). Used for program hashes.

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace occamworm {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
public:
    Sha256();
    void update(std::span<const std::uint8_t> data);
    void update(std::string_view text);
    Sha256Digest finalize();  // the object must not be reused afterwards

private:
    void compress(const std::uint8_t* block);

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffered_ = 0;
    std::uint64_t total_bytes_ = 0;
};

Sha256Digest sha256(std::string_view text);
std::string to_hex(const Sha256Digest& digest);
std::string sha256_hex(std::string_view text);

}  // namespace occamworm
