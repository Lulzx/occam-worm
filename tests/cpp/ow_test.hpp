#pragma once
// Tiny in-repo test harness (no external dependencies). Each test executable links ow_test_main.cpp,
// which runs every OW_TEST registered in that executable and returns non-zero on any failure.
//
//   OW_TEST(name) { OW_CHECK(cond); OW_CHECK_EQ(a, b); OW_CHECK_NEAR(a, b, tol); OW_CHECK_THROWS(expr, Errc::Unit); }

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "occamworm/core/error.hpp"

namespace occamworm::test {

struct TestCase {
    const char* name;
    void (*function)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> cases;
    return cases;
}

inline int& failure_count() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const char* name, void (*function)()) { registry().push_back({name, function}); }
};

inline void report_failure(const char* file, int line, const std::string& message) {
    ++failure_count();
    std::cerr << file << ":" << line << ": FAILED: " << message << "\n";
}

template <typename T>
std::string show(const T& value) {
    std::ostringstream out;
    out.precision(17);
    out << value;
    return out.str();
}

}  // namespace occamworm::test

#define OW_TEST(name)                                                                  \
    static void name();                                                                \
    static const ::occamworm::test::Registrar ow_registrar_##name(#name, &name);       \
    static void name()

#define OW_CHECK(condition)                                                                   \
    do {                                                                                      \
        if (!(condition)) {                                                                   \
            ::occamworm::test::report_failure(__FILE__, __LINE__, "check (" #condition ")"); \
        }                                                                                     \
    } while (false)

#define OW_CHECK_EQ(actual, expected)                                                                      \
    do {                                                                                                   \
        const auto& ow_a = (actual);                                                                       \
        const auto& ow_e = (expected);                                                                     \
        if (!(ow_a == ow_e)) {                                                                             \
            ::occamworm::test::report_failure(__FILE__, __LINE__,                                          \
                                              std::string(#actual " == " #expected ": got ") +            \
                                                  ::occamworm::test::show(ow_a) + ", expected " +         \
                                                  ::occamworm::test::show(ow_e));                         \
        }                                                                                                  \
    } while (false)

#define OW_CHECK_NEAR(actual, expected, tolerance)                                                        \
    do {                                                                                                  \
        const double ow_a = (actual);                                                                     \
        const double ow_e = (expected);                                                                   \
        if (!(std::fabs(ow_a - ow_e) <= (tolerance))) {                                                   \
            ::occamworm::test::report_failure(__FILE__, __LINE__,                                         \
                                              std::string(#actual " ~= " #expected ": got ") +           \
                                                  ::occamworm::test::show(ow_a) + ", expected " +        \
                                                  ::occamworm::test::show(ow_e));                        \
        }                                                                                                 \
    } while (false)

// Checks that evaluating `expression` throws occamworm::Error with the given code.
#define OW_CHECK_THROWS(expression, errc_value)                                                         \
    do {                                                                                                \
        bool ow_threw = false;                                                                          \
        try {                                                                                           \
            (void)(expression);                                                                         \
        } catch (const ::occamworm::Error& ow_error) {                                                  \
            ow_threw = true;                                                                            \
            if (ow_error.code() != (errc_value)) {                                                      \
                ::occamworm::test::report_failure(__FILE__, __LINE__,                                   \
                                                  std::string("wrong error code for " #expression ": ") + \
                                                      ow_error.what());                                 \
            }                                                                                           \
        }                                                                                               \
        if (!ow_threw) {                                                                                \
            ::occamworm::test::report_failure(__FILE__, __LINE__, "no error thrown by " #expression);   \
        }                                                                                               \
    } while (false)
