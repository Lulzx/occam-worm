#include <exception>
#include <iostream>

#include "ow_test.hpp"

int main() {
    using namespace occamworm::test;
    int unexpected = 0;
    for (const TestCase& test_case : registry()) {
        const int before = failure_count();
        try {
            test_case.function();
        } catch (const std::exception& error) {
            ++unexpected;
            std::cerr << test_case.name << ": unexpected exception: " << error.what() << "\n";
        }
        std::cout << (failure_count() == before ? "[ ok ] " : "[FAIL] ") << test_case.name << "\n";
    }
    std::cout << registry().size() << " tests, " << failure_count() << " failed checks, " << unexpected
              << " unexpected exceptions\n";
    return (failure_count() == 0 && unexpected == 0) ? 0 : 1;
}
