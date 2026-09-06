#ifndef SMARTGC_TEST_FRAMEWORK_HPP
#define SMARTGC_TEST_FRAMEWORK_HPP

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <sstream>

namespace smartgc::test {

class TestRunner {
public:
    struct TestCase {
        std::string name;
        std::function<void()> func;
    };

    static TestRunner& instance() {
        static TestRunner runner;
        return runner;
    }

    void register_test(const std::string& name, std::function<void()> func) {
        tests_.push_back({name, func});
    }

    int run_all() {
        std::cout << "==================================================\n";
        std::cout << " Running SmartGC Test Suite (" << tests_.size() << " tests)\n";
        std::cout << "==================================================\n\n";

        size_t passed = 0;
        size_t failed = 0;

        for (const auto& test : tests_) {
            std::cout << "[ RUN      ] " << test.name << "\n";
            try {
                test.func();
                std::cout << "[       OK ] " << test.name << "\n";
                passed++;
            } catch (const std::exception& ex) {
                std::cerr << "[  FAILED  ] " << test.name << "\n";
                std::cerr << "             Reason: " << ex.what() << "\n";
                failed++;
            } catch (...) {
                std::cerr << "[  FAILED  ] " << test.name << "\n";
                std::cerr << "             Reason: Unknown non-std exception\n";
                failed++;
            }
        }

        std::cout << "\n==================================================\n";
        std::cout << " Test Results: " << passed << " Passed, " << failed << " Failed\n";
        std::cout << "==================================================\n";

        return failed == 0 ? 0 : 1;
    }

private:
    std::vector<TestCase> tests_;
};

#define TEST_CASE(name) \
    void name(); \
    namespace { \
        struct name##_registrar { \
            name##_registrar() { \
                ::smartgc::test::TestRunner::instance().register_test(#name, name); \
            } \
        } name##_registrar_instance; \
    } \
    void name()

#define ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::ostringstream ss; \
            ss << "Assertion failed: (" #expr ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (false)

#define ASSERT_FALSE(expr) \
    do { \
        if (expr) { \
            std::ostringstream ss; \
            ss << "Assertion failed: NOT (" #expr ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (false)

#define ASSERT_EQ(a, b) \
    do { \
        if (!((a) == (b))) { \
            std::ostringstream ss; \
            ss << "Assertion failed: (" #a " == " #b ") [" << (a) << " != " << (b) << "] at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (false)

#define ASSERT_NE(a, b) \
    do { \
        if ((a) == (b)) { \
            std::ostringstream ss; \
            ss << "Assertion failed: (" #a " != " #b ") [" << (a) << " == " << (b) << "] at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (false)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs(static_cast<double>(a) - static_cast<double>(b)) > (eps)) { \
            std::ostringstream ss; \
            ss << "Assertion failed: |" #a " - " #b "| <= " #eps " [|" << (a) << " - " << (b) << "| = " \
               << std::abs(static_cast<double>(a) - static_cast<double>(b)) << "] at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (false)

} // namespace smartgc::test

#endif // SMARTGC_TEST_FRAMEWORK_HPP
