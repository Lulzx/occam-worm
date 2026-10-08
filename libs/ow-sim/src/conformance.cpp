#include "occamworm/sim/conformance.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "occamworm/core/error.hpp"
#include "occamworm/ir/compile.hpp"
#include "occamworm/sim/sim.hpp"

namespace occamworm {
namespace {

struct Tolerance {
    double absolute = 1e-12;
    double relative = 0.0;
    bool close(double actual, double expected) const {
        return std::fabs(actual - expected) <= absolute + relative * std::fabs(expected);
    }
};

Tolerance read_tolerance(const Json& check) {
    Tolerance tolerance;
    if (const Json* node = check.find("tolerance")) {
        if (const Json* a = node->find("abs")) {
            tolerance.absolute = a->as_double();
        }
        if (const Json* r = node->find("rel")) {
            tolerance.relative = r->as_double();
        }
    }
    return tolerance;
}

// Program text: inline "source" (string or array of lines) or a "path" relative to the case file.
std::string load_source(const Json& program, const std::string& base_directory) {
    if (const Json* source = program.find("source")) {
        if (source->is_string()) {
            return source->as_string();
        }
        std::string text;
        for (const Json& line : source->as_array()) {
            text += line.as_string() + "\n";
        }
        return text;
    }
    const std::filesystem::path path = std::filesystem::path(base_directory) / program.at("path").as_string();
    return read_text_file(path.string());
}

[[noreturn]] void mismatch(const std::string& message) { throw std::runtime_error(message); }

void compare_matrix(const std::string& label, const Json& expected, const std::vector<std::vector<double>>& actual,
                    const Tolerance& tolerance) {
    const auto& rows = expected.as_array();
    if (rows.size() != actual.size()) {
        mismatch(label + ": expected " + std::to_string(rows.size()) + " samples, got " + std::to_string(actual.size()));
    }
    for (std::size_t s = 0; s < rows.size(); ++s) {
        const auto& cols = rows[s].as_array();
        if (cols.size() != actual[s].size()) {
            mismatch(label + ": sample " + std::to_string(s) + " expected " + std::to_string(cols.size()) + " neurons, got " +
                     std::to_string(actual[s].size()));
        }
        for (std::size_t i = 0; i < cols.size(); ++i) {
            if (!tolerance.close(actual[s][i], cols[i].as_double())) {
                std::ostringstream message;
                message.precision(17);
                message << label << "[sample " << s << "][neuron " << i << "]: got " << actual[s][i] << ", expected "
                        << cols[i].as_double();
                mismatch(message.str());
            }
        }
    }
}

void check_traces(const Json& check, const Ir& ir, const SimInput& input, const SimResult& result) {
    const Tolerance tolerance = read_tolerance(check);
    const Json& expected = check.at("expected");
    if (const Json* registers = expected.find("registers")) {
        for (const auto& [name, matrix] : registers->as_object()) {
            const auto it = std::ranges::find(result.register_names, name);
            if (it == result.register_names.end()) {
                mismatch("expected register '" + name + "' is not in the program");
            }
            compare_matrix("register " + name, matrix, result.registers[static_cast<std::size_t>(it - result.register_names.begin())],
                           tolerance);
        }
    }
    if (const Json* observation = expected.find("observation")) {
        compare_matrix("observation", *observation, result.observation, tolerance);
    }
    (void)ir;
    (void)input;
}

// Relabelling neurons must permute the outputs: output of new neuron j equals output of old neuron permutation[j].
void check_permutation(const Json& check, const Ir& ir, const SimInput& input) {
    const Tolerance tolerance = read_tolerance(check);
    std::vector<std::size_t> permutation;
    for (const Json& index : check.at("permutation").as_array()) {
        permutation.push_back(static_cast<std::size_t>(index.as_int()));
    }
    if (permutation.size() != input.graph.neurons.size()) {
        mismatch("permutation must list every neuron");
    }
    SimInput base = input;
    base.observed.clear();
    const SimResult original = simulate(ir, base);
    const SimResult permuted = simulate(ir, permute_neurons(base, permutation));
    for (std::size_t r = 0; r < original.registers.size(); ++r) {
        for (std::size_t s = 0; s < original.registers[r].size(); ++s) {
            for (std::size_t j = 0; j < permutation.size(); ++j) {
                if (!tolerance.close(permuted.registers[r][s][j], original.registers[r][s][permutation[j]])) {
                    mismatch("register " + original.register_names[r] + " differs under relabelling at sample " + std::to_string(s));
                }
            }
        }
    }
    for (std::size_t s = 0; s < original.observation.size(); ++s) {
        for (std::size_t j = 0; j < permutation.size(); ++j) {
            if (!tolerance.close(permuted.observation[s][j], original.observation[s][permutation[j]])) {
                mismatch("observation differs under relabelling at sample " + std::to_string(s));
            }
        }
    }
}

// Runs the same physical scenario at several dt and compares one value with an analytic solution; the error
// must shrink by at most `max_ratio` per halving (first-order schemes give ratio ~0.5).
void check_convergence(const Json& check, const Ir& ir, const SimInput& input) {
    const double final_time = check.at("time").as_double();
    const std::string register_name = check.at("register").as_string();
    const std::string neuron = check.at("neuron").as_string();
    const double analytic = check.at("expected").as_double();
    const double max_ratio = check.at("max_ratio").as_double();
    const double max_error = check.at("max_error_finest").as_double();
    std::vector<double> errors;
    for (const Json& dt_json : check.at("dts").as_array()) {
        SimInput run = input;
        run.dt = dt_json.as_double();
        const double steps = std::round(final_time / run.dt);
        if (std::fabs(steps * run.dt - final_time) > 1e-9 * final_time) {
            mismatch("time is not a whole number of steps of dt");
        }
        run.n_steps = static_cast<long long>(steps);
        run.sample_ticks = {run.n_steps};
        run.observed = {neuron};
        const SimResult result = simulate(ir, run);
        const auto it = std::ranges::find(result.register_names, register_name);
        if (it == result.register_names.end()) {
            mismatch("unknown register '" + register_name + "'");
        }
        const double value = result.registers[static_cast<std::size_t>(it - result.register_names.begin())][0][0];
        errors.push_back(std::fabs(value - analytic));
    }
    for (std::size_t k = 1; k < errors.size(); ++k) {
        if (!(errors[k] <= max_ratio * errors[k - 1])) {
            std::ostringstream message;
            message.precision(6);
            message << "error did not shrink fast enough on halving dt: " << errors[k - 1] << " -> " << errors[k] << " (ratio limit "
                    << max_ratio << ")";
            mismatch(message.str());
        }
    }
    if (!(errors.back() <= max_error)) {
        mismatch("finest-dt error " + std::to_string(errors.back()) + " exceeds " + std::to_string(max_error));
    }
}

}  // namespace

bool run_conformance_case(const Json& testcase, const std::string& base_directory, std::string& detail) {
    try {
        const Json& check = testcase.at("check");
        const std::string type = check.at("type").as_string();
        const std::string source = load_source(testcase.at("program"), base_directory);
        if (type == "compile_error") {
            const auto compiled = try_compile(source);
            if (compiled) {
                mismatch("expected a compile error but the program compiled");
            }
            const std::string expected = check.at("code").as_string();
            if (errc_name(compiled.error().code()) != expected) {
                mismatch("expected " + expected + " but got " + std::string(errc_name(compiled.error().code())) + ": " +
                         compiled.error().message());
            }
            return true;
        }
        const Ir ir = compile(source);
        if (const Json* hash = testcase.find("program_hash")) {
            if (hash->as_string() != ir.program_hash) {
                mismatch("program hash " + ir.program_hash + " differs from the recorded " + hash->as_string());
            }
        }
        const SimInput input = sim_input_from_json(testcase.at("input"));
        if (type == "traces") {
            check_traces(check, ir, input, simulate(ir, input));
        } else if (type == "permutation") {
            check_permutation(check, ir, input);
        } else if (type == "convergence") {
            check_convergence(check, ir, input);
        } else {
            mismatch("unknown check type '" + type + "'");
        }
        return true;
    } catch (const std::exception& error) {
        detail = error.what();
        return false;
    }
}

ConformanceSummary run_conformance_suite(const std::string& directory, std::ostream& out) {
    ConformanceSummary summary;
    std::vector<std::filesystem::path> files;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            files.push_back(entry.path());
        }
    }
    if (error) {
        out << "[FAIL] cannot read suite directory '" << directory << "': " << error.message() << "\n";
        ++summary.failed;
        return summary;
    }
    std::ranges::sort(files);
    for (const auto& file : files) {
        std::string detail;
        bool ok = false;
        try {
            const Json testcase = Json::parse(read_text_file(file.string()));
            ok = run_conformance_case(testcase, file.parent_path().string(), detail);
        } catch (const std::exception& e) {
            detail = e.what();
        }
        out << (ok ? "[ ok ] " : "[FAIL] ") << file.filename().string();
        if (!ok) {
            out << ": " << detail;
            ++summary.failed;
        } else {
            ++summary.passed;
        }
        out << "\n";
    }
    out << summary.passed + summary.failed << " cases, " << summary.failed << " failed\n";
    return summary;
}

}  // namespace occamworm
