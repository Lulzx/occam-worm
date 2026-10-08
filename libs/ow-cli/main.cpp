// `ow`: command-line front end (docs/engineering/ARCHITECTURE.md §14.5, docs/language/WRL_SYNTAX.md "CLI").
//
//   ow --version
//   ow rule inspect <file.wrl>              canonical IR JSON
//   ow rule canon <file.wrl>                canonical source (the hashed bytes)
//   ow sim run --program <file> --case <case.json> [--out <file>]
//   ow sim conformance --suite <dir>
//   ow search enumerate --config <file.json> [--out <file>]

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "occamworm/core/build_info.hpp"
#include "occamworm/core/error.hpp"
#include "occamworm/ir/compile.hpp"
#include "occamworm/search/enumerate.hpp"
#include "occamworm/sim/conformance.hpp"
#include "occamworm/sim/sim.hpp"

using namespace occamworm;

namespace {

constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;

int usage() {
    std::cerr << "usage:\n"
                 "  ow --version\n"
                 "  ow rule inspect <file.wrl>\n"
                 "  ow rule canon <file.wrl>\n"
                 "  ow sim run --program <file.wrl|ir.json> --case <case.json> [--out <file>]\n"
                 "  ow sim conformance --suite <dir>\n"
                 "  ow search enumerate --config <file.json> [--out <file>]\n";
    return kExitUsage;
}

// Value of `--name <value>` among the arguments, or empty when absent.
std::string option(const std::vector<std::string>& args, const std::string& name) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == name) {
            return args[i + 1];
        }
    }
    return {};
}

Ir load_program(const std::string& path) {
    const std::string text = read_text_file(path);
    if (path.size() > 5 && path.substr(path.size() - 5) == ".json") {
        return ir_from_json(Json::parse(text));
    }
    return compile(text);
}

int write_output(const std::string& text, const std::string& out_path) {
    if (out_path.empty()) {
        std::cout << text;
        return 0;
    }
    std::ofstream file(out_path, std::ios::binary);
    if (!file) {
        std::cerr << "error: cannot write '" << out_path << "'\n";
        return kExitFailure;
    }
    file << text;
    return 0;
}

int rule_command(const std::vector<std::string>& args) {
    if (args.size() != 2 || (args[0] != "inspect" && args[0] != "canon")) {
        return usage();
    }
    const Ir ir = compile(read_text_file(args[1]));
    if (args[0] == "inspect") {
        return write_output(ir_to_json(ir).dump(2) + "\n", "");
    }
    return write_output(ir.canonical_source, "");
}

int sim_command(const std::vector<std::string>& args) {
    if (args.empty()) {
        return usage();
    }
    if (args[0] == "conformance") {
        const std::string suite = option(args, "--suite");
        if (suite.empty()) {
            return usage();
        }
        const ConformanceSummary summary = run_conformance_suite(suite, std::cout);
        return (summary.failed == 0 && summary.passed > 0) ? 0 : kExitFailure;
    }
    if (args[0] == "run") {
        const std::string case_path = option(args, "--case");
        if (case_path.empty()) {
            return usage();
        }
        const Json case_json = Json::parse(read_text_file(case_path));
        const Json& input_json = case_json.find("input") ? case_json.at("input") : case_json;
        Ir ir;
        const std::string program_path = option(args, "--program");
        if (!program_path.empty()) {
            ir = load_program(program_path);
        } else if (const Json* program = case_json.find("program")) {
            if (const Json* source = program->find("source")) {
                std::string text;
                if (source->is_string()) {
                    text = source->as_string();
                } else {
                    for (const Json& line : source->as_array()) {
                        text += line.as_string() + "\n";
                    }
                }
                ir = compile(text);
            } else {
                ir = load_program(program->at("path").as_string());
            }
        } else {
            return usage();
        }
        const SimInput input = sim_input_from_json(input_json);
        const SimResult result = simulate(ir, input);
        return write_output(sim_result_to_json(ir, input, result).dump(2) + "\n", option(args, "--out"));
    }
    return usage();
}

int search_command(const std::vector<std::string>& args) {
    if (args.empty() || args[0] != "enumerate") {
        return usage();
    }
    const std::string config_path = option(args, "--config");
    if (config_path.empty()) {
        return usage();
    }
    const EnumerationConfig config = enumeration_config_from_json(Json::parse(read_text_file(config_path)));
    std::string lines;
    const EnumerationSummary summary = enumerate_programs(config, [&](const EnumeratedProgram& program) {
        lines += program_record_json(program).dump() + "\n";
    });
    lines += summary_record_json(config, summary).dump() + "\n";
    return write_output(lines, option(args, "--out"));
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) {
        return usage();
    }
    try {
        if (args[0] == "--version") {
            std::cout << OW_BUILD_STRING << "; WRL grammar " << OW_GRAMMAR_VERSION << "\n";
            return 0;
        }
        const std::vector<std::string> rest(args.begin() + 1, args.end());
        if (args[0] == "rule") {
            return rule_command(rest);
        }
        if (args[0] == "sim") {
            return sim_command(rest);
        }
        if (args[0] == "search") {
            return search_command(rest);
        }
        return usage();
    } catch (const Error& error) {
        std::cerr << "error: " << error.what() << "\n";
        return kExitFailure;
    } catch (const std::exception& error) {
        std::cerr << "internal error: " << error.what() << "\n";
        return kExitFailure;
    }
}
