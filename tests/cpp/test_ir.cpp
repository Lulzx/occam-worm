// ow-ir: parsing, unit/type/stability checks, canonical hashing, bit accounting and IR JSON.

#include <cstdint>
#include <string>

#include "occamworm/core/numfmt.hpp"
#include "occamworm/core/sha256.hpp"
#include "occamworm/ir/compile.hpp"
#include "occamworm/ir/parser.hpp"
#include "occamworm/ir/source.hpp"
#include "ow_test.hpp"

using namespace occamworm;

namespace {

// The §5.5 illustrative rule in WRL v0.1 syntax.
const std::string kLeakAdapt = R"(# leaky E/I network neuron with an adaptation register
wrl 0.1
rule leak_exc_inh_adapt_v1
tier G1
dt_max 0.05
state v : 1 = 0
state h : 1 = 0
param tau_v : s = 0.25 in [0.1, 10] trainable bits 12
param tau_h : s = 2 in [0.1, 30] trainable bits 12
param gain : 1 = 1 in [0, 20] trainable bits 10
param adapt : 1 = 0.2 in [0, 10] trainable bits 10
input u = stimulus
input ex = sum_in(v, exc)
input inh = sum_in(v, inh)
let drive = gain * (ex - inh + u) - adapt * h
next v = leaky_integrate(v, tanh(drive), tau_v)
next h = leaky_integrate(h, relu(v), tau_h)
observe calcium_linear_v1(v, 0.5[s])
)";

// Three live registers a, b, c; `body` supplies the statements after the declarations.
std::string three_register_program(const std::string& body) {
    return "wrl 0.1\ntier G1\nstate a : 1 = 0\nstate b : 1 = 0.5\nstate c : 1 = 0\n" + body + "\nobserve identity_v1(a)\n";
}

std::string hash_of(const std::string& source) { return compile(source).program_hash; }

void check_same_hash(const std::string& left, const std::string& right) {
    OW_CHECK_EQ(hash_of(left), hash_of(right));
}

void check_different_hash(const std::string& left, const std::string& right) {
    OW_CHECK(hash_of(left) != hash_of(right));
}

}  // namespace

OW_TEST(compiles_leak_adapt_example) {
    const Ir ir = compile(kLeakAdapt);
    OW_CHECK_EQ(ir.program_hash.size(), std::size_t{64});
    OW_CHECK_EQ(ir.registers.size(), std::size_t{2});
    OW_CHECK_EQ(ir.params.size(), std::size_t{4});
    OW_CHECK_EQ(ir.rule_name, std::string("leak_exc_inh_adapt_v1"));
    OW_CHECK(ir.bits.l_struct > 0);
    OW_CHECK_EQ(ir.bits.l_params, 12 + 12 + 10 + 10);
}

OW_TEST(canonical_source_is_a_fixed_point) {
    const Ir ir = compile(kLeakAdapt);
    const Ir again = compile(ir.canonical_source);
    OW_CHECK_EQ(again.canonical_source, ir.canonical_source);
    OW_CHECK_EQ(again.program_hash, ir.program_hash);
    OW_CHECK_EQ(sha256_hex(ir.canonical_source), ir.program_hash);
}

OW_TEST(executable_source_preserves_default_parameter_values) {
    const Ir ir = compile(kLeakAdapt);
    PrintOptions full;
    full.identity = false;
    const Ir again = compile(print_source(ir, full));
    OW_CHECK_EQ(again.program_hash, ir.program_hash);
    for (std::size_t i = 0; i < ir.params.size(); ++i) {
        OW_CHECK_EQ(again.params[i].value, ir.params[i].value);
    }
}

// ---- unit, type, tier and stability rejections ---------------------------------------------------------

OW_TEST(rejects_adding_seconds_to_dimensionless) {
    const std::string source = R"(wrl 0.1
tier G1
state v : 1 = 0
param tau : s = 1 in [0.1, 2] trainable bits 8
next v = v + tau
observe identity_v1(v)
)";
    OW_CHECK_THROWS(compile(source), Errc::Unit);
}

OW_TEST(rejects_unit_errors) {
    const std::string head = "wrl 0.1\ntier G1\nstate v : 1 = 0\nstate x : V = 0\nparam tau : s = 1 in [0.1, 2] trainable bits 8\n";
    const std::string tail = "\nobserve identity_v1(v)\n";
    OW_CHECK_THROWS(compile(head + "next v = tanh(tau)" + tail), Errc::Unit);            // tanh needs dimensionless
    OW_CHECK_THROWS(compile(head + "next v = x" + tail), Errc::Unit);                    // next must match register unit
    OW_CHECK_THROWS(compile(head + "next v = leaky_integrate(v, v, v)" + tail), Errc::Type);  // tau must be param/const
    OW_CHECK_THROWS(compile(head + "param k : 1 = 1 in [0.1, 2] trainable bits 8\nnext v = leaky_integrate(v, v, k)" + tail), Errc::Unit);
    OW_CHECK_THROWS(compile(head + "next v = min(v, x)" + tail), Errc::Unit);
    OW_CHECK_THROWS(compile(head + "next v = select(x, v, v)" + tail), Errc::Unit);
    OW_CHECK_THROWS(compile(head + "next v = threshold(v, x)" + tail), Errc::Unit);
    OW_CHECK_THROWS(compile(head + "next v = v + 1[s]" + tail), Errc::Unit);
    // Multiplying units is fine: x * (1/s) has unit V/s, which cannot be assigned to a V register.
    OW_CHECK_THROWS(compile(head + "next x = x * 2[s^-1]" + tail), Errc::Unit);
    OW_CHECK_NOTHROW(compile(head + "next x = x * 2[1]" + tail));
    OW_CHECK_NOTHROW(compile(head + "next x = x + 0" + tail));  // an unannotated zero adopts the unit
    OW_CHECK_THROWS(compile(head + "next x = x + 1" + tail), Errc::Unit);
}

OW_TEST(rejects_unstable_explicit_euler) {
    const std::string head = "wrl 0.1\ntier G1\nstate v : 1 = 0\n";
    const std::string tail = "\nnext v = euler_leak(v, stimulus, tau)\nobserve identity_v1(v)\n";
    // dt_max / tau_lower = 0.05 / 0.005 = 10 > 1.
    OW_CHECK_THROWS(compile(head + "dt_max 0.05\nparam tau : s = 0.25 in [0.005, 10] trainable bits 8" + tail), Errc::Stability);
    // Exactly at the bound is accepted: dt_max / tau_lower = 1.
    OW_CHECK_NOTHROW(compile(head + "dt_max 0.05\nparam tau : s = 0.25 in [0.05, 10] trainable bits 8" + tail));
    OW_CHECK_NOTHROW(compile(head + "dt_max 0.05\nparam tau : s = 0.25 in [0.1, 10] trainable bits 8" + tail));
    // No dt_max declared.
    OW_CHECK_THROWS(compile(head + "param tau : s = 0.25 in [0.1, 10] trainable bits 8" + tail), Errc::Stability);
    // The exact integrator has no dt/tau bound but needs a strictly positive time constant.
    const std::string exact = "\nnext v = leaky_integrate(v, stimulus, tau)\nobserve identity_v1(v)\n";
    OW_CHECK_NOTHROW(compile(head + "param tau : s = 0.25 in [0.001, 10] trainable bits 8" + exact));
    OW_CHECK_THROWS(compile(head + "param tau : s = 0.25 in [0, 10] trainable bits 8" + exact), Errc::Stability);
    // A constant time constant is checked the same way.
    OW_CHECK_THROWS(compile(head + "dt_max 0.05\nnext v = euler_leak(v, stimulus, 0.01[s])\nobserve identity_v1(v)\n"), Errc::Stability);
}

OW_TEST(rejects_division_and_unknown_things) {
    const std::string head = "wrl 0.1\ntier G1\nstate v : 1 = 0\n";
    const std::string tail = "\nobserve identity_v1(v)\n";
    OW_CHECK_THROWS(compile(head + "next v = v / 2" + tail), Errc::Syntax);
    OW_CHECK_THROWS(compile(head + "next v = frobnicate(v)" + tail), Errc::Name);
    OW_CHECK_THROWS(compile(head + "next v = w" + tail), Errc::Name);
    OW_CHECK_THROWS(compile(head + "next w = v" + tail), Errc::Name);
    OW_CHECK_THROWS(compile(head + "next v = v\nnext v = v" + tail), Errc::Name);
    OW_CHECK_THROWS(compile(head + "state v : 1 = 0" + tail), Errc::Name);
    OW_CHECK_THROWS(compile(head + "let tanh = v" + tail), Errc::Name);
    OW_CHECK_THROWS(compile(head + "input x = v + 1" + tail), Errc::Type);
    OW_CHECK_THROWS(compile(head + "next v = sum_in(v, sideways)" + tail), Errc::Type);
    OW_CHECK_THROWS(compile(head + "next v = clamp(v, 2, 1)" + tail), Errc::Type);
    OW_CHECK_THROWS(compile(head + "next v = tanh(v, v)" + tail), Errc::Type);
    OW_CHECK_THROWS(compile(head + "next v = [1, 2]" + tail), Errc::Syntax);
    OW_CHECK_THROWS(compile(head + "next v = v"), Errc::Name);  // no observation
}

OW_TEST(tier_restrictions) {
    const std::string g0 = "wrl 0.1\ntier G0\nstate s : 1 = 0\n";
    const std::string tail = "\nobserve identity_v1(s)\n";
    OW_CHECK_NOTHROW(compile(g0 + "next s = lut(count_in(s, 1), [0, 1, 1])" + tail));
    OW_CHECK_THROWS(compile(g0 + "next s = tanh(s)" + tail), Errc::Tier);
    OW_CHECK_THROWS(compile(g0 + "next s = s * 0.5" + tail), Errc::Tier);
    OW_CHECK_THROWS(compile(g0 + "param p : 1 = 1 fixed\nnext s = s" + tail), Errc::Tier);
    OW_CHECK_THROWS(compile(g0 + "next s = sum_in(s, all)" + tail), Errc::Tier);
    OW_CHECK_THROWS(compile("wrl 0.1\ntier G0\nstate s : 1 = 0.5\nobserve identity_v1(s)\n"), Errc::Tier);
    OW_CHECK_THROWS(compile("wrl 0.1\ntier G1\nstate s : 1 = 0\nnext s = lut(s, [0, 1])\nobserve identity_v1(s)\n"), Errc::Tier);
    OW_CHECK_THROWS(compile(g0 + "gap s\nobserve identity_v1(s)\n"), Errc::Tier);
}

OW_TEST(syntax_errors_carry_locations) {
    try {
        compile("wrl 0.1\ntier G1\nstate v : 1 = 0\nnext v = (v + \nobserve identity_v1(v)\n");
        OW_CHECK(false);
    } catch (const Error& error) {
        OW_CHECK(error.code() == Errc::Syntax);
    }
    try {
        compile("wrl 0.1\ntier G1\nstate v : 1 = 0\nnext v = v $ 2\nobserve identity_v1(v)\n");
        OW_CHECK(false);
    } catch (const Error& error) {
        OW_CHECK_EQ(error.line(), 4);
        OW_CHECK_EQ(error.column(), 12);
    }
    OW_CHECK_THROWS(compile(""), Errc::Syntax);
    OW_CHECK_THROWS(compile("tier G1\n"), Errc::Syntax);
    OW_CHECK_THROWS(compile("wrl 0.2\ntier G1\n"), Errc::Syntax);
    OW_CHECK_THROWS(compile("wrl 0.1\n"), Errc::Syntax);
    OW_CHECK_THROWS(compile(std::string(2 * 1024 * 1024, ' ')), Errc::Limit);
}

OW_TEST(parser_accepts_comments_continuations_and_signs) {
    const std::string source = R"(wrl 0.1   # header comment
tier G1

state v : 1 = -0.5   # negative init
let t = tanh(
   v + 1)            # continuation inside parentheses
next v = clamp(t, -1, 1)
observe identity_v1(v)
)";
    const Ir ir = compile(source);
    OW_CHECK_EQ(ir.registers[0].init, -0.5);
}

OW_TEST(deeply_nested_expressions_are_rejected_not_crashed) {
    std::string nested = "wrl 0.1\ntier G1\nstate v : 1 = 0\nnext v = ";
    for (int i = 0; i < 500; ++i) {
        nested += "(";
    }
    nested += "v";
    for (int i = 0; i < 500; ++i) {
        nested += ")";
    }
    nested += "\nobserve identity_v1(v)\n";
    OW_CHECK_THROWS(compile(nested), Errc::Limit);
    std::string chain = "wrl 0.1\ntier G1\nstate v : 1 = 0\nnext v = v";
    for (int i = 0; i < 200; ++i) {
        chain += " + v";
    }
    chain += "\nobserve identity_v1(v)\n";
    OW_CHECK_THROWS(compile(chain), Errc::Limit);
}

// ---- canonical equivalences (OW-008 done-when) ---------------------------------------------------------

OW_TEST(hash_commutativity) {
    check_same_hash(three_register_program("next a = a + b"), three_register_program("next a = b + a"));
    check_same_hash(three_register_program("next a = a * b"), three_register_program("next a = b * a"));
    check_same_hash(three_register_program("next a = min(a, b)"), three_register_program("next a = min(b, a)"));
    check_same_hash(three_register_program("next a = max(a, b)"), three_register_program("next a = max(b, a)"));
}

OW_TEST(hash_associativity_by_flattening) {
    const std::string left = three_register_program("next a = (a + b) + c");
    check_same_hash(left, three_register_program("next a = a + (b + c)"));
    check_same_hash(left, three_register_program("next a = c + (b + a)"));
    check_same_hash(left, three_register_program("next a = add(a, b, c)"));
    check_same_hash(three_register_program("next a = (a * b) * c"), three_register_program("next a = c * (b * a)"));
    check_same_hash(three_register_program("next a = min(min(a, b), c)"), three_register_program("next a = min(c, min(b, a))"));
}

OW_TEST(hash_ignores_local_names_and_dead_code) {
    const std::string base = three_register_program("let t = a * b\nnext a = t + c");
    check_same_hash(base, three_register_program("let zz = b * a\nnext a = c + zz"));
    check_same_hash(base, three_register_program("next a = a * b + c"));  // inlined local
    // Dead let, dead register, unused parameter.
    check_same_hash(base, three_register_program("let unused = tanh(c)\nnext a = a * b + c"));
    const std::string with_dead_register =
        "wrl 0.1\ntier G1\nstate a : 1 = 0\nstate b : 1 = 0.5\nstate c : 1 = 0\nstate d : 1 = 7\n"
        "param unused : 1 = 1 in [0, 2] trainable bits 4\n"
        "next d = d + a\nnext a = a * b + c\nobserve identity_v1(a)\n";
    check_same_hash(base, with_dead_register);
    const Ir ir = compile(with_dead_register);
    OW_CHECK_EQ(ir.eliminated_registers.size(), std::size_t{1});
    OW_CHECK_EQ(ir.eliminated_registers[0], std::string("d"));
    OW_CHECK_EQ(ir.eliminated_params.size(), std::size_t{1});
    // A register that only feeds itself is dead too.
    check_same_hash(base, three_register_program("next a = a * b + c\nnext c = c"));
}

OW_TEST(hash_ignores_register_and_parameter_names_and_order) {
    const std::string left =
        "wrl 0.1\ntier G1\nstate x : 1 = 0\nstate y : 1 = 0.5\n"
        "param p : 1 = 1 in [0, 1] trainable bits 8\nparam q : 1 = 1 in [0, 2] trainable bits 8\n"
        "next x = p * x + q * y\nobserve identity_v1(x)\n";
    const std::string renamed =
        "wrl 0.1\ntier G1\nstate left : 1 = 0\nstate right : 1 = 0.5\n"
        "param alpha : 1 = 1 in [0, 1] trainable bits 8\nparam beta : 1 = 1 in [0, 2] trainable bits 8\n"
        "next left = alpha * left + beta * right\nobserve identity_v1(left)\n";
    const std::string reordered =
        "wrl 0.1\ntier G1\nstate y : 1 = 0.5\nstate x : 1 = 0\n"
        "param q : 1 = 1 in [0, 2] trainable bits 8\nparam p : 1 = 1 in [0, 1] trainable bits 8\n"
        "next x = p * x + q * y\nobserve identity_v1(x)\n";
    check_same_hash(left, renamed);
    check_same_hash(left, reordered);
    // But which parameter multiplies which register still matters.
    const std::string swapped_roles =
        "wrl 0.1\ntier G1\nstate x : 1 = 0\nstate y : 1 = 0.5\n"
        "param p : 1 = 1 in [0, 1] trainable bits 8\nparam q : 1 = 1 in [0, 2] trainable bits 8\n"
        "next x = q * x + p * y\nobserve identity_v1(x)\n";
    check_different_hash(left, swapped_roles);
}

OW_TEST(hash_constant_folding_and_identities) {
    check_same_hash(three_register_program("next a = a + (1 + 2)"), three_register_program("next a = 3 + a"));
    check_same_hash(three_register_program("next a = (a + 0.1) + 0.2"), three_register_program("next a = a + (0.2 + 0.1)"));
    check_same_hash(three_register_program("next a = a * (2 * 3)"), three_register_program("next a = 6 * a"));
    check_same_hash(three_register_program("next a = a * 1 + b"), three_register_program("next a = a + b"));
    check_same_hash(three_register_program("next a = a + 0 + b"), three_register_program("next a = a + b"));
    check_same_hash(three_register_program("next a = a - b"), three_register_program("next a = a + (-b)"));
    check_same_hash(three_register_program("next a = a - b"), three_register_program("next a = sub(a, b)"));
    check_same_hash(three_register_program("next a = a - b"), three_register_program("next a = -b + a"));
    check_same_hash(three_register_program("next a = a * -1"), three_register_program("next a = -a"));
    check_same_hash(three_register_program("next a = -(-a)"), three_register_program("next a = a"));
    check_same_hash(three_register_program("next a = a + b - b"), three_register_program("next a = a"));
    check_same_hash(three_register_program("next a = max(a, 0)"), three_register_program("next a = relu(a)"));
    check_same_hash(three_register_program("next a = max(a, a, b)"), three_register_program("next a = max(b, a)"));
    check_same_hash(three_register_program("next a = a * 0 + b"), three_register_program("next a = b"));
    check_same_hash(three_register_program("next a = select(1, a, b)"), three_register_program("next a = a"));
    check_same_hash(three_register_program("next a = select(c, a, a)"), three_register_program("next a = a"));
    check_same_hash(three_register_program("next a = clamp(5, 1, 3) + a"), three_register_program("next a = a + 3"));
    check_same_hash(three_register_program("next a = threshold(2, 1) + a"), three_register_program("next a = a + 1"));
    check_same_hash(three_register_program("next a = abs(abs(a))"), three_register_program("next a = abs(-a)"));
    check_same_hash(three_register_program("next a = delay(a, 0) + b"), three_register_program("next a = a + b"));
}

OW_TEST(hash_common_subexpression_elimination) {
    check_same_hash(three_register_program("next a = tanh(a * b) + tanh(b * a)"),
                    three_register_program("let t = tanh(a * b)\nnext a = t + t"));
}

OW_TEST(hash_distinguishes_different_programs) {
    check_different_hash(three_register_program("next a = a + b"), three_register_program("next a = a * b"));
    check_different_hash(three_register_program("next a = a + a"), three_register_program("next a = a"));
    check_different_hash(three_register_program("next a = a - b"), three_register_program("next a = b - a"));
    check_different_hash(three_register_program("next a = a + 1"), three_register_program("next a = a + 2"));
    check_different_hash(three_register_program("next a = tanh(a) + b"), three_register_program("next a = relu(a) + b"));
    check_different_hash(three_register_program("next a = delay(a, 1) + b"), three_register_program("next a = delay(a, 2) + b"));
    check_different_hash(three_register_program("next a = a + b"),
                         "wrl 0.1\ntier G1\nstate a : 1 = 0\nstate b : 1 = 0.75\nstate c : 1 = 0\nnext a = a + b\nobserve identity_v1(a)\n");
}

OW_TEST(hash_depends_on_declarations_that_define_the_program) {
    const std::string head = "wrl 0.1\ntier G1\nstate v : 1 = 0\n";
    const std::string tail = "\nnext v = p * v\nobserve identity_v1(v)\n";
    // A trainable parameter's starting value is not part of the program identity ...
    check_same_hash(head + "param p : 1 = 0.2 in [0, 1] trainable bits 8" + tail, head + "param p : 1 = 0.9 in [0, 1] trainable bits 8" + tail);
    // ... but its bounds, declared precision and trainability are, and so is a fixed value.
    check_different_hash(head + "param p : 1 = 0.2 in [0, 1] trainable bits 8" + tail, head + "param p : 1 = 0.2 in [0, 2] trainable bits 8" + tail);
    check_different_hash(head + "param p : 1 = 0.2 in [0, 1] trainable bits 8" + tail, head + "param p : 1 = 0.2 in [0, 1] trainable bits 9" + tail);
    check_different_hash(head + "param p : 1 = 0.2 fixed" + tail, head + "param p : 1 = 0.3 fixed" + tail);
    check_different_hash(head + "param p : 1 = 0.2 in [0, 1] trainable bits 8" + tail, head + "param p : 1 = 0.2 fixed" + tail);
}

OW_TEST(hash_changes_with_tier_dt_max_and_observation) {
    const std::string body = "state v : 1 = 0\nnext v = v\n";
    check_different_hash("wrl 0.1\ntier G1\n" + body + "observe identity_v1(v)\n", "wrl 0.1\ntier G0\n" + body + "observe identity_v1(v)\n");
    check_different_hash("wrl 0.1\ntier G1\ndt_max 0.1\n" + body + "observe identity_v1(v)\n", "wrl 0.1\ntier G1\ndt_max 0.2\n" + body + "observe identity_v1(v)\n");
    check_different_hash("wrl 0.1\ntier G1\n" + body + "observe identity_v1(v)\n", "wrl 0.1\ntier G1\n" + body + "observe calcium_linear_v1(v, 1[s])\n");
    check_different_hash("wrl 0.1\ntier G1\n" + body + "observe calcium_linear_v1(v, 1[s])\n", "wrl 0.1\ntier G1\n" + body + "observe calcium_linear_v1(v, 2[s])\n");
    check_different_hash("wrl 0.1\ntier G1\n" + body + "observe identity_v1(v)\n", "wrl 0.1\ntier G1\n" + body + "gap v\nobserve identity_v1(v)\n");
}

OW_TEST(rule_name_is_not_part_of_the_hash) {
    check_same_hash("wrl 0.1\nrule one\ntier G1\nstate v : 1 = 0\nobserve identity_v1(v)\n", "wrl 0.1\nrule two\ntier G1\nstate v : 1 = 0\nobserve identity_v1(v)\n");
}

OW_TEST(canonicalisation_is_idempotent_on_varied_programs) {
    const std::vector<std::string> programs = {
        three_register_program("next a = tanh(a + b) * c - relu(b)"),
        three_register_program("next a = select(threshold(b, 0.25), a + 1, c)\nnext c = delay(a, 3) + sum_in(b, inh)"),
        three_register_program("next a = clamp(a + stimulus, -1, 1)\nnext b = decay(b, 0.5[s])"),
        three_register_program("next a = sigmoid(a) * type_mask(motor) + b"),
    };
    for (const std::string& source : programs) {
        const Ir first = compile(source);
        const Ir second = compile(first.canonical_source);
        OW_CHECK_EQ(second.canonical_source, first.canonical_source);
        OW_CHECK_EQ(second.program_hash, first.program_hash);
    }
}

// ---- bit accounting ------------------------------------------------------------------------------------

OW_TEST(bit_accounting_hand_computed) {
    // header: tier 3 + dt_max flag 1 + stimulus unit (gamma0(0)+gamma0(0) = 2) = 6
    // registers: gamma(1)=1 + unit 2 + constant(1)=7 = 10;  parameters: gamma0(0)=1
    // instructions: gamma0(1)=3 + stimulus kind 8 = 11;  writes: gamma(1)=1;  gap 1;  observation 1+1 = 2
    const Ir ir = compile("wrl 0.1\ntier G0\nstate s : 1 = 1\nnext s = stimulus\nobserve identity_v1(s)\n");
    OW_CHECK_EQ(ir.bits.header, 6);
    OW_CHECK_EQ(ir.bits.registers, 10);
    OW_CHECK_EQ(ir.bits.parameters, 1);
    OW_CHECK_EQ(ir.bits.instructions, 11);
    OW_CHECK_EQ(ir.bits.writes, 1);
    OW_CHECK_EQ(ir.bits.gap, 1);
    OW_CHECK_EQ(ir.bits.observation, 2);
    OW_CHECK_EQ(ir.bits.l_struct, 32);
    OW_CHECK_EQ(ir.bits.l_params, 0);
}

OW_TEST(bit_accounting_charges_precision_and_dispatch) {
    // One trainable parameter at 12 bits: L_params = 12 regardless of the fitted value.
    const Ir ir = compile("wrl 0.1\ntier G1\nstate v : 1 = 0\nparam g : 1 = 1 in [0, 2] trainable bits 12\nnext v = g * v\nobserve identity_v1(v)\n");
    OW_CHECK_EQ(ir.bits.l_params, 12);
    // Precision is not free: raising `bits` raises L_params and leaves L_struct unchanged.
    const Ir finer = compile("wrl 0.1\ntier G1\nstate v : 1 = 0\nparam g : 1 = 1 in [0, 2] trainable bits 20\nnext v = g * v\nobserve identity_v1(v)\n");
    OW_CHECK_EQ(finer.bits.l_params, 20);
    OW_CHECK_EQ(finer.bits.l_struct, ir.bits.l_struct);
    // A lookup table and a type mask are charged as dispatch bits.
    const Ir g0 = compile("wrl 0.1\ntier G0\nstate s : 1 = 0\nnext s = lut(count_in(s, 1), [0, 1, 1])\nobserve identity_v1(s)\n");
    OW_CHECK(g0.bits.type_dispatch > 0);
    OW_CHECK_EQ(g0.bits.l_struct, g0.bits.l_ast + g0.bits.l_topology_overrides + g0.bits.type_dispatch);
    const Ir masked = compile("wrl 0.1\ntier G1\nstate v : 1 = 0\nnext v = type_mask(motor)\nobserve identity_v1(v)\n");
    OW_CHECK_EQ(masked.bits.type_dispatch, elias_gamma_bits(5) + 8 * 5);
}

OW_TEST(more_complex_programs_cost_more_bits) {
    const Ir small = compile(three_register_program("next a = a + b"));
    const Ir large = compile(three_register_program("next a = tanh(a + b) * c - relu(b)"));
    OW_CHECK(large.bits.l_struct > small.bits.l_struct);
}

// ---- IR JSON -------------------------------------------------------------------------------------------

OW_TEST(ir_json_contains_documented_fields) {
    const Ir ir = compile(kLeakAdapt);
    const Json json = ir_to_json(ir);
    OW_CHECK_EQ(json.at("schema").as_string(), std::string(kIrSchema));
    OW_CHECK_EQ(json.at("grammar_version").as_string(), std::string("0.1"));
    OW_CHECK_EQ(json.at("program_hash").as_string(), ir.program_hash);
    OW_CHECK(!json.at("compiler_build").as_string().empty());
    OW_CHECK_EQ(json.at("registers").as_array().size(), std::size_t{2});
    OW_CHECK_EQ(json.at("parameters").as_array().size(), std::size_t{4});
    OW_CHECK(json.at("instructions").as_array().size() > 5);
    OW_CHECK_EQ(json.at("l_struct").at("total_bits").as_int(), static_cast<long long>(ir.bits.l_struct));
    OW_CHECK_EQ(json.at("l_params").at("total_bits").as_int(), 44LL);
    const Json& first = json.at("instructions").as_array()[0];
    OW_CHECK(first.find("op") && first.find("args") && first.find("attrs") && first.find("unit") && first.find("id"));
}

OW_TEST(ir_json_round_trips) {
    for (const std::string& source : {kLeakAdapt,
                                      std::string("wrl 0.1\ntier G0\nstate s : 1 = 0\nnext s = lut(add(mul(s, 2), count_in(s, 1)), [0, 1, 1, 0, 1, 1])\nobserve identity_v1(s)\n"),
                                      three_register_program("next a = select(threshold(b, 0.25), a + 1, c)\nnext c = delay(a, 3) + sum_in(b, inh) + type_mask(x)")}) {
        const Ir ir = compile(source);
        const Json json = Json::parse(ir_to_json(ir).dump(2));
        const Ir again = ir_from_json(json);
        OW_CHECK_EQ(print_source(again, {}), ir.canonical_source);
        OW_CHECK_EQ(ir_to_json(again).dump(), ir_to_json(ir).dump());
    }
}
