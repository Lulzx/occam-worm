// ow-core: number formatting, JSON, units and the CSR graph (including its documented orientation).

#include <cmath>
#include <string>

#include "occamworm/core/graph.hpp"
#include "occamworm/core/json.hpp"
#include "occamworm/core/numfmt.hpp"
#include "occamworm/core/units.hpp"
#include "ow_test.hpp"

using namespace occamworm;

OW_TEST(canonical_number_format) {
    OW_CHECK_EQ(canonical_number(0.0), std::string("0e0"));
    OW_CHECK_EQ(canonical_number(-0.0), std::string("0e0"));
    OW_CHECK_EQ(canonical_number(1.0), std::string("1e0"));
    OW_CHECK_EQ(canonical_number(0.25), std::string("2.5e-1"));
    OW_CHECK_EQ(canonical_number(-12.0), std::string("-1.2e1"));
    OW_CHECK_EQ(canonical_number(100.0), std::string("1e2"));
    OW_CHECK_EQ(canonical_number(0.1), std::string("1e-1"));
    OW_CHECK_EQ(canonical_number(5e-3), std::string("5e-3"));
    OW_CHECK_EQ(canonical_number(1e300), std::string("1e300"));
}

OW_TEST(canonical_number_round_trips) {
    for (const double v : {0.1, 1.0 / 3.0, 123456.789, 2.2250738585072014e-308, 1.7976931348623157e308, -4.9e-324}) {
        const auto parsed = parse_number(canonical_number(v));
        OW_CHECK(parsed.has_value());
        if (parsed) {
            OW_CHECK_EQ(*parsed, v);
        }
    }
}

OW_TEST(parse_number_rejects_malformed) {
    for (const char* bad : {"", "-", "1.", ".5", "1e", "0x10", "inf", "nan", "1 ", " 1", "1e999", "--1"}) {
        OW_CHECK(!parse_number(bad).has_value());
    }
    OW_CHECK(parse_number("1e-3").has_value());
    OW_CHECK(parse_number("+2").has_value());
}

OW_TEST(constant_bit_costs) {
    // sign(1) + gamma(digits) + 4*digits + zigzag-gamma0(exponent)
    OW_CHECK_EQ(constant_bits(1.0), 1 + 1 + 4 + 1);      // digits "1", e=0 -> gamma0(0)=1
    OW_CHECK_EQ(constant_bits(0.25), 1 + 3 + 8 + 3);     // digits "25", e=-1 -> zigzag 1 -> gamma0(1)=3
    OW_CHECK_EQ(constant_bits(0.0), 1 + 1 + 4 + 1);
    OW_CHECK_EQ(elias_gamma_bits(1), 1);
    OW_CHECK_EQ(elias_gamma_bits(2), 3);
    OW_CHECK_EQ(elias_gamma_bits(7), 5);
    OW_CHECK_EQ(elias_gamma_bits(8), 7);
    OW_CHECK_EQ(elias_gamma0_bits(0), 1);
}

OW_TEST(json_round_trip_and_order) {
    const Json parsed = Json::parse(R"({"b":[1,2.5,"x\n",true,null],"a":{"k":-3e2}})");
    OW_CHECK_EQ(parsed.dump(), std::string(R"({"b":[1,2.5,"x\n",true,null],"a":{"k":-300}})"));
    OW_CHECK_EQ(parsed.at("a").at("k").as_int(), -300LL);
    OW_CHECK(Json::parse(parsed.dump()).dump() == parsed.dump());
}

OW_TEST(json_rejects_malformed) {
    for (const char* bad : {"", "{", "[1,]", "{\"a\":1,}", "01", "\"\\x\"", "{\"a\":1,\"a\":2}", "[1] 2", "nul", "\"\x01\"", "1e999"}) {
        OW_CHECK_THROWS(Json::parse(bad), Errc::Json);
    }
    OW_CHECK_THROWS(Json::parse(std::string(500, '[')), Errc::Json);  // depth limit
}

OW_TEST(json_unicode_escapes) {
    const Json parsed = Json::parse(R"("\u00e9\ud83d\ude00")");
    OW_CHECK_EQ(parsed.as_string(), std::string("\xC3\xA9\xF0\x9F\x98\x80"));
}

OW_TEST(json_pretty_inlines_scalar_arrays) {
    Json object = Json::object();
    object.set("v", Json::parse("[1,2,3]"));
    OW_CHECK_EQ(object.dump(2), std::string("{\n  \"v\": [1, 2, 3]\n}"));
}

OW_TEST(unit_parse_and_print) {
    OW_CHECK(parse_unit("dimensionless").dimensionless());
    OW_CHECK(parse_unit("1").dimensionless());
    OW_CHECK(parse_unit("s") == (Unit{1, 0}));
    OW_CHECK(parse_unit("1/s") == (Unit{-1, 0}));
    OW_CHECK(parse_unit("V/s") == (Unit{-1, 1}));
    OW_CHECK(parse_unit("s^-1") == parse_unit("1/s"));
    OW_CHECK(parse_unit("V*s^2/s") == (Unit{1, 1}));
    OW_CHECK_EQ(to_string(Unit{}), std::string("1"));
    OW_CHECK_EQ(to_string(Unit{-1, 1}), std::string("V*s^-1"));
    OW_CHECK_EQ(to_string(Unit{2, 0}), std::string("s^2"));
    OW_CHECK(parse_unit(to_string(Unit{-3, 2})) == (Unit{-3, 2}));
}

OW_TEST(unit_parse_rejects) {
    for (const char* bad : {"", "m", "s^", "s^x", "s^99", "s*", "/s", "s s", "S"}) {
        OW_CHECK_THROWS(parse_unit(bad), Errc::Unit);
    }
}

namespace {
GraphSpec three_neuron_spec() {
    GraphSpec spec;
    spec.neurons = {{"A", "sensory"}, {"B", "inter"}, {"C", "inter"}};
    // B receives from C then from A, listed in non-canonical order; C receives from A.
    spec.chemical = {{"C", "B", 0.5, -1, 2}, {"A", "B", 1.0, 1, 0}, {"A", "C", 2.0, 1, 1}};
    spec.gap = {{"C", "B", 3.0}};
    return spec;
}
}  // namespace

OW_TEST(graph_csr_rows_are_postsynaptic) {
    const Graph graph = Graph::build(three_neuron_spec());
    OW_CHECK_EQ(graph.size(), std::size_t{3});
    // Row of B (index 1) lists its two inputs sorted by presynaptic index: A (0) then C (2).
    OW_CHECK_EQ(graph.in_offsets[1], std::size_t{0});
    OW_CHECK_EQ(graph.in_offsets[2], std::size_t{2});
    OW_CHECK_EQ(graph.pre[0], std::size_t{0});
    OW_CHECK_EQ(graph.weight[0], 1.0);
    OW_CHECK_EQ(graph.sign[0], 1);
    OW_CHECK_EQ(graph.pre[1], std::size_t{2});
    OW_CHECK_EQ(graph.sign[1], -1);
    OW_CHECK_EQ(graph.delay[1], 2);
    // Row of C (index 2) lists its single input A.
    OW_CHECK_EQ(graph.in_offsets[3] - graph.in_offsets[2], std::size_t{1});
    OW_CHECK_EQ(graph.pre[2], std::size_t{0});
    OW_CHECK_EQ(graph.delay[2], 1);
    // Row of A (index 0) is empty: nothing projects onto A.
    OW_CHECK_EQ(graph.in_offsets[1] - graph.in_offsets[0], std::size_t{0});
    OW_CHECK_EQ(graph.max_edge_delay, 2);
}

OW_TEST(graph_gap_is_symmetric) {
    const Graph graph = Graph::build(three_neuron_spec());
    // Junction B-C appears in both rows.
    OW_CHECK_EQ(graph.gap_offsets[2] - graph.gap_offsets[1], std::size_t{1});
    OW_CHECK_EQ(graph.gap_neighbor[graph.gap_offsets[1]], std::size_t{2});
    OW_CHECK_EQ(graph.gap_offsets[3] - graph.gap_offsets[2], std::size_t{1});
    OW_CHECK_EQ(graph.gap_neighbor[graph.gap_offsets[2]], std::size_t{1});
    OW_CHECK_EQ(graph.gap_conductance[graph.gap_offsets[2]], 3.0);
}

OW_TEST(graph_validation) {
    GraphSpec spec = three_neuron_spec();
    spec.chemical.push_back({"A", "Z", 1.0, 1, 0});
    OW_CHECK_THROWS(Graph::build(spec), Errc::Graph);
    spec = three_neuron_spec();
    spec.chemical[0].sign = 0;
    OW_CHECK_THROWS(Graph::build(spec), Errc::Graph);
    spec = three_neuron_spec();
    spec.chemical[0].delay = -1;
    OW_CHECK_THROWS(Graph::build(spec), Errc::Graph);
    spec = three_neuron_spec();
    spec.chemical[0].weight = -1.0;
    OW_CHECK_THROWS(Graph::build(spec), Errc::Graph);
    spec = three_neuron_spec();
    spec.gap[0].conductance = std::nan("");
    OW_CHECK_THROWS(Graph::build(spec), Errc::Graph);
    spec = three_neuron_spec();
    spec.gap.push_back({"A", "A", 1.0});
    OW_CHECK_THROWS(Graph::build(spec), Errc::Graph);
    spec = three_neuron_spec();
    spec.neurons.push_back({"A", "x"});
    OW_CHECK_THROWS(Graph::build(spec), Errc::Graph);
    spec = three_neuron_spec();
    spec.neurons[0].id = "bad id";
    OW_CHECK_THROWS(Graph::build(spec), Errc::Graph);
}

OW_TEST(graph_edge_deletion) {
    GraphSpec spec = three_neuron_spec();
    delete_chemical_edges(spec, "A", "B");
    OW_CHECK_EQ(spec.chemical.size(), std::size_t{2});
    delete_gap_edges(spec, "B", "C");
    OW_CHECK_EQ(spec.gap.size(), std::size_t{0});
    OW_CHECK_THROWS(delete_chemical_edges(spec, "A", "nobody"), Errc::Graph);
}

OW_TEST(graph_json_round_trip) {
    const GraphSpec spec = three_neuron_spec();
    const GraphSpec again = graph_spec_from_json(graph_spec_to_json(spec));
    OW_CHECK_EQ(graph_spec_to_json(again).dump(), graph_spec_to_json(spec).dump());
}
