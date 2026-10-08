#pragma once
// Typed neuron graph in CSR form (§6.2).
//
// Orientation (versioned, tested in tests/cpp/test_graph.cpp):
//   * A chemical edge pre -> post is stored in the row of its POSTSYNAPTIC neuron `post`:
//     `in_offsets[post] .. in_offsets[post+1]` indexes the entries, and `pre[k]` is the presynaptic source.
//     A row therefore lists the INPUTS of a neuron, which is what `sum_in` iterates.
//   * Within a row, entries are sorted by (pre index, delay, sign, weight), ascending. This fixes the
//     floating-point summation order independently of the order in which edges were listed.
//   * A gap junction {a, b, g} is undirected and stored twice, once in each endpoint's row of the gap CSR
//     (`gap_offsets`, `gap_neighbor`, `gap_conductance`), sorted by (neighbor index, conductance).
//   * CSR orientation version: kCsrOrientationVersion.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "occamworm/core/json.hpp"

namespace occamworm {

constexpr int kCsrOrientationVersion = 1;
constexpr int kMaxEdgeDelayTicks = 100000;

struct NeuronSpec {
    std::string id;
    std::string type = "generic";
};

struct ChemicalEdgeSpec {
    std::string pre;
    std::string post;
    double weight = 1.0;  // non-negative magnitude
    int sign = 1;         // +1 excitatory, -1 inhibitory
    int delay = 0;        // whole ticks, >= 0
};

struct GapEdgeSpec {
    std::string a;
    std::string b;
    double conductance = 0.0;  // non-negative, units s^-1 when states are dimensionless rates
};

struct GraphSpec {
    std::vector<NeuronSpec> neurons;
    std::vector<ChemicalEdgeSpec> chemical;
    std::vector<GapEdgeSpec> gap;
};

// {"neurons":[{"id","type"?}], "chemical":[{"pre","post","weight","sign","delay"}], "gap":[{"a","b","g"}]}
GraphSpec graph_spec_from_json(const Json& json);
Json graph_spec_to_json(const GraphSpec& spec);

// Removes every chemical edge pre->post / every gap junction between a and b. Unknown ids are errors.
void delete_chemical_edges(GraphSpec& spec, std::string_view pre, std::string_view post);
void delete_gap_edges(GraphSpec& spec, std::string_view a, std::string_view b);

class Graph {
public:
    // Validates and builds the CSR arrays. Throws Error(Errc::Graph).
    static Graph build(const GraphSpec& spec);

    std::size_t size() const { return ids.size(); }
    std::optional<std::size_t> index_of(std::string_view id) const;

    std::vector<std::string> ids;
    std::vector<std::string> types;

    // Chemical CSR (row = postsynaptic neuron).
    std::vector<std::size_t> in_offsets;  // size N+1
    std::vector<std::size_t> pre;         // size E
    std::vector<double> weight;           // size E, >= 0
    std::vector<int> sign;                // size E, +1 or -1
    std::vector<int> delay;               // size E, ticks >= 0

    // Gap CSR (symmetric adjacency, each junction appears in both endpoint rows).
    std::vector<std::size_t> gap_offsets;  // size N+1
    std::vector<std::size_t> gap_neighbor;
    std::vector<double> gap_conductance;

    int max_edge_delay = 0;
};

}  // namespace occamworm
