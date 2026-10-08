#include "occamworm/core/graph.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

#include "occamworm/core/error.hpp"

namespace occamworm {
namespace {

[[noreturn]] void fail(const std::string& message) { throw Error(Errc::Graph, message); }

bool valid_name(std::string_view name) {
    if (name.empty() || name.size() > 64) {
        return false;
    }
    return std::ranges::all_of(name, [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
               c == ':' || c == '-';
    });
}

const Json& require(const Json& object, std::string_view key) {
    const Json* found = object.find(key);
    if (found == nullptr) {
        fail("missing key '" + std::string(key) + "'");
    }
    return *found;
}

}  // namespace

GraphSpec graph_spec_from_json(const Json& json) {
    GraphSpec spec;
    try {
        for (const Json& item : json.at("neurons").as_array()) {
            NeuronSpec neuron;
            neuron.id = item.at("id").as_string();
            if (const Json* type = item.find("type")) {
                neuron.type = type->as_string();
            }
            spec.neurons.push_back(std::move(neuron));
        }
        if (const Json* chemical = json.find("chemical")) {
            for (const Json& item : chemical->as_array()) {
                ChemicalEdgeSpec edge;
                edge.pre = require(item, "pre").as_string();
                edge.post = require(item, "post").as_string();
                edge.weight = item.find("weight") ? item.at("weight").as_double() : 1.0;
                edge.sign = item.find("sign") ? static_cast<int>(item.at("sign").as_int()) : 1;
                edge.delay = item.find("delay") ? static_cast<int>(item.at("delay").as_int()) : 0;
                spec.chemical.push_back(std::move(edge));
            }
        }
        if (const Json* gap = json.find("gap")) {
            for (const Json& item : gap->as_array()) {
                GapEdgeSpec edge;
                edge.a = require(item, "a").as_string();
                edge.b = require(item, "b").as_string();
                edge.conductance = require(item, "g").as_double();
                spec.gap.push_back(std::move(edge));
            }
        }
    } catch (const Error& error) {
        if (error.code() == Errc::Json) {
            throw Error(Errc::Graph, "malformed graph description: " + error.message());
        }
        throw;
    }
    return spec;
}

Json graph_spec_to_json(const GraphSpec& spec) {
    Json out = Json::object();
    Json neurons = Json::array();
    for (const NeuronSpec& n : spec.neurons) {
        Json item = Json::object();
        item.set("id", n.id);
        item.set("type", n.type);
        neurons.push(std::move(item));
    }
    out.set("neurons", std::move(neurons));
    Json chemical = Json::array();
    for (const ChemicalEdgeSpec& e : spec.chemical) {
        Json item = Json::object();
        item.set("pre", e.pre);
        item.set("post", e.post);
        item.set("weight", e.weight);
        item.set("sign", e.sign);
        item.set("delay", e.delay);
        chemical.push(std::move(item));
    }
    out.set("chemical", std::move(chemical));
    Json gap = Json::array();
    for (const GapEdgeSpec& e : spec.gap) {
        Json item = Json::object();
        item.set("a", e.a);
        item.set("b", e.b);
        item.set("g", e.conductance);
        gap.push(std::move(item));
    }
    out.set("gap", std::move(gap));
    return out;
}

namespace {
bool has_neuron(const GraphSpec& spec, std::string_view id) {
    return std::ranges::any_of(spec.neurons, [&](const NeuronSpec& n) { return n.id == id; });
}
}  // namespace

void delete_chemical_edges(GraphSpec& spec, std::string_view pre, std::string_view post) {
    if (!has_neuron(spec, pre) || !has_neuron(spec, post)) {
        fail("delete_chemical names an unknown neuron");
    }
    std::erase_if(spec.chemical, [&](const ChemicalEdgeSpec& e) { return e.pre == pre && e.post == post; });
}

void delete_gap_edges(GraphSpec& spec, std::string_view a, std::string_view b) {
    if (!has_neuron(spec, a) || !has_neuron(spec, b)) {
        fail("delete_gap names an unknown neuron");
    }
    std::erase_if(spec.gap, [&](const GapEdgeSpec& e) { return (e.a == a && e.b == b) || (e.a == b && e.b == a); });
}

std::optional<std::size_t> Graph::index_of(std::string_view id) const {
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == id) {
            return i;
        }
    }
    return std::nullopt;
}

Graph Graph::build(const GraphSpec& spec) {
    Graph graph;
    std::map<std::string, std::size_t, std::less<>> index;
    for (const NeuronSpec& neuron : spec.neurons) {
        if (!valid_name(neuron.id)) {
            fail("invalid neuron id '" + neuron.id + "'");
        }
        if (!valid_name(neuron.type)) {
            fail("invalid neuron type '" + neuron.type + "'");
        }
        if (!index.emplace(neuron.id, graph.ids.size()).second) {
            fail("duplicate neuron id '" + neuron.id + "'");
        }
        graph.ids.push_back(neuron.id);
        graph.types.push_back(neuron.type);
    }
    const std::size_t n = graph.ids.size();
    const auto lookup = [&](const std::string& id) -> std::size_t {
        const auto it = index.find(id);
        if (it == index.end()) {
            fail("edge references unknown neuron '" + id + "'");
        }
        return it->second;
    };

    struct ChemEntry {
        std::size_t post;
        std::size_t pre;
        int delay;
        int sign;
        double weight;
    };
    std::vector<ChemEntry> chem;
    for (const ChemicalEdgeSpec& edge : spec.chemical) {
        if (!std::isfinite(edge.weight) || edge.weight < 0.0) {
            fail("chemical weight must be finite and non-negative");
        }
        if (edge.sign != 1 && edge.sign != -1) {
            fail("chemical sign must be +1 or -1");
        }
        if (edge.delay < 0 || edge.delay > kMaxEdgeDelayTicks) {
            fail("chemical delay must be an integer in [0, " + std::to_string(kMaxEdgeDelayTicks) + "]");
        }
        chem.push_back({lookup(edge.post), lookup(edge.pre), edge.delay, edge.sign, edge.weight});
        graph.max_edge_delay = std::max(graph.max_edge_delay, edge.delay);
    }
    std::ranges::sort(chem, [](const ChemEntry& l, const ChemEntry& r) {
        return std::tie(l.post, l.pre, l.delay, l.sign, l.weight) < std::tie(r.post, r.pre, r.delay, r.sign, r.weight);
    });
    graph.in_offsets.assign(n + 1, 0);
    for (const ChemEntry& e : chem) {
        ++graph.in_offsets[e.post + 1];
    }
    for (std::size_t i = 0; i < n; ++i) {
        graph.in_offsets[i + 1] += graph.in_offsets[i];
    }
    for (const ChemEntry& e : chem) {  // already grouped by post, so append in order
        graph.pre.push_back(e.pre);
        graph.weight.push_back(e.weight);
        graph.sign.push_back(e.sign);
        graph.delay.push_back(e.delay);
    }

    struct GapEntry {
        std::size_t row;
        std::size_t neighbor;
        double conductance;
    };
    std::vector<GapEntry> gap;
    for (const GapEdgeSpec& edge : spec.gap) {
        if (!std::isfinite(edge.conductance) || edge.conductance < 0.0) {
            fail("gap conductance must be finite and non-negative");
        }
        const std::size_t a = lookup(edge.a);
        const std::size_t b = lookup(edge.b);
        if (a == b) {
            fail("gap junction cannot connect a neuron to itself");
        }
        gap.push_back({a, b, edge.conductance});
        gap.push_back({b, a, edge.conductance});
    }
    std::ranges::sort(gap, [](const GapEntry& l, const GapEntry& r) {
        return std::tie(l.row, l.neighbor, l.conductance) < std::tie(r.row, r.neighbor, r.conductance);
    });
    graph.gap_offsets.assign(n + 1, 0);
    for (const GapEntry& e : gap) {
        ++graph.gap_offsets[e.row + 1];
    }
    for (std::size_t i = 0; i < n; ++i) {
        graph.gap_offsets[i + 1] += graph.gap_offsets[i];
    }
    for (const GapEntry& e : gap) {
        graph.gap_neighbor.push_back(e.neighbor);
        graph.gap_conductance.push_back(e.conductance);
    }
    return graph;
}

}  // namespace occamworm
