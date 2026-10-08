#pragma once
// Exact scalar semantics of the WRL operators. The canonicaliser (constant folding) and the reference
// interpreter both call these functions, so folding can never disagree with execution. Every function
// uses only IEEE-754 basic operations, except sigmoid / exp-based integrators, which use libm exp.

#include <cmath>
#include <cstddef>

namespace occamworm::scalar {

inline double relu(double x) { return x > 0.0 ? x : 0.0; }
inline double min2(double accumulated, double next) { return next < accumulated ? next : accumulated; }
inline double max2(double accumulated, double next) { return next > accumulated ? next : accumulated; }
inline double clamp(double x, double lo, double hi) { return min2(max2(x, lo), hi); }
inline double threshold(double x, double theta) { return x >= theta ? 1.0 : 0.0; }
inline double select(double condition, double a, double b) { return condition != 0.0 ? a : b; }
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Index into a table of n >= 1 entries: floor(x) clamped to [0, n-1]. x must be finite.
inline std::size_t lut_index(double x, std::size_t n) {
    const double floored = std::floor(x);
    if (floored <= 0.0) {
        return 0;
    }
    if (floored >= static_cast<double>(n - 1)) {
        return n - 1;
    }
    return static_cast<std::size_t>(floored);
}

// dx/dt = (target - x) / tau with target held constant over one step of length dt (exact solution).
inline double leaky_integrate(double x, double target, double tau, double dt) {
    const double decay = std::exp(-(dt / tau));
    return target + (x - target) * decay;
}

// Forward Euler for the same ODE: x + (dt / tau) * (target - x).
inline double euler_leak(double x, double target, double tau, double dt) {
    const double rate = dt / tau;
    return x + rate * (target - x);
}

}  // namespace occamworm::scalar
