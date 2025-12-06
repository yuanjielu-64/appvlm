#ifndef Antipatrea__PseudoRandom_HPP_
#define Antipatrea__PseudoRandom_HPP_

#include <random>

namespace Antipatrea {
    // Random number generator
    inline double RandomUniformReal(double min, double max) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen);
    }
}

#endif
