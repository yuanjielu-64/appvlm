#ifndef Antipatrea__Timer_HPP_
#define Antipatrea__Timer_HPP_

#include <chrono>

namespace Antipatrea {
    namespace Timer {
        using Clock = std::chrono::high_resolution_clock::time_point;

        inline void Start(Clock& clock) {
            clock = std::chrono::high_resolution_clock::now();
        }

        inline double Elapsed(const Clock& start) {
            return std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start).count();
        }
    }
}

#endif
