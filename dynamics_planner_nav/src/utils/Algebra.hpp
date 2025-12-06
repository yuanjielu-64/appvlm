#ifndef Antipatrea__Algebra_HPP_
#define Antipatrea__Algebra_HPP_

#include <cmath>

namespace Antipatrea {
    namespace Algebra {
        // Point distance calculation
        inline double PointDistance(const int n, const double p1[], const double p2[]) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                double diff = p1[i] - p2[i];
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }
    }
}

#endif
