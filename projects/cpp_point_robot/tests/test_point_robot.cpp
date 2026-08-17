#include "point_robot.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

bool approximately_equal(double left, double right) {
    constexpr double tolerance = 1e-9;
    return std::abs(left - right) < tolerance;
}

int main() {
    const Vec2 result = next_position(
        Vec2{0.0, 0.0},
        Vec2{1.0, 0.5},
        2.0
    );

    if (
        !approximately_equal(result.x, 2.0)
        || !approximately_equal(result.y, 1.0)
    ) {
        std::cerr << "forward motion test failed\n";
        return 1;
    }

    bool exception_was_thrown = false;

    try {
        next_position(
            Vec2{0.0, 0.0},
            Vec2{1.0, 0.5},
            -1.0
        );
    } catch (const std::invalid_argument&) {
        exception_was_thrown = true;
    }

    if (!exception_was_thrown) {
        std::cerr << "negative dt test failed\n";
        return 1;
    }

    return 0;
}