#include "point_robot.hpp"

#include <iostream>

int main() {
    const Vec2 position{0.0, 0.0};
    const Vec2 velocity{1.0, 0.5};
    const double dt = 1.0;

    const Vec2 result = next_position(position, velocity, dt);

    std::cout
        << "Next position: ("
        << result.x
        << ", "
        << result.y
        << ")\n";

    return 0;
}