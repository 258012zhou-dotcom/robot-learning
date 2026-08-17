#include "point_robot.hpp"

#include <stdexcept>

Vec2 next_position(
    const Vec2& position,
    const Vec2& velocity,
    double dt
) {
    if (dt < 0.0) {
        throw std::invalid_argument("dt must be non-negative");
    }

    return {
        position.x + velocity.x * dt,
        position.y + velocity.y * dt,
    };
}