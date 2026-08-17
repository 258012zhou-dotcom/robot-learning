#pragma once

struct Vec2 {
    double x;
    double y;
};

Vec2 next_position(
    const Vec2& position,
    const Vec2& velocity,
    double dt
);