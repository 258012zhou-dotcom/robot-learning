def next_position(position, velocity, dt):
    """计算一维机器人经过 dt 秒后的新位置。"""
    return position + velocity * dt
