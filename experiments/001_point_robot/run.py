"""运行二维点机器人的恒定速度轨迹仿真并保存图片。"""

from pathlib import Path

import matplotlib.pyplot as plt

from robot_learning.point_robot import simulate_constant_velocity


def main() -> None:
    """生成实验轨迹并保存到 outputs 目录。"""
    trajectory = simulate_constant_velocity(
        initial_position=[0, 0],
        velocity=[1, 0.5],
        dt=0.1,
        steps=100,
    )

    output_path = Path("outputs/001_point_robot/trajectory.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="trajectory")
    plt.scatter(*trajectory[0], label="start", zorder=3)
    plt.scatter(*trajectory[-1], label="end", zorder=3)
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("2D Point Robot: Constant Velocity")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"轨迹形状：{trajectory.shape}")
    print(f"最终位置：{trajectory[-1]}")
    print(f"图片已保存到：{output_path}")


if __name__ == "__main__":
    main()
