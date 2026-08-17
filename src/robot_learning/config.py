"""实验配置的读取与校验。"""

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class PointRobotConfig:
    """二维点机器人恒定速度实验的不可变配置。"""

    experiment_name: str
    seed: int
    initial_position: tuple[float, float]
    velocity: tuple[float, float]
    dt: float
    steps: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "PointRobotConfig":
        """从 JSON 读取的映射创建并校验配置。"""
        if not isinstance(config, Mapping):
            raise ValueError("configuration must be a mapping")

        required_fields = (
            "experiment_name",
            "seed",
            "initial_position",
            "velocity",
            "dt",
            "steps",
        )
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"missing required fields: {', '.join(missing_fields)}")

        experiment_name = config["experiment_name"]
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")

        seed = config["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer, not bool")

        initial_position = cls._parse_vector(
            config["initial_position"], "initial_position"
        )
        velocity = cls._parse_vector(config["velocity"], "velocity")

        dt = config["dt"]
        if not isinstance(dt, Real) or isinstance(dt, bool) or not dt > 0:
            raise ValueError("dt must be a positive number")

        steps = config["steps"]
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise ValueError("steps must be an integer, not bool")
        if steps < 0:
            raise ValueError("steps must be non-negative")

        return cls(
            experiment_name=experiment_name,
            seed=seed,
            initial_position=initial_position,
            velocity=velocity,
            dt=float(dt),
            steps=steps,
        )

    @staticmethod
    def _parse_vector(value: Any, field_name: str) -> tuple[float, float]:
        """校验并转换一个恰好包含两个数的二维向量。"""
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{field_name} must contain exactly two numbers")

        try:
            values = tuple(value)
        except TypeError as error:
            raise ValueError(
                f"{field_name} must contain exactly two numbers"
            ) from error

        if len(values) != 2 or any(
            not isinstance(component, Real) or isinstance(component, bool)
            for component in values
        ):
            raise ValueError(f"{field_name} must contain exactly two numbers")

        return float(values[0]), float(values[1])
