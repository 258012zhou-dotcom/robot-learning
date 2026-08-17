import pytest

from robot_learning.config import PointRobotConfig


def valid_mapping() -> dict[str, object]:
    return {
        "experiment_name": "001_point_robot",
        "seed": 42,
        "initial_position": [0, 0],
        "velocity": [1, 0.5],
        "dt": 0.1,
        "steps": 100,
    }


def test_config_is_created_from_a_valid_mapping():
    config = PointRobotConfig.from_mapping(valid_mapping())

    assert config.experiment_name == "001_point_robot"
    assert config.seed == 42
    assert config.initial_position == (0.0, 0.0)
    assert config.velocity == (1.0, 0.5)
    assert config.dt == 0.1
    assert config.steps == 100


def test_config_rejects_missing_required_field():
    config = valid_mapping()
    del config["velocity"]

    with pytest.raises(ValueError, match="missing required fields: velocity"):
        PointRobotConfig.from_mapping(config)


@pytest.mark.parametrize("field_name", ["initial_position", "velocity"])
def test_config_rejects_non_two_dimensional_vectors(field_name: str):
    config = valid_mapping()
    config[field_name] = [0, 1, 2]

    with pytest.raises(ValueError, match=f"{field_name} must contain exactly two numbers"):
        PointRobotConfig.from_mapping(config)


@pytest.mark.parametrize("dt", [0, -0.1, True, "0.1"])
def test_config_rejects_invalid_dt(dt: object):
    config = valid_mapping()
    config["dt"] = dt

    with pytest.raises(ValueError, match="dt must be a positive number"):
        PointRobotConfig.from_mapping(config)


@pytest.mark.parametrize("steps", [-1, True, 1.5, "100"])
def test_config_rejects_invalid_steps(steps: object):
    config = valid_mapping()
    config["steps"] = steps

    with pytest.raises(ValueError):
        PointRobotConfig.from_mapping(config)
