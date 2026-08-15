from robot_learning.point_robot import next_position


def test_robot_moves_forward():
    assert next_position(position=0, velocity=2, dt=0.5) == 1


def test_robot_moves_backward():
    assert next_position(position=3, velocity=-1, dt=2) == 1
