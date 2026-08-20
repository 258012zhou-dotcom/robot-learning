"""二维刚体坐标变换。"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rotation_matrix(angle: float) -> NDArray[np.float64]:
    """返回逆时针旋转 angle 弧度的二维旋转矩阵。"""
    if not isinstance(angle, Real) or isinstance(angle, bool):
        raise ValueError("angle must be a real number")
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")

    cosine = np.cos(float(angle))
    sine = np.sin(float(angle))
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )


@dataclass(frozen=True)
class RigidTransform2D:
    """把局部坐标转换到父坐标的二维刚体变换。"""

    angle: float
    translation: tuple[float, float]

    def __post_init__(self) -> None:
        rotation_matrix(self.angle)
        translation = np.asarray(self.translation, dtype=float)
        if translation.shape != (2,):
            raise ValueError("translation must have shape (2,)")
        if not np.all(np.isfinite(translation)):
            raise ValueError("translation must contain finite values")

    @property
    def rotation(self) -> NDArray[np.float64]:
        """局部方向到父坐标方向的旋转矩阵。"""
        return rotation_matrix(self.angle)

    def apply_to_point(self, point: ArrayLike) -> NDArray[np.float64]:
        """对位置应用旋转和平移。"""
        point_array = self._validate_vector(point, "point")
        return self.rotation @ point_array + np.asarray(self.translation)

    def apply_to_vector(self, vector: ArrayLike) -> NDArray[np.float64]:
        """对方向或速度只应用旋转，不应用平移。"""
        vector_array = self._validate_vector(vector, "vector")
        return self.rotation @ vector_array

    def inverse(self) -> "RigidTransform2D":
        """返回父坐标到局部坐标的逆变换。"""
        inverse_rotation = self.rotation.T
        inverse_translation = (
            -inverse_rotation @ np.asarray(self.translation)
        )
        return RigidTransform2D(
            angle=-float(self.angle),
            translation=tuple(inverse_translation),
        )

    def compose(self, local_transform: "RigidTransform2D") -> "RigidTransform2D":
        """组合变换：先应用 local_transform，再应用当前变换。"""
        if not isinstance(local_transform, RigidTransform2D):
            raise TypeError("local_transform must be RigidTransform2D")

        composed_translation = self.apply_to_point(
            local_transform.translation
        )
        return RigidTransform2D(
            angle=float(self.angle) + float(local_transform.angle),
            translation=tuple(composed_translation),
        )

    @property
    def homogeneous_matrix(self) -> NDArray[np.float64]:
        """返回用于齐次坐标计算的 3×3 变换矩阵。"""
        matrix = np.eye(3, dtype=float)
        matrix[:2, :2] = self.rotation
        matrix[:2, 2] = np.asarray(self.translation)
        return matrix

    @staticmethod
    def _validate_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (2,):
            raise ValueError(f"{name} must have shape (2,)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain finite values")
        return vector
