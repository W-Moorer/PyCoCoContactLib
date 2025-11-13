from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


class ContactModelError(Exception):
    pass


@dataclass
class HertzLikeParameters:
    k: float = 1e6
    n: float = 1.5
    d: float = 0.0

    def validate(self) -> None:
        if not np.isfinite(self.k) or self.k <= 0.0:
            raise ValueError("k must be positive and finite")
        if not np.isfinite(self.n) or self.n <= 0.0:
            raise ValueError("n must be positive and finite")
        if not np.isfinite(self.d) or self.d < 0.0:
            raise ValueError("d must be non-negative and finite")


@dataclass
class ContactForceResult:
    total_force: np.ndarray
    total_moment: np.ndarray
    point_forces: np.ndarray
    pressures: np.ndarray
    scale_factor: float


class ContactForceCalculator:
    def __init__(self, params: Optional[HertzLikeParameters] = None) -> None:
        self.params = params or HertzLikeParameters()
        self.params.validate()

    def compute_point_force_magnitude(self, delta: float, delta_dot: float = 0.0) -> float:
        if not np.isfinite(delta) or not np.isfinite(delta_dot):
            raise ValueError("delta/delta_dot must be finite")
        if delta < 0.0:
            raise ValueError("delta must be >= 0")
        if delta == 0.0 and delta_dot <= 0.0:
            return 0.0
        F_elastic = self.params.k * (delta ** self.params.n)
        F_damping = self.params.d * delta_dot
        F = F_elastic + F_damping
        return float(max(F, 0.0))

    def compute_distributed_forces(
        self,
        deltas: Sequence[float],
        normals: np.ndarray,
        areas: Sequence[float],
        delta_dots: Optional[Sequence[float]] = None,
        reference_point: Optional[np.ndarray] = None,
        enforce_consistency: bool = False,
    ) -> ContactForceResult:
        deltas = np.asarray(deltas, dtype=float).reshape(-1)
        areas = np.asarray(areas, dtype=float).reshape(-1)
        normals = np.asarray(normals, dtype=float)
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError("normals must be (N,3)")
        N = deltas.size
        if normals.shape[0] != N or areas.size != N:
            raise ValueError("size mismatch")
        if np.any(deltas < 0.0):
            raise ValueError("delta >= 0")
        if np.any(areas < 0.0):
            raise ValueError("areas >= 0")
        if delta_dots is None:
            delta_dots_arr = np.zeros_like(deltas)
        else:
            delta_dots_arr = np.asarray(delta_dots, dtype=float).reshape(-1)
            if delta_dots_arr.size != N:
                raise ValueError("delta_dots size mismatch")
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("zero normal")
        unit_normals = normals / norms[:, None]
        pressures = np.zeros(N, dtype=float)
        for i in range(N):
            pressures[i] = self.compute_point_force_magnitude(deltas[i], delta_dots_arr[i])
        if np.allclose(pressures, 0.0) or np.allclose(areas, 0.0):
            zero = np.zeros(3, dtype=float)
            return ContactForceResult(
                total_force=zero.copy(),
                total_moment=zero.copy(),
                point_forces=np.zeros((N, 3), dtype=float),
                pressures=pressures,
                scale_factor=1.0,
            )
        point_forces = (pressures[:, None] * areas[:, None]) * unit_normals
        total_force = point_forces.sum(axis=0)
        total_moment = np.zeros(3, dtype=float)
        scale_factor = 1.0
        if enforce_consistency:
            delta_ref = float(deltas.max(initial=0.0))
            if delta_ref > 0.0:
                F_target = self.params.k * (delta_ref ** self.params.n)
                F_current = float(np.linalg.norm(total_force))
                if F_current > 0.0 and F_target > 0.0:
                    scale_factor = F_target / F_current
                    pressures *= scale_factor
                    point_forces *= scale_factor
                    total_force *= scale_factor
        return ContactForceResult(
            total_force=total_force,
            total_moment=total_moment,
            point_forces=point_forces,
            pressures=pressures,
            scale_factor=scale_factor,
        )


def compute_distributed_forces(deltas, normals, areas, delta_dots=None, enforce_consistency=False):
    calc = ContactForceCalculator(HertzLikeParameters())
    return calc.compute_distributed_forces(deltas=deltas, normals=normals, areas=areas,
                                           delta_dots=delta_dots, enforce_consistency=enforce_consistency)
