from typing import Any


class DistributedIntegrator:
    def __init__(self, model: Any, quadrature: Any):
        self.model = model
        self.quadrature = quadrature

    def integrate(self, contact_points_set: Any):
        deltas = contact_points_set["deltas"]
        normals = contact_points_set["normals"]
        areas = contact_points_set["areas"]
        from .calculator import compute_distributed_forces
        return compute_distributed_forces(deltas, normals, areas)

