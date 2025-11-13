from dataclasses import dataclass
import numpy as np
from typing import Optional

from ..core.mesh import Mesh
from ..core.bvh import MedianSplitBVHBuilder
from ..detection.sampling import collect_contact_samples


class Dunavant7:
    def values(self):
        from ..quadrature import DUNAVANT7
        return DUNAVANT7


@dataclass
class ContactResult:
    total_force: np.ndarray
    max_penetration: float
    contact_points: np.ndarray
    pressures: np.ndarray


class ContactSolver:
    def __init__(self, model=None, bvh_builder=None, integrator=None):
        from ..force.models import HertzModel
        from ..force.integrator import DistributedIntegrator
        self.model = model or HertzModel()
        self.bvh_builder = bvh_builder or MedianSplitBVHBuilder()
        self.integrator = integrator or DistributedIntegrator(self.model, Dunavant7())

    def _sample_contact_points(self, mesh_a: Mesh, mesh_b: Mesh):
        bvh_a = self.bvh_builder.build(mesh_a)
        bvh_b = self.bvh_builder.build(mesh_b)
        pts, deltas, normals, areas, stats = collect_contact_samples(
            mesh_a, mesh_b, bvh_a.nodes, bvh_a.root, bvh_b.nodes, bvh_b.root, skin=0.0, qrule=None
        )
        return {
            "points": pts,
            "deltas": deltas,
            "normals": normals,
            "areas": areas,
            "stats": stats,
        }

    def compute(self, mesh_a: Mesh, mesh_b: Mesh, skin: float = 0.0):
        cps = self._sample_contact_points(mesh_a, mesh_b)
        res = self.integrator.integrate(cps)
        delta_max = float(cps["deltas"].max(initial=0.0)) if cps["deltas"].size else 0.0
        return ContactResult(
            total_force=np.asarray(res.total_force, float),
            max_penetration=delta_max,
            contact_points=cps["points"],
            pressures=np.asarray(res.pressures, float),
        )
