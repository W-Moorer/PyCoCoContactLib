from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from ..core.mesh import Mesh
from ..core.bvh import BVHNode, BVHBuilder, MedianSplitBVHBuilder, find_candidates_with_pose
from ..utils.transforms import apply_pose_to_mesh
from ..detection.narrow_phase import MollerDetector
from ..detection.sampling import collect_contact_samples
from ..force.integrator import DistributedIntegrator


@dataclass
class StepContactResult:
    total_force: np.ndarray
    max_penetration: float
    contact_points: np.ndarray
    pressures: np.ndarray
    candidates: List[Tuple[int, int]]
    refined_pairs: List[Tuple[int, int]]
    stats: Dict[str, float]


@dataclass
class ContactAdapter:
    meshA: Mesh
    meshB: Mesh
    bvhA_nodes: List[BVHNode]
    bvhA_root: int
    bvhB_nodes: List[BVHNode]
    bvhB_root: int
    builder: BVHBuilder

    @staticmethod
    def from_meshes(meshA: Mesh, meshB: Mesh, builder: Optional[BVHBuilder] = None):
        b = builder or MedianSplitBVHBuilder()
        bvhA = b.build(meshA)
        bvhB = b.build(meshB)
        return ContactAdapter(meshA, meshB, bvhA.nodes, bvhA.root, bvhB.nodes, bvhB.root, b)

    def step(self, originA, quatA, originB, quatB, skin: float = 0.0) -> StepContactResult:
        cand = find_candidates_with_pose(self.meshA, self.bvhA_nodes, self.bvhA_root, originA, quatA,
                                         self.meshB, self.bvhB_nodes, self.bvhB_root, originB, quatB, skin)
        A_w = apply_pose_to_mesh(self.meshA, originA, quatA)
        B_w = apply_pose_to_mesh(self.meshB, originB, quatB)
        mol = MollerDetector().check_pairs(A_w, B_w, cand)
        pts, deltas, normals, areas, stats = collect_contact_samples(
            A_w, B_w, self.bvhA_nodes, self.bvhA_root, self.bvhB_nodes, self.bvhB_root, skin
        )
        res = DistributedIntegrator(model=None, quadrature=None).integrate({
            "points": pts, "deltas": deltas, "normals": normals, "areas": areas
        })
        delta_max = float(np.max(deltas)) if deltas.size else 0.0
        return StepContactResult(
            total_force=np.asarray(res.total_force, float),
            max_penetration=delta_max,
            contact_points=pts,
            pressures=np.asarray(res.pressures, float),
            candidates=cand,
            refined_pairs=mol,
            stats=stats,
        )


