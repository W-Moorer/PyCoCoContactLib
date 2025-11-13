from typing import List, Tuple
from ..core.mesh import Mesh
from ..core.bvh import BVHBuilder, traverse_bvhs


class BroadPhaseDetector:
    def __init__(self, bvh_builder: BVHBuilder):
        self.bvh_builder = bvh_builder

    def find_candidates(self, mesh_a: Mesh, mesh_b: Mesh, skin: float = 0.0) -> List[Tuple[int, int]]:
        bvh_a = self.bvh_builder.build(mesh_a)
        bvh_b = self.bvh_builder.build(mesh_b)
        return traverse_bvhs(bvh_a.nodes, mesh_a, bvh_a.root, bvh_b.nodes, mesh_b, bvh_b.root, skin=skin)
