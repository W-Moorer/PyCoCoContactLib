import unittest
import numpy as np

from PyCoCoContactLib.io.obj_loader import load_obj
from PyCoCoContactLib.core.bvh import MedianSplitBVHBuilder, find_candidates_with_pose
from PyCoCoContactLib.detection.broad_phase import BroadPhaseDetector
from PyCoCoContactLib.detection.narrow_phase import MollerDetector
from PyCoCoContactLib.utils.transforms import apply_pose_to_mesh


class BVHPoseUpdateTests(unittest.TestCase):
    def test_reuse_bvh_candidates_match_rebuild(self):
        A = load_obj("models/ballMesh.obj")
        B = load_obj("models/ballMesh.obj")
        builder = MedianSplitBVHBuilder()
        bvhA = builder.build(A)
        bvhB = builder.build(B)
        originA = (0.0, 0.0, 0.26)
        originB = (0.0, 0.0, -0.26)
        quatI = (1.0, 0.0, 0.0, 0.0)
        cand_reuse = find_candidates_with_pose(A, bvhA.nodes, bvhA.root, originA, quatI,
                                               B, bvhB.nodes, bvhB.root, originB, quatI, skin=0.0)
        self.assertGreater(len(cand_reuse), 0)
        A_w = apply_pose_to_mesh(A, originA, quatI)
        B_w = apply_pose_to_mesh(B, originB, quatI)
        res_force = MollerDetector().check_pairs(A_w, B_w, cand_reuse)
        self.assertGreaterEqual(len(res_force), 1)


if __name__ == "__main__":
    unittest.main()
