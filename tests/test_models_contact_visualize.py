import os
import unittest
import numpy as np

from PyCoCoContactLib import ContactSolver, Mesh
from PyCoCoContactLib.io.obj_loader import load_obj
from PyCoCoContactLib.utils.transforms import apply_pose_to_mesh


def set_reference(mesh: Mesh, ref_center, quat_wxyz) -> Mesh:
    com = mesh.center_of_mass()
    V0 = np.asarray(mesh.V, float) - com[None, :]
    m0 = Mesh(V=V0, F=np.asarray(mesh.F, int))
    return apply_pose_to_mesh(m0, origin=ref_center, quat_wxyz=quat_wxyz)


class ModelsContactVisualizeTests(unittest.TestCase):
    def test_models_contact_and_optional_visualize(self):
        ballA = load_obj("models/ballMesh.obj")
        ballB = load_obj("models/ballMesh.obj")

        ballA_ref = set_reference(ballA, (0.0, 0.0, 198.0), (1.0, 0.0, 0.0, 0.0))
        ballB_ref = set_reference(ballB, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

        solver = ContactSolver()
        res = solver.compute(ballA_ref, ballB_ref)
        self.assertGreater(float(np.linalg.norm(res.total_force)), 0.0)
        self.assertGreaterEqual(float(res.max_penetration), 0.0)

        if os.environ.get("PYCOCO_VIZ", "0") == "1":
            try:
                from PyCoCoContactLib.visualization.pyvista_backend import visualize_contact
                visualize_contact(ballA_ref, ballB_ref, res.contact_points, res.pressures, res.total_force)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
