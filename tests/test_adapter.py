import unittest
import numpy as np

from PyCoCoContactLib.io.obj_loader import load_obj
from PyCoCoContactLib.api.adapter import ContactAdapter


class ContactAdapterTests(unittest.TestCase):
    def test_adapter_step_contact(self):
        A = load_obj("models/ballMesh.obj")
        B = load_obj("models/ballMesh.obj")
        adapter = ContactAdapter.from_meshes(A, B)
        res = adapter.step((0.0, 0.0, 0.26), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -0.26), (1.0, 0.0, 0.0, 0.0))
        self.assertGreater(float(np.linalg.norm(res.total_force)), 0.0)
        self.assertGreaterEqual(float(res.max_penetration), 0.0)



if __name__ == "__main__":
    unittest.main()
