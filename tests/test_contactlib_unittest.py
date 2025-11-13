import unittest
import numpy as np

from PyCoCoContactLib import ContactSolver
from PyCoCoContactLib.core.mesh import Mesh
from PyCoCoContactLib.core.sphere import make_icosphere


class ContactLibBasicTests(unittest.TestCase):
    def test_sphere_sphere_contact_positive_force(self):
        A = make_icosphere(R=0.5, center=(0, 0, 0.26), subdivisions=2)
        B = make_icosphere(R=0.5, center=(0, 0, -0.26), subdivisions=2)
        meshA = Mesh(V=A.V, F=A.F)
        meshB = Mesh(V=B.V, F=B.F)
        solver = ContactSolver()
        res = solver.compute(meshA, meshB)
        self.assertGreater(np.linalg.norm(res.total_force), 0.0)
        self.assertGreaterEqual(res.max_penetration, 0.0)

    def test_no_contact_zero_force(self):
        A = make_icosphere(R=0.4, center=(0, 0, 2.0), subdivisions=1)
        B = make_icosphere(R=0.4, center=(0, 0, -2.0), subdivisions=1)
        meshA = Mesh(V=A.V, F=A.F)
        meshB = Mesh(V=B.V, F=B.F)
        solver = ContactSolver()
        res = solver.compute(meshA, meshB)
        self.assertAlmostEqual(float(np.linalg.norm(res.total_force)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
