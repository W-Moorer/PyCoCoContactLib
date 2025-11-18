import numpy as np

from rcsim.contact import RapidMesh


class RigidBody:
    def __init__(self, mass, position, velocity, is_static: bool = False):
        self.mass = float(mass)
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.velocity = np.asarray(velocity, dtype=float).reshape(3)
        self.is_static = bool(is_static)


class RigidMeshBody:
    def __init__(self, mesh: RapidMesh, mass, position, velocity, is_static: bool = False):
        assert isinstance(mesh, RapidMesh)
        self.mesh = mesh
        self.body = RigidBody(mass, position, velocity, is_static)
        self.R = np.eye(3, dtype=float)