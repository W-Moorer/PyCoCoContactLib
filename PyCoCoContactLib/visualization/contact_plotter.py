import numpy as np
from ..core.mesh import Mesh


class ContactVisualizer:
    def plot(self, mesh_a: Mesh, mesh_b: Mesh, contact_points, pressures):
        from .pyvista_backend import visualize_contact
        zero = np.zeros(3)
        visualize_contact(mesh_a, mesh_b, contact_points, pressures, zero)

    def show(self):
        pass

