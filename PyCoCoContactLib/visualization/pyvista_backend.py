import numpy as np
import pyvista as pv


def mesh_to_pyvista(mesh):
    V = np.asarray(mesh.V, float)
    F = np.asarray(mesh.F, np.int64)
    n_faces = F.shape[0]
    faces = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), F]).ravel()
    return pv.PolyData(V, faces)


def visualize_contact(meshA, meshB, contact_points, pressures, total_force):
    pl = pv.Plotter()
    pl.set_background("white")
    polyA = mesh_to_pyvista(meshA)
    polyB = mesh_to_pyvista(meshB)
    pl.add_mesh(polyA, color="lightgray", opacity=0.4, show_edges=True, label="Mesh A")
    pl.add_mesh(polyB, color="white", opacity=0.6, show_edges=True, label="Mesh B")
    if contact_points.size > 0:
        cloud = pv.PolyData(contact_points)
        cloud["pressure"] = pressures
        pl.add_mesh(cloud, scalars="pressure", render_points_as_spheres=True, point_size=10.0, cmap="viridis")
        pl.add_scalar_bar(title="Contact pressure")
        if np.linalg.norm(total_force) > 0:
            origin = contact_points.mean(axis=0)
            boundsA = polyA.bounds
            boundsB = polyB.bounds
            all_bounds = [
                min(boundsA[0], boundsB[0]), max(boundsA[1], boundsB[1]),
                min(boundsA[2], boundsB[2]), max(boundsA[3], boundsB[3]),
                min(boundsA[4], boundsB[4]), max(boundsA[5], boundsB[5])
            ]
            bbox_diagonal = np.linalg.norm([
                all_bounds[1] - all_bounds[0],
                all_bounds[3] - all_bounds[2],
                all_bounds[5] - all_bounds[4]
            ])
            force_magnitude = np.linalg.norm(total_force)
            base_scale = bbox_diagonal * 0.01
            force_scale = np.log1p(force_magnitude) * 0.1
            arrow_scale = base_scale + force_scale
            arrow = pv.Arrow(start=origin, direction=total_force, scale=arrow_scale)
            pl.add_mesh(arrow, color="red", label="Total force")
    pl.add_axes(line_width=2)
    pl.add_legend(bcolor="white")
    pl.show()
