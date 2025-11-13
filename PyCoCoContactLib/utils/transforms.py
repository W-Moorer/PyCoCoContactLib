import numpy as np
from ..core.mesh import Mesh


def quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=float).reshape(4)
    w, x, y, z = q
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = q / norm
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)
    return R


def apply_pose_to_mesh(mesh: Mesh, origin, quat_wxyz) -> Mesh:
    V_local = np.asarray(mesh.V, float)
    p = np.asarray(origin, float).reshape(3)
    R = quat_wxyz_to_rotmat(quat_wxyz)
    V_world = V_local @ R.T + p[None, :]
    return Mesh(V=V_world, F=np.asarray(mesh.F, int).copy())
