from typing import List, Tuple
import numpy as np
from ..core.mesh import Mesh


def triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    L = np.linalg.norm(n) + 1e-18
    return n / L


def _tri_tri_intersect_segment(a0, a1, a2, b0, b1, b2, eps=1e-12):
    n1 = triangle_normal(a0, a1, a2)
    n2 = triangle_normal(b0, b1, b2)
    c1 = float(np.dot(n1, a0))
    c2 = float(np.dot(n2, b0))
    d = np.cross(n1, n2)
    L2 = float(np.dot(d, d))
    if L2 < 1e-18:
        return False, None, None
    p0 = np.cross(c1 * n2 - c2 * n1, d) / (L2 + 1e-18)
    dirv = d / (np.sqrt(L2) + 1e-18)

    def interval(v0, v1, v2):
        t0 = float(np.dot(v0 - p0, dirv))
        t1 = float(np.dot(v1 - p0, dirv))
        t2 = float(np.dot(v2 - p0, dirv))
        return min(t0, t1, t2), max(t0, t1, t2)

    a0i, a1i = interval(a0, a1, a2)
    b0i, b1i = interval(b0, b1, b2)
    t0 = max(a0i, b0i)
    t1 = min(a1i, b1i)
    if t1 < t0 - eps:
        return False, None, None
    p = p0 + t0 * dirv
    q = p0 + t1 * dirv
    if np.linalg.norm(q - p) < 1e-12:
        return False, None, None
    return True, p, q


def filter_pairs_by_moller(meshA: Mesh, meshB: Mesh, pairs: List[Tuple[int, int]]):
    out = []
    for i, j in pairs:
        a0 = meshA.V[meshA.F[i, 0]]
        a1 = meshA.V[meshA.F[i, 1]]
        a2 = meshA.V[meshA.F[i, 2]]
        b0 = meshB.V[meshB.F[j, 0]]
        b1 = meshB.V[meshB.F[j, 1]]
        b2 = meshB.V[meshB.F[j, 2]]
        hit, _, _ = _tri_tri_intersect_segment(a0, a1, a2, b0, b1, b2)
        if hit:
            out.append((i, j))
    return out


class MollerDetector:
    def check_pairs(self, mesh_a: Mesh, mesh_b: Mesh, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return filter_pairs_by_moller(mesh_a, mesh_b, pairs)
