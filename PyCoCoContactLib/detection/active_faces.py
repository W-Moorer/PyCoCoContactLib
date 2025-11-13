from typing import List, Tuple
import numpy as np
from ..core.mesh import Mesh
from ..core.bvh import BVHBuilder
from .narrow_phase import filter_pairs_by_moller


def vertex_normals_area_weighted(mesh: Mesh):
    N = np.zeros_like(mesh.V, float)
    for (i0, i1, i2) in mesh.F:
        a, b, c = mesh.V[i0], mesh.V[i1], mesh.V[i2]
        n = np.cross(b - a, c - a)
        N[i0] += n
        N[i1] += n
        N[i2] += n
    L = np.linalg.norm(N, axis=1) + 1e-18
    N = N / L[:, None]
    return N


def barycentric_coords(a, b, c, p):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0) + 1e-18
    d11 = np.dot(v1, v1) + 1e-18
    d01 = np.dot(v0, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01 + 1e-18
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def point_aabb_dist2(p, lo, hi):
    d2 = 0.0
    for k in range(3):
        if p[k] < lo[k]:
            d = lo[k] - p[k]
            d2 += d * d
        elif p[k] > hi[k]:
            d = p[k] - hi[k]
            d2 += d * d
    return d2


def nearest_triangle_bvh(mesh: Mesh, bvh, root, p):
    best_j = -1
    best_q = None
    best_d2 = 1e30
    stack = [root]
    while stack:
        i = stack.pop()
        node = bvh[i]
        if point_aabb_dist2(p, node.lo, node.hi) > best_d2:
            continue
        if node.tri_idx >= 0:
            j = node.tri_idx
            i0, i1, i2 = mesh.F[j, 0], mesh.F[j, 1], mesh.F[j, 2]
            q = closest_point_on_triangle(p, mesh.V[i0], mesh.V[i1], mesh.V[i2])
            d2 = float(np.dot(p - q, p - q))
            if d2 < best_d2:
                best_d2 = d2
                best_q = q
                best_j = j
        else:
            stack += [node.left, node.right]
    return best_j, best_q, float(np.sqrt(best_d2 + 1e-30))


def closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3 + 1e-18)
        return a + v * ab
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6 + 1e-18)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-18)
        return b + w * (c - b)
    denom = 1.0 / (va + vb + vc + 1e-18)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    return u * a + v * b + w * c


def extract_active_faces(mesh_a: Mesh, mesh_b: Mesh, bvh_builder: BVHBuilder, skin: float = 0.0) -> Tuple[List[int], list, list]:
    bvh_a = bvh_builder.build(mesh_a)
    bvh_b = bvh_builder.build(mesh_b)
    from .broad_phase import BroadPhaseDetector
    cand = BroadPhaseDetector(bvh_builder).find_candidates(mesh_a, mesh_b, skin)
    mol = filter_pairs_by_moller(mesh_a, mesh_b, cand)
    active = set([i for (i, _) in cand])
    active.update([i for (i, _) in mol])
    VN_B = vertex_normals_area_weighted(mesh_b)
    eps = 1e-12
    candA = set([i for (i, _) in cand])
    for iA in candA:
        a = mesh_a.V[mesh_a.F[iA, 0]]
        b = mesh_a.V[mesh_a.F[iA, 1]]
        c = mesh_a.V[mesh_a.F[iA, 2]]
        centroid = (a + b + c) / 3.0
        for v in (a, b, c, centroid):
            jB, q, _ = nearest_triangle_bvh(mesh_b, bvh_b.nodes, bvh_b.root, v)
            if jB < 0:
                continue
            i0, i1, i2 = mesh_b.F[jB, 0], mesh_b.F[jB, 1], mesh_b.F[jB, 2]
            u, vb, wb = barycentric_coords(mesh_b.V[i0], mesh_b.V[i1], mesh_b.V[i2], q)
            nB = u * VN_B[i0] + vb * VN_B[i1] + wb * VN_B[i2]
            nB = nB / (np.linalg.norm(nB) + 1e-18)
            d_signed = float(np.dot(nB, v - q))
            if d_signed < -eps:
                active.add(iA)
                break
    return list(active), cand, mol
