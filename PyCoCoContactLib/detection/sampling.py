import numpy as np
from ..core.mesh import Mesh
from ..core.bvh import triangle_area, triangle_normal, MedianSplitBVHBuilder
from .active_faces import extract_active_faces, vertex_normals_area_weighted, nearest_triangle_bvh, barycentric_coords
from ..quadrature import DUNAVANT7


def collect_contact_samples(meshA: Mesh,
                            meshB: Mesh,
                            bvhA,
                            rootA,
                            bvhB,
                            rootB,
                            skin: float = 0.0,
                            qrule=None):
    qr = DUNAVANT7 if qrule is None else qrule
    builder = MedianSplitBVHBuilder()
    act, cand, mol = extract_active_faces(meshA, meshB, builder, skin)
    VN_B = vertex_normals_area_weighted(meshB)
    contact_points = []
    deltas = []
    normals = []
    areas = []
    for iA in act:
        a = meshA.V[meshA.F[iA, 0]]
        b = meshA.V[meshA.F[iA, 1]]
        c = meshA.V[meshA.F[iA, 2]]
        ntri = triangle_normal(a, b, c)
        Atri = triangle_area(a, b, c)
        for w, (l1, l2, l3) in qr:
            x = l1 * a + l2 * b + l3 * c
            jB, q, _ = nearest_triangle_bvh(meshB, bvhB, rootB, x)
            if jB < 0:
                continue
            i0, i1, i2 = meshB.F[jB, 0], meshB.F[jB, 1], meshB.F[jB, 2]
            u, vb, wb = barycentric_coords(meshB.V[i0], meshB.V[i1], meshB.V[i2], q)
            nB = u * VN_B[i0] + vb * VN_B[i1] + wb * VN_B[i2]
            nB = nB / (np.linalg.norm(nB) + 1e-18)
            d_signed = float(np.dot(nB, x - q))
            if d_signed >= 0.0:
                continue
            delta = -d_signed
            dA_proj = w * Atri * abs(np.dot(nB, ntri))
            if dA_proj <= 0.0:
                continue
            contact_points.append(q)
            deltas.append(delta)
            normals.append(nB)
            areas.append(dA_proj)
    contact_points = np.asarray(contact_points, float).reshape(-1, 3)
    deltas = np.asarray(deltas, float).reshape(-1)
    normals = np.asarray(normals, float).reshape(-1, 3)
    areas = np.asarray(areas, float).reshape(-1)
    stats = dict(
        samples=contact_points.shape[0],
        pairs_in=len(cand),
        pairs_moller=len(mol),
        active_A=len(act),
    )
    return contact_points, deltas, normals, areas, stats
