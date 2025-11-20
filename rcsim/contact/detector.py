import numpy as np
from typing import List, Tuple, Optional

from .mesh import RapidMesh
from .aabb_bvh import Aabb, BvhTree
from .contact_types import ContactManifold, ContactPoint, CollisionResult, ContactType
from .tri_geom import (
    _normalize,
    triTriIntersect,
    triTriIntersectLine,
    fitPlane,
)
from rcsim.io.perf import perf


class MeshPairContactDetector:
    def __init__(self, meshA: RapidMesh, meshB: RapidMesh) -> None:
        self.meshA = meshA
        self.meshB = meshB
        self._dbg_last_plane = None
        self._dbg_last_p0 = None
        self.max_penetration = 0.0
        self.is_collide = False

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.asarray(v, float)
        L = float(np.linalg.norm(n))
        if L < 1e-12:
            return np.array([0.0, 0.0, 1.0], float)
        return n / L

    def _ray_box_intersect(self, origin: np.ndarray, direction: np.ndarray, box: Aabb) -> Tuple[bool, float, float]:
        t0 = -float('inf')
        t1 = float('inf')
        for i in range(3):
            d = float(direction[i])
            if abs(d) > 1e-16:
                inv = 1.0 / d
                t_near = (float(box.m_min[i]) - float(origin[i])) * inv
                t_far = (float(box.m_max[i]) - float(origin[i])) * inv
                if t_near > t_far:
                    t_near, t_far = t_far, t_near
                t0 = max(t0, t_near)
                t1 = min(t1, t_far)
            else:
                if (float(origin[i]) < float(box.m_min[i])) or (float(origin[i]) > float(box.m_max[i])):
                    return False, 0.0, 0.0
        return (t0 <= t1), t0, t1

    def _find_ray_candidates(self, bvh: BvhTree, origin: np.ndarray, direction: np.ndarray, pose: Tuple[np.ndarray, np.ndarray], out: List[int], node_index: int = 0) -> None:
        box = Aabb()
        bvh.get_node_bound(node_index, box)
        box.update(pose)
        hit, _, _ = self._ray_box_intersect(origin, direction, box)
        if not hit:
            return
        if bvh.is_leaf_node(node_index):
            out.append(bvh.get_node_data(node_index))
            return
        self._find_ray_candidates(bvh, origin, direction, pose, out, bvh.get_left_node(node_index))
        self._find_ray_candidates(bvh, origin, direction, pose, out, bvh.get_right_node(node_index))

    def _ray_intersects_triangle(self, P: np.ndarray, D: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        E1 = B - A
        E2 = C - A
        N = np.cross(E1, E2)
        nL = float(np.linalg.norm(N))
        if nL < 1e-16:
            return False, 0.0, np.zeros(3, dtype=float)
        N = N / nL
        denom = float(np.dot(N, D))
        if abs(denom) < 1e-16:
            return False, 0.0, np.zeros(3, dtype=float)
        t = float(np.dot(A - P, N)) / denom
        X = P + t * D
        v0 = B - A
        v1 = C - A
        v2 = X - A
        d00 = float(np.dot(v0, v0))
        d01 = float(np.dot(v0, v1))
        d11 = float(np.dot(v1, v1))
        d20 = float(np.dot(v2, v0))
        d21 = float(np.dot(v2, v1))
        denom2 = d00 * d11 - d01 * d01
        if abs(denom2) < 1e-16:
            return False, 0.0, np.zeros(3, dtype=float)
        v = (d11 * d20 - d01 * d21) / denom2
        w = (d00 * d21 - d01 * d20) / denom2
        u = 1.0 - v - w
        if (u >= -1e-12) and (v >= -1e-12) and (w >= -1e-12):
            return True, t, X
        return False, 0.0, np.zeros(3, dtype=float)

    def _calc_ray_trimesh_intersection(self, origin: np.ndarray, direction: np.ndarray, mesh: RapidMesh, pose: Tuple[np.ndarray, np.ndarray]) -> Tuple[bool, np.ndarray, int]:
        candidates: List[int] = []
        self._find_ray_candidates(mesh.bvh, origin, direction, pose, candidates, 0)
        best_t = float('inf')
        best_pt = None
        best_tri = -1
        tX, RX = pose
        for tri_index in candidates:
            tri = mesh.triangles[tri_index]
            A = RX @ mesh.vertices[tri[0]] + tX
            B = RX @ mesh.vertices[tri[1]] + tX
            C = RX @ mesh.vertices[tri[2]] + tX
            ok, t, X = self._ray_intersects_triangle(origin, direction, A, B, C)
            if ok and t >= 0.0 and t < best_t:
                best_t = t
                best_pt = X
                best_tri = tri_index
        if best_pt is None:
            return False, np.zeros(3, dtype=float), -1
        return True, best_pt, best_tri

    def geneContactInfoTriangle(self, inertPts: List[np.ndarray], ptNum: int, poseA: Tuple[np.ndarray, np.ndarray], poseB: Tuple[np.ndarray, np.ndarray]) -> Optional[ContactPoint]:
        if ptNum <= 0:
            return None
        ctPos = np.zeros(3, dtype=float)
        for i in range(ptNum):
            ctPos += inertPts[i]
        ctPos *= (1.0 / float(ptNum))
        Normal = np.zeros(3, dtype=float)
        ok = fitPlane(inertPts, ptNum, ctPos, Normal)
        if not ok:
            return None
        Normal = self._normalize(Normal)
        hitA, pA_world, triA = self._calc_ray_trimesh_intersection(ctPos, Normal, self.meshA, poseA)
        hitB, pB_world, triB = self._calc_ray_trimesh_intersection(ctPos, Normal, self.meshB, poseB)
        if not (hitA and hitB):
            return None
        pene = pB_world - pA_world
        L = float(np.linalg.norm(pene))
        nvec = np.array([0.0, 0.0, 1.0], dtype=float) if L < 1e-12 else (pene / L)
        tA, RA = poseA
        tB, RB = poseB
        RA_T = RA.T
        RB_T = RB.T
        v0A, v1A, v2A = self._triangle_vertices_local(self.meshA, triA)
        v0B, v1B, v2B = self._triangle_vertices_local(self.meshB, triB)
        pA_local = RA_T @ (pA_world - tA)
        pB_local = RB_T @ (pB_world - tB)
        baryA = self._barycentric(pA_local, v0A, v1A, v2A)
        baryB = self._barycentric(pB_local, v0B, v1B, v2B)
        phi = -L
        self.max_penetration = max(self.max_penetration, L)
        return ContactPoint(
            triA=triA,
            triB=triB,
            baryA=baryA,
            baryB=baryB,
            pA_world=pA_world,
            pB_world=pB_world,
            n=nvec,
            phi=phi,
            area_weight=1.0,
            lifetime=0,
        )

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.asarray(v, float)
        L = float(np.linalg.norm(n))
        if L < 1e-12:
            return np.array([0.0, 0.0, 1.0], float)
        return n / L

    @staticmethod
    def _barycentric(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0
        d00 = float(np.dot(v0v1, v0v1))
        d01 = float(np.dot(v0v1, v0v2))
        d11 = float(np.dot(v0v2, v0v2))
        d20 = float(np.dot(v0p, v0v1))
        d21 = float(np.dot(v0p, v0v2))
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-14:
            return np.array([1.0, 0.0, 0.0], float)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return np.array([u, v, w], float)

    def _triangle_vertices_local(self, mesh: RapidMesh, tri_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i, j, k = mesh.triangles[tri_index]
        return mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]

    def _world_from_bary(self, mesh: RapidMesh, tri_index: int, bary: np.ndarray, pose: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        trans, R = pose
        v0, v1, v2 = self._triangle_vertices_local(mesh, tri_index)
        p_local = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
        return R @ p_local + trans

    def _build_candidates_from_pairs(
        self,
        poseA: Tuple[np.ndarray, np.ndarray],
        poseB: Tuple[np.ndarray, np.ndarray],
        pairs: List[Tuple[int, int]],
        max_pairs_to_process: int = 200,
    ) -> List[ContactPoint]:
        candidates: List[ContactPoint] = []
        tA, RA = poseA
        tB, RB = poseB
        ordered_pairs: List[Tuple[int, int]] = []
        if pairs:
            centers: List[Tuple[int, int, float]] = []
            for ia, ib in pairs:
                triA = self.meshA.triangles[ia]
                triB = self.meshB.triangles[ib]
                cA_local = (self.meshA.vertices[triA[0]] + self.meshA.vertices[triA[1]] + self.meshA.vertices[triA[2]]) / 3.0
                cB_local = (self.meshB.vertices[triB[0]] + self.meshB.vertices[triB[1]] + self.meshB.vertices[triB[2]]) / 3.0
                cA_world = RA @ cA_local + tA
                cB_world = RB @ cB_local + tB
                d = float(np.linalg.norm(cB_world - cA_world))
                centers.append((ia, ib, d))
            centers.sort(key=lambda x: x[2])
            ordered_pairs = [(ia, ib) for ia, ib, _ in centers]
        max_pairs_to_process = min(300, len(ordered_pairs))
        segments: List[Tuple[np.ndarray, np.ndarray, int, int, List[Tuple[np.ndarray, np.ndarray]]]] = []
        for ia, ib in ordered_pairs[:max_pairs_to_process]:
            triA = self.meshA.triangles[ia]
            triB = self.meshB.triangles[ib]
            nodesA = [RA @ self.meshA.vertices[triA[k]] + tA for k in range(3)]
            nodesB = [RB @ self.meshB.vertices[triB[k]] + tB for k in range(3)]
            p0 = np.zeros(3, dtype=float)
            p1 = np.zeros(3, dtype=float)
            if not triTriIntersectLine(nodesA, nodesB, p0, p1):
                continue
            nA = np.cross(nodesA[1] - nodesA[0], nodesA[2] - nodesA[0])
            nB = np.cross(nodesB[1] - nodesB[0], nodesB[2] - nodesB[0])
            if float(np.dot(np.cross(nA, nB), p1 - p0)) < 0.0:
                seg = (p1, p0, ia, ib, [])
            else:
                seg = (p0, p1, ia, ib, [])
            segments.append(seg)
        if not segments:
            return candidates
        tol = max(1e-8, 0.25 * min(float(getattr(self.meshA, 'tri_size', 0.0)), float(getattr(self.meshB, 'tri_size', 0.0))))
        unused = list(range(len(segments)))
        curves: List[List[int]] = []
        while unused:
            idx = unused.pop()
            curve = [idx]
            a, b = segments[idx][0], segments[idx][1]
            extended = True
            while extended:
                extended = False
                for k in list(unused):
                    s0, s1 = segments[k][0], segments[k][1]
                    if np.linalg.norm(s0 - b) <= tol:
                        curve.append(k)
                        b = s1
                        unused.remove(k)
                        extended = True
                        break
                    if np.linalg.norm(s1 - b) <= tol:
                        curve.append(k)
                        b = s0
                        unused.remove(k)
                        extended = True
                        break
                    if np.linalg.norm(s1 - a) <= tol:
                        curve.insert(0, k)
                        a = s0
                        unused.remove(k)
                        extended = True
                        break
                    if np.linalg.norm(s0 - a) <= tol:
                        curve.insert(0, k)
                        a = s1
                        unused.remove(k)
                        extended = True
                        break
            curves.append(curve)
        for curve in curves:
            pts = []
            cp_pairs_all: List[Tuple[np.ndarray, np.ndarray]] = []
            ia0, ib0 = segments[curve[0]][2], segments[curve[0]][3]
            for idc in curve:
                s = segments[idc]
                pts.append(s[0])
                cp_pairs_all.extend(s[4])
            if not pts:
                continue
            center = sum(pts, np.zeros(3, dtype=float)) / float(len(pts))
            P = np.stack(pts)
            C = P - center[None, :]
            H = C.T @ C
            w, V = np.linalg.eigh(H)
            n_plane = V[:, int(np.argmin(w))]
            n_plane = self._normalize(n_plane)
            if cp_pairs_all:
                pA_avg = sum((p1 for (p1, _) in cp_pairs_all), np.zeros(3, dtype=float)) / float(len(cp_pairs_all))
                pB_avg = sum((p2 for (_, p2) in cp_pairs_all), np.zeros(3, dtype=float)) / float(len(cp_pairs_all))
            else:
                pA_avg = center - 0.5 * n_plane
                pB_avg = center + 0.5 * n_plane
            depth = float(np.dot(pB_avg - pA_avg, n_plane))
            phi = -abs(depth)
            v0A, v1A, v2A = self._triangle_vertices_local(self.meshA, ia0)
            v0B, v1B, v2B = self._triangle_vertices_local(self.meshB, ib0)
            RA_T = RA.T
            RB_T = RB.T
            pA_local = RA_T @ (pA_avg - tA)
            pB_local = RB_T @ (pB_avg - tB)
            baryA = self._barycentric(pA_local, v0A, v1A, v2A)
            baryB = self._barycentric(pB_local, v0B, v1B, v2B)
            candidates.append(ContactPoint(
                triA=ia0,
                triB=ib0,
                baryA=baryA,
                baryB=baryB,
                pA_world=pA_avg,
                pB_world=pB_avg,
                n=n_plane,
                phi=phi,
                area_weight=1.0,
                lifetime=0,
            ))
        return candidates

    def _project_prev_manifold(
        self,
        poseA: Tuple[np.ndarray, np.ndarray],
        poseB: Tuple[np.ndarray, np.ndarray],
        prev_manifold: Optional[ContactManifold],
    ) -> List[ContactPoint]:
        return []

    def _reduce_candidates_to_manifold(self, candidates: List[ContactPoint], max_points: int = 4, weights: Optional[List[float]] = None) -> ContactManifold:
        manifold = ContactManifold()
        if not candidates:
            return manifold
        N = len(candidates)
        if weights is not None and len(weights) == N:
            s = float(sum(max(0.0, w) for w in weights))
            if s <= 1e-18:
                for cp in candidates:
                    cp.area_weight = 1.0 / float(N)
                    manifold.points.append(cp)
            else:
                for cp, w in zip(candidates, weights):
                    cp.area_weight = max(0.0, float(w)) / s
                    manifold.points.append(cp)
        else:
            for cp in candidates:
                cp.area_weight = 1.0 / float(N)
                manifold.points.append(cp)
        return manifold

    def narrowPhaseDetectionTriMesh(self, poseA: Tuple[np.ndarray, np.ndarray], poseB: Tuple[np.ndarray, np.ndarray]) -> ContactManifold:
        manifold = ContactManifold()
        pairs: List[Tuple[int, int]] = []
        self.meshA.bvh.find_collision(self.meshB.bvh, poseA, poseB, pairs)
        if not pairs:
            return manifold
        tA, RA = poseA
        tB, RB = poseB
        ordered_pairs: List[Tuple[int, int]] = []
        centers: List[Tuple[int, int, float]] = []
        for ia, ib in pairs:
            triA = self.meshA.triangles[ia]
            triB = self.meshB.triangles[ib]
            cA_local = (self.meshA.vertices[triA[0]] + self.meshA.vertices[triA[1]] + self.meshA.vertices[triA[2]]) / 3.0
            cB_local = (self.meshB.vertices[triB[0]] + self.meshB.vertices[triB[1]] + self.meshB.vertices[triB[2]]) / 3.0
            cA_world = RA @ cA_local + tA
            cB_world = RB @ cB_local + tB
            d = float(np.linalg.norm(cB_world - cA_world))
            centers.append((ia, ib, d))
        centers.sort(key=lambda x: x[2])
        ordered_pairs = [(ia, ib) for ia, ib, _ in centers]
        max_pairs_to_process = min(300, len(ordered_pairs))
        segments: List[Tuple[np.ndarray, np.ndarray, int, int, List[Tuple[np.ndarray, np.ndarray]]]] = []
        for ia, ib in ordered_pairs[:max_pairs_to_process]:
            triA = self.meshA.triangles[ia]
            triB = self.meshB.triangles[ib]
            nodesA = [RA @ self.meshA.vertices[triA[k]] + tA for k in range(3)]
            nodesB = [RB @ self.meshB.vertices[triB[k]] + tB for k in range(3)]
            res = CollisionResult()
            triTriIntersect(nodesA, nodesB, res)
            if res.contactResult == ContactType.SEPARATE:
                continue
            nA = np.cross(nodesA[1] - nodesA[0], nodesA[2] - nodesA[0])
            nB = np.cross(nodesB[1] - nodesB[0], nodesB[2] - nodesB[0])
            p0 = res.isectpt1
            p1 = res.isectpt2
            if float(np.dot(np.cross(nA, nB), p1 - p0)) < 0.0:
                seg = (p1, p0, ia, ib, res.contPtsPairs.copy())
            else:
                seg = (p0, p1, ia, ib, res.contPtsPairs.copy())
            segments.append(seg)
        if not segments:
            return manifold
        tol = max(1e-8, 0.25 * min(float(getattr(self.meshA, 'tri_size', 0.0)), float(getattr(self.meshB, 'tri_size', 0.0))))
        unused = list(range(len(segments)))
        curves: List[List[int]] = []
        while unused:
            idx = unused.pop()
            curve = [idx]
            a, b = segments[idx][0], segments[idx][1]
            extended = True
            while extended:
                extended = False
                for k in list(unused):
                    s0, s1 = segments[k][0], segments[k][1]
                    if np.linalg.norm(s0 - b) <= tol:
                        curve.append(k)
                        b = s1
                        unused.remove(k)
                        extended = True
                        break
                    if np.linalg.norm(s1 - b) <= tol:
                        curve.append(k)
                        b = s0
                        unused.remove(k)
                        extended = True
                        break
                    if np.linalg.norm(s1 - a) <= tol:
                        curve.insert(0, k)
                        a = s0
                        unused.remove(k)
                        extended = True
                        break
                    if np.linalg.norm(s0 - a) <= tol:
                        curve.insert(0, k)
                        a = s1
                        unused.remove(k)
                        extended = True
                        break
            curves.append(curve)
        candidates: List[ContactPoint] = []
        weights: List[float] = []
        for curve in curves:
            pts = []
            cp_pairs_all: List[Tuple[np.ndarray, np.ndarray]] = []
            ia0, ib0 = segments[curve[0]][2], segments[curve[0]][3]
            for idc in curve:
                s = segments[idc]
                pts.append(s[0])
                cp_pairs_all.extend(s[4])
            if not pts:
                continue
            wlen = 0.0
            for idc in curve:
                s = segments[idc]
                wlen += float(np.linalg.norm(s[1] - s[0]))
            center = sum(pts, np.zeros(3, dtype=float)) / float(len(pts))
            P = np.stack(pts)
            C = P - center[None, :]
            H = C.T @ C
            w, V = np.linalg.eigh(H)
            n_plane = V[:, int(np.argmin(w))]
            n_plane = self._normalize(n_plane)
            if not cp_pairs_all:
                continue
            pA_avg = sum((p1 for (p1, _) in cp_pairs_all), np.zeros(3, dtype=float)) / float(len(cp_pairs_all))
            pB_avg = sum((p2 for (_, p2) in cp_pairs_all), np.zeros(3, dtype=float)) / float(len(cp_pairs_all))
            depth = float(np.dot(pB_avg - pA_avg, n_plane))
            phi = -abs(depth)
            tA, RA = poseA
            tB, RB = poseB
            v0A, v1A, v2A = self._triangle_vertices_local(self.meshA, ia0)
            v0B, v1B, v2B = self._triangle_vertices_local(self.meshB, ib0)
            RA_T = RA.T
            RB_T = RB.T
            pA_local = RA_T @ (pA_avg - tA)
            pB_local = RB_T @ (pB_avg - tB)
            baryA = self._barycentric(pA_local, v0A, v1A, v2A)
            baryB = self._barycentric(pB_local, v0B, v1B, v2B)
            cp = ContactPoint(
                triA=ia0,
                triB=ib0,
                baryA=baryA,
                baryB=baryB,
                pA_world=pA_avg,
                pB_world=pB_avg,
                n=n_plane,
                phi=phi,
                area_weight=1.0,
                lifetime=0,
            )
            candidates.append(cp)
            weights.append(wlen)
        return self._reduce_candidates_to_manifold(candidates, max_points=4, weights=weights)

    def query_manifold(
        self,
        poseA: Tuple[np.ndarray, np.ndarray],
        poseB: Tuple[np.ndarray, np.ndarray],
        prev_manifold: Optional[ContactManifold] = None,
        max_points: int = 4,
    ) -> ContactManifold:
        manifold = self.narrowPhaseDetectionTriMesh(poseA, poseB)
        self.is_collide = len(manifold.points) > 0
        return manifold

    def closest(self, poseA: Tuple[np.ndarray, np.ndarray], poseB: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, np.ndarray]:
        manifold = self.query_manifold(poseA, poseB, prev_manifold=None, max_points=4)
        if manifold.is_empty():
            cA = poseA[1] @ self.meshA.center_local + poseA[0]
            cB = poseB[1] @ self.meshB.center_local + poseB[0]
            diff = cA - cB
            dist = float(np.linalg.norm(diff))
            n = (diff / dist) if dist > 0 else np.array([0.0, 0.0, 1.0], float)
            d_sph = dist - (self.meshA.radius + self.meshB.radius)
            return max(0.0, d_sph), n
        best = min(manifold.points, key=lambda cp: cp.phi)
        return best.phi, best.n


class RapidContactDetectionLib:
    def __init__(self, mesh_path: str):
        vs, fs = load_obj(mesh_path)
        self.mesh = RapidMesh(vs, fs)

    @staticmethod
    def _penetration_from_result(res: 'CollisionResult') -> Tuple[float, np.ndarray]:
        n = res.contactNormal.copy()
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            n = np.array([0.0, 0.0, 1.0])
            nn = 1.0
        else:
            n = n / nn
        depths = []
        for p1, p2 in res.contPtsPairs:
            depths.append(float(np.dot(p2 - p1, n)))
        depth = float(np.mean(depths)) if depths else 0.0
        return depth, n

    def _triangle_vertices_local(self, tri_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i, j, k = self.mesh.triangles[tri_index]
        return self.mesh.vertices[i], self.mesh.vertices[j], self.mesh.vertices[k]

    @staticmethod
    def _barycentric(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0
        d00 = float(np.dot(v0v1, v0v1))
        d01 = float(np.dot(v0v1, v0v2))
        d11 = float(np.dot(v0v2, v0v2))
        d20 = float(np.dot(v0p, v0v1))
        d21 = float(np.dot(v0p, v0v2))
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-14:
            return np.array([1.0, 0.0, 0.0], float)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return np.array([u, v, w], float)

    def _world_from_bary(self, tri_index: int, bary: np.ndarray, pose: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        trans, R = pose
        v0, v1, v2 = self._triangle_vertices_local(tri_index)
        p_local = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
        return R @ p_local + trans

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.asarray(v, float)
        L = float(np.linalg.norm(n))
        if L < 1e-12:
            return np.array([0.0, 0.0, 1.0], float)
        return n / L

    def closest_optimized(self, poseA: Tuple[np.ndarray, np.ndarray], poseB: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, np.ndarray]:
        cA = poseA[1] @ self.mesh.center_local + poseA[0]
        cB = poseB[1] @ self.mesh.center_local + poseB[0]
        diff = cA - cB
        dist = float(np.linalg.norm(diff))
        if dist > 0:
            nB = diff / dist
        else:
            nB = np.array([0.0, 0.0, 1.0])
        d_sph = dist - 2.0 * self.mesh.radius
        if d_sph > self.mesh.tri_size * 3.0:
            return max(0.0, d_sph), nB
        pairs: List[Tuple[int, int]] = []
        self.mesh.bvh.find_collision(self.mesh.bvh, poseA, poseB, pairs)
        if not pairs:
            return max(0.0, d_sph), nB
        best_distance = float('inf')
        best_normal = nB.copy()
        max_pairs_to_process = min(100, len(pairs))
        for ia, ib in pairs[:max_pairs_to_process]:
            nodesA = self.mesh.get_triangle_nodes(ia, poseA)
            nodesB = self.mesh.get_triangle_nodes(ib, poseB)
            distance, clA, clB, normal = triangle_distance(nodesA, nodesB)
            if distance < best_distance:
                best_distance = distance
                best_normal = normal
        return best_distance, best_normal

    def closest_high_accuracy(self, poseA: Tuple[np.ndarray, np.ndarray], poseB: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, np.ndarray]:
        pairs: List[Tuple[int, int]] = []
        self.mesh.bvh.find_collision(self.mesh.bvh, poseA, poseB, pairs)
        if not pairs:
            cA = poseA[1] @ self.mesh.center_local + poseA[0]
            cB = poseB[1] @ self.mesh.center_local + poseB[0]
            diff = cA - cB
            dist = float(np.linalg.norm(diff))
            nB = diff / dist if dist > 0 else np.array([0.0, 0.0, 1.0])
            return max(0.0, dist - 2.0 * self.mesh.radius), nB
        max_pairs_to_process = min(200, len(pairs))
        best_pen_depth = 0.0
        best_pen_normal = np.array([0.0, 0.0, 1.0])
        best_sep_dist = float('inf')
        best_sep_normal = np.array([0.0, 0.0, 1.0])
        for ia, ib in pairs[:max_pairs_to_process]:
            nodesA = self.mesh.get_triangle_nodes(ia, poseA)
            nodesB = self.mesh.get_triangle_nodes(ib, poseB)
            res = CollisionResult()
            triTriIntersect(nodesA, nodesB, res)
            if res.contactResult != ContactType.SEPARATE:
                depth, n = self._penetration_from_result(res)
                if depth > best_pen_depth:
                    best_pen_depth = depth
                    best_pen_normal = n
                continue
            res2 = CollisionResult()
            TriangleDistance2(nodesA, nodesB, res2)
            if res2.closetPtsPairs:
                clA, clB = res2.closetPtsPairs[0]
                d = float(np.linalg.norm(clA - clB))
                if d < best_sep_dist:
                    best_sep_dist = d
                    best_sep_normal = _normalize(clB - clA) if d > 0 else best_sep_normal
        if best_pen_depth > 0.0:
            return -best_pen_depth, best_pen_normal
        return best_sep_dist, best_sep_normal
