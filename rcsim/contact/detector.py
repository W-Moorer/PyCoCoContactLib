import numpy as np
from typing import List, Tuple, Optional

from .mesh import RapidMesh
from .contact_types import ContactManifold, ContactPoint, CollisionResult, ContactType
from .tri_geom import (
    _normalize,
    triangle_distance,
    TriangleDistance2,
    triTriIntersect,
)


class MeshPairContactDetector:
    def __init__(self, meshA: RapidMesh, meshB: RapidMesh) -> None:
        self.meshA = meshA
        self.meshB = meshB

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
        RA_T = RA.T
        RB_T = RB.T
        for ia, ib in pairs[:max_pairs_to_process]:
            nodesA = self.meshA.get_triangle_nodes(ia, poseA)
            nodesB = self.meshB.get_triangle_nodes(ib, poseB)
            res = CollisionResult()
            triTriIntersect(nodesA, nodesB, res)
            if res.contactResult != ContactType.separate and res.contPtsPairs:
                for pA_world, pB_world in res.contPtsPairs:
                    v = pB_world - pA_world
                    dist = float(np.linalg.norm(v))
                    n = self._normalize(v)
                    phi = -dist
                    pA_local = RA_T @ (pA_world - tA)
                    pB_local = RB_T @ (pB_world - tB)
                    v0A, v1A, v2A = self._triangle_vertices_local(self.meshA, ia)
                    v0B, v1B, v2B = self._triangle_vertices_local(self.meshB, ib)
                    baryA = self._barycentric(pA_local, v0A, v1A, v2A)
                    baryB = self._barycentric(pB_local, v0B, v1B, v2B)
                    cp = ContactPoint(
                        triA=ia,
                        triB=ib,
                        baryA=baryA,
                        baryB=baryB,
                        pA_world=np.asarray(pA_world, float),
                        pB_world=np.asarray(pB_world, float),
                        n=n,
                        phi=phi,
                        area_weight=1.0,
                        lifetime=0,
                    )
                    candidates.append(cp)
                continue
            res2 = CollisionResult()
            TriangleDistance2(nodesA, nodesB, res2)
            if res2.closetPtsPairs:
                pA_world, pB_world = res2.closetPtsPairs[0]
                v = pA_world - pB_world
                dist = float(np.linalg.norm(v))
                n = (v / dist) if dist > 1e-12 else np.array([0.0, 0.0, 1.0], float)
                phi = dist
                pA_local = RA_T @ (pA_world - tA)
                pB_local = RB_T @ (pB_world - tB)
                v0A, v1A, v2A = self._triangle_vertices_local(self.meshA, ia)
                v0B, v1B, v2B = self._triangle_vertices_local(self.meshB, ib)
                baryA = self._barycentric(pA_local, v0A, v1A, v2A)
                baryB = self._barycentric(pB_local, v0B, v1B, v2B)
                cp = ContactPoint(
                    triA=ia,
                    triB=ib,
                    baryA=baryA,
                    baryB=baryB,
                    pA_world=np.asarray(pA_world, float),
                    pB_world=np.asarray(pB_world, float),
                    n=n,
                    phi=phi,
                    area_weight=1.0,
                    lifetime=0,
                )
                candidates.append(cp)
        return candidates

    def _project_prev_manifold(
        self,
        poseA: Tuple[np.ndarray, np.ndarray],
        poseB: Tuple[np.ndarray, np.ndarray],
        prev_manifold: Optional[ContactManifold],
    ) -> List[ContactPoint]:
        if prev_manifold is None or prev_manifold.is_empty():
            return []
        candidates: List[ContactPoint] = []
        for cp_prev in prev_manifold.points:
            pA_world = self._world_from_bary(self.meshA, cp_prev.triA, cp_prev.baryA, poseA)
            pB_world = self._world_from_bary(self.meshB, cp_prev.triB, cp_prev.baryB, poseB)
            n = self._normalize(pB_world - pA_world)
            dist = float(np.linalg.norm(pB_world - pA_world))
            nodesA = self.meshA.get_triangle_nodes(cp_prev.triA, poseA)
            nodesB = self.meshB.get_triangle_nodes(cp_prev.triB, poseB)
            res = CollisionResult()
            triTriIntersect(nodesA, nodesB, res)
            phi = -dist if res.contactResult != ContactType.separate else dist
            candidates.append(ContactPoint(
                triA=cp_prev.triA,
                triB=cp_prev.triB,
                baryA=cp_prev.baryA.copy(),
                baryB=cp_prev.baryB.copy(),
                pA_world=pA_world,
                pB_world=pB_world,
                n=n,
                phi=phi,
                area_weight=cp_prev.area_weight,
                lifetime=cp_prev.lifetime + 1,
            ))
        return candidates

    def _reduce_candidates_to_manifold(self, candidates: List[ContactPoint], max_points: int = 4) -> ContactManifold:
        manifold = ContactManifold()
        if not candidates:
            return manifold
        tri_size = 0.5 * (self.meshA.tri_size + self.meshB.tri_size)
        tangent_thresh = 0.25 * tri_size
        candidates_sorted = sorted(candidates, key=lambda cp: cp.phi)
        selected: List[ContactPoint] = []
        for cp in candidates_sorted:
            if not selected:
                selected.append(cp)
                if len(selected) >= max_points:
                    break
                continue
            too_close = False
            for other in selected:
                dp = cp.pA_world - other.pA_world
                t = dp - float(np.dot(dp, other.n)) * other.n
                if float(np.linalg.norm(t)) < tangent_thresh:
                    too_close = True
                    break
            if not too_close:
                selected.append(cp)
                if len(selected) >= max_points:
                    break
        if not selected:
            return manifold
        N = len(selected)
        for cp in selected:
            cp.area_weight = 1.0 / float(N)
            manifold.points.append(cp)
        return manifold

    def query_manifold(
        self,
        poseA: Tuple[np.ndarray, np.ndarray],
        poseB: Tuple[np.ndarray, np.ndarray],
        prev_manifold: Optional[ContactManifold] = None,
        max_points: int = 4,
    ) -> ContactManifold:
        pairs: List[Tuple[int, int]] = []
        self.meshA.bvh.find_collision(self.meshB.bvh, poseA, poseB, pairs)
        projected = self._project_prev_manifold(poseA, poseB, prev_manifold)
        if not pairs and not projected:
            return ContactManifold()
        candidates = projected + self._build_candidates_from_pairs(poseA, poseB, pairs)
        manifold = self._reduce_candidates_to_manifold(candidates, max_points=max_points)
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
            if res.contactResult != ContactType.separate:
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
                    best_sep_normal = _normalize(clA - clB) if d > 0 else best_sep_normal
        if best_pen_depth > 0.0:
            return -best_pen_depth, best_pen_normal
        return best_sep_dist, best_sep_normal