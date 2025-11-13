from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from .mesh import Mesh
from ..utils.transforms import quat_wxyz_to_rotmat


@dataclass
class BVHNode:
    lo: np.ndarray
    hi: np.ndarray
    left: int
    right: int
    tri_idx: int


class BVH:
    def __init__(self, nodes: List[BVHNode], root: int):
        self.nodes = nodes
        self.root = root


def triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    L = np.linalg.norm(n) + 1e-18
    return n / L


def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def aabb_of_triangle(a, b, c):
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    return lo, hi


def aabb_overlap(lo1, hi1, lo2, hi2, skin=0.0):
    return np.all(hi1 + skin >= lo2) and np.all(hi2 + skin >= lo1)


def build_bvh(mesh: Mesh) -> Tuple[List[BVHNode], int]:
    boxes = []
    cents = []
    for i in range(mesh.F.shape[0]):
        a, b, c = mesh.V[mesh.F[i, 0]], mesh.V[mesh.F[i, 1]], mesh.V[mesh.F[i, 2]]
        lo, hi = aabb_of_triangle(a, b, c)
        boxes.append((lo, hi))
        cents.append((lo + hi) * 0.5)
    nodes: List[BVHNode] = []
    ids = list(range(mesh.F.shape[0]))

    def make(ids):
        los = np.array([boxes[i][0] for i in ids])
        his = np.array([boxes[i][1] for i in ids])
        lo = np.min(los, axis=0)
        hi = np.max(his, axis=0)
        if len(ids) <= 2:
            idx = len(nodes)
            nodes.append(BVHNode(lo, hi, -1, -1, ids[0]))
            if len(ids) == 2:
                idx2 = len(nodes)
                nodes.append(BVHNode(*boxes[ids[1]], -1, -1, ids[1]))
                parent = BVHNode(np.minimum(lo, boxes[ids[1]][0]), np.maximum(hi, boxes[ids[1]][1]), idx, idx2, -1)
                nodes.append(parent)
                return len(nodes) - 1
            return idx
        ext = hi - lo
        axis = int(np.argmax(ext))
        ss = sorted(ids, key=lambda i: cents[i][axis])
        mid = len(ss) // 2
        L = make(ss[:mid])
        R = make(ss[mid:])
        parent = BVHNode(np.minimum(nodes[L].lo, nodes[R].lo), np.maximum(nodes[L].hi, nodes[R].hi), L, R, -1)
        idx = len(nodes)
        nodes.append(parent)
        return idx

    root = make(ids)
    return nodes, root


def traverse_bvhs(bvhA: List[BVHNode], meshA: Mesh, rootA: int,
                  bvhB: List[BVHNode], meshB: Mesh, rootB: int, skin: float = 0.0):
    out = []
    stack = [(rootA, rootB)]
    while stack:
        ia, ib = stack.pop()
        na, nb = bvhA[ia], bvhB[ib]
        if not aabb_overlap(na.lo, na.hi, nb.lo, nb.hi, skin):
            continue
        if na.tri_idx >= 0 and nb.tri_idx >= 0:
            out.append((na.tri_idx, nb.tri_idx))
            continue
        if na.tri_idx < 0 and nb.tri_idx < 0:
            stack += [(na.left, nb.left), (na.left, nb.right), (na.right, nb.left), (na.right, nb.right)]
        elif na.tri_idx < 0:
            stack += [(na.left, ib), (na.right, ib)]
        else:
            stack += [(ia, nb.left), (ia, nb.right)]
    return out


class BVHBuilder:
    def build(self, mesh: Mesh) -> BVH:
        nodes, root = build_bvh(mesh)
        return BVH(nodes, root)


class MedianSplitBVHBuilder(BVHBuilder):
    pass


def _update_leaf(node: BVHNode, mesh: Mesh, R: np.ndarray, p: np.ndarray):
    j = node.tri_idx
    i0, i1, i2 = mesh.F[j, 0], mesh.F[j, 1], mesh.F[j, 2]
    a = mesh.V[i0] @ R.T + p
    b = mesh.V[i1] @ R.T + p
    c = mesh.V[i2] @ R.T + p
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    node.lo = lo
    node.hi = hi


def update_bvh_aabb_with_pose(nodes: List[BVHNode], root: int, mesh: Mesh, origin, quat_wxyz):
    R = quat_wxyz_to_rotmat(quat_wxyz)
    p = np.asarray(origin, dtype=float).reshape(3)
    for n in nodes:
        if n.tri_idx >= 0:
            _update_leaf(n, mesh, R, p)
    def post(i: int):
        n = nodes[i]
        if n.tri_idx >= 0:
            return n.lo, n.hi
        loL, hiL = post(n.left)
        loR, hiR = post(n.right)
        n.lo = np.minimum(loL, loR)
        n.hi = np.maximum(hiL, hiR)
        return n.lo, n.hi
    post(root)


def find_candidates_with_pose(meshA: Mesh, bvhA_nodes: List[BVHNode], bvhA_root: int, originA, quatA,
                              meshB: Mesh, bvhB_nodes: List[BVHNode], bvhB_root: int, originB, quatB,
                              skin: float = 0.0) -> List[Tuple[int, int]]:
    update_bvh_aabb_with_pose(bvhA_nodes, bvhA_root, meshA, originA, quatA)
    update_bvh_aabb_with_pose(bvhB_nodes, bvhB_root, meshB, originB, quatB)
    return traverse_bvhs(bvhA_nodes, meshA, bvhA_root, bvhB_nodes, meshB, bvhB_root, skin=skin)
