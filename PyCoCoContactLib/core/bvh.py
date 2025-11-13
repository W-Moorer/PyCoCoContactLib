from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from .mesh import Mesh


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
