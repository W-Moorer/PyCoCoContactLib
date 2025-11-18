import numpy as np
from typing import List, Tuple

from .aabb_bvh import Aabb, AabbWithData, BvhTree


def load_obj(path: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    vs: List[np.ndarray] = []
    fs: List[Tuple[int, int, int]] = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if not line or line.startswith('#'):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v' and len(parts) >= 4:
                    vs.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], float))
                elif parts[0] == 'f' and len(parts) >= 4:
                    idx = []
                    for k in range(1, 4):
                        tok = parts[k]
                        if '/' in tok:
                            tok = tok.split('/')[0]
                        idx.append(int(tok) - 1)
                    fs.append((idx[0], idx[1], idx[2]))
    except Exception as e:
        print(f"Error loading OBJ file {path}: {e}")
    return vs, fs


class RapidMesh:
    def __init__(self, vertices: List[np.ndarray], triangles: List[Tuple[int, int, int]]):
        self.vertices = [np.asarray(v, float) for v in vertices]
        self.triangles = [(int(i), int(j), int(k)) for (i, j, k) in triangles]
        self.tri_areas: List[float] = []
        for tri in self.triangles:
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            area = 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
            self.tri_areas.append(area)
        self.total_area = float(sum(self.tri_areas))
        tri_aabbs: List[AabbWithData] = []
        for ti, tri in enumerate(self.triangles):
            aabb = AabbWithData()
            aabb.merge_vec(self.vertices[tri[0]])
            aabb.merge_vec(self.vertices[tri[1]])
            aabb.merge_vec(self.vertices[tri[2]])
            aabb.m_data = ti
            tri_aabbs.append(aabb)
        self.bvh = BvhTree()
        self.bvh.build(tri_aabbs)
        self.aabb = Aabb()
        for v in self.vertices:
            self.aabb.merge_vec(v)
        extents = self.aabb.m_max - self.aabb.m_min
        self.center_local = 0.5 * (self.aabb.m_min + self.aabb.m_max)
        self.radius = 0.5 * float(np.max(extents))
        total_size = 0.0
        for tri in self.triangles:
            mn = np.array([np.finfo(float).max] * 3)
            mx = -mn
            for k in range(3):
                v = self.vertices[tri[k]]
                mn = np.minimum(mn, v)
                mx = np.maximum(mx, v)
            total_size += float(np.linalg.norm(mx - mn))
        self.tri_size = total_size / max(1, len(self.triangles))

    def get_triangle_nodes(self, tri_index: int, pose: Tuple[np.ndarray, np.ndarray]) -> List[np.ndarray]:
        trans, mat = pose
        tri = self.triangles[tri_index]
        nodes = [mat @ self.vertices[tri[k]] + trans for k in range(3)]
        return nodes

    @classmethod
    def from_obj(cls, path: str) -> 'RapidMesh':
        vs, fs = load_obj(path)
        return cls(vs, fs)


class OptimizedTriMesh:
    def __init__(self, vertices: List[np.ndarray], triangles: List[Tuple[int, int, int]], if_modify_box: bool = True):
        self.m_nodes = [np.asarray(v, float) for v in vertices]
        self.m_triangles = [(int(i), int(j), int(k)) for (i, j, k) in triangles]
        self.m_bvh = BvhTree()
        self.m_triSize = 0.0
        self.m_aabb = Aabb()
        self._compute_tri_size()
        self.gen_bvh(if_modify_box)
        self._compute_aabb()
    
    def _compute_tri_size(self):
        tri_size = np.zeros(3)
        for tri in self.m_triangles:
            aabb = Aabb()
            aabb.invalidate()
            for j in range(3):
                aabb.merge_vec(self.m_nodes[tri[j]])
            tri_size += (aabb.m_max - aabb.m_min)
        if self.m_triangles:
            tri_size /= len(self.m_triangles)
            self.m_triSize = max(tri_size[0], tri_size[1], tri_size[2])
        else:
            self.m_triSize = 0.0
    
    def _compute_aabb(self):
        self.m_aabb.invalidate()
        for node in self.m_nodes:
            self.m_aabb.merge_vec(node)
    
    def gen_bvh(self, if_modify_box: bool) -> None:
        tri_num = len(self.m_triangles)
        aabbs = []
        for tri_id in range(tri_num):
            data = AabbWithData()
            data.invalidate()
            for i in range(3):
                a = self.get_tri_node(tri_id, i)
                data.merge_vec(a)
            if if_modify_box:
                data.modify()
            data.m_data = tri_id
            aabbs.append(data)
        self.m_bvh.build(aabbs)
    
    def get_tri_node(self, tri_id: int, node_id: int) -> np.ndarray:
        return self.m_nodes[self.m_triangles[tri_id][node_id]]
    
    def get_bvh(self) -> BvhTree:
        return self.m_bvh
    
    def get_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.m_aabb.m_min, self.m_aabb.m_max