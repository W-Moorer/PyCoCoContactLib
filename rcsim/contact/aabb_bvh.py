import numpy as np
from typing import List, Tuple


class Aabb:
    def __init__(self, m_min: np.ndarray | None = None, m_max: np.ndarray | None = None):
        if m_min is None or m_max is None:
            self.invalidate()
        else:
            self.m_min = np.asarray(m_min, float)
            self.m_max = np.asarray(m_max, float)
        self.m_min0 = self.m_min.copy()
        self.m_max0 = self.m_max.copy()
        self.m_is_first_update = True

    def merge(self, box: 'Aabb') -> None:
        self.m_min = np.minimum(self.m_min, box.m_min)
        self.m_max = np.maximum(self.m_max, box.m_max)

    def merge_vec(self, data: np.ndarray) -> None:
        v = np.asarray(data, float)
        self.m_min = np.minimum(self.m_min, v)
        self.m_max = np.maximum(self.m_max, v)

    def invalidate(self) -> None:
        big = np.array([3.402823466e+38] * 3, dtype=float)
        self.m_min = big.copy()
        self.m_max = -big.copy()

    def modify(self, margin: float = 1e-6) -> None:
        if margin > 0:
            self.m_min -= margin
            self.m_max += margin

    def update(self, pose: Tuple[np.ndarray, np.ndarray]) -> None:
        if self.m_is_first_update:
            self.m_min0 = self.m_min.copy()
            self.m_max0 = self.m_max.copy()
            self.m_is_first_update = False
        trans, mat = pose
        trans = np.asarray(trans, float)
        mat = np.asarray(mat, float)
        x = [self.m_min0[0], self.m_max0[0]]
        y = [self.m_min0[1], self.m_max0[1]]
        z = [self.m_min0[2], self.m_max0[2]]
        self.invalidate()
        for xi in x:
            for yi in y:
                for zi in z:
                    pt = np.array([xi, yi, zi], float)
                    pt_u = mat @ pt + trans
                    self.m_min = np.minimum(self.m_min, pt_u)
                    self.m_max = np.maximum(self.m_max, pt_u)

    def intersect(self, other: 'Aabb') -> bool:
        if (self.m_min[0] > other.m_max[0]) or (self.m_max[0] < other.m_min[0]):
            return False
        if (self.m_min[1] > other.m_max[1]) or (self.m_max[1] < other.m_min[1]):
            return False
        if (self.m_min[2] > other.m_max[2]) or (self.m_max[2] < other.m_min[2]):
            return False
        return True

    def distance_vec(self, other: 'Aabb') -> np.ndarray:
        d = np.zeros(3, float)
        if self.m_min[0] > other.m_max[0]:
            d[0] = self.m_min[0] - other.m_max[0]
        elif self.m_max[0] < other.m_min[0]:
            d[0] = -(other.m_min[0] - self.m_max[0])
        if self.m_min[1] > other.m_max[1]:
            d[1] = self.m_min[1] - other.m_max[1]
        elif self.m_max[1] < other.m_min[1]:
            d[1] = -(other.m_min[1] - self.m_max[1])
        if self.m_min[2] > other.m_max[2]:
            d[2] = self.m_min[2] - other.m_max[2]
        elif self.m_max[2] < other.m_min[2]:
            d[2] = -(other.m_min[2] - self.m_max[2])
        return d


class AabbWithData(Aabb):
    def __init__(self):
        super().__init__()
        self.m_data: int | None = None


class BvhTree:
    class Node:
        def __init__(self) -> None:
            self.m_bound = Aabb()
            self.m_index = 0
        
        def is_leaf(self) -> bool: 
            return self.m_index >= 0
        
        def set_data_index(self, index: int) -> None: 
            self.m_index = int(index)
        
        def get_data_index(self) -> int: 
            return int(self.m_index)
        
        def set_escape_index(self, index: int) -> None: 
            self.m_index = -int(index)
        
        def get_escape_index(self) -> int: 
            return -int(self.m_index)

    def __init__(self) -> None:
        self._node_num = 0
        self._nodes: List[BvhTree.Node] = []

    def get_node_count(self) -> int: 
        return int(self._node_num)
    
    def is_leaf_node(self, idx: int) -> bool: 
        return self._nodes[idx].is_leaf()
    
    def set_node_bound(self, idx: int, bound: Aabb) -> None: 
        self._nodes[idx].m_bound = bound
    
    def get_node_bound(self, idx: int, out: Aabb) -> None:
        out.m_min = self._nodes[idx].m_bound.m_min.copy()
        out.m_max = self._nodes[idx].m_bound.m_max.copy()
    
    def get_node_data(self, idx: int) -> int: 
        return self._nodes[idx].get_data_index()
    
    def get_left_node(self, idx: int) -> int: 
        return idx + 1
    
    def get_right_node(self, idx: int) -> int:
        return (idx + 2) if self._nodes[idx + 1].is_leaf() else (idx + 1 + self._nodes[idx + 1].get_escape_index())
    
    def get_escape_node_index(self, idx: int) -> int: 
        return self._nodes[idx].get_escape_index()

    def build(self, info: List[AabbWithData]) -> None:
        self._node_num = 0
        ids = list(range(len(info)))
        self._nodes = [BvhTree.Node() for _ in range(max(0, len(info) * 2 - 1))]
        if ids:
            self._build_impl(info, ids, 0, len(ids))

    def _build_impl(self, info: List[AabbWithData], ids: List[int], from_idx: int, to_idx: int) -> None:
        cur_index = self._node_num
        self._node_num += 1
        if (to_idx - from_idx) == 1:
            bound = Aabb(info[ids[from_idx]].m_min, info[ids[from_idx]].m_max)
            self.set_node_bound(cur_index, bound)
            self._nodes[cur_index].set_data_index(ids[from_idx])
            return
        
        split_axis = self._calc_split_axis(info, ids, from_idx, to_idx)
        split_index = self._sort_calc_split_index(info, ids, from_idx, to_idx, split_axis)
        
        node_bound = Aabb()
        node_bound.invalidate()
        for i in range(from_idx, to_idx):
            b = Aabb(info[ids[i]].m_min, info[ids[i]].m_max)
            node_bound.merge(b)
        
        self.set_node_bound(cur_index, node_bound)
        self._build_impl(info, ids, from_idx, split_index)
        self._build_impl(info, ids, split_index, to_idx)
        self._nodes[cur_index].set_escape_index(self._node_num - cur_index)

    @staticmethod
    def _calc_split_axis(info: List[AabbWithData], ids: List[int], from_idx: int, to_idx: int) -> int:
        num = to_idx - from_idx
        means = np.zeros(3, float)
        for i in range(from_idx, to_idx):
            center = 0.5 * (info[ids[i]].m_max + info[ids[i]].m_min)
            means += center
        means *= 1.0 / float(num)
        
        variance = np.zeros(3, float)
        for i in range(from_idx, to_idx):
            center = 0.5 * (info[ids[i]].m_max + info[ids[i]].m_min)
            diff2 = center - means
            variance += diff2 * diff2
        
        variance *= 1.0 / max(1.0, float(num - 1))
        
        if variance[0] >= variance[1] and variance[0] >= variance[2]:
            return 0
        elif variance[1] >= variance[2]:
            return 1
        else:
            return 2

    @staticmethod
    def _sort_calc_split_index(info: List[AabbWithData], ids: List[int], from_idx: int, to_idx: int, split_axis: int) -> int:
        split_index = from_idx
        num = to_idx - from_idx
        means = np.zeros(3, float)
        for i in range(from_idx, to_idx):
            center = 0.5 * (info[ids[i]].m_max + info[ids[i]].m_min)
            means += center
        means *= 1.0 / float(num)
        split_value = means[split_axis]
        
        for i in range(from_idx, to_idx):
            center = 0.5 * (info[ids[i]].m_max + info[ids[i]].m_min)
            if center[split_axis] > split_value:
                ids[i], ids[split_index] = ids[split_index], ids[i]
                split_index += 1
        
        range_balanced = num // 3
        unbalanced = (split_index <= (from_idx + range_balanced)) or (split_index >= (to_idx - 1 - range_balanced))
        if unbalanced:
            split_index = from_idx + (num >> 1)
        
        return split_index

    def find_collision(self, other: 'BvhTree', pose1: Tuple[np.ndarray, np.ndarray], 
                      pose2: Tuple[np.ndarray, np.ndarray], pairs: List[Tuple[int, int]]) -> None:
        if self.get_node_count() == 0 or other.get_node_count() == 0:
            return
        self._find_collision_pair(other, pose1, pose2, pairs, 0, 0)

    def _find_collision_pair(self, b1: 'BvhTree', pose0: Tuple[np.ndarray, np.ndarray], 
                           pose1: Tuple[np.ndarray, np.ndarray], pairs: List[Tuple[int, int]], 
                           n0: int, n1: int) -> None:
        res, _ = self._node_collision(b1, n0, n1, pose0, pose1)
        if not res:
            return
        
        if self.is_leaf_node(n0):
            if b1.is_leaf_node(n1):
                pairs.append((self.get_node_data(n0), b1.get_node_data(n1)))
                return
            else:
                self._find_collision_pair(b1, pose0, pose1, pairs, n0, b1.get_left_node(n1))
                self._find_collision_pair(b1, pose0, pose1, pairs, n0, b1.get_right_node(n1))
                return
        else:
            if b1.is_leaf_node(n1):
                self._find_collision_pair(b1, pose0, pose1, pairs, self.get_left_node(n0), n1)
                self._find_collision_pair(b1, pose0, pose1, pairs, self.get_right_node(n0), n1)
                return
            else:
                self._find_collision_pair(b1, pose0, pose1, pairs, self.get_left_node(n0), b1.get_left_node(n1))
                self._find_collision_pair(b1, pose0, pose1, pairs, self.get_left_node(n0), b1.get_right_node(n1))
                self._find_collision_pair(b1, pose0, pose1, pairs, self.get_right_node(n0), b1.get_left_node(n1))
                self._find_collision_pair(b1, pose0, pose1, pairs, self.get_right_node(n0), b1.get_right_node(n1))
                return

    def _node_collision(self, b2: 'BvhTree', n1: int, n2: int, 
                       pose1: Tuple[np.ndarray, np.ndarray], pose2: Tuple[np.ndarray, np.ndarray]) -> Tuple[bool, np.ndarray]:
        box1 = Aabb()
        self.get_node_bound(n1, box1)
        box1.update(pose1)
        
        box2 = Aabb()
        b2.get_node_bound(n2, box2)
        box2.update(pose2)
        
        if box1.intersect(box2):
            return True, np.array([np.finfo(float).max] * 3, float)
        else:
            return False, box1.distance_vec(box2)