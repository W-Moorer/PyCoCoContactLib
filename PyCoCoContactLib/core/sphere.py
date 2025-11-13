import numpy as np
import math
from .mesh import Mesh


def make_icosphere(R=0.5, center=(0, 0, 0), subdivisions=3):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], float)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], int)
    def normalize_rows(V):
        L = np.linalg.norm(V, axis=1).reshape(-1, 1) + 1e-18
        return V / L
    V = normalize_rows(verts)
    F = faces.copy()
    def midpoint(i, j, cache, V):
        key = (min(i, j), max(i, j))
        if key in cache:
            return cache[key], V
        p = (V[i] + V[j]) * 0.5
        V = np.vstack([V, p])
        idx = V.shape[0] - 1
        cache[key] = idx
        return idx, V
    for _ in range(subdivisions):
        cache = {}
        newF = []
        for (i, j, k) in F:
            a, V = midpoint(i, j, cache, V)
            b, V = midpoint(j, k, cache, V)
            c, V = midpoint(k, i, cache, V)
            newF += [[i, a, c], [a, j, b], [c, b, k], [a, b, c]]
        V = normalize_rows(V)
        F = np.array(newF, int)
    V = R * V + np.asarray(center, float)
    return Mesh(V=V, F=F)

