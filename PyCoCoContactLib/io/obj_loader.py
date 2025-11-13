import os
import numpy as np
from ..core.mesh import Mesh


def load_obj(path: str) -> Mesh:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                def parse_index(tok):
                    return int(tok.split("/")[0])
                idx = [parse_index(t) for t in parts]
                idx = [i - 1 for i in idx]
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    if not vertices or not faces:
        raise ValueError(f"OBJ '{path}' 中没有有效的顶点或面。")
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=int)
    return Mesh(V=V, F=F)
