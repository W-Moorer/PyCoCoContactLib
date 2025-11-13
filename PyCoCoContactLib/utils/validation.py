def ensure_mesh(mesh):
    if not hasattr(mesh, "V") or not hasattr(mesh, "F"):
        raise TypeError("mesh must have V and F")

