from ..io.obj_loader import load_obj
from .contact_solver import ContactSolver


def run_obj_demo(pathA: str, pathB: str):
    meshA = load_obj(pathA)
    meshB = load_obj(pathB)
    solver = ContactSolver()
    return solver.compute(meshA, meshB)

