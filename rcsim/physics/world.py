import numpy as np
from dataclasses import dataclass, field

from rcsim.contact import MeshPairContactDetector, ContactManifold
from rcsim.io.perf import perf
from .rigid_bodies import RigidMeshBody


@dataclass
class ContactPair:
    i: int
    j: int
    detector: MeshPairContactDetector
    manifold: ContactManifold = field(default_factory=ContactManifold)


class ContactWorld:
    def __init__(self, bodies, g: float = -9.8, k_contact: float = 1e5, c_damp: float = 0.0, half_wave_damp: bool = False):
        self.bodies = list(bodies)
        self.g = float(g)
        self.k_contact = float(k_contact)
        self.c_damp = float(c_damp)
        self.half_wave_damp = bool(half_wave_damp)
        self.pairs: list[ContactPair] = []

    def build_all_pairs(self) -> None:
        self.pairs.clear()
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                det = MeshPairContactDetector(self.bodies[i].mesh, self.bodies[j].mesh)
                self.pairs.append(ContactPair(i=i, j=j, detector=det))

    def compute_forces(self) -> np.ndarray:
        with perf.section("world.compute_forces"):
            n = len(self.bodies)
            F = np.zeros((n, 3), dtype=float)
            for i, rb in enumerate(self.bodies):
                body = rb.body
                if not body.is_static and body.mass > 0.0:
                    F[i, 2] += body.mass * self.g
            for pair in self.pairs:
                i, j = pair.i, pair.j
                A = self.bodies[i]
                B = self.bodies[j]
                bodyA, bodyB = A.body, B.body
                poseA = (bodyA.position, A.R)
                poseB = (bodyB.position, B.R)
                with perf.section("detector.query_manifold"):
                    manifold = pair.detector.query_manifold(poseA, poseB, prev_manifold=pair.manifold, max_points=4)
                pair.manifold = manifold
                rel_v = bodyA.velocity - bodyB.velocity
                for cp in manifold.points:
                    nvec = cp.n
                    L = float(np.linalg.norm(nvec))
                    nvec = np.array([0.0, 0.0, 1.0], dtype=float) if L < 1e-12 else (nvec / L)
                    if cp.phi >= 0.0:
                        continue
                    pen_i = -cp.phi
                    vn_i = float(np.dot(rel_v, nvec))
                    if self.half_wave_damp:
                        vn_comp = vn_i if vn_i < 0.0 else 0.0
                    else:
                        vn_comp = vn_i
                    Fn_raw_i = self.k_contact * pen_i - self.c_damp * vn_comp
                    Fn_i = max(Fn_raw_i, 0.0)
                    f = (Fn_i * cp.area_weight) * nvec
                    if not bodyA.is_static:
                        F[i] += f
                    if not bodyB.is_static:
                        F[j] -= f
            return F