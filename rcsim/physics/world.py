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
    def __init__(
        self,
        bodies,
        g: float = -9.8,
        k_contact: float = 1e5,
        c_damp: float = 0.0,
        half_wave_damp: bool = False,
        mu_fric: float = 0.3,
        c_tangent: float = 0.0,
        exponent: float = 1.0,
        gn: float = 0.0,
        d_max: float = 0.0,
        rebound_factor: float = 0.0,
        damp_type: str = "boundary",
        dp_exp: float = 1.0,
        indentation_exp: float = 1.0,
        thres_vel_static: float = 0.0,
        thres_vel_dynamic: float = 0.0,
        mu_static: float = None,
        mu_dynamic: float = None,
    ):
        self.bodies = list(bodies)
        self.g = float(g)
        self.k_contact = float(k_contact)
        self.c_damp = float(c_damp)
        self.half_wave_damp = bool(half_wave_damp)
        self.mu_fric = float(mu_fric)
        self.c_tangent = float(c_tangent)
        self.exponent = float(exponent)
        self.gn = float(gn)
        self.d_max = float(d_max)
        self.rebound_factor = float(rebound_factor)
        self.damp_type = str(damp_type)
        self.dp_exp = float(dp_exp)
        self.indentation_exp = float(indentation_exp)
        self.thres_vel_static = float(thres_vel_static)
        self.thres_vel_dynamic = float(thres_vel_dynamic)
        self.mu_static = float(mu_static) if mu_static is not None else float(mu_fric)
        self.mu_dynamic = float(mu_dynamic) if mu_dynamic is not None else float(mu_fric)
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

            def computeFn(x: float, xdot: float) -> float:
                if x <= 0.0:
                    return 0.0
                fspring = self.k_contact * (x ** self.exponent)
                if self.damp_type == "boundary":
                    fdamp = (self.gn if (x > 0.0 and (self.d_max <= 0.0 or x <= self.d_max)) else 0.0) * xdot
                elif self.damp_type == "indentation":
                    sgn = 0.0 if abs(xdot) < 1e-18 else (1.0 if xdot > 0.0 else -1.0)
                    fdamp = self.gn * sgn * (abs(xdot) ** self.dp_exp) * (x ** self.indentation_exp)
                else:
                    fdamp = self.c_damp * xdot
                force = fspring - fdamp
                if self.rebound_factor > 0.0:
                    force = max(force, self.rebound_factor * fspring)
                return force

            def muFunction(relVt: float) -> float:
                if relVt <= self.thres_vel_static:
                    return self.mu_static
                if self.thres_vel_dynamic > self.thres_vel_static and relVt >= self.thres_vel_dynamic:
                    return self.mu_dynamic
                if self.thres_vel_dynamic > self.thres_vel_static:
                    t = (relVt - self.thres_vel_static) / max(1e-18, (self.thres_vel_dynamic - self.thres_vel_static))
                    return self.mu_static + (self.mu_dynamic - self.mu_static) * t
                return self.mu_dynamic

            def computeFt(relVt: float, fn: float) -> float:
                return muFunction(relVt) * abs(fn)

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

                force_fresh = bool(getattr(self, "_force_fresh_manifold", False))
                near_band = getattr(self, "_step_max_move", None)
                had_contact_prev = False
                if pair.manifold is not None:
                    for _cp in pair.manifold.points:
                        if _cp.phi < 0.0:
                            had_contact_prev = True
                            break
                if near_band is not None:
                    if had_contact_prev:
                        pair.detector._near_band = 0.0
                    else:
                        try:
                            pair.detector._near_band = 0.5 * float(near_band)
                        except Exception:
                            pair.detector._near_band = float(near_band)

                with perf.section("detector.query_manifold"):
                    manifold = pair.detector.query_manifold(poseA, poseB)
                pair.manifold = manifold

                if not force_fresh:
                    max_move = getattr(self, "_step_max_move", None)
                    need_refresh = False
                    if max_move is not None:
                        lim = float(max_move) + 1e-3
                        for cp in manifold.points:
                            if cp.phi < -lim or float(np.linalg.norm(cp.n)) < 1e-8:
                                need_refresh = True
                                break
                    if need_refresh:
                        with perf.section("detector.query_manifold"):
                            manifold = pair.detector.query_manifold(poseA, poseB)
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
                    fn_i = computeFn(pen_i, vn_i)
                    vt = rel_v - vn_i * nvec
                    vt_norm = float(np.linalg.norm(vt))
                    if vt_norm > 1e-12:
                        if self.c_tangent > 0.0:
                            ft_visc = (-self.c_tangent) * vt
                            ft_mag = float(np.linalg.norm(ft_visc))
                            ft_max = computeFt(vt_norm, fn_i)
                            Ft = ft_visc * (ft_max / ft_mag) if ft_mag > ft_max else ft_visc
                        else:
                            Ft = -computeFt(vt_norm, fn_i) * (vt / vt_norm)
                    else:
                        Ft = np.zeros(3, dtype=float)
                    f = (fn_i * nvec + Ft) * cp.area_weight
                    if not bodyA.is_static:
                        F[i] -= f
                    if not bodyB.is_static:
                        F[j] += f
            return F