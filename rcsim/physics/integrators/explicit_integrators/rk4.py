import numpy as np
from typing import Callable, Optional

from rcsim.physics.world import ContactWorld
from rcsim.io.perf import perf


def step_sub(world: ContactWorld, dt: float) -> None:
    n = len(world.bodies)
    x0 = np.stack([rb.body.position.copy() for rb in world.bodies], axis=0)
    v0 = np.stack([rb.body.velocity.copy() for rb in world.bodies], axis=0)
    masses = np.array([rb.body.mass for rb in world.bodies], dtype=float).reshape(n)
    static_mask = np.array([rb.body.is_static for rb in world.bodies], dtype=bool).reshape(n)
    inv_m = np.zeros(n, dtype=float)
    inv_m[(~static_mask) & (masses > 0.0)] = 1.0 / masses[(~static_mask) & (masses > 0.0)]

    def set_state(X: np.ndarray, V: np.ndarray) -> None:
        for i, rb in enumerate(world.bodies):
            b = rb.body
            if b.is_static:
                continue
            b.position = X[i].copy()
            b.velocity = V[i].copy()

    def compute_a(X: np.ndarray, V: np.ndarray) -> np.ndarray:
        set_state(X, V)
        F = world.compute_forces()
        a = F * inv_m[:, None]
        return a

    k1x = v0
    k1v = compute_a(x0, v0)

    x2 = x0 + 0.5 * dt * k1x
    v2 = v0 + 0.5 * dt * k1v
    k2x = v2
    k2v = compute_a(x2, v2)

    x3 = x0 + 0.5 * dt * k2x
    v3 = v0 + 0.5 * dt * k2v
    k3x = v3
    k3v = compute_a(x3, v3)

    x4 = x0 + dt * k3x
    v4 = v0 + dt * k3v
    k4x = v4
    k4v = compute_a(x4, v4)

    x1 = x0 + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v1 = v0 + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    for i, rb in enumerate(world.bodies):
        b = rb.body
        if b.is_static:
            continue
        b.position = x1[i]
        b.velocity = v1[i]


def simulate(
    world: ContactWorld,
    t_end: float,
    dt_frame: float,
    dt_sub: float,
    on_frame: Optional[Callable[[int, float, ContactWorld], None]] = None,
    on_substep: Optional[Callable[[float, ContactWorld], None]] = None,
) -> None:
    t = 0.0
    frame = 0
    while t < t_end - 1e-12:
        try:
            perf.set_meta(algorithm="rk4", t_end=t_end, dt_frame=dt_frame, dt_sub=dt_sub, bodies=len(world.bodies))
        except Exception:
            pass
        remaining = dt_frame
        while remaining > 1e-12:
            dt = min(dt_sub, remaining)
            with perf.section("integrator.step_sub"):
                step_sub(world, dt)
            remaining -= dt
            t += dt
            if on_substep is not None:
                with perf.section("callback.on_substep"):
                    on_substep(t, world)
        if on_frame is not None:
            with perf.section("callback.on_frame"):
                on_frame(frame, t, world)
        frame += 1

