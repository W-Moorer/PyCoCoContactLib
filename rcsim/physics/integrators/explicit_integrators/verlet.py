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

    F0 = world.compute_forces()
    inv_m = np.zeros(n, dtype=float)
    inv_m[(~static_mask) & (masses > 0.0)] = 1.0 / masses[(~static_mask) & (masses > 0.0)]
    a0 = F0 * inv_m[:, None]

    v_half = v0.copy()
    v_half[~static_mask] = v0[~static_mask] + 0.5 * dt * a0[~static_mask]
    x1 = x0.copy()
    x1[~static_mask] = x0[~static_mask] + dt * v_half[~static_mask]

    for i, rb in enumerate(world.bodies):
        b = rb.body
        if b.is_static:
            continue
        b.velocity = v_half[i]
        b.position = x1[i]

    F1 = world.compute_forces()
    a1 = F1 * inv_m[:, None]

    for i, rb in enumerate(world.bodies):
        b = rb.body
        if b.is_static:
            continue
        v1 = b.velocity + 0.5 * dt * a1[i]
        b.velocity = v1


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
            perf.set_meta(algorithm="verlet", t_end=t_end, dt_frame=dt_frame, dt_sub=dt_sub, bodies=len(world.bodies))
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