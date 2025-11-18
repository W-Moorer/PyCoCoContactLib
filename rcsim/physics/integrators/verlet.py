import numpy as np
from typing import Callable, Optional

from rcsim.physics.world import ContactWorld


def step_sub(world: ContactWorld, dt: float) -> None:
    n = len(world.bodies)
    x0 = [rb.body.position.copy() for rb in world.bodies]
    v0 = [rb.body.velocity.copy() for rb in world.bodies]
    F0 = world.compute_forces()
    a0 = []
    for i, rb in enumerate(world.bodies):
        body = rb.body
        if body.is_static or body.mass <= 0.0:
            a0.append(np.zeros(3, dtype=float))
        else:
            a0.append(F0[i] / body.mass)
    for i, rb in enumerate(world.bodies):
        body = rb.body
        if body.is_static:
            continue
        v_half = v0[i] + 0.5 * dt * a0[i]
        x1 = x0[i] + dt * v_half
        body.velocity = v_half
        body.position = x1
    F1 = world.compute_forces()
    a1 = []
    for i, rb in enumerate(world.bodies):
        body = rb.body
        if body.is_static or body.mass <= 0.0:
            a1.append(np.zeros(3, dtype=float))
        else:
            a1.append(F1[i] / body.mass)
    for i, rb in enumerate(world.bodies):
        body = rb.body
        if body.is_static:
            continue
        v_half = body.velocity
        v1 = v_half + 0.5 * dt * a1[i]
        body.velocity = v1


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
        remaining = dt_frame
        while remaining > 1e-12:
            dt = min(dt_sub, remaining)
            step_sub(world, dt)
            remaining -= dt
            t += dt
            if on_substep is not None:
                on_substep(t, world)
        if on_frame is not None:
            on_frame(frame, t, world)
        frame += 1