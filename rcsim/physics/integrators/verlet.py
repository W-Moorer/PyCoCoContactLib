import numpy as np
from typing import Callable, Optional

from rcsim.physics.world import ContactWorld


def step_sub(world: ContactWorld, dt: float) -> None:
    n = len(world.bodies)
    X0, V0, M, S = world.get_state_arrays()
    F0 = world.compute_forces_array()
    a0 = np.zeros_like(V0)
    mask = (~S) & (M > 0.0)
    a0[mask] = F0[mask] / M[mask][:, None]
    Vh = V0 + 0.5 * dt * a0
    X1 = X0 + dt * Vh
    world.set_state_arrays(X1, Vh)
    F1 = world.compute_forces_array()
    a1 = np.zeros_like(V0)
    a1[mask] = F1[mask] / M[mask][:, None]
    V1 = Vh + 0.5 * dt * a1
    world.set_state_arrays(X1, V1)


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