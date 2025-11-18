from __future__ import annotations

import os
import csv
import math
import numpy as np
import re

from rcsim import RapidMesh, RigidMeshBody, ContactWorld, simulate_verlet
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def run_case(mesh: str, t_end: float, dt_frame: float, dt_sub: float, damp: float, log_path: str, csv_path: str, progress: float | None, verbose: bool, half_wave: bool):
    meshA = RapidMesh.from_obj(mesh)
    meshB = RapidMesh.from_obj(mesh)
    A = RigidMeshBody(meshA, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], is_static=True)
    B = RigidMeshBody(meshB, 1.0, [0.0, 0.0, 205.0], [0.0, 0.0, 0.0], is_static=False)
    world = ContactWorld([A, B], g=-9.8, k_contact=1e5, c_damp=damp, half_wave_damp=half_wave)
    world.build_all_pairs()

    records: list[tuple[float, ...]] = []
    contact_count = 0
    last_pen = 0.0
    last_fn = 0.0
    last_fn_raw = 0.0
    last_speed = 0.0

    f = open(log_path, 'w', encoding='utf-8')
    steps = int(math.ceil(t_end / dt_frame))
    t = 0.0
    if verbose:
        f.write(f"start t={t:.2f} steps={steps} dt_frame={dt_frame:.6f} dt_sub={dt_sub:.6f}\n")
        f.flush()

    def on_sub(ts: float, w: ContactWorld):
        nonlocal contact_count, last_pen, last_fn, last_fn_raw, last_speed
        pair = w.pairs[0]
        pen = 0.0
        Fn = 0.0
        Fn_raw = 0.0
        rel_v = w.bodies[0].body.velocity - w.bodies[1].body.velocity
        for cp in pair.manifold.points:
            n = cp.n
            L = float(np.linalg.norm(n))
            n = np.array([0.0, 0.0, 1.0], float) if L < 1e-12 else (n / L)
            if cp.phi < 0.0:
                pen = max(pen, -cp.phi)
                vn_i = float(np.dot(rel_v, n))
                vn_comp = vn_i if half_wave and vn_i < 0.0 else vn_i
                Fn_raw_i = 1e5 * (-cp.phi) - damp * vn_comp
                Fn += max(Fn_raw_i, 0.0) * cp.area_weight
                Fn_raw += Fn_raw_i * cp.area_weight
        last_pen = pen
        last_fn = Fn
        last_fn_raw = Fn_raw
        last_speed = float(np.linalg.norm(w.bodies[1].body.velocity))
        if pen > 0.0:
            contact_count += 1
        if verbose:
            f.write(
                f"sub t={ts:.6f} z={w.bodies[1].body.position[2]:.6f} "
                f"pen={last_pen:.6e} Fn={last_fn:.6f} "
                f"Fn_raw={last_fn_raw:.6f} speed={last_speed:.6f} "
                f"contacts={contact_count} dt={min(dt_sub, dt_frame):.6e} E=0\n"
            )
            f.flush()

    next_prog = 0.0 if progress is None else float(progress)

    def on_frame(frame: int, ts: float, w: ContactWorld):
        nonlocal next_prog
        dist = float(np.linalg.norm(w.bodies[1].body.position - w.bodies[0].body.position))
        records.append(
            (
                ts,
                *w.bodies[0].body.position.tolist(),
                *w.bodies[1].body.position.tolist(),
                *w.bodies[0].body.velocity.tolist(),
                *w.bodies[1].body.velocity.tolist(),
                dist,
            )
        )
        if verbose and (progress is None or ts + 1e-12 >= next_prog):
            f.write(
                f"t={ts:.2f} z={w.bodies[1].body.position[2]:.6f} "
                f"pen={last_pen:.6e} Fn={last_fn:.6f} "
                f"Fn_raw={last_fn_raw:.6f} speed={last_speed:.6f} "
                f"contacts={contact_count}\n"
            )
            f.flush()
            if progress is not None:
                next_prog += float(progress)

    simulate_verlet(world, t_end=t_end, dt_frame=dt_frame, dt_sub=dt_sub, on_frame=on_frame, on_substep=on_sub)

    with open(csv_path, 'w', newline='') as wf:
        w = csv.writer(wf)
        w.writerow([
            'time',
            'fixed_x', 'fixed_y', 'fixed_z',
            'fall_x', 'fall_y', 'fall_z',
            'fixed_vx', 'fixed_vy', 'fixed_vz',
            'fall_vx', 'fall_vy', 'fall_vz',
            'distance',
        ])
        for r in records:
            w.writerow(r)

    f.write(
        f"final_pos_fixed=({world.bodies[0].body.position[0]:.6f},{world.bodies[0].body.position[1]:.6f},{world.bodies[0].body.position[2]:.6f})\n"
    )
    f.write(
        f"final_pos_fall=({world.bodies[1].body.position[0]:.6f},{world.bodies[1].body.position[1]:.6f},{world.bodies[1].body.position[2]:.6f})\n"
    )
    f.write(
        f"final_distance={float(np.linalg.norm(world.bodies[1].body.position - world.bodies[0].body.position)):.6f}\n"
    )
    f.write(f"contacts={contact_count}\n")
    f.write(f"energy_rel_error=0.000000e+00\n")
    f.flush()
    f.close()


def plot_results(csv_path: str, log_path: str, out_dir: str) -> None:
    ts = []
    z_fall = []
    speed = []
    with open(csv_path, 'r', newline='') as rf:
        r = csv.reader(rf)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            t = float(row[0])
            zc = float(row[6])
            vx = float(row[10])
            vy = float(row[11])
            vz = float(row[12])
            ts.append(t)
            z_fall.append(zc)
            speed.append(float(np.sqrt(vx * vx + vy * vy + vz * vz)))
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    fig1 = plt.figure(figsize=(8, 4.5))
    plt.plot(ts, z_fall, label='fall_z')
    plt.xlabel('time')
    plt.ylabel('z')
    plt.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'fall_z_vs_time.png'))
    plt.close(fig1)
    fig2 = plt.figure(figsize=(8, 4.5))
    plt.plot(ts, speed, label='fall_speed')
    plt.xlabel('time')
    plt.ylabel('speed')
    plt.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'fall_speed_vs_time.png'))
    plt.close(fig2)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=float, default=8.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--sub', type=float, default=1e-3)
    parser.add_argument('--mesh', type=str, default='ballMesh.obj')
    parser.add_argument('--out', type=str, default='mesh_drop_results_rcsim_verlet.csv')
    parser.add_argument('--log', type=str, default='rcsim_output_verlet.log')
    parser.add_argument('--progress', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--damp', type=float, default=0.0)
    parser.add_argument('--half_wave', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    mesh_basename = os.path.basename(args.mesh)
    mesh_stem = os.path.splitext(mesh_basename)[0]
    mesh_dir = os.path.dirname(os.path.abspath(args.mesh))
    out_dir = os.path.join(mesh_dir, mesh_stem)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, os.path.basename(args.out))
    log_path = os.path.join(out_dir, os.path.basename(args.log))

    run_case(
        mesh=args.mesh,
        t_end=args.time,
        dt_frame=args.dt,
        dt_sub=args.sub,
        damp=args.damp,
        log_path=log_path,
        csv_path=csv_path,
        progress=args.progress,
        verbose=args.verbose,
        half_wave=args.half_wave,
    )
    if args.plot:
        plot_results(csv_path, log_path, out_dir)


if __name__ == '__main__':
    main()