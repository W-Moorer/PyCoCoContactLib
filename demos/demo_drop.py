from __future__ import annotations

import os
import csv
import math
import time
import numpy as np
import re

from rcsim import RapidMesh, RigidMeshBody, ContactWorld, simulate_verlet, simulate_rk4, simulate_generalized_alpha
from demos.optimized_generalized_alpha import optimized_simulate
from rcsim.io.perf import perf
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def run_case(
    mesh: str,
    t_end: float,
    dt_frame: float,
    dt_sub: float,
    damp: float,
    log_path: str,
    csv_path: str,
    progress: float | None,
    verbose: bool,
    half_wave: bool,
    integrator: str = "verlet",
    ga_rho_inf: float = 0.5,
    ga_tol: float = 1e-8,
    ga_iter: int = 20,
    ga_relax: float = 1.0,
    ga_newton: bool = False,
    ga_eps_x: float = 1e-6,
    ga_eps_v: float = 1e-6,
    ga_cap: float = 1.0e-3,
    ga_min_dt: float = 1.0e-6,
    ga_growth_thr: float = 1.0e-3,
    ga_growth_scale: float = 0.1,
    ga_init_scale: float = 1.0,
    ga_init_max: float = 5.0e-3,
    ga_grace: int = 1,
):
    meshA = RapidMesh.from_obj(mesh)
    meshB = RapidMesh.from_obj(mesh)
    A = RigidMeshBody(meshA, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], is_static=True)
    B = RigidMeshBody(meshB, 1.0, [0.0, 0.0, 205.0], [0.0, 0.0, 0.0], is_static=False)
    world = ContactWorld(
        [A, B],
        g=-9.8,
        k_contact=1e5,
        c_damp=damp,
        half_wave_damp=half_wave,
        damp_type="linear",
        exponent=1.0,
        rebound_factor=0.0,  # 关闭反弹因子，实现纯弹性碰撞
    )
    world.build_all_pairs()

    records: list[tuple[float, ...]] = []
    contact_count = 0
    last_pen = 0.0
    last_fn = 0.0
    last_fn_raw = 0.0
    last_speed = 0.0
    last_ts = 0.0
    E0 = None

    f = open(log_path, 'w', encoding='utf-8')
    steps = int(math.ceil(t_end / dt_frame))
    t = 0.0
    if verbose:
        f.write(f"start t={t:.2f} steps={steps} dt_frame={dt_frame:.6f} dt_sub={dt_sub:.6f}\n")
        f.flush()

    def on_sub(ts: float, w: ContactWorld):
        nonlocal contact_count, last_pen, last_fn, last_fn_raw, last_speed, last_ts, E0
        def calc_energy() -> float:
            E = 0.0
            for rb in w.bodies:
                b = rb.body
                if not b.is_static and b.mass > 0.0:
                    v2 = float(np.dot(b.velocity, b.velocity))
                    E += 0.5 * b.mass * v2
                    E += (-b.mass * w.g) * b.position[2]
            pen_sum = 0.0
            for pair in w.pairs:
                for cp in pair.manifold.points:
                    if cp.phi < 0.0:
                        pen_sum += 0.5 * w.k_contact * ((-cp.phi) ** 2) * cp.area_weight
            E += pen_sum
            return float(E)
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
                vn_comp = vn_i if (not half_wave or vn_i < 0.0) else 0.0
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
            dt_act = ts - last_ts
            E = calc_energy()
            if E0 is None:
                E0 = float(E)
            f.write(
                f"sub t={ts:.6f} z={w.bodies[1].body.position[2]:.6f} "
                f"pen={last_pen:.6e} Fn={last_fn:.6f} "
                f"Fn_raw={last_fn_raw:.6f} speed={last_speed:.6f} "
                f"contacts={contact_count} dt={dt_act:.6e} E={E:.6f}\n"
            )
            f.flush()
        last_ts = ts

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

    integ = integrator.lower()
    if integ == "rk4":
        simulate_rk4(world, t_end=t_end, dt_frame=dt_frame, dt_sub=dt_sub, on_frame=on_frame, on_substep=on_sub)
    elif integ == "genalpha" or integ == "generalized_alpha":
        simulate_generalized_alpha(
            world,
            t_end=t_end,
            dt_frame=dt_frame,
            dt_sub=dt_sub,
            on_frame=on_frame,
            on_substep=on_sub,
            rho_inf=ga_rho_inf,
            tol=ga_tol,
            max_iter=ga_iter,
            relax=ga_relax,
            use_newton=ga_newton,
            diff_eps_x=ga_eps_x,
            diff_eps_v=ga_eps_v,
            contact_dt_cap=ga_cap,
            min_dt=ga_min_dt,
            contact_growth_thr=ga_growth_thr,
            growth_scale=ga_growth_scale,
            init_thr_scale=ga_init_scale,
            init_thr_max=ga_init_max,
            contact_grace_steps=ga_grace,
        )
    elif integ == "optimized_genalpha" or integ == "optimized_generalized_alpha":
        optimized_simulate(
            world,
            t_end=t_end,
            dt_frame=dt_frame,
            dt_sub=dt_sub,
            on_substep=on_sub,
            use_performance_optimization=True,
            contact_dt_cap=ga_cap,
            min_dt=ga_min_dt,
            contact_growth_thr=ga_growth_thr,
            growth_scale=ga_growth_scale,
            init_thr_scale=ga_init_scale,
            init_thr_max=ga_init_max,
            contact_grace_steps=ga_grace,
            rho_inf=ga_rho_inf,  # 关键修复：传递rho_inf参数
            max_iter=ga_iter,    # 关键修复：传递max_iter参数
            tol=ga_tol           # 关键修复：传递tol参数
        )
    else:
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
    try:
        E_end = 0.0
        for rb in world.bodies:
            b = rb.body
            if not b.is_static and b.mass > 0.0:
                v2 = float(np.dot(b.velocity, b.velocity))
                E_end += 0.5 * b.mass * v2
                E_end += (-b.mass * world.g) * b.position[2]
        pen_sum = 0.0
        for pair in world.pairs:
            for cp in pair.manifold.points:
                if cp.phi < 0.0:
                    pen_sum += 0.5 * world.k_contact * ((-cp.phi) ** 2) * cp.area_weight
        E_end += pen_sum
        rel = 0.0 if (E0 is None or abs(E0) < 1e-12) else abs((E_end - E0) / E0)
        f.write(f"energy_rel_error={rel:.6e}\n")
    except Exception:
        f.write(f"energy_rel_error=nan\n")
    f.flush()
    f.close()
    try:
        perf_path = os.path.join(os.path.dirname(csv_path), 'perf_metrics.csv')
        perf.write_csv(perf_path)
    except Exception:
        pass


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


def plot_per_body(csv_path: str, out_dir: str) -> None:
    with open(csv_path, 'r', newline='') as rf:
        r = csv.reader(rf)
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        ts = []
        fix = []
        fall = []
        fixv = []
        fallv = []
        for row in r:
            if not row:
                continue
            ts.append(float(row[idx['time']]))
            fix.append([float(row[idx['fixed_x']]), float(row[idx['fixed_y']]), float(row[idx['fixed_z']])])
            fall.append([float(row[idx['fall_x']]), float(row[idx['fall_y']]), float(row[idx['fall_z']])])
            fixv.append([float(row[idx['fixed_vx']]), float(row[idx['fixed_vy']]), float(row[idx['fixed_vz']])])
            fallv.append([float(row[idx['fall_vx']]), float(row[idx['fall_vy']]), float(row[idx['fall_vz']])])
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    ts_arr=np.array(ts)
    fx=np.array(fix); cx=np.array(fall)
    fvx=np.array(fixv); cvx=np.array(fallv)
    def acc(v):
        if len(ts_arr)>=2:
            return np.vstack([np.gradient(v[:,0], ts_arr), np.gradient(v[:,1], ts_arr), np.gradient(v[:,2], ts_arr)]).T
        else:
            return np.zeros_like(v)
    fax=acc(fvx); cax=acc(cvx)
    def save_axis_series(prefix, arr, ylabel):
        labels=['x','y','z']
        for k in range(3):
            fig=plt.figure(figsize=(8,4.5));
            plt.plot(ts_arr, arr[:,k], label=labels[k])
            plt.xlabel('time');
            plt.ylabel(ylabel)
            plt.legend();
            plt.tight_layout();
            fig.savefig(os.path.join(out_dir, f"{prefix}_{labels[k]}.png"))
            plt.close(fig)
    save_axis_series('fixed_pos', fx, 'pos')
    save_axis_series('fixed_vel', fvx, 'vel')
    save_axis_series('fixed_acc', fax, 'acc')
    save_axis_series('fall_pos', cx, 'pos')
    save_axis_series('fall_vel', cvx, 'vel')
    save_axis_series('fall_acc', cax, 'acc')

def _load_obj_simple(mesh_path: str):
    vs = []
    fs = []
    with open(mesh_path, 'r', encoding='utf-8', errors='ignore') as rf:
        for line in rf:
            l = line.strip()
            if not l:
                continue
            if l.startswith('v '):
                a = l.split()
                if len(a) >= 4:
                    vs.append([float(a[1]), float(a[2]), float(a[3])])
            elif l.startswith('f '):
                a = l.split()
                idxs = []
                for tok in a[1:]:
                    p = tok.split('/')
                    try:
                        idxs.append(max(0, int(p[0]) - 1))
                    except Exception:
                        pass
                if len(idxs) >= 3:
                    fs.append([idxs[0], idxs[1], idxs[2]])
                    for k in range(3, len(idxs)):
                        fs.append([idxs[0], idxs[k - 1], idxs[k]])
    return np.asarray(vs, dtype=float), np.asarray(fs, dtype=int)

def save_video(csv_path: str, mesh_path: str, out_mp4: str, *, fps: int = 30, stride: int = 1, out_frames_dir: str | None = None) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception:
        return
    try:
        import imageio_ffmpeg
        matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    with open(csv_path, 'r', newline='') as rf:
        r = csv.reader(rf)
        h = next(r)
        idx = {n:i for i,n in enumerate(h)}
        times=[]; fix=[]; fall=[]
        for j,row in enumerate(r):
            if not row: continue
            if (j % max(1,int(stride)))!=0: continue
            times.append(float(row[idx['time']]))
            fix.append([float(row[idx['fixed_x']]), float(row[idx['fixed_y']]), float(row[idx['fixed_z']])])
            fall.append([float(row[idx['fall_x']]), float(row[idx['fall_y']]), float(row[idx['fall_z']])])
    V,F = _load_obj_simple(mesh_path)
    V = np.asarray(V)
    F = np.asarray(F)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    fix_arr = np.asarray(fix)
    fall_arr = np.asarray(fall)
    fix_min = fix_arr.min(axis=0) if len(fix_arr) else np.zeros(3)
    fix_max = fix_arr.max(axis=0) if len(fix_arr) else np.zeros(3)
    fall_min = fall_arr.min(axis=0) if len(fall_arr) else np.zeros(3)
    fall_max = fall_arr.max(axis=0) if len(fall_arr) else np.zeros(3)
    glob_min = np.minimum(vmin + fix_min, vmin + fall_min)
    glob_max = np.maximum(vmax + fix_max, vmax + fall_max)
    center = (glob_min + glob_max) / 2.0
    max_span = float(np.max(glob_max - glob_min))
    pad = max_span * 0.05
    half = 0.5 * max_span + pad
    def set_limits_global_equal():
        ax.set_xlim(center[0]-half, center[0]+half)
        ax.set_ylim(center[1]-half, center[1]+half)
        ax.set_zlim(center[2]-half, center[2]+half)
    tris_fixed = [[V[i] for i in tri] for tri in F]
    tris_fall = [[V[i] for i in tri] for tri in F]
    col_fix = Poly3DCollection(tris_fixed, facecolor=(0.2,0.4,0.8,0.95), edgecolor=(0.1,0.1,0.1,0.6), linewidths=0.3)
    col_fall = Poly3DCollection(tris_fall, facecolor=(0.85,0.2,0.2,0.95), edgecolor=(0.1,0.1,0.1,0.6), linewidths=0.3)
    ax.add_collection3d(col_fix)
    ax.add_collection3d(col_fall)
    ax.view_init(elev=25, azim=35)
    set_limits_global_equal()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    def set_frame(k: int):
        fp = np.asarray(fix[k])
        cp = np.asarray(fall[k])
        col_fix.set_verts([[V[i]+fp for i in tri] for tri in F])
        col_fall.set_verts([[V[i]+cp for i in tri] for tri in F])
        ax.set_title(f"t={times[k]:.2f}s")
    frames = None
    try:
        writer = FFMpegWriter(fps=fps)
        with writer.saving(fig, out_mp4, dpi=600):
            for k in range(len(times)):
                set_frame(k)
                writer.grab_frame()
    except Exception:
        if out_frames_dir is not None:
            frames = os.path.join(out_frames_dir, 'frames')
            os.makedirs(frames, exist_ok=True)
            for k in range(len(times)):
                set_frame(k)
                fig.savefig(os.path.join(frames, f'frame_{k:05d}.png'), dpi=600)
            try:
                import subprocess
                exe = None
                try:
                    import imageio_ffmpeg
                    exe = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    exe = None
                if exe is None:
                    import shutil
                    exe = shutil.which('ffmpeg')
                cmd = [
                    exe if exe is not None else 'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(frames, 'frame_%05d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_mp4,
                ]
                subprocess.run(cmd, check=True)
            except Exception:
                pass
    plt.close(fig)
    
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
    parser.add_argument('--integrator', type=str, choices=['verlet', 'rk4', 'genalpha', 'generalized_alpha', 'optimized_genalpha', 'optimized_generalized_alpha'], default='verlet')
    parser.add_argument('--rho_inf', type=float, default=0.5)
    parser.add_argument('--ga_tol', type=float, default=1e-8)
    parser.add_argument('--ga_iter', type=int, default=20)
    parser.add_argument('--ga_relax', type=float, default=1.0)
    parser.add_argument('--ga_newton', action='store_true')
    parser.add_argument('--ga_eps_x', type=float, default=1e-6)
    parser.add_argument('--ga_eps_v', type=float, default=1e-6)
    parser.add_argument('--ga_cap', type=float, default=1.0e-3)
    parser.add_argument('--ga_min_dt', type=float, default=1.0e-6)
    parser.add_argument('--ga_growth_thr', type=float, default=1.0e-3)
    parser.add_argument('--ga_growth_scale', type=float, default=0.1)
    parser.add_argument('--ga_init_scale', type=float, default=1.0)
    parser.add_argument('--ga_init_max', type=float, default=5.0e-3)
    parser.add_argument('--ga_grace', type=int, default=1)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--name', type=str, default='output')
    args = parser.parse_args()

    mesh_basename = os.path.basename(args.mesh)
    mesh_stem = os.path.splitext(mesh_basename)[0]
    base_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    base_out = os.path.abspath(base_out)
    outputs_root = os.path.join(base_out, 'outputs')
    os.makedirs(outputs_root, exist_ok=True)
    ts_name = time.strftime('%Y%m%d_%H%M%S')
    name = getattr(args, 'name', 'output')
    viz_dir = os.path.join(outputs_root, f"{name}_{mesh_stem}_{ts_name}")
    os.makedirs(viz_dir, exist_ok=True)
    csv_path = os.path.join(viz_dir, os.path.basename(args.out))
    log_path = os.path.join(viz_dir, os.path.basename(args.log))

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
        integrator=args.integrator,
        ga_rho_inf=args.rho_inf,
        ga_tol=args.ga_tol,
        ga_iter=args.ga_iter,
        ga_relax=args.ga_relax,
        ga_newton=args.ga_newton,
        ga_eps_x=args.ga_eps_x,
        ga_eps_v=args.ga_eps_v,
        ga_cap=args.ga_cap,
        ga_min_dt=args.ga_min_dt,
        ga_growth_thr=args.ga_growth_thr,
        ga_growth_scale=args.ga_growth_scale,
        ga_init_scale=args.ga_init_scale,
        ga_init_max=args.ga_init_max,
        ga_grace=args.ga_grace,
    )
    # Always export per-body pos/vel/acc component PNGs
    try:
        plot_per_body(csv_path, viz_dir)
    except Exception:
        pass
    if args.plot:
        plot_results(csv_path, log_path, viz_dir)
    if args.video:
        save_video(csv_path, args.mesh, os.path.join(viz_dir, 'animation.mp4'), fps=args.fps, stride=args.stride, out_frames_dir=viz_dir)


if __name__ == '__main__':
    main()