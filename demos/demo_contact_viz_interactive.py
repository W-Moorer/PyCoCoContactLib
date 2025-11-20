import os
import re
import argparse
import numpy as np
import pyvista as pv

from rcsim import RapidMesh, MeshPairContactDetector


def parse_log_line(path: str, line_no: int) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    idx = max(0, min(len(lines) - 1, line_no - 1))
    s = lines[idx].strip()
    m = re.search(r"sub t=([0-9eE\.+\-]+) z=([0-9eE\.+\-]+) pen=([0-9eE\.+\-]+) Fn=([0-9eE\.+\-]+) Fn_raw=([0-9eE\.+\-]+) speed=([0-9eE\.+\-]+) contacts=(\d+) dt=([0-9eE\.+\-]+)", s)
    if not m:
        raise RuntimeError("log line parse failed")
    t = float(m.group(1))
    z = float(m.group(2))
    pen = float(m.group(3))
    fn = float(m.group(4))
    fn_raw = float(m.group(5))
    speed = float(m.group(6))
    contacts = int(m.group(7))
    dt = float(m.group(8))
    return {"t": t, "z": z, "pen": pen, "fn": fn, "fn_raw": fn_raw, "speed": speed, "contacts": contacts, "dt": dt}


def build_polydata(mesh: RapidMesh, trans: np.ndarray, R: np.ndarray) -> pv.PolyData:
    verts = [R @ v + trans for v in mesh.vertices]
    pts = np.asarray(verts, dtype=float)
    faces = []
    for i, j, k in mesh.triangles:
        faces.extend([3, i, j, k])
    faces = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(pts, faces)


def compute_force_and_pen(manifold, rel_v, k_contact=1e5, c_damp=0.0, half_wave=False):
    pen = 0.0
    Fn = 0.0
    for cp in manifold.points:
        n = cp.n
        L = float(np.linalg.norm(n))
        n = np.array([0.0, 0.0, 1.0], dtype=float) if L < 1e-12 else (n / L)
        if cp.phi < 0.0:
            pen_i = -cp.phi
            pen = max(pen, pen_i)
            vn_i = float(np.dot(rel_v, n))
            vn_comp = vn_i if (not half_wave or vn_i < 0.0) else 0.0
            Fn_raw_i = k_contact * pen_i - c_damp * vn_comp
            Fn += max(Fn_raw_i, 0.0) * cp.area_weight
    return pen, Fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    parser.add_argument('--line', type=int, required=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--plane_scale', type=float, default=6.0)
    parser.add_argument('--k_contact', type=float, default=1e5)
    parser.add_argument('--c_damp', type=float, default=0.0)
    parser.add_argument('--half_wave', action='store_true')
    parser.add_argument('--no_near_band', action='store_true')
    parser.add_argument('--show_edges', action='store_true')
    parser.add_argument('--edge_color', type=str, default='black')
    parser.add_argument('--sim_pen', type=float, default=None)
    parser.add_argument('--sim_fn', type=float, default=None)
    args = parser.parse_args()

    info = parse_log_line(args.log, args.line)
    meshA = RapidMesh.from_obj(args.mesh)
    meshB = RapidMesh.from_obj(args.mesh)
    tA = np.array([0.0, 0.0, 0.0], dtype=float)
    RA = np.eye(3, dtype=float)
    tB = np.array([0.0, 0.0, info["z"]], dtype=float)
    RB = np.eye(3, dtype=float)

    det = MeshPairContactDetector(meshA, meshB)
    near_band = 0.5 * info["speed"] * info["dt"]
    use_near_band = (not args.no_near_band) and (info.get("contacts", 0) == 0) and (info.get("pen", 0.0) <= 0.0)
    if use_near_band:
        det._near_band = min(meshA.tri_size, near_band) if meshA.tri_size > 0.0 else near_band
    manifold = det.query_manifold((tA, RA), (tB, RB), prev_manifold=None, max_points=64)

    polyA = build_polydata(meshA, tA, RA)
    polyB = build_polydata(meshB, tB, RB)
    plotter = pv.Plotter(off_screen=False, window_size=(1280, 720))
    plotter.add_mesh(polyA, color='lightgray', opacity=0.2, show_edges=args.show_edges, edge_color=args.edge_color)
    plotter.add_mesh(polyB, color='steelblue', opacity=0.35, show_edges=args.show_edges, edge_color=args.edge_color)

    nvec = np.array([0.0, 0.0, 1.0], dtype=float)
    p0 = 0.5 * ((RA @ meshA.center_local + tA) + (RB @ meshB.center_local + tB))
    ptsA = None
    ptsB = None
    mids = None
    if manifold.points:
        ns = np.stack([cp.n for cp in manifold.points], axis=0)
        n = np.mean(ns, axis=0)
        L = float(np.linalg.norm(n))
        nvec = np.array([0.0, 0.0, 1.0], dtype=float) if L < 1e-12 else (n / L)
        mids = np.stack([0.5 * (cp.pA_world + cp.pB_world) for cp in manifold.points], axis=0)
        p0 = np.mean(mids, axis=0)
        ptsA = np.stack([cp.pA_world for cp in manifold.points], axis=0)
        ptsB = np.stack([cp.pB_world for cp in manifold.points], axis=0)
        plotter.add_points(ptsA, color='red', point_size=12)
        plotter.add_points(ptsB, color='yellow', point_size=12)
        plotter.add_points(mids, color='white', point_size=14)

    size = args.plane_scale * max(meshA.tri_size, meshB.tri_size)
    plane = pv.Plane(center=p0, direction=nvec, i_size=size, j_size=size)
    plotter.add_mesh(plane, color='orange', opacity=0.25)
    plotter.add_arrows(np.asarray([p0]), np.asarray([nvec]), mag=size * 0.25, color='orange')

    rel_v = np.array([0.0, 0.0, -info["speed"]], dtype=float)
    pen_calc, Fn_calc = compute_force_and_pen(manifold, rel_v, k_contact=args.k_contact, c_damp=args.c_damp, half_wave=args.half_wave)
    sim_pen = info["pen"] if args.sim_pen is None else args.sim_pen
    sim_fn = info["fn"] if args.sim_fn is None else args.sim_fn
    contacts_sim = int(info.get("contacts", 0))
    contacts_calc = len(manifold.points)
    d_pen = pen_calc - sim_pen
    d_fn = Fn_calc - sim_fn
    h_text = (f"\nh={min(meshA.tri_size, near_band):.6e}" if (meshA.tri_size > 0.0 and use_near_band) else "\nh=disabled")
    text = (
        f"t={info['t']:.6f}\nz={info['z']:.6f}"
        f"\npen_calc={pen_calc:.6e}\npen_sim={sim_pen:.6e}\nΔpen={d_pen:.6e}"
        f"\nFn_calc={Fn_calc:.6f}\nFn_sim={sim_fn:.6f}\nΔFn={d_fn:.6f}"
        f"\ncontacts_calc={contacts_calc}\ncontacts_sim={contacts_sim}"
    ) + h_text
    plotter.add_text(text, position='upper_right', font_size=14, color='black')
    plotter.show()


if __name__ == '__main__':
    main()