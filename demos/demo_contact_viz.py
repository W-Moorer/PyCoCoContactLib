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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    parser.add_argument('--line', type=int, required=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--plane_scale', type=float, default=6.0)
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
    det._near_band = min(meshA.tri_size, near_band) if meshA.tri_size > 0.0 else near_band
    manifold = det.query_manifold((tA, RA), (tB, RB), prev_manifold=None, max_points=64)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
    polyA = build_polydata(meshA, tA, RA)
    polyB = build_polydata(meshB, tB, RB)
    plotter.add_mesh(polyA, color='lightgray', opacity=0.2, show_edges=False)
    plotter.add_mesh(polyB, color='steelblue', opacity=0.35, show_edges=False)

    nvec = np.array([0.0, 0.0, 1.0], dtype=float)
    p0 = 0.5 * ((RA @ meshA.center_local + tA) + (RB @ meshB.center_local + tB))
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
    plotter.show(screenshot=os.path.abspath(args.out))


if __name__ == '__main__':
    main()