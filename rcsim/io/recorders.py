import csv


def csv_recorder(csv_path):
    def _on_frame(frame, t, world):
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for i, rb in enumerate(world.bodies):
                x, y, z = rb.body.position
                w.writerow([frame, t, i, x, y, z])
    return _on_frame