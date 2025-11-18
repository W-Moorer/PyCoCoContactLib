import time
import csv
from contextlib import contextmanager


class Perf:
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self.meta = {}

    @contextmanager
    def section(self, label: str):
        s = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - s
            self.totals[label] = self.totals.get(label, 0.0) + dt
            self.counts[label] = self.counts.get(label, 0) + 1

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            self.meta[k] = v

    def reset(self):
        self.totals.clear()
        self.counts.clear()
        self.meta.clear()

    def write_csv(self, path: str):
        total = sum(self.totals.values())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "total_sec", "count", "avg_ms", "percent", "algorithm", "t_end", "dt_frame", "dt_sub", "bodies"])
            algo = self.meta.get("algorithm", "")
            t_end = self.meta.get("t_end", "")
            dt_frame = self.meta.get("dt_frame", "")
            dt_sub = self.meta.get("dt_sub", "")
            bodies = self.meta.get("bodies", "")
            for k in sorted(self.totals.keys()):
                tot = float(self.totals.get(k, 0.0))
                cnt = int(self.counts.get(k, 0))
                avg_ms = (tot / cnt * 1000.0) if cnt > 0 else 0.0
                pct = (tot / total * 100.0) if total > 0 else 0.0
                w.writerow([k, f"{tot:.9f}", cnt, f"{avg_ms:.6f}", f"{pct:.4f}", algo, t_end, dt_frame, dt_sub, bodies])


perf = Perf()