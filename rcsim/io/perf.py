import time
import csv
from contextlib import contextmanager


class Perf:
    def __init__(self):
        self.totals_exclusive = {}
        self.totals_inclusive = {}
        self.counts = {}
        self.meta = {}
        self._stack = []
        self.totals = self.totals_exclusive

    @contextmanager
    def section(self, label: str):
        frame = {"label": label, "start": time.perf_counter(), "child": 0.0}
        self._stack.append(frame)
        try:
            yield
        finally:
            dt = time.perf_counter() - frame["start"]
            exclusive = dt - frame["child"]
            self.totals_exclusive[label] = self.totals_exclusive.get(label, 0.0) + max(0.0, exclusive)
            self.totals_inclusive[label] = self.totals_inclusive.get(label, 0.0) + dt
            self.counts[label] = self.counts.get(label, 0) + 1
            self._stack.pop()
            if self._stack:
                self._stack[-1]["child"] += dt

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            self.meta[k] = v

    def reset(self):
        self.totals_exclusive.clear()
        self.totals_inclusive.clear()
        self.counts.clear()
        self.meta.clear()
        self._stack.clear()

    def write_csv(self, path: str):
        total_exc = sum(self.totals_exclusive.values())
        total_inc = sum(self.totals_inclusive.values())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "key",
                "exclusive_sec",
                "inclusive_sec",
                "count",
                "avg_ms",
                "exclusive_percent",
                "algorithm",
                "t_end",
                "dt_frame",
                "dt_sub",
                "bodies",
            ])
            algo = self.meta.get("algorithm", "")
            t_end = self.meta.get("t_end", "")
            dt_frame = self.meta.get("dt_frame", "")
            dt_sub = self.meta.get("dt_sub", "")
            bodies = self.meta.get("bodies", "")
            for k in sorted(self.totals_exclusive.keys()):
                tot_exc = float(self.totals_exclusive.get(k, 0.0))
                tot_inc = float(self.totals_inclusive.get(k, 0.0))
                cnt = int(self.counts.get(k, 0))
                avg_ms = (tot_exc / cnt * 1000.0) if cnt > 0 else 0.0
                pct = (tot_exc / total_exc * 100.0) if total_exc > 0 else 0.0
                w.writerow([k, f"{tot_exc:.9f}", f"{tot_inc:.9f}", cnt, f"{avg_ms:.6f}", f"{pct:.4f}", algo, t_end, dt_frame, dt_sub, bodies])
            w.writerow(["TOTAL", f"{total_exc:.9f}", f"{total_inc:.9f}", "", "", "100.00", algo, t_end, dt_frame, dt_sub, bodies])


perf = Perf()