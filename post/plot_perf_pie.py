import os
import csv
import argparse

def read_perf(path: str):
    data = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {n: i for i, n in enumerate(header)}
        for row in r:
            if not row:
                continue
            key = row[idx["key"]]
            total_sec = float(row[idx["total_sec"]])
            percent = float(row[idx["percent"]])
            data.append((key, total_sec, percent))
    return data

def make_legend_labels(items):
    labels = []
    for key, tot, pct in items:
        labels.append(f"{key} ({pct:.2f}%, {tot:.3f}s)")
    return labels

def plot_pie(items, out_png: str, title: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except Exception:
        pass
    sizes = [pct for _, _, pct in items]
    legend_labels = make_legend_labels(items)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    wedges, _texts = ax.pie(
        sizes,
        labels=None,
        autopct=None,
        startangle=90,
        counterclock=False,
        textprops={'fontsize': 11},
    )
    import math
    centers = []
    for w, (_, _, pct) in zip(wedges, items):
        ang = (w.theta2 + w.theta1) / 2.0
        rad = math.radians(ang)
        x = math.cos(rad)
        y = math.sin(rad)
        centers.append((x, y, pct))
    for (x, y, pct) in centers:
        if pct >= 6.0:
            ax.text(
                0.65 * x,
                0.65 * y,
                f"{pct:.2f}%",
                ha='center',
                va='center',
                fontfamily='Times New Roman',
                fontsize=11,
            )
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(0.93, 0.5), prop={'family': 'Times New Roman', 'size': 10})
    ax.set_title(title, fontfamily='Times New Roman', pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perf", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    items = read_perf(args.perf)
    items_sorted = sorted(items, key=lambda x: x[2], reverse=True)
    title = os.path.basename(os.path.dirname(args.perf))
    if args.out is None:
        base_dir = os.path.dirname(args.perf)
        out_png = os.path.join(base_dir, "perf_pie.png")
    else:
        out_png = args.out
    plot_pie(items_sorted, out_png, title)

if __name__ == "__main__":
    main()