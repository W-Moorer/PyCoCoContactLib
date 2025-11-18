import csv
import sys


def load_cols(path: str):
    with open(path, 'r', newline='') as rf:
        r = csv.reader(rf)
        header = next(r)
        cols = {name: i for i, name in enumerate(header)}
        rows = [list(map(float, row)) for row in r if row]
    return rows, cols


def compare(pb: str, pn: str):
    b, cb = load_cols(pb)
    n, cn = load_cols(pn)
    if cb != cn:
        raise RuntimeError('headers differ')
    if len(b) != len(n):
        raise RuntimeError('length differ')
    def col(name: str):
        i = cb[name]
        diffs = [abs(b[j][i] - n[j][i]) for j in range(len(b))]
        rmse = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
        return max(diffs), rmse
    names = ['fall_z', 'distance', 'fall_vz']
    return {nm: col(nm) for nm in names}


def main(argv):
    if len(argv) != 3:
        print('Usage: compare_csv.py <baseline.csv> <new.csv>')
        return 1
    pb, pn = argv[1], argv[2]
    res = compare(pb, pn)
    print(res)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))