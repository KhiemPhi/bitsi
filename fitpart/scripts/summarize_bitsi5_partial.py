"""Summarize paired development results, with object-paired bootstrap intervals."""
import json
import sys
import numpy as np


def summarize(path, split=None):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if split:
        rows = [r for r in rows if r["split"] == split]
    if not rows:
        print("No completed development objects.")
        return
    print(f"{len(rows)} development objects; intervals resample objects (2,000 draws, seed 0).")
    print("| views | metric | v4 | v5 partial | delta [95% bootstrap CI] |")
    print("|---|---|---:|---:|---:|")
    for tag in ("full", "K3", "K6"):
        for metric in ("sym_miou", "ari", "n_pred", "oversegmentation", "undersegmentation"):
            values = []
            for row in rows:
                pair = []
                for name in ("v4", "v5_partial"):
                    s = row["evaluations"][tag][name]["full"]
                    if metric == "oversegmentation":
                        value = float(s["n_pred"] > s["n_gt"])
                    elif metric == "undersegmentation":
                        value = float(s["n_pred"] < s["n_gt"])
                    else:
                        value = s[metric]
                    pair.append(value)
                values.append(pair)
            v = np.array(values)
            delta = v[:, 1] - v[:, 0]
            rng = np.random.default_rng(0)
            samples = delta[rng.integers(len(rows), size=(2000, len(rows)))].mean(1)
            lo, hi = np.quantile(samples, [.025, .975])
            print(f"| {tag} | {metric} | {v[:,0].mean():.3f} | {v[:,1].mean():.3f} | {delta.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    for name in ("v4", "v5_partial"):
        times = [r["evaluations"][tag][name]["seconds"] for r in rows for tag in ("full", "K3", "K6")]
        print(f"{name}: mean fit {np.mean(times):.2f}s, median {np.median(times):.2f}s")


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
