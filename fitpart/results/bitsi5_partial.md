# BITSI v5: observed-only cut experiment

This is an opt-in development candidate, not a replacement for the frozen robot
pipeline. The earlier v5 normal-axis and Chebyshev-child experiments remain intact.

## What changed

`BITSI5(..., partial_cuts=True)` partitions measured outer radii in 24 angular bins
instead of the Fourier amplitudes of completed cross-sections. Missing bins stay
NaN. The Chebyshev likelihood excludes them, retains the original slab coordinates
across gaps, and requires at least three shared measured bins across a cut. Each
piece needs at least three observed slabs. Noise is estimated from consecutive
observed triples, with point spacing as a floor.

The initial `partial_bin_pooling="sum"` experiment treated angular residuals as
independent. Early development results showed more over-segmentation. The `"mean"`
candidate divides standardized radii by sqrt(24), reducing the residual evidence
without changing parameter penalties. This is a heuristic for correlated angular
measurements, not a derived probabilistic independence model. The divisor is fixed:
changing visibility must not itself change a profile's scale. Both runs are retained.

The original envelope fit and child code already exclude unobserved angular bins.
They still have other partial-observation weaknesses; this experiment only changes
cuts. No box/voxel state is consumed, no missing surface is reconstructed, and no
handle-recovery improvement is claimed. A true boundary hidden by a gap cannot be
localized by this candidate. Color and topology cut channels are not supported in
the new mode; unsupported configurations raise an error.

## Usage

```python
from fitpart.bitsi5 import BITSI5

model = BITSI5(
    up=(0, 0, 1),       # actual gravity direction in the point-cloud frame
    axis_rule="residual",
    spine_coding="free",
    partial_cuts=True,
    partial_bin_pooling="mean",
)
labels, root = model.fit(points, normals, colors)
```

Existing defaults are unchanged. `partial_cuts=False` with residual axis/free
spine retains v4 behavior. Each fit resets v5 diagnostic traces.

## Development protocol

- First available object of each tabletop category, restricted to `dev`/`ooddev`:
  32 objects. No held-out objects were evaluated.
- Identical deterministic clouds, gravity prior, child rules and cut penalty for
  each pair. Full scans are placed in the tabletop frame with gravity provided;
  these are not the gravity-free full-cloud benchmarks from section 24.
- Simulated orbit views at K=3 and K=6, using the existing sensing/fusion code.
  These are not robot-reachability-constrained trials.
- Save both observed-cloud and nearest-neighbor-transferred full-object scores.
  Summary tables use full-object scores, object means, and paired object-bootstrap
  intervals (2,000 draws, seed 0). With one object per category, category and object
  means coincide. The small pilot cannot establish generalization.
- Raw angular-sum run: `results/all/v5_partial_pilot.jsonl` (created before pooling
  was exposed as an option; equivalent to `--pooling sum`).
- Correlation-downweighted run: `results/all/v5_partial_mean_pilot.jsonl`.

Reproduce from this directory, using the sam3 Python environment:

```bash
python scripts/dev_bitsi5_partial.py results/all/NEW_RUN.jsonl --per-category 1 --workers 4 --pooling mean
python scripts/summarize_bitsi5_partial.py results/all/NEW_RUN.jsonl
```

Output files are exclusive-create. The evaluation refuses held-out split names.
For regression checks, from the parent directory:

```bash
python -m fitpart.scripts.test_partial_sections
```

Seven checks cover unknown bins, smooth profiles with gaps, real observed steps,
disjoint angular support, unknown-channel cost, completely missing profiles,
v4 compatibility when disabled, and repeated-fit diagnostic reset.

The existing v4 envelope code emits empty-angle-bin and correlation warnings on
some inputs. It is unchanged in this comparison; those warnings remain a separate
robustness issue.

## Completed pilot: do not promote

All 32 objects completed for both variants (96 paired cloud evaluations per
variant). No failed objects were dropped. Results for the downweighted candidate:

| Input | v4 sym mIoU | v5 sym mIoU | paired delta, 95% bootstrap CI | v4 ARI | v5 ARI |
|---|---:|---:|---|---:|---:|
| Full tabletop scan | 0.322 | 0.297 | -0.025 [-0.076, +0.019] | 0.331 | 0.263 |
| 3 views | 0.263 | 0.284 | +0.022 [-0.040, +0.080] | 0.180 | 0.154 |
| 6 views | 0.293 | 0.251 | -0.042 [-0.100, +0.010] | 0.268 | 0.127 |

At K3, over-segmentation falls from 25.0% to 9.4%, but under-segmentation rises
from 53.1% to 71.9%. This is not an unqualified segmentation improvement. At K6,
the ARI delta is -0.141 [-0.253, -0.045]. Mean fit time is 4.08 s versus 1.74 s
for v4 in this four-worker run.

On the nine swept-category development objects alone, K3 sym mIoU is 0.451 versus
0.447 and K6 is 0.518 versus 0.423 (v4 versus candidate). The small aggregate K3
gain therefore does not establish a gain on the project's central object family.

The unweighted sum ablation is worse: sym mIoU 0.151 / 0.214 / 0.189 for full / K3
/ K6, compared with v4's 0.322 / 0.263 / 0.293. Raw angular profiles and a missing
data mask alone are insufficient. The downweighting was chosen after seeing early
sum-run development results, so its confidence intervals are exploratory, not
confirmatory. Neither run justifies a held-out evaluation or a frozen default.

Next hypothesis: preserve the robust fitted section representation, but propagate
observability/fit uncertainty into its cut likelihood. Separately add the explicit
unknown/free/occupied visibility interface to envelope and child inference. Those
changes are not implemented here.
