# Overnight results (2026-10-06, single shared RTX 4060 Ti)

All numbers are from this night's runs on subsets (see "Scope"); none are final paper
numbers. Class means average over categories. "Hungarian mIoU" is class-agnostic:
GT regions are matched one-to-one to predicted segments and unmatched GT counts as 0.

## Scope

| Dataset | Objects | Notes |
|---|---|---|
| ShapeNetPart test | 128 (8/category) geometry-only; 67 (4/category) full method | unit-normalized, 2.5k points, no color |
| PartNet-E test (PartSLIP) | 180 (4 per category, 45 categories) | 20k-point random subsample of each cloud; fetched by random-access reads of the remote zip, not the full 9.4 GB |

## 1. Geometry: Chebyshev cuts vs the ITSI jump threshold (ShapeNetPart, 128 objects, CPU only)

| Segmentation along the longest PCA axis | Cuts used | Class-mean Hungarian mIoU | Segments |
|---|---|---|---|
| Chebyshev inflections, top K-1 (oracle K) | strongest K-1 | **0.452** | K |
| Jump threshold delta = 0.2, top K-1 (oracle K) | largest K-1 jumps | 0.347 | K |
| Chebyshev, cuts persisting at the 2 coarsest degrees (no threshold, no K) | persistence | **0.415** | ~4.0 |
| Chebyshev, all cuts | all | 0.390 | many |
| Jump threshold delta = 0.2, all cuts | all | 0.323 | many |
| BITSI v0 (existing `main.py` pipeline: dominant axis, BIC degree, strength threshold, IBR fusion) | own | 0.402 | ~3.3 |

Same protocol on PartNet-E (180 objects, 20k points; GT = instances + "rest"):

| Longest-axis slabs | Class-mean Hungarian mIoU | Segments |
|---|---|---|
| Chebyshev, top K-1 (oracle K) | **0.285** | K |
| Jump delta = 0.2, top K-1 (oracle K) | 0.268 | K |
| Chebyshev persistence (levels <= 2), top degree capped at 16 | 0.249 | ~3.5 |
| Chebyshev persistence, uncapped (chopping picks high degrees on dense clouds) | 0.230 | ~8.3 |
| Chebyshev, all cuts / jump, all cuts | 0.201 / 0.219 | many |
| BITSI v0 (own pipeline, own segment count) | 0.259 | ~3.5 |

With the top degree capped at 16, persistence gives 0.425 (~3.6 segments) on ShapeNetPart.
So the threshold-free detector matches the tuned BITSI v0 (0.425 vs 0.402 on ShapeNetPart,
0.249 vs 0.259 on PartNet-E) without its thresholds, about 100x faster. The gap over jumps
under oracle K is large on ShapeNetPart (+0.105) and small on PartNet-E (+0.017).
Recommendation: cap `max_deg` at 16 (the default 64 was used for every other number here).

Recursive (divisive) geometry with oracle K, recall@0.5 / mIoU: Chebyshev 0.506 / 0.325,
axis-only 0.504 / 0.331, single degree 0.471 / 0.327, extrema 0.423 / 0.311, jump 0.375 / 0.301.

Speed: BITC profiles + cuts for 11 directions take ~0.05 s/object (vectorized), vs 4.2 s/object
for BITSI v0 end to end (18 s/object on 20k-point PartNet-E clouds).

Figure: `figures/fig3_knife.{png,pdf}`. The strongest inflection cut sits 1% of the
object length from the true blade/handle boundary; the jump threshold fires 20 cuts.

## 2. Full method, class-agnostic (ShapeNetPart, 67 objects; tree cut at linkage theta, no oracle)

| Variant | theta = 0.85 | 0.9 | 0.93 |
|---|---|---|---|
| FitPart full (lambda = 0.5, divisive Chebyshev A_geo) | 0.398 | **0.507** | 0.522 |
| no geometry (A_geo = 1) | 0.342 | 0.517 | 0.574 |
| jump-threshold geometry | 0.342 | 0.519 | 0.573 |
| semantics only (lambda = 1) | 0.579 | 0.560 | 0.536 |
| BITSI v0, same 66 objects (own segment count) | | 0.402 | |

Long-range DINOv2 edges matter (0.507 -> 0.317 without them). Geometry over-splits flat
segmentations: semantics alone peaks higher (0.579 at theta 0.85).

## 3. PartNet-E, language selection (180 objects)

Semantic mIoU = part-name queries; "acc." = category-accumulated part IoU (sum I / sum U,
PartSLIP-style); "per-shape" = mean per-shape part IoU (absent in both = 1, inflated).
Selection acc. = selected region IoU >= 0.5 with the GT part, per part present.

| Method | Sem. mIoU (acc.) | Sem. mIoU (per-shape) | Selection acc. |
|---|---|---|---|
| FitPart + SAM 3 (multi-instance) | 0.247 | 0.316 | 0.237 |
| FitPart, no geometry + SAM 3 | 0.247 | 0.318 | 0.239 |
| FitPart, jump-threshold geometry + SAM 3 | 0.250 | 0.319 | 0.233 |
| **PartField hierarchy + same SAM 3 selection** | **0.269** | **0.340** | 0.236 |
| SAM 3 lifted to points, no hierarchy (score >= 0.3) | 0.207 | 0.315 | 0.237 |
| SAM 3 lifted to points, no hierarchy (score >= 0.5) | 0.153 | 0.280 | 0.177 |
| FitPart + SigLIP crop relevance (draft Sec. 4.5) | 0.106 | 0.155 | 0.108 |

Class-agnostic (instances + one "rest" region, Hungarian):

| Method | Hierarchy recall@0.5 | Oracle-K mIoU | mIoU at theta = 0.9 |
|---|---|---|---|
| FitPart full | 0.760 | 0.408 | 0.399 (5.7 segs) |
| FitPart, no geometry | 0.763 | 0.417 | 0.409 (4.0 segs) |
| semantics only (lambda = 1) | 0.763 | 0.417 | 0.345 (9.9 segs) |
| geometry only (lambda = 0) | 0.364 | 0.224 | 0.228 |
| no long-range edges | 0.716 | 0.332 | 0.360 |
| PartField (its KMeans protocol, oracle K) | | 0.408 | |
| PartField hierarchy on our superpoints | 0.781 | 0.454 | |
| BITSI v0 (own segment count) | | | 0.259 |

Graspable-part queries (handles, knobs, shafts, necks, grips; n = 44; selection under the
feasibility constraint; metric scale from category priors, G = 8 cm):

| Method | Mean IoU | Correct (IoU >= 0.5) | Any selection graspable | Correct and graspable |
|---|---|---|---|---|
| FitPart + SAM 3 | 0.249 | 0.182 | 0.955 | 0.182 |
| FitPart, no geometry + SAM 3 | 0.229 | 0.159 | 0.932 | 0.159 |
| FitPart, jump geometry + SAM 3 | 0.237 | 0.159 | 0.955 | 0.159 |
| PartField + SAM 3 | 0.248 | 0.182 | 0.955 | 0.182 |
| SAM 3 lifted, no hierarchy (>= 0.5) | 0.211 | **0.250** | 0.273 | **0.250** |
| FitPart + SigLIP | 0.119 | 0.091 | 0.864 | 0.091 |

Without the feasibility constraint, "any selection graspable" drops from 0.955 to 0.932 (SAM 3)
and from 0.864 to 0.818 (SigLIP); correctness is unchanged.

"Exists but cannot be grasped" (Table 2 of the draft; `scripts/eval_infeasible.py`). A GT part
is ungraspable if its min-volume OBB is thicker than G after metric scaling (50 of 319 part
queries: doors, seats, drawers, shades, ...). FitPart + SAM 3 selects among feasible nodes
with relevance floor r_min:

| r_min | Ungraspable parts reported "infeasible" | Graspable parts falsely reported "infeasible" |
|---|---|---|
| 0.2 | 0.160 | 0.033 |
| 0.3 | 0.200 | 0.019 |
| 0.4 | 0.220 | 0.011 |
| 0.5 | 0.180 | 0.011 |
| relative, alpha = 0.5 | 0.300 | 0.041 |
| relative, alpha = 0.7 | **0.460** | 0.067 |
| relative, alpha = 0.9 | 0.560 | 0.100 |

With an absolute floor, a feasible fragment of the big part (a door edge, a seat rim) still
clears it, so the report fires rarely. The relative rule reports "infeasible" when the most
relevant node is infeasible and the best feasible node scores below alpha times it; it
roughly doubles detection at a few percent of false alarms. alpha is a new parameter to
state in Sec. 4.5.

Task instructions (n = 122; "rewrite" = oracle part phrase standing in for LLM rewriting):

| Query | Acc. (IoU >= 0.5) | Mean IoU |
|---|---|---|
| SAM 3, raw instruction | 0.016 | 0.047 |
| SAM 3, rewritten part phrase | **0.180** | **0.219** |
| SigLIP, raw instruction | 0.041 | 0.065 |
| SigLIP, rewritten part phrase | 0.082 | 0.111 |

Figures: `figures/fig4_partnete.{png,pdf}` (semantic mIoU vs correct-and-graspable),
`figures/qual_partnete.png` (8 queries x 4 methods). Full tables, per-category numbers and
theta sweeps: `results/partnete_summary.md`, `results/shapenetpart_summary.md`.

PartField features inside FitPart as the semantic affinity (same SAM 3 selection):
acc. mIoU 0.258 (PartField replaces lifted affinity), 0.264 (PartField, no geometry),
0.254 (lifted + PartField averaged), vs 0.247 FitPart and 0.269 plain PartField hierarchy.
Geometry still adds nothing on semantic metrics, and on graspable-part queries these
variants drop to 0.159 correct-and-graspable.

## Findings that change the draft

1. **The Chebyshev cut detector is the strongest standalone result**, but a modest one:
   it beats the jump threshold under the same protocol (+0.105 on ShapeNetPart, +0.017 on
   PartNet-E with oracle K), matches the tuned old pipeline without thresholds, and is
   ~100x faster.
2. **Semantic quality comes from the 2D foundation models, not from geometry.** Across
   PartNet-E and ShapeNetPart, removing A_geo barely changes semantic metrics.
   Geometry helps on graspable-part queries (handles, knobs) but by small margins on
   small n. The draft's abstract and intro over-credit geometry for part quality.
3. **SigLIP crop relevance is too weak** on rendered point clouds (grey, untextured):
   the hierarchy contains the right node (mean best-node IoU ~0.73 on the first
   objects), but SigLIP picks it rarely. SAM 3 text-prompted masks lifted onto the
   hierarchy roughly double selection accuracy. Sec. 4.5 should be rewritten around it.
4. **The current hierarchy does not beat the strongest baselines.** PartField's learned
   features give a better hierarchy under the same SAM 3 selection (0.269 vs 0.247 acc.
   mIoU), and plain SAM 3 lifting is more often correct on graspable-part queries (0.250
   vs 0.182), although it returns nothing graspable for most queries (0.273 any-graspable
   vs 0.955). As built, FitPart's hierarchy is competitive, not state of the art.
5. **The feasibility constraint rarely changes the selected part** on PartNet-E
   queries (any-graspable 0.932 -> 0.955, correctness unchanged), and the "exists but
   cannot be grasped" report catches ~20% of ungraspable parts with an absolute floor, 46%
   with a relative rule (6.7% false alarms).
6. **GPR given a correct selection is ~1.0 everywhere**, so it cannot separate methods.
   Use "correct and graspable" (and graspable-any) instead.

## Open issues / to confirm

- Category size priors in `scales.py` put normalized clouds in metres for GPR (assumption).
- Task list `data/partnete_tasks.json` is hand-written; "rewrite" is an oracle stand-in for the LLM.
- The PartNet-E metric here is computed on 20k-point subsamples; PartSLIP's exact protocol
  (full clouds, its own label handling) must be matched before comparing to published numbers.
- Baselines not run yet: PartSLIP, PartSLIP++, OpenMask3D, Find3D, SAMPart3D, PartDistill,
  CoACD, superquadrics; no TaskGrasp/LASO/HANDAL/real-robot experiments.

## What I would do next (in order)

1. Decide the paper's claim. The evidence supports "threshold-free, multi-scale cut
   detection for gripper-relevant decomposition" (Sec. 1 of this file) far better than
   "geometry improves open-vocabulary part segmentation" (Secs. 2-3).
2. Close the gap to PartField's hierarchy. Using its features as FitPart's semantic
   affinity was tried (0.258-0.264 vs 0.269); next is its clustering on our superpoints
   with Chebyshev cuts only as split proposals for gripper-relevant parts.
3. Grow the "exists but cannot be grasped" evaluation (50 queries now) into a proper
   benchmark, with the relative rule, so the Sec. 3 constraint is tested where it matters.
4. Replace the oracle rewrite with a real LLM call and run the full PartNet-E
   test set at full resolution with PartSLIP's protocol.

## 4. Learned geometric-topological part detector (added later on 2026-10-06)

Leave-one-category-out on PartNet-E (180 objects): 15 part names shared by >= 2 categories
(base, blade, body, button, cap, door, drawer, handle, leg, lid, screen, seat, spout, switch,
wheel), 242 queries. Accuracy = selected node has IoU >= 0.5 with the part or one instance.

| Detector | Acc. | Mean IoU with GT part |
|---|---|---|
| SAM 3 (soft-union score) | 0.331 | 0.268 |
| Learned, geometry only | 0.237 | 0.210 |
| Learned, geometry + topology | 0.286 | 0.237 |
| Learned, geometry + topology + Chebyshev + context | 0.266 | 0.222 |
| Learned fusion (all + SAM 3 + SigLIP scores as features) | 0.299 | 0.251 |
| **SAM 3, else learned geometry + topology (SAM 3 max < 0.05)** | **0.390** | **0.333** |
| same, gate at 0.15 / 0.30 | 0.382 / 0.361 | 0.316 / 0.300 |

The learned detector beats SAM 3 on lid (0.63 vs 0.46), seat (0.75 vs 0.38), switch (0.27 vs
0.00), door (0.16 vs 0.03) and body; SAM 3 wins on handle, leg, spout, wheel, blade, base. Per-part
table: `results/part_detector_loco.md`. The gate value (0.05) was the a-priori "SAM 3 empty"
cut-off; 0.15 and 0.30 are shown for transparency. Classifier settings were not tuned.

## 5. Multi-channel sweep cuts (geometry + topology + color), go/no-go tests (2026-10-06)

Channels per slab: BITC (geometry); beta_0 and beta_1 of the cross-section (topology); mean CIELAB
L, a, b (color, with an absolute slope floor of 30 Lab units). Same multi-scale Chebyshev
analysis per channel; cuts pooled across channels (`affinity.detect_all_cuts_mc`).

**Test 1, standalone** (PartNet-E 180 objects, longest-PCA-axis slabs, Hungarian mIoU):

| Channels | Oracle K | Threshold-free | Segments |
|---|---|---|---|
| geometry | 0.264 | 0.248 | 3.5 |
| topology | 0.273 | 0.245 | 5.4 |
| color | 0.276 | 0.240 | 6.8 |
| geometry + topology | 0.277 | 0.235 | 7.4 |
| geometry + topology + color | 0.278 | 0.216 | 11.5 |
| jump threshold, oracle K | 0.268 | | |

Ceiling of any longest-axis slab segmentation (majority GT label per slab): 0.420 (0.533 with the
best of 3 axes per object). Go/no-go bar was 0.33: not met.

**Test 2, plugged into strong pipelines as A_geo** (90 objects, 2 per category; SAM 3 on colored
full-resolution renders; accumulated semantic mIoU with refinement; paired bootstrap over categories):

| Hierarchy semantics | no geometry | Chebyshev geometry | geo + topo | geo + color | geo + topo + color |
|---|---|---|---|---|---|
| Lifted SAM/DINOv2 | 0.381 | 0.386 | 0.389 | 0.392 | **0.402** |
| PartField features | 0.411 | 0.406 | **0.415** | 0.394 | 0.400 |

- Lifted pipeline: multi-channel cuts +0.021 over no geometry, 95% CI [+0.004, +0.038],
  better in 20 categories, worse in 8.
- PartField pipeline: geo + topo +0.004, CI [-0.020, +0.024] (12 better / 13 worse);
  all channels -0.011, CI [-0.032, +0.005].
- Selection accuracy and correct-and-graspable (n = 22) unchanged by geometry.

Verdict: the cuts help the weaker (2D-lifted) pipeline significantly but do not improve the
strongest one, so "a plug-in that consistently improves strong pipelines" is not supported.

## 6. SIFT/HOG-inspired sweep events (`sweep_events.py`, 2026-10-06)

Significance test (step z-score against robust channel noise, keep z > 3), HOG-style per-slab
histograms (normal orientation vs sweep axis; CIELAB colour) compared across k = 3 slab windows
with the Hellinger distance, and direction selection by total significant evidence.

**Standalone** (PartNet-E 180, longest-axis slabs; ceiling 0.420):

| Variant | Threshold-free (z > 3) | Segments | Oracle K |
|---|---|---|---|
| **HOG-style histograms only** | **0.322** | 7.6 | 0.322 |
| all channels | 0.304 | 8.8 | 0.319 |
| geometry + topology (significance-scored) | 0.259 | 1.7 | 0.261 |
| geometry (significance-scored) | 0.238 | 1.3 | 0.239 |
| all channels, auto direction | 0.279 | 10.6 | 0.321 |
| previous: Chebyshev geometry / BITSI v0 | 0.248 / 0.259 | 3.5 / ~3.5 | 0.264-0.285 |

**Inside the pipelines** (90 objects, full-res colored SAM 3, accumulated mIoU with refinement):

| Hierarchy semantics | no geometry | Chebyshev | HOG events | all events |
|---|---|---|---|---|
| Lifted SAM/DINOv2 | 0.381 | 0.386 | 0.389 | 0.388 |
| PartField | **0.411** | 0.406 | 0.404 | 0.400 |

Paired bootstrap vs no geometry: lifted +0.008 [-0.003, +0.021]; PartField -0.007 [-0.016, +0.000].

Verdict: the histogram channels are the best training-free sweep detector so far (+7.4 points
over Chebyshev geometry, +6.3 over BITSI v0, threshold-free), but they do not improve the
foundation-model pipelines.

## 7. Part-count-free cuts by change-point selection + gripper channel/stop (`changepoint.py`)

Per-slab feature vector (BITC, beta_0, beta_1, mean Lab, normal and colour histograms as
sqrt-proportions, gripper channel = fraction of the slab in cross-section pieces of minor width
<= G), each feature divided by its robust noise; optimal partitioning with piecewise-linear
segments and a BIC penalty (2D+1) log S x scale. A piecewise-constant model over-segments
(~32 segments on a 12-object sample) because tapers cost a staircase of cuts.
Penalty scale chosen on ShapeNetPart (64 objects): single sweep x4 (0.440, 4.0 segments; vs
Chebyshev persistence 0.425, BITSI v0 0.402), recursive x16 (0.339, 2.7 segments).

**PartNet-E 180, penalty transferred unchanged:**

| Variant | Hungarian mIoU | Segments |
|---|---|---|
| single sweep, all boundaries as candidates | 0.275 | 8.2 |
| single sweep, Chebyshev candidates (+ gripper channel) | 0.274 | 7.2 |
| single sweep, Chebyshev candidates, no gripper channel | 0.272 | 7.2 |
| recursive, no gripper stop | 0.232 | 17.8 |
| recursive + gripper stop ("stop when it fits") | 0.238 | 8.0 |
| reference: HOG events z > 3 / Chebyshev persistence / BITSI v0 | 0.322 / 0.248 / 0.259 | 7.6 / 3.5 / ~3.5 |

Graspable GT parts (n = 44; leaf with IoU >= 0.5 / and feasible): recursive 0.045 / 0.023;
recursive + gripper stop 0.114 / 0.091.

Verdict: change-point selection transfers across datasets and beats Chebyshev persistence and
BITSI v0 without any part count, but not the simpler HOG z > 3 detector. The gripper channel adds
nothing measurable; the gripper stopping rule halves over-segmentation in the recursion and
raises graspable-part recovery (0.023 -> 0.091), still low in absolute terms. Recursion is worse
than a single sweep everywhere: planar cuts on sub-pieces fragment parts.

## 8. Topological BITSI (`topo_bitsi.py`, 2026-10-06)

B_d(t) = sum_k rho_k(t) * chi(C_k(t)): per connected component of the slab cross-section, a
local boundary-fit ratio (against the component's own 2-D box) weighted by its Euler
characteristic. Topological events (beta_0 / beta_1 changes that persist) are breakpoints.

**Additivity** (PartNet-E 180, 3 PCA directions; symmetric error
sum|obj - sum parts| / (sum|obj| + sum|parts|), 0 = additive):

| Channel | Fine grid (2.5h cells, 1 slab) | Coarse (6h cells, 5-slab window) |
|---|---|---|
| point count | 0.000 | 0.000 |
| beta_0 | 0.075 | 0.248 |
| chi | 0.066 | 0.466 |
| sum rho | 0.086 | 0.394 |
| **topological BITSI (sum rho chi)** | **0.086** | 0.570 |
| old BITSI (global box x beta_0) | 0.360 | 0.565 |

**Noise** (longest axis): raw grid 158.8 topological events / up to 41.7 components per slab;
coarse settings (chosen without labels) 5.0 events / 2.2 components.

**Cuts without part count** (coarse settings, longest axis): topological breakpoints 0.290
(6.0 segments); HOG z > 3 0.322 (7.6); union 0.274 (11.3). References: Chebyshev 0.248,
BITSI v0 0.259.

Verdict: on a fine grid topological BITSI is ~4x more additive than BITSI (0.086 vs 0.360),
supporting "shapes are sums of BITSI stacks"; but fine grids are noisy on 20k-point clouds, and
the coarse grid that removes the noise merges touching parts and destroys additivity. Topological
breakpoints alone segment better than Chebyshev / BITSI v0 but not better than HOG events.

## 9. Denoising / full resolution for topological BITSI (`denoise.py`, 2026-10-06)

Fine grid (2.5h cells, 1 slab, h = 20k-subsample spacing for every input so the grid is
identical); cuts = persistent (2-slab) topological breakpoints on the longest axis, evaluated on
the same 20k points and labels. PartNet-E 180 objects; full-res clouds ~368k points.
Denoising = statistical outlier removal (20 nn, 2 sigma) + projection onto local PCA planes
(16 nn) + voxel resampling at 0.5h (keeps 69% of points).

| Input | Events | Median / max beta_0 per slab | Additivity chi | Additivity topo-BITSI | Additivity old BITSI | Cuts mIoU (segments) |
|---|---|---|---|---|---|---|
| 20k subsample | 20.8 | 13.7 / 23.5 | 0.066 | 0.086 | 0.360 | 0.176 (21.8) |
| full-res raw | 13.3 | 1.3 / 3.1 | 0.484 | 0.552 | 0.570 | 0.266 (14.3) |
| full-res denoised | 18.2 | 1.5 / 4.4 | 0.277 | 0.356 | 0.540 | 0.252 (19.2) |

References: coarse-grid topological breakpoints on 20k 0.290 (6.0); HOG z > 3 0.322; BITSI v0 0.259.
A 12-object sample had suggested 0.316 for full-res raw; the full set does not confirm it.

Findings:
- Density, not noise, fragments slabs: full-res gives real cross-sections (median beta_0 1.3 vs
  13.7). Denoising does not help (PartNet-E clouds are sampled from meshes; little positional noise).
- The 20k "additivity" (0.086) was an artifact of fragmentation: tiny fragments never touch, so
  sums match trivially. On clean cross-sections, touching parts merge and Euler characteristic
  obeys chi(A u B) = chi(A) + chi(B) - chi(A n B): stacks are additive only up to contact terms
  (error ~0.5). Unmixing would have to model part contacts explicitly.
- Topological breakpoints on full-res still over-cut (14 segments) and trail HOG events.

## 10. Zero-shot geometric-topological detector (`scripts/train_zeroshot.py`, 2026-10-06)

Detector = gradient-boosted trees on node geometry + topology features, used only when SAM 3
(coloured renders) finds nothing (best node score < 0.05). PartNet-E 180 objects.

| Method | Queries | Accuracy | Where SAM 3 is empty |
|---|---|---|---|
| SAM 3 alone | 242 | 0.393 | 0.110 |
| Z2: trained on SAM 3 pseudo-labels, leave-category-out | 242 | 0.393 | 0.110 |
| Z2: same, leave-object-out | 323 | 0.384 (SAM 3 0.381) | 0.108 |
| Supervised reference (PartNet-E labels, leave-category-out) | 242 | 0.413 | 0.153 |
| SAM 3 alone, 7 shared types | 101 | 0.545 | 0.185 |
| Z1: trained on ShapeNetPart labels (82 objects), 7 shared types | 101 | 0.554 | 0.222 |
| Z1, categories without ShapeNet overlap | 72 | 0.569 (SAM 3 0.556) | 0.240 (0.200) |

Paired bootstrap vs SAM 3 on identical queries: Z1 +0.010 [-0.020, +0.040] (2 queries better, 1
worse); Z1 non-overlap +0.014 [-0.028, +0.069]; supervised reference +0.021 [-0.017, +0.058]
(13 better, 8 worse).

- Z2 fails because SAM 3 fails per part name, not per object: of 118 SAM-3-empty queries only 26
  had enough confident pseudo-labels elsewhere to train on (median 3.5 objects), and there the
  detector was right exactly as often as SAM 3's own pick (5/26).
- Z1 helps a few queries; not significant. With 49 training objects it looked like +1.9; with
  82 it is +1.0.
- With coloured SAM 3 renders even the supervised fallback is no longer significant (+2.1; it was
  +5.9 against the grey-render SAM 3).

Verdict: no significant zero-shot gain; the CVPR "shape fixes VLM failures" claim is not supported
at this scale.

## 11. MuJoCo grasp simulator: ground-truth sanity check (`sim/mujoco_grasp.py`, 2026-10-06)

Setup: HANDAL real meshes (17 categories, GT handle / rest part meshes), CoACD convex pieces
(<= 16), category mass priors, floating Franka-like parallel gripper (8 cm, 70 N, pads mu 1.5),
6 drop orientations, shared antipodal sampler, floor-aware filtering; trial = approach, close,
lift 15 cm, hold 1 s; success = rose >= 10 cm and still held. 2 objects per category (34).

| | GT handle | GT rest | Whole object |
|---|---|---|---|
| All categories (successes / executed grasps) | 332/571 (0.58) | 548/933 (0.59) | 529/883 (0.60) |
| Power drill | 0.75 | 0.28 | 0.43 |
| Hammer | 0.52 | 0.17 | 0.51 |
| Pots / pans | 0.22 | 0.19 | 0.21 |
| Wrenches, pliers, spatula, whisk, ladle | ~0.9-1.0 | ~0.9-1.0 | ~0.9-1.0 |

Verdict: the simulator is physically sensible (handle >> head on drills and hammers), but "lift
success" does not discriminate parts overall: light tools can be lifted from anywhere and heavy
pans fail from anywhere. A part-selection benchmark needs a task in which the part matters
(articulated opening, tool use, handover, transport with reorientation).

**Screw (pivot) task**, same protocol: grasp without lifting, then drive the hand along
exp(xi theta) applied to the grasp pose (ITSI `pivoting_motion`), axis along the settled object's
long bottom edge, pitch 0, theta = 90 deg; success = object within 20 deg / 3 cm of the commanded
screw displacement and still held.

| | GT handle | GT rest | Whole object |
|---|---|---|---|
| All (34 objects) | 177/571 (0.31) | 250/933 (0.27) | 264/883 (0.30) |
| Pots / pans | 0.32 | 0.18 | 0.21 |
| Ladle | 0.65 | 0.37 | 0.41 |
| Power drill | 0.30 | 0.14 | 0.32 |
| Mug | 0.21 | 0.36 | 0.28 |

A generic geometric screw (pivot about the bottom edge) discriminates parts only weakly: the task
is not tied to the object's function. Screws must be anchored to functional parts (pour about the
rim, swing about the wrist, turn about the shaft) for part choice to matter.

## 12. "Pour from the mug": task screw + SOCP metric in simulation (2026-10-06)

HANDAL mugs (12 real scans, no colour). Task screw from GT geometry: upright mug, lift 10 cm
(translation), rotate 90 deg (pitch 0) about the horizontal axis through the rim opposite the
handle. Grasps: shared antipodal sampler on each method's region, floor-filtered, ordered randomly
or by tau = ITSI SOCP screw metric (ported; mu 0.5, e_r 0.02, sigma 0.02, F 70 N, mass 0.35 kg:
parameter values assumed). Up to 10 executions; success = within 20 deg / 3 cm of the commanded
motion and held. Video: `figures/pour_obj000002.mp4`.

| Method | First-grasp success | Successes / 10 attempts |
|---|---|---|
| Whole object, ranked by tau | 1.00 (12/12) | 0.68 |
| BITSI segments + tau (ITSI-style) | 0.92 (11/12) | **0.72** |
| Whole object, random | 0.83 (10/12) | 0.62 |
| GT handle, random (oracle) | 0.83 | 0.59 |
| GT handle, ranked by tau (oracle) | 0.67 | 0.69 |
| GT body, random | 0.75 | 0.53 |
| FitPart + SAM 3 "handle", ranked by tau | 0.58 | 0.54 |
| FitPart + SAM 3 "handle", random | 0.42 | 0.43 |
| SAM 3 lifted "handle", random | 0.08 | 0.11 |

Handle IoU of the selections: FitPart + SAM 3 0.65 mean (9/12 >= 0.5); SAM 3 lifted 0 on 6/12;
BITSI best segment 0.26 mean (2/12).

Findings (n = 12, wide intervals): the task metric tau helps (whole object 0.62 -> 0.68 of
attempts, first grasp 0.83 -> 1.00); BITSI segmentation + tau is best per attempt (0.72). For this
light mug, pouring does not require the handle (GT handle 0.59 vs GT body 0.53 random), so
"select the handle" methods gain nothing and imperfect handle masks lose grasps.

## 13. Reeb-graph (topology-informed) slicing, first attempts (2026-10-06)

Pieces per slab linked across slabs = Reeb graph; cycles = loops (handles). On the 12 HANDAL mugs,
handle recovery (best part IoU >= 0.5): coarse grid 2/12; label-free finer scale 3/12 (13-70
spurious loops); + persistence pruning of short arcs 2/12; + solid cross-sections 2/12. BITSI v0
2/12, FitPart + SAM 3 9/12. Not working yet; needs diagnosis (sweep direction, surface vs volume).

## 14. Label-free sweep axis and the surface-of-revolution residual (12 HANDAL mugs)

Axis: `axes.revolution_axis`. For each candidate direction, it projects the normals into the perpendicular plane and fits the common point their lines pass through (on a surface of revolution every normal line meets the axis). The score is the fraction of points within 2% of the diameter. A 600-direction hemisphere search plus local refinement picks the axis. `axes.closed_end_sign` puts the end with points near the axis (a disc, the base) down; an open rim has none. In the cloud frame, the rim is visibly up on 10/12 mugs (figures/upright_cloud_top.png; obj_000001 and 007 are unclear). The normal-tensor axis (smallest eigenvector of sum n n^T) failed (4/12 within 10 deg), because base, inner-wall and handle normals corrupt it.

After dropping in that orientation in MuJoCo, tilt from vertical is <= 12 deg on 9/12. obj_000005, 006 and 025 tip over (53, 58, 27 deg) because their CoACD hulls have rounded bases.

Handle as the residual of the body primitive (`rev_residual.py`): along the axis, 40 slabs; the body's outer radius per slab is the 90th percentile of the radius of points whose normal line meets the axis, median-filtered. Points more than 3h beyond it are kept, and only the dominant azimuth cluster is retained. The parameters are first-try values and were not tuned on these mugs.

| Handle detector (no labels) | IoU >= 0.5 | mean IoU |
|---|---|---|
| Surface-of-revolution residual (geometry only) | **11/12** | **0.78** |
| FitPart + SAM 3 "handle" | 9/12 | — |
| Reeb arcs along the symmetry axis, best of 16 (cell, window, solid) settings | 3/12 | 0.28 |
| BITSI v0 segments (best segment) | 2/12 | — |

The Reeb approach fails for a measurable reason: in each horizontal slice the handle is about 5 mm from the body wall, which is below a grid cell that doesn't also fragment the sparse surface. The residual doesn't need the gap.

## 15. Primitive decomposition on PartNet-E rotationally symmetric categories (`primitive.py`)

Categories were fixed before any results: Bottle, Bucket, Kettle, KitchenPot, Lamp, Globe, Pen, TrashCan, Dispenser. 177 objects (the first 20 test objects per category; KitchenPot has 17). Metric: class-agnostic Hungarian mIoU, with instances plus one "rest" region. Bootstrap resamples objects within each category, 3000 rounds. Script: `scripts/eval_primitive.py`.

Method: fit the body as a surface of revolution (axis from normal lines; radius profile R(s)). Off-axis parts are the residual beyond R(s) + 3h. They are split into connected components, and a component is kept only if it covers <= 50% of the azimuth, i.e. it breaks the symmetry. The rest of the body is cut along an axis.

| Method | class-mean mIoU |
|---|---|
| **Primitive residual parts + HOG events on the body (longest PCA axis)** | **0.558** |
| Primitive residual parts + HOG events along the symmetry axis | 0.554 |
| HOG events, longest PCA axis (previous best training-free) | 0.539 |
| HOG events along the symmetry axis, no residual | 0.509 |
| BITSI v0 (existing pipeline), same 177 objects | 0.432 |
| Primitive: Chebyshev cuts of R(s) + residual | 0.443 |
| Whole object | 0.340 |

- Residual + HOG (PCA axis) vs HOG: **+0.018 [+0.009, +0.027]**, better in 5 of 9 categories. Kettle +0.17, Globe +0.02, KitchenPot +0.03, Bucket +0.01. Lamp −0.03, TrashCan −0.02, Dispenser −0.01, Pen −0.004.
- Adding the residual on the same symmetry axis: **+0.045 [+0.033, +0.058]**. The off-axis residual is the part of the method that works.
- Chebyshev inflection cuts of R(s) miss lid and cap boundaries (0.443). On a pot the lid shows up as a step in R(s) (figures/primitive_qual.png), so cuts at steps, not inflections, are the next thing to test.
- Where the primitive does not apply, it hurts: Lamp (mean axis score 0.45) and TrashCan (0.34, many are boxy). The axis score is a label-free applicability signal, but no gate has been tested yet.

Development note: the azimuth-coverage rule and the "decide residual parts before cutting the body" order were added after a smoke test on 4 objects per category of the same categories. Those 36 objects are part of the 177.

## 16. BITSI v2: shapes as trees of swept primitives (`bitsi2/`, 2026-10-06)

Primitive = spine + cross-section field r(s, phi).
- Spine: the symmetry axis, the longest PCA axis, or a graph-geodesic principal curve for children.
- Cross-section: per-slab Fourier series in phi. The order (<= 4) is chosen by a truncated-loss BIC, and protruding angle bins are rejected one-sided.

Off-axis children are envelope residuals that break the symmetry (<= 50% of the azimuth). They are fit recursively with geodesic spines. Coaxial cuts are significant (z > 3) normal/colour histogram events along the root spine. Topology: the number of contact patches between a child and its parent; 2+ means a loop.

**Development** (177 objects, first 20 per category of Bottle, Bucket, Kettle, KitchenPot, Lamp, Globe, Pen, TrashCan, Dispenser; settings were chosen here; `scripts/eval_bitsi2.py`):

| Variant | class-mean mIoU |
|---|---|
| first version: section-step + histogram cuts, children cut, revolution spine | 0.432 |
| histogram cuts only | 0.583 |
| + children kept whole | 0.590 |
| **+ root spine chosen by least residual (frozen defaults)** | **0.596** |
| + radius-step cuts (z > 3 / z > 5) | 0.550 / 0.586 |
| order 0 only (surface of revolution) | 0.564 |
| reference on the same objects: BITSI v1 primitive / HOG events / BITSI v0 | 0.558 / 0.539 / 0.432 |

**Held-out test 1, the 36 other PartNet-E categories** (144 objects, 4 per category; `scripts/eval_bitsi2_heldout.py`):

| Method | mIoU |
|---|---|
| HOG events | 0.262 |
| BITSI v1 primitive | 0.258 |
| **BITSI v2** | **0.248** |
| BITSI v0 | 0.219 |

v2 − HOG: −0.014 [−0.028, −0.001], better in 17/36 categories. v2 − v0: +0.028 [+0.015, +0.043], better in 28/36. On non-swept objects (furniture, electronics) the primitive assumption is wrong and v2 loses to plain sweep events.

**Held-out test 2, unseen objects of the 9 development categories** (179 objects: test objects 21+; KitchenPot has none; Kettle has 1):

| Category | n | BITSI v2 | BITSI v1 | HOG | whole |
|---|---|---|---|---|---|
| Bottle | 29 | **0.672** | 0.657 | 0.660 | 0.452 |
| Bucket | 8 | 0.464 | 0.528 | **0.529** | 0.465 |
| Dispenser | 29 | 0.601 | 0.597 | **0.617** | 0.327 |
| Globe | 33 | **0.737** | 0.676 | 0.667 | 0.400 |
| Kettle | 1 | **0.886** | 0.632 | 0.526 | 0.190 |
| Lamp | 17 | 0.450 | **0.501** | 0.482 | 0.113 |
| Pen | 20 | **0.651** | 0.529 | 0.521 | 0.400 |
| TrashCan | 42 | 0.449 | 0.451 | 0.449 | 0.381 |

- Class mean: v2 0.614 vs HOG 0.557 (+0.057 [+0.041, +0.075]). This is inflated by the single Kettle object.
- Object-weighted: 0.589 vs 0.566, **+0.023 [+0.004, +0.041]**. BITSI v0 on the same 179 objects: 0.475 object-weighted, 0.444 class mean.
- Class mean without Kettle: +0.014 [−0.005, +0.033], not significant.
- The gains come from Globe and Pen. Bucket (thin wire handles are missed), Lamp (not a swept body) and Dispenser lose.

**Topology** (`scripts/eval_bitsi2_topology.py`, dev objects of Kettle, KitchenPot, Bucket):

| Child's GT part | loop (2+ patches) | 1 patch | 0 patches |
|---|---|---|---|
| handle | 22 | 11 | 10 |
| body/unlabelled (false child) | 21 | 9 | 0 |
| spout | 3 | 11 | 1 |
| lid | 3 | 3 | 0 |

- GT handles recovered as a child (IoU >= 0.5): 29/77.
- Spouts are mostly free ends (11/15). About half of handles are loops.
- False children are frequent and often look like loops, so "loop" alone is not yet a handle detector.

**Verdict:** v2 improves on sweep events for swept objects, significantly by object weighting but only modestly. It hurts on objects that are not swept, and on thin handles. Two next steps:
1. An applicability rule: fall back to sweep events when the root residual is large or the spine fit is poor.
2. Better child detection: thin structures, and rejecting body fragments.

## 17. Franka Panda on a table: segment, plan, grasp and pour (`sim/panda_scene.py`, `scripts/panda_pour.py`)

Setup:
- Robot: MuJoCo Menagerie Franka Panda. The gripper servo is stiffened 100x so a closed grip reaches the actuator's 100 N limit; the stock servo gives about 1 N on a 1 cm handle.
- Mug: placed on a table in its label-free upright pose and held by friction only.

Pipeline:
1. Segment the part. Candidate grasps on handle parts close across the handle.
2. Rank grasps by tau.
3. Take the first grasp with a collision-free IK solution, approach path and pour path, and RRT-Connect from home to the pre-grasp.
4. Execute on the joint servos: approach, close, lift 10 cm, then pour 90 deg about the rim screw axis.

Success means the final pose is within 20 deg / 3 cm of the commanded screw motion and the mug is held by both fingers. All methods share the same task screw, built from the GT handle side.

| Part used for the grasp | Success | Planned | Executed but failed (dropped/slipped) | No feasible grasp + plan |
|---|---|---|---|---|
| **BITSI v2 loop child (ours, no labels)** | **9/12** | 9 | 0 | 05, 06, 25 |
| Whole object (any point) | 7/12 | 11 | 07, 22, 23, 25 | 05 |
| FitPart + SAM 3 "handle" | 3/12 | 3 | 0 | 9 mugs |
| SAM 3 lifted directly | 2/12 | 2 | 0 | 10 mugs (empty or partial masks) |

- Every BITSI v2 grasp that was planned succeeded.
- Its 3 failures are the 3 mugs that tip over when placed upright, because their CoACD hulls have rounded bases (section 14).
- Whole-object grasps are always reachable but slip during the pour on 4/11 mugs.
- FitPart/SAM 3 handle masks are often partial or bleed into the body, so no across-the-handle grasp on them is reachable and collision-free.
- 12 mugs, one task: indicative, not yet statistically strong.

With the top-25 tau cutoff instead of all candidates: BITSI v2 7/12, whole 7/12, FitPart 2/12, SAM 3 2/12.

Videos: `figures/panda_pour_obj000021_{bitsi2,fitpart,whole}.mp4`, `figures/panda_pour_obj000002_bitsi2.mp4`.

## 18. Fixes after the method review (2026-10-06)

### 18a. Panda pour without label leakage, same samplers for all

Section 17 built every method's pour screw from the GT handle side, and applied the across-the-handle sampler only to part methods. Now:
- Each method's screw comes from its own output: pour over the rim opposite its part, or opposite its grasp when it has no part.
- Every method gets both samplers: across-the-part and plain antipodal.
- The GT handle feeds only an evaluation column: whether the pour goes away from the handle (within 30 deg).

| Part used | Screw motion done | Pours away from the handle | No plan |
|---|---|---|---|
| **BITSI v2 loop child** | 9/12 | **9/12** (all within 1 deg) | 05, 06, 25 (mugs that tip on placement) |
| Whole object | **10/12** | 5/12 (others 68–135 deg off) | — (22 and 23 slipped) |
| FitPart + SAM 3 "handle" | 5/12 | 5/12 | 7 mugs |
| SAM 3 direct | 2/12 | 2/12 | 10 mugs |

Without its handle, the whole-object baseline completes the motion more often but pours sideways or over the handle half the time. Videos: `figures/fair_pour_obj000021_{bitsi2,whole,fitpart}.mp4`, `figures/fair_pour_obj000004_{bitsi2,whole}.mp4`.

### 18b. Segmentation scores that see over-segmentation (`metrics.partition_scores`, `scripts/eval_partition.py`)

Symmetric mIoU = matched IoU / max(#GT, #pred), so every extra segment costs. ARI = adjusted Rand index. Class means.

| Split | Method | mIoU (GT-avg) | symmetric mIoU | ARI | segments |
|---|---|---|---|---|---|
| dev (177, GT 2.7 parts) | BITSI v2 | **0.596** | 0.246 | **0.364** | 7.4 |
| | HOG events | 0.539 | 0.214 | 0.322 | 8.8 |
| | BITSI v0 | 0.432 | 0.315 | 0.165 | 3.6 |
| | whole object | 0.340 | **0.340** | 0.006 | 1.0 |
| in-domain held-out (179, GT 2.4) | BITSI v2 | **0.614** | 0.268 | **0.372** | 7.7 |
| | BITSI v1 | 0.571 | 0.193 | 0.361 | 10.9 |
| | HOG events | 0.557 | 0.220 | 0.350 | 9.4 |
| | BITSI v0 | 0.444 | 0.326 | 0.218 | 3.5 |
| | whole object | 0.341 | **0.341** | 0.008 | 1.0 |
| out-of-domain (144, GT 9.1) | BITSI v2 | 0.248 | 0.092 | 0.089 | 10.1 |
| | HOG events | **0.262** | 0.129 | **0.136** | 7.2 |
| | BITSI v0 | 0.219 | 0.178 | 0.087 | 3.6 |
| | whole object | 0.198 | **0.198** | 0.007 | 1.0 |

In-domain held-out comparisons:
- v2 − HOG: symmetric mIoU +0.048 [+0.028, +0.068]; ARI +0.023 [−0.011, +0.054].
- v2 − v0: symmetric mIoU −0.058 [−0.081, −0.036]; ARI +0.154 [+0.126, +0.181].

**Reading:**
- Every sweep method over-segments about 3x (7–11 segments vs 2.4 GT parts). On symmetric mIoU even the whole object as one part beats them all, and BITSI v0's few cuts beat v2.
- ARI, which rewards grouping the right points together, ranks v2 first in-domain.
- The GT-averaged mIoU used in sections 15–16 hid the over-segmentation.
- The cut detector (histogram events, z > 3) is the weak point. It needs a part-count-aware stopping rule, not more channels.

## 19. BITSI v3: primitive-relative channels, Chebyshev pieces, MDL cuts (`bitsi3.py`, 2026-10-06)

v3 = BITSI v2 with the coaxial cut detector replaced:
- Channels along the spine, each divided by its noise. The noise is estimated from local-quadratic residual scatter and inflated by (1+rho)/(1-rho) for slab-to-slab autocorrelation. With plain first-difference noise, BIC split everywhere: residuals were 5–14x that estimate, and lag-1 autocorrelation was 0.4–0.7.
- Each candidate piece gets a Chebyshev series per channel, with BIC-chosen degree <= 8.
- An exact dynamic-programming partition minimises code length + penalty x log S per cut.
- Candidate channels: cross-section amplitudes a0, |c1..c4| ("section"); primitive-relative topological BITC B = sum rho_k chi_k, beta0, beta1 ("topo"); normal/colour histograms ("hist").

**Selection on dev** (177 objects; rule fixed before running: maximise mean(symmetric mIoU, ARI), with mIoU >= v2 − 0.03), excerpt:

| Variant | mIoU | sym mIoU | ARI | segments |
|---|---|---|---|---|
| **section, penalty 10 (selected)** | 0.582 | 0.382 | 0.407 | 4.9 |
| section, penalty 30 (fails the mIoU rule) | 0.566 | 0.434 | 0.419 | 4.0 |
| section + topo, penalty 30 | 0.577 | 0.381 | 0.377 | 4.5 |
| topo, penalty 10 | 0.575 | 0.353 | 0.333 | 4.9 |
| section + topo + hist, penalty 30 | 0.560 | 0.271 | 0.266 | 6.4 |
| v2 | 0.596 | 0.246 | 0.364 | 7.4 |

The histogram channels over-split under MDL. The topological BITC channels do not add to the cross-section channels.

**Held-out, scored once with frozen defaults:**

| Split | Method | mIoU | sym mIoU | ARI | segments |
|---|---|---|---|---|---|
| in-domain (179, GT 2.4) | **BITSI v3** | 0.607 | **0.414** | **0.459** | 5.3 |
| | BITSI v2 | **0.614** | 0.268 | 0.372 | 7.7 |
| | HOG events | 0.557 | 0.220 | 0.350 | 9.4 |
| | BITSI v0 | 0.444 | 0.326 | 0.218 | 3.5 |
| | whole object | 0.341 | 0.341 | 0.008 | 1.0 |
| out-of-domain (144, GT 9.1) | BITSI v3 | 0.232 | 0.099 | 0.098 | 9.0 |
| | HOG events | **0.262** | 0.129 | **0.136** | 7.2 |
| | BITSI v0 | 0.219 | 0.178 | 0.087 | 3.6 |

In-domain comparisons:

| v3 minus | mIoU | sym mIoU | ARI |
|---|---|---|---|
| v2 | −0.007 [−0.033, +0.019] | **+0.147 [+0.114, +0.180]** | **+0.086 [+0.040, +0.132]** |
| HOG | — | +0.195 [+0.161, +0.227] | +0.109 [+0.072, +0.149] |
| v0 | +0.163 [+0.132, +0.196] | +0.089 [+0.052, +0.127] | +0.240 [+0.189, +0.290] |

Out-of-domain, v3 still loses to HOG (sym mIoU −0.030, ARI −0.037). There is no applicability rule yet.

**Robot** (fair setup, section 18a): v3 completes the motion on 9/12 mugs and pours away from the handle on all 9. The children are the same as v2's, so this matches v2. The same 3 tipped mugs fail.

Videos:
- `figures/bitsi3_explainer_kettle.mp4`: the sweep, the Chebyshev pieces, the MDL cut at the lid step, the children with attachment topology.
- `figures/v3_pour_obj0000{21,03}_bitsi3.mp4`.

## 20. BITSI v4: children, applicability and child cuts by code length (`bitsi4.py`, 2026-10-07)

**Protocol, fixed before any v4 run.**
- Changes under test (see `bitsi4.py` docstring): (a) children decided by description length (absorb
  into the body's cross-section vs own swept primitive) instead of the 3h / 1% / 50%-of-angle rules;
  (b) applicability gate: fall back to sweep events (longest PCA axis) when the fraction u of points
  neither inside the body envelope nor in a child exceeds `gate`; (c) MDL cuts inside children.
- Development objects: dev (first 20 of the 9 swept categories, 177) + ooddev (objects 5-8 of each
  of the other 36 categories, 142; new, now that the full PartNet-E test set is local).
- Grid: children {v3 rule, mdl} x child cuts {no, yes} x gate {inf, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3}; penalty 10.
- Selection rule: maximise the category mean over the 45 dev + ooddev categories of
  mean(symmetric mIoU, ARI), subject to dev GT-averaged mIoU >= v3's dev value - 0.03.
- Held-out, scored once with the frozen choice: indomain (179), outdomain (first 4 of the 36, 144),
  oodrest (all remaining objects of the 36, 1264). Penalty sensitivity {3, 10, 30} is reported on
  held-out for the frozen variant, not used for selection.

**Selection on dev + ooddev** (319 objects; `results/all/v4_dev.jsonl`, `scripts/select_bitsi4.py`), excerpt:

| Variant | gate | dev mIoU | dev sym | dev ARI | ooddev sym | ooddev ARI | objective |
|---|---|---|---|---|---|---|---|
| v3 | | 0.582 | 0.382 | 0.407 | 0.108 | 0.089 | 0.158 |
| sweep events (fallback) | | 0.539 | 0.214 | 0.322 | 0.124 | 0.126 | 0.154 |
| **children by code length, no child cuts (selected)** | inf | 0.577 | **0.402** | 0.431 | **0.169** | 0.098 | **0.190** |
| same | 0.1 | 0.581 | 0.378 | 0.432 | 0.129 | 0.120 | 0.181 |
| same, with child cuts | inf | 0.564 | 0.363 | 0.431 | 0.167 | 0.097 | 0.185 |
| v3 children + gate 0.02 | 0.02 | 0.582 | 0.374 | 0.418 | 0.124 | 0.124 | 0.178 |

- v3 children, no cuts, no gate reproduces v3 exactly (sanity check of the refactor).
- The gate does not help: u does not separate swept from non-swept objects; any threshold that sends
  the non-swept objects to the fallback also sends many swept ones (fallback rate 0.54 at 0.1).
- Child cuts over-split children (handles, spouts).
- Frozen v4 = code-length children, no child cuts, no gate, penalty 10 (now the `BITSI4()` defaults).

**Held-out, scored once with the frozen v4** (`results/all/v4_heldout.jsonl`; class means; 95% bootstrap
intervals, objects resampled within category):

| Split | Method | mIoU | sym mIoU | ARI | segments |
|---|---|---|---|---|---|
| in-domain (179) | **BITSI v4** | **0.621** | **0.444** | **0.482** | 4.0 |
| | BITSI v3 | 0.607 | 0.414 | 0.459 | 4.9 |
| | sweep events | 0.557 | 0.220 | 0.350 | 9.3 |
| out-of-domain, all 1408 objects of the 36 other categories | **BITSI v4** | 0.228 | **0.175** | 0.099 | 3.9 |
| | BITSI v3 | 0.232 | 0.105 | 0.096 | 8.8 |
| | sweep events | **0.245** | 0.124 | **0.113** | 7.0 |

- In-domain, v4 - v3: sym mIoU +0.030 [+0.015, +0.044], ARI +0.023 [+0.015, +0.031], mIoU +0.014 [+0.007, +0.022].
- Out-of-domain, v4 - v3: sym mIoU +0.071 [+0.064, +0.077], ARI +0.003 [-0.004, +0.011].
- Out-of-domain, v4 - sweep events: sym mIoU +0.052 [+0.043, +0.060] but ARI -0.014 [-0.027, -0.000],
  mIoU -0.017 [-0.025, -0.010]. v4 now segments close to the part count everywhere (3.9 segments), which
  wins on symmetric mIoU; on non-swept objects sweep events still group points slightly better.
- v4 is the first BITSI version that beats v3 on every in-domain metric and that beats sweep events on
  any out-of-domain metric.

## 21. Eye-in-hand partial views: where to look next (`eyeinhand.py`, `scripts/eval_eyeinhand.py`, 2026-10-07)

**Protocol, fixed before the run** (pipeline debugged on dev objects only).
- Objects: held-out PartNet-E objects of tabletop categories (category size <= 0.6 m): in-domain (179)
  + out-of-domain first 4 of each of 23 tabletop non-swept categories (92); 271 total. Upright on the
  table at category metric size, random yaw.
- Sensor: D405-like wrist camera (424 x 240, 87 x 58 deg), splat z-buffer visibility, depth noise
  0.5 mm + 2 mm z^2, per-view registration error 1 mm / 0.3 deg, TSDF-like 3 mm voxel-averaged fusion.
- Robots: Franka Panda; Kinova Gen3 + Robotiq 2F-85 (added on request). Allowed viewpoints = the
  96-view grid (az 15 deg, el 15/35/55/75, distance fitting the object in view) filtered by IK +
  collision checks for that robot.
- Strategies (all start at the same HOME view from the robot side): orbit (fixed coarse-to-fine
  azimuths), nbv_cov (unobserved-voxel next-best-view), nbv_bitsi (ours: view that sees the most
  unobserved cells of the surface predicted by the fitted BITSI v4 primitives); "all reachable" as the
  upper bound.
- Segmenter: frozen BITSI v4 (no change for partial data); BITSI v3 and sweep events at K = 1, 3, 8.
- Primary comparison: nbv_bitsi vs nbv_cov and vs orbit at K = 3 views, on full-object symmetric mIoU
  and ARI (labels carried to every surface sample, so unseen parts cost), in-domain and out-of-domain
  separately, bootstrap 95% intervals. Secondary: curves over K, coverage, camera travel, the
  adaptive stop of nbv_bitsi (< 5% of predicted cells unobserved).
- Small-scale debugging note: concatenation fusion left a ~2 mm-thick doubled surface and made MDL cuts
  unstable; switched to voxel averaging before the run. A gravity-aligned root-axis candidate was
  tried on 6 dev objects, changed nothing, and was removed.

## 22. Robot looks, segments, grasps and pours from its own wrist-camera cloud (`scripts/eih_pour.py`)

Panda with a wrist camera (MuJoCo depth + segmentation renders; same noise, registration error and
3 mm fusion as section 21) on 36 HANDAL meshes (12 mugs, 12 measuring cups, 12 pots/pans). The robot
drives through up to 6 BITSI next-best views (RRT-Connect between views); BITSI v4 (loop child),
Find3D ("handle of a <category>") and the whole object then all use the SAME fused partial cloud for
the grasp + pour of section 18a (task success = pour completed AND away from the GT handle).

Two mounting bugs found on the first mug (obj_000021) before any run: the camera sat on the finger
axis (a finger covered 7.5% of every image; moved to the side of the hand: object pixels x3).

**Run 1 (frozen v4):**

| Category | BITSI v4 task success | Find3D | Whole object |
|---|---|---|---|
| mug | 3/12 | **7/12** | 1/12 |
| measuring cup | 0/12 | **5/12** | 0/12 |
| pots/pans | 3/12 | 3/12 | 3/12 |
| all | 6/36 | **15/36** | 4/36 |

BITSI's selected part rarely overlaps the handle on partial clouds (IoU ~0.1 on most objects).
Diagnosis on two mugs: (a) the root axis flips to horizontal (PCA) when the surface is incomplete;
(b) a thin handle's child test is narrowly rejected (405 vs 393 nats).

**Fix (a) checked on PartNet-E dev objects, not on HANDAL:** the gravity axis (table normal) as a third
root-axis candidate, eye-in-hand orbit views, 177 dev objects, full-object scores:
K = 3: sym mIoU 0.408 -> 0.421 (+0.013 [+0.004, +0.024]), ARI +0.002 [-0.016, +0.021];
K = 6: sym +0.005 [-0.003, +0.013], ARI +0.009 [-0.003, +0.021]. Never worse; adopted for the robot
setting only (`BITSI4(up=...)`; off for canonical clouds). Section 21's pre-registered runs keep frozen v4.
Failure (b) is not addressed.

**Section 21 results, Panda** (`results/eih/panda.jsonl`; 270 of 271 objects, one has no reachable
view; class means, full-object scores = labels carried to every surface sample):

| | in-domain sym / ARI | out-of-domain sym / ARI | in-domain coverage |
|---|---|---|---|
| 1 view (HOME, all strategies) | 0.383 / 0.246 | 0.200 / 0.092 | 0.180 |
| orbit, K = 3 | **0.438 / 0.353** | 0.185 / 0.071 | 0.396 |
| volumetric NBV, K = 3 | 0.389 / 0.279 | 0.193 / 0.069 | 0.427 |
| BITSI NBV (ours), K = 3 | 0.374 / 0.270 | 0.183 / 0.062 | 0.425 |
| orbit / vol. NBV / BITSI NBV, K = 8 | 0.432 / 0.430 / 0.452 (sym) | 0.195 / 0.180 / 0.176 (sym) | 0.585 / 0.568 / 0.553 |
| all reachable views (~50 in-domain) | 0.485 / 0.418 | 0.179 / 0.092 | 0.667 |
| full cloud, no occlusion (section 20) | 0.444 / 0.482 (seen = all) | 0.175 / 0.099 | 1.0 |

**Primary comparison (pre-registered, K = 3): negative for BITSI NBV.**
- In-domain: BITSI NBV - volumetric NBV sym -0.015 [-0.043, +0.013], ARI -0.009 [-0.050, +0.035];
  BITSI NBV - orbit sym -0.065 [-0.090, -0.040], ARI -0.084 [-0.126, -0.041].
- Out-of-domain: all differences within +-0.01, intervals include 0.

Reading:
- The fixed orbit's second view is the opposite side; both NBVs choose low, adjacent views that add
  as much coverage but constrain the root axis less. Segmentation depends on the axis fit more than on
  coverage at small K.
- v4's quality is not monotonic in K (e.g. BITSI NBV 0.432 at K = 2, 0.374 at K = 3, 0.452 at K = 8):
  MDL cuts on partial clouds are unstable.
- The adaptive stop (< 5% of predicted cells unobserved) almost never fires (15% in-domain, 0% out):
  cells near the table and behind handles are never observable; the criterion needs a reachability-
  aware definition.
- v4 stays ahead of v3 on the same partial clouds in-domain (K = 8 sym 0.452 vs 0.419) and of sweep
  events on sym mIoU; out-of-domain, sweep events are slightly better on ARI, as on full clouds.

**Section 21 results, Kinova Gen3** (`results/eih/kinova.jsonl`): 39 of 271 objects have no reachable
view for the Gen3 at this table placement (excluded); the rest reach 38 (in-domain) / 47 (out) of 96
views on average vs 50 / 63 for the Panda, and never the far side of the object.

| K = 3, full object | in-domain sym / ARI (145) | out-of-domain sym / ARI (87) |
|---|---|---|
| 1 view (HOME) | 0.385 / 0.268 | 0.208 / 0.081 |
| orbit | 0.390 / 0.312 | 0.212 / 0.109 |
| volumetric NBV | 0.406 / 0.301 | 0.203 / 0.092 |
| **BITSI NBV (ours)** | **0.445 / 0.356** | 0.209 / 0.099 |
| all reachable (~38 / 47 views) | 0.467 / 0.403 | 0.182 / 0.098 |

**Primary comparison, Kinova: positive in-domain.** BITSI NBV - volumetric NBV: sym +0.038 [+0.020, +0.057],
ARI +0.055 [+0.024, +0.085]; BITSI NBV - orbit: sym +0.054 [+0.031, +0.076], ARI +0.044 [+0.003, +0.084].
Out-of-domain: all within +-0.01, intervals include 0.

**Across robots:** BITSI NBV helps when the reachable viewpoints are restricted (Gen3: the orbit's
planned views are replaced by nearest reachable ones, and the shape model's prediction of where the
surface is missing picks better substitutes), and loses to the opposite-side orbit when nearly the
whole view sphere is reachable (Panda). The pre-registered claim "BITSI NBV beats volumetric NBV at
K = 3" holds for one of two robots, in-domain only. Different object subsets per robot (reachability).

**Run 2 (v4 + gravity axis, robot setting):** BITSI v4 5/36 (mug 2, cup 0, pan 3), Find3D 12/36
(mug 5, cup 6, pan 1), whole 4/36. No object where BITSI succeeds and both others fail. BITSI's part
vs the GT handle: median IoU 0.12.

Why (two mugs, saved fused clouds): with half the surface missing, the least-residual root-axis choice
prefers a horizontal axis even when the vertical one is offered (a one-sided surface has a smaller
"outside the envelope" fraction about the wrong axis), and the child test for thin handles stays just
below break-even (lower candidate margin or a larger graph radius do not change it). Both rules assume
complete cross-sections; partial data needs them to score only observed angle bins. Not attempted on
these test objects. Conclusion: in this eye-in-hand pour task Find3D (given the part name) is the
better part detector; BITSI's handle detection does not survive partial, noisy depth.

**Find3D on the same partial PartNet-E clouds (Panda, full-object sym / ARI):** in-domain, K = 3: v4
0.374-0.438 / 0.270-0.353 vs Find3D 0.334-0.337 / 0.028-0.049 across the three strategies; K = 8:
v4 0.430-0.452 / 0.355-0.398 vs Find3D 0.341-0.350 / 0.047-0.065. Out-of-domain the two are within
~0.02 on sym mIoU (Find3D slightly ahead at K = 8), v4 ahead on ARI. So on partial clouds BITSI v4
still segments PartNet-E better than Find3D overall; what fails in section 22 is specifically the
handle-as-child detection on the HANDAL meshes. (Per-cell n differs where an object has fewer
reachable views than K; `scripts/score_eih_baselines.py`.)

**Full scans instead of the robot's partial view** (`scripts/full_pour.py`; same 36 objects, 20k mesh
samples, same grasp + pour; task success = pour completed away from the handle):

| Category | BITSI v4 | Find3D | Whole object |
|---|---|---|---|
| mug | **8/12** | 5/12 | 3/12 |
| measuring cup | 4/12 | **5/12** | 1/12 |
| pots/pans | 2/12 | 2/12 | 2/12 |
| all | **14/36** | 12/36 | 6/36 |

BITSI succeeds where both others fail on 6 objects (cups 16, 20; mugs 22, 23, 24; pan 43); Find3D
succeeds where BITSI fails on 6 (cups 11, 12, 18; mug 6; pans 31, 33). Roughly a tie with Find3D
overall (n = 36), ahead on mugs. With the robot's own partial views BITSI drops to 5-6/36 (above).
Re-running single objects is not fully deterministic: re-rendered with video, mug 22's Find3D run
succeeded (it failed in the batch run). Videos are only shown for objects whose outcome reproduced.
Source of the run-to-run variation: Find3D's own preprocessing draws an unseeded random 5000-point
subsample, so its handle mask differs between runs; BITSI and the planner are deterministic. Repeats:
pan 43 (BITSI succeeds, Find3D fails, whole has no plan: 3/3 runs) and mug 24 (BITSI succeeds, Find3D
and whole pour toward the handle: 2/2 runs) reproduce; cup 16 (Find3D failed 2 of 3 runs) and mugs 22,
23 (Find3D succeeded on re-run) do not. Videos: `figures/pour_videos/compare_{pan_obj000043,mug_obj000024}.mp4`.
The 36-object counts above are single runs; Find3D's count varies by about +-2 between runs.

## 23. BITSI v4 vs every baseline (full clouds; `scripts/summarize_all.py`, 2026-10-07)

All methods on the same objects; class means; v4 - method with 95% bootstrap intervals. A method that
raises on an object is scored as returning the whole object (`FAIL_AS_WHOLE=1`; v0: 39 ShapeNetPart
objects, superquadrics: 247 ShapeNetPart + 40 PartNet-E); dropping those objects instead changes no
conclusion. Find3D gets the GT part names of the category as queries; PartField is run with the GT
part count (oracle K, its KMeans protocol) and, as a control, with K = v4's own segment count.

**sym mIoU / ARI:**

| Method | PartNet-E in-domain (179) | PartNet-E out-of-domain (1408) | ShapeNetPart (2543) |
|---|---|---|---|
| **BITSI v4 (ours, no labels, no training)** | 0.444 / **0.482** | 0.175 / 0.099 | 0.379 / 0.262 |
| BITSI v3 | 0.414 / 0.459 | 0.105 / 0.096 | 0.288 / 0.257 |
| BITSI v0 (original) | 0.326 / 0.218 | 0.168 / 0.080 | 0.294 / 0.178 |
| sweep events | 0.220 / 0.350 | 0.124 / 0.113 | 0.309 / 0.100 |
| EMS superquadrics | 0.417 / 0.227 | 0.191 / 0.089 | 0.236 / 0.144 |
| Find3D (GT part names) | 0.352 / 0.071 | 0.200 / 0.039 | 0.373 / 0.212 |
| PartField (oracle K) | **0.544** / 0.407 | **0.363** / **0.284** | **0.546** / **0.427** |
| PartField (K = v4's count)* | 0.350 / 0.292 | 0.261 / 0.278 | 0.436 / 0.382 |

\* control run on held-out PartNet-E in-domain + first 4 per out-of-domain category (144) and 8 per
ShapeNetPart category (128); v4 on the same objects: 0.444 / 0.482, 0.180 / 0.108, 0.378 / 0.263.

- v4 beats every training-free baseline and Find3D on ARI everywhere, and BITSI v0 on every metric.
- PartField with oracle K is the strongest method overall. With the SAME number of segments as v4,
  v4 is better in-domain (sym +0.094 [+0.066, +0.122], ARI +0.190 [+0.140, +0.240]) and worse out of
  domain (sym -0.080, ARI -0.170) and on ShapeNetPart (sym -0.058, ARI -0.119). Learned features
  group points better on general objects; BITSI's swept-primitive model wins on swept, tool-like ones.
- CoACD: see below when complete (subset: 8 per ShapeNetPart category + held-out PartNet-E).

## 24. BITSI v5: axis and child rules for incomplete cross-sections (`bitsi5.py`, 2026-10-07)

Motivation: section 22 (two diagnosis mugs, HANDAL; no tuning on them): on partial views the
least-residual axis choice prefers a wrong axis, and the spine cost (3 numbers per slab) makes thin
handles fail the child test. Changes (see `bitsi5.py`): (a) root axis = candidate with the highest
normal-line consistency if >= axis_tau, else v4's least-residual rule; (b) child spine coded as a
Chebyshev series per coordinate (BIC degree <= 8).

**Protocol, fixed before the dev run.**
- Grid: axis rule {v4 residual, normals with tau in {0.15, 0.20, 0.25, 0.30}} x spine coding {v4 free,
  Chebyshev} (10 variants; v4 = residual + free).
- Dev data: (i) full clouds, dev + ooddev (319); (ii) eye-in-hand clouds (orbit, K = 3 and 6, gravity
  axis offered as in the robot setting), dev + tabletop ooddev objects.
- Objective = mean of [full-cloud mean(sym mIoU, ARI)] and [eye-in-hand full-object mean(sym mIoU, ARI),
  averaged over K = 3, 6], each a category mean. Constraint: full-cloud objective >= v4's - 0.005.
- Held-out, scored once with the frozen choice: full clouds (indomain, outdomain, oodrest), the section-21
  eye-in-hand runs for both robots (same views), and the section-22 HANDAL pour (partial + full).

**CoACD** (complete; subset because it takes ~5 min per object: first 8 per ShapeNetPart category, first 2
held-out objects per PartNet-E category; default threshold 0.05 on a Poisson surface of the cloud):
ShapeNetPart (128) sym / ARI / segments 0.062 / 0.115 / 89.8 vs v4 0.378 / 0.263 / 2.9; PartNet-E
held-out (87) 0.051 / 0.103 / 62.5 vs v4 0.224 / 0.165 / 3.9 (v4 - CoACD sym +0.173 [+0.151, +0.196],
ARI +0.063 [+0.025, +0.100]). CoACD makes collision pieces, not parts: ~60-90 pieces per object.

**Dev result (319 objects; 267 with eye-in-hand clouds; `results/all/v5_dev.jsonl`, `scripts/select_bitsi5.py`):**

| Variant | full-cloud obj. | eye-in-hand obj. | objective |
|---|---|---|---|
| v4 (residual axis, free spine) | 0.190 | **0.222** | 0.206 |
| normals tau 0.25, free spine (selected by the rule) | **0.197** | 0.217 | **0.207** |
| normals tau 0.20, free spine | 0.199 | 0.215 | 0.207 |
| residual axis, Chebyshev spine | 0.189 | 0.219 | 0.204 |
| normals tau 0.25, Chebyshev spine | 0.197 | 0.215 | 0.206 |

The rule selects the normal-line axis (tau 0.25) with v4's spine coding, by +0.001 over v4: within
noise. The axis rule helps full clouds (+0.007) and hurts the partial clouds it was designed for
(-0.005); the Chebyshev spine coding never helps (more true thin handles, more spurious children).
The two diagnosis mugs were not representative. Neither change fixes the partial-view failure;
held-out evaluation not run yet (pending decision).

## 25. Gripper-aware root axis (`BITSI5(axis_rule="gripper")`, requested, 2026-10-07)

Rule: the root axis is the minimum-volume-OBB axis along which a parallel jaw (opening G = 8 cm,
Panda) can wrap the object: slice along each OBB axis (24 slices); a slice is wrappable when the
narrower of its two cross-section widths (2-98% extent, metres) <= G; keep the axis with the most
points in wrappable slices (ties within 0.05 -> longer axis); none wrappable -> v4's rule. Full clouds
use the category size prior for metric scale.

On the 5 dumped eye-in-hand cases (diagnosis only): measuring cup 12 handle segment IoU 0.26 -> 0.95
(axis through the handle), mug 21 0.54, mugs 3 and 5 unchanged (an 8-9 cm mug body fits the jaw on
every axis, so the rule cannot discriminate).

**Protocol, fixed before running:** dev grid and objective of section 24 with two added variants
(gripper/free, gripper/cheb). For grasping, the part is: the loop child if any; else the thinnest
coaxial segment (smallest OBB l3) other than the largest segment, with >= 2% of the points; else
the whole object. Pour tests (36 HANDAL objects, partial and full scans) are run for the dev-selected
variant AND for gripper/free regardless (it was requested), reported as is.

**Pour, robot's own partial views (36 HANDAL objects, `results/eih_pour/viz5/runs.jsonl`):** BITSI v5g
(gripper-wrap axis + the fixed part rule) 6/36 (cup 2, mug 2, pan 2), Find3D 13/36, whole 4/36;
BITSI v4 earlier 5/36. When the body is wider than the jaw the rule finds the handle (cup 12: IoU
0.95, success; cup 13: 0.97); when the whole object fits the jaw (most mugs) every axis scores
~1.0 and the thinnest segment is a rim slice (mug 3: IoU 0.00, poured toward the handle).
Videos: `figures/eih_gripper_axis/` (one success, one failure; chosen to illustrate, not sampled).

**Dev segmentation (section-24 objective, 319 objects):** gripper/free full 0.195, eye-in-hand 0.188,
objective 0.192; gripper/cheb 0.188; vs v4 0.206 and normals0.25/free 0.207. As a segmentation axis the
gripper-wrap rule is worse on partial clouds (-0.034 vs v4): the axis the jaw can wrap is often not
the axis along which parts are stacked. It is a grasp-axis rule, not a part-segmentation axis rule.

## 26. Kinova Gen3 + Robotiq 2F-85 grasp-and-pour (`sim/kinova_scene.py`, 2026-10-07)

`KinovaScene` is a drop-in for `PandaScene` (shared table / object / cameras via
`panda_scene.add_environment`; `ROBOT=kinova` in `eih_pour.py`, `full_pour.py`; `run(..., robot=)`).
Bugs found and fixed while bringing it up (mug 21 / mug 4, full scans):
- gripper convention: the 2F-85 closes at ctrl 255 (Panda: 0); `grip()` maps Panda-convention commands;
- the 2F-85 tendon actuator (5 N) made 4x stiffer/stronger; 20x (as for the Panda servo) fought the
  soft 4-bar constraints and the fingers stopped 4 cm apart in free space;
- Gen3's `base_link` has no joint, so MuJoCo does not filter its contacts with `shoulder_link`: friction
  locked joint 1 (0.44 rad short, motor saturated at 105 Nm, TCP 22 cm off). Excluded the pair -> TCP
  error 1 mm. (Section 21's Kinova reachability used IK + checks that ignore robot-robot contacts, so it
  is unaffected.)
- gravity compensation on (the real Gen3 controller has it; it did not cause the error above).
Check: mug 21 full scan, BITSI v4 task success (5 deg / 1.1 cm off the commanded pour); whole object
held but 86 deg off.

**Kinova pour results (36 HANDAL objects; task success = pour completed away from the handle):**

| | BITSI v4 | Find3D | whole object |
|---|---|---|---|
| Kinova, full scans | **7** (mug 6, cup 0, pan 1) | 3 | 4 |
| Panda, full scans (section 22) | **14** | 12 | 6 |
| Kinova, own wrist views | 4 (mug 3, pan 1) | **9** | 1 |
| Panda, own wrist views (run 2) | 5 | **12** | 4 |

Same ordering as on the Panda: BITSI leads on full scans, Find3D on partial views. All rates are
lower on the Kinova: fewer reachable, collision-free grasp plans (BITSI planned 20/36 full, 23/36
partial vs 29 / 25 on the Panda). Videos: `figures/eih_kinova/` (pan 43 and mug 23: BITSI succeeds;
cup 12: BITSI fails, Find3D succeeds; chosen to illustrate).

## 27. Task screws for 12 more tasks, Kinova trials (`fitpart/tasks/`, 2026-10-07)

Screws inferred from BITSI's parts (no labels): turn, hinge, slide, press, squeeze, pivot (procedural
articulated objects with exact joints; PartNet-Mobility needs a SAPIEN account), lift, swing, scoop,
stir, hand-over (HANDAL tools lying on the table), cut (procedural knife). Development (tuned on
instances 0-5 / first objects; trials use unseen instances 101-110 and all tool objects):
- BITSI's Fourier cross-section (order <= 4) cannot follow box corners, so knobs, handles and buttons on
  box-like objects were lost in the corner residual; added a rectangular-section child detector (robust
  5-95% oriented box; points > 3h outside it) used when the round model finds no children.
- tool handles: the longest run of wrappable (<= 4.5 cm) slices along the tool's long axis (BITSI's
  segment-based rule picked hammer heads / drill barrels when the handle split into segments).
- tongs pivot: the end where the raised arm is lowest (geometry, not BITSI parts).
- success checks use ground truth only: joint displacement (turn >= 120 deg, hinge >= 45 deg, slide
  >= 8 cm, press >= 4 mm, squeeze >= 14 deg), tool pose (pivot >= 35 deg, swing/scoop >= 25 deg,
  stir >= 90 deg yaw, cut stroke >= 7 cm, lift: lowest point >= 10 cm up), held, and for tool use the
  executed grasp must be on the GT handle (hand-over: off the handle).
- Robotiq grip: 50 (12.5x stock) did not help lifting and broke the cap turn; kept 20.

**Trials, batch 1 (Kinova, unseen instances; `results/tasks/trials.jsonl`): 62 / 130.**

| task (object) | success | inferred-axis error vs GT (median) | main failure |
|---|---|---|---|
| turn (bottle cap) | **10/10** | 0.3 deg | - |
| press (button) | **10/10** | 8.5 deg | - |
| pivot (box) | **10/10** | - | - |
| slide (drawer) | **8/10** | 0.5 deg | 2 no feasible plan |
| stir (whisk) | 7/12 | - | 5 grasped off the handle |
| lift (drill) | 5/12 | - | not lifted / dropped |
| hinge (lid box) | 4/10 | 30.9 deg | hinge axis wrong on 6 |
| squeeze (tongs) | 4/10 | 0.1 deg | pivot line ~2 cm off -> grasp slips |
| hand-over (screwdriver) | 3/12 | - | dropped (grasp on the thin shaft) |
| scoop (ladle) | 1/12 | - | 8 no plan, 3 wrong grasp part |
| swing (hammer) | 0/12 | - | BUG: whole hammer taken as handle -> NaN axis |
| cut (knife) | 0/10 | - | BUG: zero stroke direction |

Two bugs found on these trial objects (so the re-run below is not fully held-out): the tool-handle rule
used the narrower cross-section width, so a hammer lying flat or a knife counted as "handle" end to
end; fixed with the wider width (a head / bowl is wide in one direction) and, for knives, the thicker
end. The cut check also measured the stroke along the inferred direction; now along the GT blade.

## 28. Language command -> screw (`fitpart/lang/`)

3900 template commands (13 tasks x 300), labels = task + direction (19 classes); magnitude by regex
(176/176 correct); grasp part from the task unless named. Test sets use templates and/or object nouns
never seen in training, plus 40 hand-written commands:

| test set | TF-IDF + LR (task / task+dir) | frozen SigLIP + MLP (task / task+dir) |
|---|---|---|
| same templates, new samples (780) | 1.000 / 1.000 | 1.000 / 1.000 |
| held-out templates (780) | 0.978 / 0.954 | 0.981 / 0.928 |
| held-out templates + held-out objects (780) | 0.787 / 0.786 | **0.905 / 0.827** |
| hand-written (40) | **0.950 / 0.950** | 0.925 / 0.850 |

SigLIP generalises to unseen object nouns, TF-IDF is better on direction words ("unscrew" vs
"tighten"); SigLIP was chosen for the robot demo (by the held-out templates + objects set, before any
robot run). End to end (14 fresh commands -> SigLIP slots -> BITSI screw -> Kinova): all 14 parsed
correctly (incl. "tighten" -> turn -1, "close the drawer" -> slide -1, "45 degrees", "about 10 cm");
executed successfully: turn x2, hinge, slide x2, press, pivot (45 deg asked, 44.5 achieved), stir;
the rest failed on the tool bugs above (re-run below).

**Trials after the tool fixes (`results/tasks/trials_v2.jsonl`; same objects, so not fully held-out,
except the 7 hammers never used before):** cut 10/10 (was 0), swing 10/12 (was 0) and 5/7 on the 7
unseen hammers, scoop 7/12 (was 1), lift 7/12 (was 5), stir 4/12 (was 7: the wider-width rule takes part
of the whisk's wire head as handle), hand-over 3/12 (unchanged: the screwdriver slips from the thin
shaft). Current per-task rates: turn 10/10, press 10/10, pivot 10/10, cut 10/10, slide 8/10, swing 10/12,
lift 7/12, scoop 7/12, hinge 4/10, squeeze 4/10, stir 4/12, hand-over 3/12 -> 87/130 trials.

**End to end (re-run after the fixes; `results/lang/e2e/`):** 14/14 commands parsed correctly, 9/14
executed successfully: turn (open), turn (tighten, -1), hinge, slide (10 cm asked, 9.6 achieved), slide
(close, -1), press, pivot (45 deg asked, 44.5 achieved), stir, cut. Failed: squeeze (8 of 14 deg), lift and
scoop (no feasible plan on those objects), swing (13 of 25 deg), hand-over (handle 64 deg off).
Videos: `figures/tasks/task_screws_reel.mp4` (one success per task, every one re-rendered and reproduced;
cards show each task's overall rate), `figures/tasks/language_commands_reel.mp4` (all 12 executed
commands, failures included).

**Swing and scoop corrected (user feedback): maximise the head's moment about the wrist.** The previous
screws pivoted about the handle's centroid (swing) or the bowl (scoop) with a random grasp on the handle.
Now the screw axis passes through the grasp (the wrist is the pivot; horizontal, perpendicular to the
tool) and grasp candidates are tried in order of the head's moment arm about it (perpendicular distance),
longest first, i.e. the far end of the handle; a candidate is kept only if its plan is feasible.
(`results/tasks/trials_lever.jsonl`)

| | swing (19 hammers) | scoop (12 ladles) |
|---|---|---|
| success, moment-ranked grasp, pivot at the grasp | **18/19** | 5/12 |
| success before (random grasp, pivot at handle centre / bowl) | 15/19 | 7/12 |
| median lever arm (longest feasible) | 28.5 cm (30.3) | 18.3 cm (27.8) |
| grasp position along the handle (1 = far end), median | 0.82 | 0.45 |
| head arc (lever x rotation), median | 15.7 cm | 9.5 cm |

Scoop: a 35 deg rotation about the far end of a long ladle often has no feasible arm plan (4/12), and when
executed the ladle turns inside the 2F-85 grip (rotation 13-25 deg while held): the larger moment arm is
also a larger torque at the fingers. Video: `figures/tasks/swing_scoop_lever_reel.mp4`.

**Reverted (user request):** the moment-arm change above (pivot at the grasp, grasps ranked by lever arm)
is undone; swing and scoop are back to the section-27 screws (swing: pivot at the handle centroid; scoop:
pivot at the bowl; random grasp order), so the current rates are those of `trials_v2.jsonl`
(swing 10/12, scoop 7/12). Task videos now show the screw: in the scene (red axis + curl for rotations,
arrow for translations, drawn where it acts, after the lift) and in a side panel the grasp heatmap as in
the pour videos: tau (ITSI SOCP screw metric, `sim/screw_metric.py`, now also for translations: a force
along the axis instead of a moment about it) of every candidate under the task screw, chosen grasp in
blue. tau is shown only; it does not change which grasp is executed. Mass/CoM for tau: HANDAL category
mass at the cloud centroid (tools), 0.25 / 0.10 kg (pivot box / knife), 0.05 kg at the moving part
(articulated objects).

## 29. OMPL motion planning (`sim/ompl_planner.py`, 2026-10-07)

Free-space joint motions now use OMPL's RRT-Connect (OMPL 1.7 Python bindings; 2.0.1's segfault here)
with path simplification, motion validation at 0.5% resolution, and self-collision via a MoveIt-style
allowed-collision matrix (robot pairs in contact in >= 95% of 200 random poses or at home are allowed).
Approach, lift and task-screw segments remain sequential-IK chains (as MoveIt's computeCartesianPath)
and are now collision-checked (approach: everything; lift/screw: everything but finger-object contact).
Full MoveIt was not used: it needs ROS 2, per-robot MoveIt configs and a MuJoCo bridge.

Same 130 Kinova trials (same objects, same grasp order): old planner 87/130, OMPL 92/130. Identical on
11 tasks, trial by trial; squeeze 4/10 -> 9/10 (5 fail -> success, none the other way; cause not
verified - the OMPL path to the pre-grasp differs from the old straight-line-first RRT). "No feasible
plan" failures are the same 9 objects with both planners: they come from IK / Cartesian-segment
feasibility, not free-space planning. The new checks rejected no plan the old planner executed.
OMPL is now the default in `tasks/run.py` (`PLANNER=rrt` restores the original).
Current per-task rates: turn 10/10, press 10/10, pivot 10/10, cut 10/10, squeeze 9/10, slide 8/10,
swing 10/12, lift 7/12, scoop 7/12, hinge 4/10, stir 4/12, hand-over 3/12.

## 30. Pivot screw baselines and 4-panel videos (`figures/four_panel/`)

Pivot ("tilt the box back 45 degrees", parsed by the language model: pivot, +1, 45 deg), Kinova, 10 unseen
boxes, same grasp part and axis direction, only the pivot point differs (`SCREW_BASELINE`):
BITSI (far bottom edge) 9/10 (41-55 deg), pivot at the gripper 0/10 (3-15 deg), pivot at the box centroid
0/10 (20-24 deg). The pivot rule uses the box's bottom edge and top, not BITSI's MDL cuts (which, on a
plain box, are cuts along a diagonal spine); the video says so.
Pour (Kinova, full scan, mug 21; "pour the water out of the mug" -> pour, +1, 90 deg, handle): BITSI
success, Find3D and whole object end in the wrong pose (outcomes reproduced on the re-render; BITSI's
handle there is a child with 1 attachment patch, used because no loop child was found).
Videos: `pivot_4panel.mp4`, `pour_4panel.mp4` (BITSI | baseline | baseline | BITSI's cuts on that object).

## 31. BITSI segments on partial clouds: what goes wrong (`figures/partial/partial_segments.png`)

Figure: 4 successes and 4 failures (in-domain, Panda, BITSI next-best views, K = 3; best/worst sym mIoU,
one per category), GT parts vs BITSI segments (coloured by the Hungarian-matched GT part; grey = extra
segments) vs the full object with the unobserved surface faint. v4 is re-fit on each saved cloud and
reproduces the logged score (within 0.001; the saved clouds are float32 / float16).
Successes: spine 1-6 deg from vertical. Failures: all over-segmented (5-8 segments vs 2 GT parts);
3 of 4 with a tilted spine (31, 89, 39 deg: the MDL cuts then slice across the body), 1 (globe on a legged
stand, spine 5 deg) from a non-swept stand cut into pieces.
Across all 177 in-domain partial clouds (`results/eih/partial_tilt.json`): spine 0-10 deg from vertical
(n = 121) sym 0.424, over-segmented 50%; > 10 deg (n = 56) sym 0.32-0.34, over-segmented 76-77%;
Spearman(tilt, sym mIoU) = -0.34. A tilted root axis costs ~0.1 sym mIoU, but over-segmentation also
affects half of the well-aligned cases: the MDL cuts themselves are unstable on partial data.

## 32. Getting the root axis right in the robot setting (`BITSI4(up=..., up_prior=..., up_cone=...)`)

Motivation: section 31 (tilted root spines cost ~0.1 sym mIoU on partial clouds). The section-22 gravity
option offered the vertical axis as a third candidate that had to win by 0.01 residual, so it rarely did.
New rule: PREFER the gravity-aligned axis unless another candidate's residual fraction is smaller by more
than delta; optionally the gravity-aligned axis is the best symmetry axis within a cone around up.
**Protocol, fixed before running:** dev = eye-in-hand clouds of dev + tabletop ooddev objects (orbit views,
K = 3 and 6, full-object scores, as section 24); grid = v4, gravity-as-3rd-candidate, prefer-vertical
delta in {0.02, 0.05, 0.10}, always vertical, cone 15 deg with delta 0.05 / always; selection = mean of
(sym mIoU, ARI) over K = 3, 6 (category means). Held-out: the frozen rule is scored once on the saved
section-21 partial clouds (Panda and Kinova, all strategies, K = 1, 3, 8) against v4.

**Dev (267 objects):** prefer vertical delta 0.05 selected, objective 0.224 vs v4 0.210 (gravity as 3rd
candidate 0.222, always vertical 0.211, cone variants 0.204-0.212); vs v4 at K = 3: sym +0.011
[+0.001, +0.021], ARI +0.022 [+0.003, +0.040].
**Held-out, scored once (2796 saved section-21 partial clouds, both robots, all strategies, K = 3 and 8;
`results/eih/axis_heldout.jsonl`):** spines tilted > 10 deg, in-domain 20-31% -> 5-8%; out-of-domain
90-93% -> 50-57%. In-domain sym mIoU: Panda K=3 0.399 -> 0.448 (+0.049 [-0.009, +0.094]), K=8 0.443 ->
0.470 (+0.028 [+0.007, +0.050]); Kinova K=3 0.415 -> 0.449 (+0.034 [-0.019, +0.092]), K=8 +0.023
[-0.010, +0.063]; ARI +0.025 to +0.048 (intervals include 0; 8 categories). Out-of-domain about +0.01.
The axis is now right on most in-domain partial clouds; the remaining loss is over-segmentation.
Adopted: `BITSI4(up=...)` now defaults to up_prior = 0.05 (every robot pipeline passes up; full clouds
are unaffected).

**Partial-view pour after the axis fix (Panda, 36 HANDAL objects, `results/eih_pour/panda_axis.jsonl`):**
BITSI 9/36 (mug 4, cup 1, pan 4; was 5-6), Find3D 13 (was 12-15), whole object 8 (was 4). The new axis rule
also changes BITSI's next-best views, and the whole-object baseline doubled on the same clouds, so part of
the gain is better views rather than better handle detection: BITSI's part still rarely matches the handle
(median IoU 0.15, was 0.12). BITSI still trails Find3D from partial views; the remaining cause is the child /
cut stage on partial data (over-segmentation, handles not separated), not the axis.

## 33. Bounding-box completion next-best-view (user's proposal) and a bug in the volumetric baseline

Proposal: fit a box to what has been seen; if the points cling to one face, the rest of the box is still
unknown, so go to the faces that are empty and fill the box's voxels until it is complete.
Implementation (`eyeinhand.BoxNBV`; parameters fixed before any run, not tuned): gravity-aligned box
(z = up, x/y = footprint principal axes, 2-98% extents), footprint squared by extending it AWAY from the
cameras (the unseen side), 20% margin on all sides but the bottom; occupancy grid by ray casting; gain of a
view = unknown voxels inside the current box reached before an occupied voxel; box refitted after every
view; stop when < 5% of the box's voxels are unknown.

**Bug found while building it (affects section 21):** `VoxelNBV.integrate` carved free space along every
pixel ray to the maximum range, through the object, so the hidden side of the object was marked free
after the first view (e.g. 11,120 of 11,951 box voxels "free" after one view of a bottle). Fixed: a pixel
ray now stops at the first occupied voxel (after view 1: 32% of the box unknown; after the opposite view:
10%). The section-21 volumetric-NBV baseline ("nbv_cov") used the buggy carving and is re-run below;
orbit and BITSI-NBV did not use it.

## 34. BITSI v5 observed-only cut pilot (2026-10-07)

Implemented an opt-in masked angular-radius cut model in `bitsi5.py` and
`partial_sections.py`, retaining v4's gravity-prior axis and child rules for the
paired experiment. Missing bins contribute no fit evidence; cuts require observed
angular support on both sides. Seven regression checks pass.

Two variants completed on 32 development-only tabletop objects, full scans and
orbit K=3,6. The raw sum of angular evidence over-segments. Downweighting correlated
angular evidence gives full-object sym mIoU **0.297 / 0.284 / 0.251** (full/K3/K6)
versus v4 **0.322 / 0.263 / 0.293**. K3 over-segmentation falls, but under-segmentation
rises; K6 ARI drops from 0.268 to 0.127. **Do not promote this candidate.**
Existing defaults remain unchanged and held-out objects were not evaluated.

Protocol, uncertainty intervals, limitations, invocation, and raw result paths:
[BITSI v5 partial-cloud experiment](results/bitsi5_partial.md).
