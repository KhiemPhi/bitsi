# FitPart (overnight build for the "Parts That Fit" CVPR draft)

Training-free, gripper-aware, language-guided part selection on point clouds.
Geometry proposes (multi-scale Chebyshev cuts on BITC profiles), the gripper
constrains (feasibility + collision-aware antipodal grasps), and language selects
(SigLIP relevance over a SAM/DINOv2-lifted hierarchy).

Nothing in the existing BITSI code was modified; everything lives in `fitpart/`.

## Modules (draft section in brackets)

| File | What it does |
|---|---|
| `profiles.py` | BITC profile along any direction, vectorized (about 1000x faster than `accumulate_slices`); PCA + hemisphere sweep directions [4.3] |
| `cheb.py` | Chebyshev fits, chopping-rule top degree, degree schedule, colleague-matrix roots of p'', persistence + strength s(c); ITSI jump-threshold and extrema ablations [4.3] |
| `affinity.py` | Global-cut A_geo (draft formula), mask co-occurrence, normal smoothness [4.2, 4.3] |
| `divisive.py` | Recursive (ITSI-style) best-first splitting with Chebyshev cuts [4.3, new] |
| `superpoints.py` | Normal-aware geodesic Voronoi superpoints [4.1] |
| `hierarchy.py` | Average-linkage merge tree on the superpoint graph; flat cuts by K or theta [4.4] |
| `render.py` | GPU point-splat renderer with z-buffer, pixel->point maps, visibility [4.2] |
| `lifting.py` | SAM ViT-H masks (fp16 encoder), A_mask (draft "finest" + multi-granularity "jaccard"), DINOv2 features, SigLIP/CLIP mask-crop embeddings [4.2] |
| `selection.py` | LERF relevance, mask-matched node embeddings, constrained selection [4.5] |
| `feasibility.py` | Box test on min-volume OBB + antipodal test [3] |
| `grasping.py` | Collision-aware antipodal grasp sampler for a Franka-sized parallel jaw (GPR proxy, shared sampler for Table 3) [5.1] |
| `scales.py` | Category size priors to put normalized clouds in metres (ASSUMPTION, confirm) |
| `sam3_select.py` | Language selection with SAM 3 text-prompted masks lifted onto hierarchy nodes (default engine in the results); per-superpoint no-hierarchy baseline [4.5] |
| `pipeline.py` | `FitPart.lift()` (expensive, cacheable) + `assemble()` (cheap, per ablation) + `select()` |
| `metrics.py` | Hungarian class-agnostic mIoU, hierarchy recall, superpoint purity |
| `baselines/bitsi_v0.py` | The existing BITSI pipeline (main.py) wrapped to give per-point labels |
| `scripts/partfield_features.py` | PartField (ICCV 2025) features as a baseline hierarchy (own `partfield` env) |
| `data/partnete_tasks.json` | Hand-written task instructions per PartNet-E category with an oracle part rewrite (REVIEW) |
| `figures/` | `fig3_knife` (Chebyshev vs thresholded cuts), `qual_partnete` (qualitative selections) |

## Running (env: `sam3`; the BITSI v0 baseline needs `pointcloud` for numba)

```bash
# geometry only, CPU, seconds
python fitpart/scripts/eval_divisive_shapenetpart.py OUT 8 12
python fitpart/scripts/fig3_cuts.py fitpart/figures/fig3_knife Knife 1 0.2
# baseline: existing BITSI pipeline, Hungarian mIoU
~/miniconda3/envs/pointcloud/bin/python fitpart/scripts/eval_bitsi_v0.py OUT 8 8
# full method: stage 1 (GPU, ~30 s/object, resumable), 1b SAM 3 queries, 1c PartField baseline,
# stage 2 (CPU, all ablations + baselines; FITPART_TEXT_DEVICE=cpu keeps it off the GPU)
HF_HUB_OFFLINE=1 python fitpart/scripts/lift_cache.py partnete CACHE 4
HF_HUB_OFFLINE=1 python fitpart/scripts/sam3_cache.py CACHE ~/data/PartNetE/PartNetE_meta.json 10 fitpart/data/partnete_tasks.json
(cd PartField && LD_LIBRARY_PATH=~/miniconda3/envs/partfield/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \
   ~/miniconda3/envs/partfield/bin/python /home/khiem/bitsi/fitpart/scripts/partfield_features.py CACHE WORK)
FITPART_TEXT_DEVICE=cpu FITPART_TASKS=fitpart/data/partnete_tasks.json HF_HUB_OFFLINE=1 \
   python fitpart/scripts/eval_cache.py CACHE OUT 0.9 ~/data/PartNetE/PartNetE_meta.json
python fitpart/scripts/fig4_scatter.py OUT/rows.pkl fitpart/figures/fig4
# PartNet-E subset without the 9.4 GB download (random-access reads of the remote zip)
python fitpart/scripts/fetch_partnete_subset.py 4
```

## Deviations from the draft (decide whether to adopt or revert)

1. **Strength normalization.** The draft's s(c) divides |p'| by the max over the same
   profile, so flat, noisy directions still produce cuts of strength ~1. Code default
   divides by the steepest slope over all directions of the object (`norm="object"`),
   in BITC units per unit of normalized position (per unit length favoured short axes).
2. **End margin.** Inflections within two slabs of either end are dropped (fits ring
   at the domain ends; such cuts only shave slivers).
3. **Divisive geometry.** Global planar cuts fed into bottom-up merging barely beat
   no cuts (a plane at a mug-handle junction also slices the body). Default A_geo
   now comes from a recursive tree: 1 - strength of the split that first separates
   two superpoints.
4. **Mask affinity.** Default is multi-granularity Jaccard over all SAM masks; the
   draft's finest-mask version is kept as an ablation (`mask_mode="finest"`).
5. **Node embeddings for selection.** Averaging superpoint embeddings made tiny nodes
   win every query. Default scores a node with the SAM crops that match it in 3D
   (IoU-weighted), so "the mug" is judged by whole-object crops and "the handle" by
   handle crops.
6. **Relevance temperature.** SigLIP dot products are scaled by 100 before the LERF
   softmax and near-ties are judged in log-odds; with raw cosines the canonical
   phrases win everywhere on untextured renders and r saturates.
7. **Semantic labels need a background prompt.** PartNet-E leaves most points
   unlabeled (-1); segments matching "a <category>" better than any part name are
   labeled background.
8. **SAM prompts.** 16x16 points per view (draft: unspecified); 32x32 is 4x slower on
   a shared 4060 Ti.
9. **SAM 3 as the selection engine.** SigLIP crop relevance on untextured renders picks
   the right node rarely, even when the hierarchy contains it (oracle node IoU ~0.73 vs
   selected ~0.23). SAM 3 text-prompted masks lifted onto nodes roughly double selection
   accuracy. SigLIP stays as an ablation. The draft's Sec. 4.5 would need rewriting.
10. **Multi-instance selection** for semantic segmentation (both pot handles, all chair
   legs): the best node plus disjoint nodes scoring >= 0.5 x the best. Grasp selection
   stays single-node.
11. **Thin-structure grasp fallback.** Wire handles (4 mm) have unreliable normals; the
   feasibility and grasp checks also accept jaws closing across locally thin structure
   (local PCA thickness < 1 cm), still collision-checked against the whole object.
12. **Feasibility vs graspability.** The draft's box test (l3 <= G) rejects a bucket body
   even though a rim pinch exists. GPR is reported on the draft's GT graspable parts.

Results of the overnight runs, with the protocol and caveats, are in `RESULTS.md`; raw summary
tables are in `results/`.
