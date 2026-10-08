# FitPart, as implemented (2026-10-06)

Input: a point cloud P (N points, normals estimated if missing), its category (for metric
scale), and a part phrase q. Output: a hierarchy of candidate parts, the selected part,
and whether a parallel-jaw gripper of opening G can hold it.

## 1. BITC profiles along many directions (`profiles.py`)

Sweep directions D: the three PCA axes of P plus 9 Fibonacci-hemisphere directions, dropping
any within 15 degrees of one already chosen (~11 in practice).

For a direction d, build the frame [d, u, v] (u, v = principal axes of P projected on the
plane normal to d) and the object's box in that frame. With h the median nearest-neighbour
spacing (so everything is scale-free), slice along d into slabs of thickness 1.5h. For slab k:

    g(t_k) = ( |{p in slab k : dist(p, box faces) < h}| / |slab k| ) * C_k

where C_k is the number of 8-connected components of the slab's occupied cross-section grid
(cells of 2.5h). Empty slabs give 0. This is the existing BITSI quantity (boundary ratio x
occupied components), vectorized and generalized to any direction.

## 2. Multi-scale Chebyshev fit and inflection cuts (`cheb.py`)

Map slab centres to x in [-1, 1] and the profile to y in [0, 1]. Fit by least squares

    p_n(x) = sum_{k=0..n} c_k T_k(x)

**Top degree n_L (chopping rule).** Fit at n_max = min(64, S/2) (S = #slabs; we recommend capping
at 16). Form the monotone coefficient envelope e_k = max_{j>=k} |c_j| / max|c|, estimate the noise
plateau as the median of e_k over the upper half of k, and take n_L = the smallest k with
e_k <= 3 x plateau (clipped to [4, n_max]).

**Degree schedule.** n_l = max(3, round(f n_L)) for f in {1/4, 1/2, 3/4, 1}, deduplicated: L <= 4 levels.

**Cuts at a level.** Real roots of p_n'' in [-1+m, 1-m], found as eigenvalues of the colleague
matrix (numpy `chebroots`), kept when p'' changes sign there. m = two slabs: fits ring at the
domain ends, and a cut there only shaves a sliver.

**Persistence.** Levels are processed coarse to fine; a root within eta = 2 slabs of an existing cut
is the same cut. Its persistence l(c) is the coarsest level where it appears.

**Strength.**

    s(c) = (L - l(c) + 1)/L * |p'_{n_l(c)}(x_c)| * y_range / S_ref

i.e. the draft's persistence factor times the slope at the cut, in raw BITC units per unit of
normalized position, divided by a reference slope S_ref: the steepest fitted slope over all
directions of the object (for global cuts), or of the root object (inside the recursion).
The draft divided by the max slope of the *same* profile; that gives strength ~1 to the
steepest wiggle of a flat, noisy direction, and per unit length it favours short axes.

**Threshold-free flat segmentation.** Cuts with l(c) <= 2 on the longest axis.

## 3. Recursive (divisive) geometric tree and A_geo (`divisive.py`, `pipeline.py`)

Global planar cuts also cut unrelated regions (a plane at a mug's handle junction crosses the
body), so cuts are found recursively, ITSI-style. Starting from the whole object, each piece is
re-swept in all directions and its strongest cut (max s) is a candidate split. Splits are applied
best-first across pieces (priority = s): the piece is cut by the plane, each side is broken into
connected components of the kNN graph (Section 4), fragments below 2% of N are attached to the
nearest kept part, until 32 leaves or no valid split.

For adjacent superpoints i, j (each assigned to its majority leaf), with LCA their lowest common
ancestor in this tree:

    A_geo(i, j) = 1 - s(split at LCA) / max split strength

## 4. Superpoints (`superpoints.py`)

M = clip(N/8, 50, 500) seeds by farthest-point sampling; a symmetric kNN graph (k = 10) with edge
length ||p_i - p_j|| (1 + 2(1 - |n_i . n_j|)); every point joins the seed closest along the graph
(multi-source Dijkstra). Superpoints are adjacent when a kNN edge joins them.

## 5. Semantic affinity from 2D models (`render.py`, `lifting.py`)

Render K = 10 orthographic views (point splats with a z-buffer, 512 px; elevations alternating
30 / -20 degrees). Per view:

- SAM ViT-H automatic masks (16 x 16 prompts, fp16 image encoder). A superpoint belongs to a mask
  if >= 50% of its visible pixels fall inside it.
  A_mask(i, j) = (# masks containing both) / (# masks containing either), counted over views where
  both are visible (multi-granularity Jaccard; the draft's finest-mask version is an ablation).
- DINOv2-base patch features, upsampled, averaged per superpoint over all views:
  A_feat(i, j) = clip(cos, 0, 1).

    A_sem = beta A_mask + (1 - beta) A_feat   (beta = 0.5; A_feat alone where never co-visible)

## 6. Candidate hierarchy (`hierarchy.py`)

Edges: superpoint adjacency plus long-range edges where A_feat >= 0.85 (the two halves of a
handle seen in different views); long-range edges get the median A_geo.

    w(i, j) = lambda A_sem(i, j) + (1 - lambda) A_geo(i, j)   (lambda = 0.5)

Greedy average-linkage merging (cluster linkage = mean w over the edges joining them) yields a
binary tree H; every node is a candidate part. Flat segmentation: replay merges while linkage
>= theta (0.9), absorbing fragments < 2% of N into the nearest kept cluster.

## 7. Language selection with SAM 3 (`sam3_select.py`)

SAM 3 is prompted with the part phrase in every view (detection threshold 0.3). For view v,
superpoint i gets u_i = max score of the masks covering it. With s_i the superpoint's point count
and V_v(h) the visible points of node h in view v:

    IoU_v(h) = sum_{i in h} u_i s_i / ( sum_i u_i s_i + V_v(h) - sum_{i in h} u_i s_i )
    r(h, q)  = sum_v (V_v(h)/|h|) IoU_v(h) / sum_v (V_v(h)/|h|)

(soft union of all masks in a view, so both handles of a pot count as one answer; views weighted
by how much of the node they show). The selected node is the argmax over nodes with >= 1% of the
points; near-ties (0.1 in log-odds) go to the smaller node. For semantic segmentation, further
disjoint nodes scoring >= 0.5 x the best are added (all legs, all handles).

## 8. Learned geometric-topological part detector (`node_features.py`, `scripts/train_part_detector.py`)

For every candidate node (0.1% - 50% of the points), 26 features from the point cloud alone
(object rescaled to metres with a category size prior):

- geometry (13): share of the object; PCA-box extents l1 >= l2 >= l3; l1 relative to the object;
  elongation l1/l2; flatness l2/l3; local cross-section (median second extent of local PCA in a
  4 cm ball: a wire handle is ~1 cm even when its box is 15 cm wide); mean/min/max height along
  the up axis; verticality of the main axis; share of the node on its boundary.
- topology (7): beta_0 of the node's superpoint graph; number of separate attachment patches to
  the rest (connected components of its boundary superpoints); H1 persistence (Vietoris-Rips,
  ripser, 120 points, metric) of the node alone (largest and second largest, / l1); largest H1 of
  the node plus its one-ring of neighbouring superpoints, and its gain over the node alone (a
  handle closes a loop with the body); the H0 gap.
- Chebyshev (4): the node's own BITC profile along its main axis, fit as in Section 2: number of
  cuts, number with l(c) <= 2, strongest cut, chopped degree.
- context (2): size relative to its parent; linkage at which it was formed.

One gradient-boosted tree classifier per part name (scikit-learn HistGradientBoosting, 200
iterations, learning rate 0.05, 15 leaves, balanced classes); positives are nodes with IoU >= 0.5
with the part or one of its instances. At test time the detector scores every candidate.

**Gating.** Use SAM 3's choice when its best node score is >= 0.05; otherwise the detector's.

## 9. Gripper feasibility and grasps (`feasibility.py`, `grasping.py`)

    feasible(h) <=> l3(min-volume OBB of h) <= G  and  an antipodal contact pair exists

Antipodal: two contacts within G whose connecting line lies in both friction cones (mu = 0.5,
sign-free normals). Wires and rims, where normals are unreliable, fall back to jaws closing across
locally thin structure (local PCA thickness < 1 cm). A grasp counts only if Franka-sized finger
and palm boxes do not intersect the rest of the object. Selection is restricted to feasible nodes;
"exists but cannot be grasped" is reported when the most relevant node is infeasible and the best
feasible node scores below alpha = 0.7 times it.
