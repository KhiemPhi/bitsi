# bitsi_sim — headless grasp validation (P5)

Headless PyBullet environment that validates **target-part grasps** produced by a
point-cloud decomposition. It is the sim half of the P5 "learned cutting policy"
paper: the RL policy is trained on the *analytic* grasp metric (no sim in the
loop), and `bitsi_sim` validates those grasps under physics + provides the
sim→real bridge to the Kinova Gen3 / Robotiq 2F-85.

## Design

- **Floating gripper.** Grasp success is evaluated with a free-floating parallel
  jaw sized to the Robotiq 2F-85 (85 mm stroke), not a full arm. This is the
  standard grasp-benchmark setup (ACRONYM / Contact-GraspNet): it measures grasp
  *quality* without conflating it with arm reachability/IK. A hook
  (`gripper.load_kinova_gripper`) is left for later reachability experiments with
  the real Gen3 URDF from `datasets/P5/ros_kortex`.
- **Target-part metric.** `evaluate_target_part_grasp` requires the object to be
  lifted **and** the contacts to land on the target part **without** colliding
  with the rest of the object — this is what ties decomposition quality to grasp
  success (a bad cut that merges the target into a neighbour → no clean grasp).
- **Analytic metric = training reward.** `grasp_sampler.grasp_success_rate`
  (antipodal + force-closure) needs only numpy/scipy, so it is the reward for the
  cutting policy; the sim only validates.

## Layout

| file | role |
|------|------|
| `config.py`      | `GripperSpec` (Robotiq 2F-85), `SimConfig` |
| `world.py`       | `BulletWorld` — headless DIRECT scene, table, object loading |
| `gripper.py`     | `FloatingGripper` (generated URDF) + Kinova hook |
| `grasp_sampler.py` | antipodal / force-closure grasp sampling (analytic metric) |
| `grasp_eval.py`  | `evaluate_grasp`, `evaluate_target_part_grasp` |
| `mesh_utils.py`  | VHACD convex decomposition + collision-mesh cache |
| `geometry.py`    | `Grasp` dataclass + frame math (pure numpy) |
| `demo_grasp.py`  | headless smoke test |

## Setup (server, no display)

The base fbcode python has no pip; create an isolated env:

```bash
# option A: venv (fbcode python has ensurepip)
python3 -m venv ~/.venvs/bitsi_sim
source ~/.venvs/bitsi_sim/bin/activate
pip install -r bitsi_sim/requirements.txt

# option B: conda
conda create -n bitsi_sim python=3.10 -y && conda activate bitsi_sim
pip install -r bitsi_sim/requirements.txt
```

Then run the smoke test (fully headless):

```bash
python -m bitsi_sim.demo_grasp          # p.DIRECT, no X server
python -m bitsi_sim.demo_grasp --egl    # + GPU offscreen rendering on the A100
```

Expected (validated on this server, A100, pybullet 3.2.7, headless DIRECT + EGL):

```
[plain]  success=True  dz=0.109  contacts=5  final_width=0.044
[target] success=True  dz=0.111  on_target_frac=1.00  collided_nontarget=False
OK: headless sim ran end-to-end.
```

Both cases lift deterministically; the target-part classifier assigns each contact
to its nearest part and flags a contact that seats on a non-target part.

## Real meshes

For grasp validation on real objects, load a mesh with VHACD collision:

```python
from bitsi_sim import BulletWorld, SimConfig
from bitsi_sim.mesh_utils import load_mesh_object
with BulletWorld(SimConfig()) as w:
    obj = load_mesh_object(w, "datasets/P4/shapenetcore/<id>/model.obj",
                           scale=0.15, position=(0, 0, w.cfg.table_height + 0.05))
```

Datasets (see `download_datasets.py --paper P5`): ShapeNetCore (train-consistent),
Google Scanned Objects + YCB (sim→real), ACRONYM (grasp-metric cross-check).

## Gripper params to VERIFY

`config.GripperSpec` uses the 85 mm stroke (datasheet) but the finger pad
length/width/thickness/reach are estimates marked `VERIFY` — confirm against the
`ros_kortex` 2F-85 meshes before the hardware experiments so sim collision
matches the real gripper.

## Known caveat: grasping bare primitives

Grasp stability was validated on a table-top box. Bare **primitive** boxes have
flat face-face contacts that jitter, so an adversarial grasp deliberately
straddling two synthetic "parts" may seat only one finger and under-report the
non-target collision. This is a primitive-contact artifact, not a metric bug (the
collision path fires correctly once both pads seat), and it largely disappears on
real **VHACD meshes** (ShapeNetCore / YCB / GSO) which have richer contact
geometry. Tuning knobs in `SimConfig` / `GripperSpec` if you need tighter primitive
grasps: `close_steps`, `settle_steps`, `max_force`, `finger_*`, object `mass` /
`lateral_friction`. Validate on real meshes before trusting numbers for the paper.
