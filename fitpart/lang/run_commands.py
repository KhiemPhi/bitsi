"""End to end: user command -> (SigLIP + MLP) task / direction / magnitude / part -> BITSI screw on the
robot's point cloud -> Kinova executes it. The model is the one trained by lang/model.py (chosen on the
held-out-templates-and-objects test before any robot run).
usage (sam3 env): python -m fitpart.lang.run_commands MODEL_DIR OUT_DIR
"""
import json, os, pickle, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fitpart.lang.model import SiglipModel, to_screw_spec

COMMANDS = [
    "open the bottle for me", "tighten the cap on the bottle", "lift the lid of that box",
    "pull the drawer open about 10 cm", "close the drawer", "press the button", "squeeze the tongs",
    "tilt the box back 45 degrees", "pick up the drill", "swing the hammer", "scoop some soup with the ladle",
    "stir with the whisk", "slice with the knife", "hand me the screwdriver",
]
ART = dict(turn="bottle", hinge="lidbox", slide="drawer", press="button", squeeze="tongs", pivot="pivotbox", cut="knife")
TOOL = dict(lift=("power_drill", "obj_000001"), swing=("hammer", "obj_000004"), scoop=("ladle", "obj_000002"),
            stir=("whisk", "obj_000022"), handover=("screwdriver", "obj_000002"), pour=("mug", "obj_000021"))
GRASPSIM = os.path.expanduser("~/miniconda3/envs/graspsim/bin/python")


def main(model_dir, out_dir):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    m = SiglipModel()
    st = pickle.load(open(f"{model_dir}/siglip_head.pkl", "rb"))
    m.classes = st["classes"]
    m.head = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                 torch.nn.Linear(256, len(m.classes)))
    m.head.load_state_dict(st["head"])
    m.head.eval()
    labs = m.predict(COMMANDS)
    out = f"{out_dir}/commands.jsonl"
    for i, (cmd, lab) in enumerate(zip(COMMANDS, labs)):
        spec = to_screw_spec(cmd, lab)
        task = spec["task"]
        key = str(300 + i) if task in ART else TOOL.get(task, (None, None))[1]
        print(f"[{i}] {cmd!r} -> {spec}", flush=True)
        if key is None:
            continue
        r = subprocess.run([GRASPSIM, "-m", "fitpart.tasks.run", task, key, out, "--spec", json.dumps(spec),
                            "--video", f"{out_dir}/cmd{i:02d}_{task}.mp4"],
                           cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           capture_output=True, text=True)
        last = [l for l in r.stdout.splitlines() if l.startswith("{")]
        print("    ", last[-1][:240] if last else r.stderr[-400:], flush=True)
    json.dump(dict(commands=COMMANDS, predicted=labs), open(f"{out_dir}/predictions.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
