#!/usr/bin/env python3
"""
download_datasets.py — fetch (or plan the fetch of) datasets for papers P1–P4.

DRY-RUN BY DEFAULT. Nothing is downloaded unless you pass --execute.

Papers:
  P1  Selective Grasping                 (ICRA)  — grasping object sets + baseline weights
  P2  Acting to Certify                  (RSS)   — articulated objects + articulation baselines
  P3  Geometry-Calibrated Selective Seg  (RA-L)  — ScanNet/Replica + SAM2 + YOLO11-seg
  P4  Part-Masked Pretraining            (CVPR)  — ShapeNet(Part) + ModelNet40 + Point-MAE

Usage:
  python download_datasets.py                      # dry-run, all papers, plan only
  python download_datasets.py --paper P3           # only P3 datasets
  python download_datasets.py --paper P3 --execute # actually download P3's fetchable items
  python download_datasets.py --include-optional    # also plan optional/large extras
  python download_datasets.py --list                # print the registry as a table and exit

Methods per dataset:
  http     direct archive download (+ optional unpack)
  git      git clone (code / repos with in-repo download scripts)
  script   run a provided download script that ships in a cloned repo
  pip      install via pip (data pulled lazily by the package)
  manual   license/account-gated: print instructions, then verify expected path exists
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ROOT = Path(os.environ.get("BITSI_DATA_ROOT", "./datasets")).resolve()


@dataclass
class Dataset:
    key: str
    paper: str                 # "P1".."P4"
    method: str                # http | git | script | pip | manual
    dest: str                  # subdir under ROOT
    url: str = ""              # archive URL / git remote / pip spec
    unpack: bool = False       # unzip/untar after http download
    optional: bool = False     # skipped unless --include-optional
    gated: bool = False        # requires license/account (never auto-fetched)
    # For method=="script": repo to clone, then the command to run inside it.
    repo: str = ""
    post_cmd: list[str] = field(default_factory=list)
    notes: str = ""


REGISTRY: list[Dataset] = [
    # ----------------------------- P3 (draft-ready first) ------------------
    Dataset(
        key="replica",
        paper="P3",
        method="script",
        dest="P3/replica",
        repo="https://github.com/facebookresearch/Replica-Dataset.git",
        post_cmd=["bash", "download.sh", "{dest}"],
        notes="Public. Cloud-hosted scenes; clone repo then run its download.sh.",
    ),
    Dataset(
        key="scannet",
        paper="P3",
        method="manual",
        dest="P3/scannet",
        url="https://github.com/ScanNet/ScanNet",
        gated=True,
        notes=(
            "GATED. Sign the ScanNet Terms of Use and email the signed PDF to the "
            "maintainers to receive download-scannet.py. Then:\n"
            "        python download-scannet.py -o {dest} --type _vh_clean_2.ply\n"
            "      (2D/instance labels also needed for GT parity in 5.2)."
        ),
    ),
    Dataset(
        key="sam2-checkpoints",
        paper="P3",
        method="script",
        dest="P3/sam2",
        repo="https://github.com/facebookresearch/sam2.git",
        post_cmd=["bash", "checkpoints/download_ckpts.sh"],
        notes="SAM 2 teacher weights via the repo's download_ckpts.sh (dl.fbaipublicfiles.com).",
    ),
    Dataset(
        key="yolo11-seg",
        paper="P3",
        method="http",
        dest="P3/yolo",
        # Ultralytics also auto-downloads on first use; explicit URL for review.
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",  # VERIFY tag/asset
        notes="Student backbone. `pip install ultralytics` will otherwise fetch this lazily.",
    ),

    # ----------------------------- P4 -------------------------------------
    Dataset(
        key="shapenetpart",
        paper="P4",
        method="http",
        dest="P4/shapenetpart",
        # Stanford-hosted PointNet mirror — no account needed (unlike full ShapeNetCore).
        url="https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",  # VERIFY
        unpack=True,
        notes="ShapeNetPart seg benchmark (normals) — main P4 eval (5.2).",
    ),
    Dataset(
        key="modelnet40",
        paper="P4",
        method="http",
        dest="P4/modelnet40",
        url="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",  # VERIFY
        unpack=True,
        notes="Non-headline classification control (5.5).",
    ),
    Dataset(
        key="shapenetcore-full",
        paper="P4",
        method="manual",
        dest="P4/shapenetcore",
        url="https://shapenet.org",
        gated=True,
        optional=True,
        notes="GATED/optional. Full ShapeNetCore for pretraining needs a shapenet.org account.",
    ),
    Dataset(
        key="objaverse-lvis",
        paper="P4",
        method="pip",
        dest="P4/objaverse",
        url="objaverse",
        optional=True,
        notes="Optional scale-up. `import objaverse; objaverse.load_lvis_annotations()`; large.",
    ),
    Dataset(
        key="point-mae",
        paper="P4",
        method="git",
        dest="P4/Point-MAE",
        url="https://github.com/Pang-Yatian/Point-MAE.git",
        notes="Reference backbone + MAE head (code; pretrained weights linked in its README).",
    ),

    # ----------------------------- P1 -------------------------------------
    Dataset(
        key="ycb-objects",
        paper="P1",
        method="http",
        dest="P1/ycb",
        url="http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/",  # VERIFY: per-object model URLs
        notes="Object set for grasp trials. Site hosts per-object meshes; a small loop over object ids.",
    ),
    Dataset(
        key="contact-graspnet",
        paper="P1",
        method="git",
        dest="P1/contact_graspnet",
        url="https://github.com/NVlabs/contact_graspnet.git",
        notes="Baseline (6.4). Weights are on Google Drive — see repo README (needs gdown). VERIFY.",
    ),
    Dataset(
        key="gpd",
        paper="P1",
        method="git",
        dest="P1/gpd",
        url="https://github.com/atenpas/gpd.git",
        notes="Baseline (6.4). Build from source; pretrained caffe/onnx model linked in README.",
    ),
    Dataset(
        key="acronym",
        paper="P1",
        method="git",
        dest="P1/acronym",
        url="https://github.com/NVlabs/acronym.git",
        optional=True,
        notes="Optional. Grasp dataset used to train CGN; grasp .h5 hosted off-repo (VERIFY), needs ShapeNet meshes.",
    ),
    Dataset(
        key="graspnet-1b",
        paper="P1",
        method="manual",
        dest="P1/graspnet1b",
        url="https://graspnet.net",
        gated=True,
        optional=True,
        notes="GATED/optional. Strong public anchor; requires registration at graspnet.net.",
    ),

    # ----------------------------- P2 -------------------------------------
    Dataset(
        key="partnet-mobility",
        paper="P2",
        method="manual",
        dest="P2/partnet_mobility",
        url="https://sapien.ucsd.edu/downloads",
        gated=True,
        notes="GATED. Articulated object set; requires a SAPIEN account + usage agreement.",
    ),
    Dataset(
        key="shape2motion",
        paper="P2",
        method="git",
        dest="P2/shape2motion",
        url="https://github.com/wangxiaogang866/Shape2Motion.git",
        notes="Articulation data; download links in repo README. VERIFY.",
    ),
    Dataset(
        key="ditto",
        paper="P2",
        method="git",
        dest="P2/Ditto",
        url="https://github.com/UT-Austin-RPL/Ditto.git",
        notes="Articulation baseline (joint-type accuracy). Data links in README.",
    ),
    Dataset(
        key="screwnet",
        paper="P2",
        method="git",
        dest="P2/ScrewNet",
        url="https://github.com/Pat-Lab-UOB/ScrewNet.git",  # VERIFY org/repo
        optional=True,
        notes="Second articulation baseline. VERIFY exact repo.",
    ),
]


# ---------------------------------------------------------------------------
# Execution helpers  (all no-ops in dry-run)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], execute: bool, cwd: Path | None = None) -> None:
    printable = " ".join(cmd)
    if cwd:
        printable = f"(cd {cwd} && {printable})"
    if not execute:
        print(f"    [dry-run] would run: {printable}")
        return
    print(f"    running: {printable}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _http_get(url: str, out: Path, execute: bool) -> None:
    if not execute:
        print(f"    [dry-run] would download: {url}\n              -> {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"    downloading: {url} -> {out}")
    urllib.request.urlretrieve(url, out)  # noqa: S310 (reviewed URLs only)


def _unpack(archive: Path, dest: Path, execute: bool) -> None:
    if not execute:
        print(f"    [dry-run] would unpack: {archive} -> {dest}")
        return
    print(f"    unpacking: {archive} -> {dest}")
    shutil.unpack_archive(str(archive), str(dest))


def handle(ds: Dataset, root: Path, execute: bool) -> None:
    dest = root / ds.dest
    tag = "OPTIONAL" if ds.optional else "core"
    print(f"\n[{ds.paper}] {ds.key}  ({ds.method}, {tag})")
    if ds.notes:
        print(f"    note: {ds.notes.format(dest=dest)}")

    # Idempotency: skip anything already present.
    if dest.exists() and any(dest.iterdir() if dest.is_dir() else [dest]):
        print(f"    SKIP: {dest} already exists and is non-empty.")
        return

    if ds.method == "manual" or ds.gated:
        print(f"    MANUAL/GATED — not auto-fetched. Expected final path: {dest}")
        print(f"    Source: {ds.url}")
        return

    if ds.method == "http":
        fname = ds.url.split("/")[-1] or f"{ds.key}.bin"
        archive = dest / fname
        _http_get(ds.url, archive, execute)
        if ds.unpack:
            _unpack(archive, dest, execute)

    elif ds.method == "git":
        _run(["git", "clone", "--depth", "1", ds.url, str(dest)], execute)

    elif ds.method == "script":
        _run(["git", "clone", "--depth", "1", ds.repo, str(dest)], execute)
        cmd = [c.format(dest=str(dest)) for c in ds.post_cmd]
        _run(cmd, execute, cwd=dest)

    elif ds.method == "pip":
        _run([sys.executable, "-m", "pip", "install", ds.url], execute)

    else:
        print(f"    ERROR: unknown method {ds.method!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", choices=["P1", "P2", "P3", "P4"], help="limit to one paper")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"download root (default {DEFAULT_ROOT})")
    ap.add_argument("--execute", action="store_true", help="actually download (default: dry-run/plan only)")
    ap.add_argument("--include-optional", action="store_true", help="also handle optional/large extras")
    ap.add_argument("--include-gated", action="store_true", help="print gated instructions even without a paper filter")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    args = ap.parse_args()

    items = [d for d in REGISTRY if (not args.paper or d.paper == args.paper)]
    if not args.include_optional:
        items = [d for d in items if not d.optional]

    if args.list:
        print(f"{'paper':5} {'key':22} {'method':7} {'gated':5} {'opt':3} dest")
        for d in REGISTRY:
            print(f"{d.paper:5} {d.key:22} {d.method:7} {str(d.gated):5} {str(d.optional):3} {d.dest}")
        return 0

    mode = "EXECUTE" if args.execute else "DRY-RUN (no downloads)"
    print(f"=== dataset fetch plan — mode: {mode} — root: {args.root} ===")
    for d in items:
        handle(d, args.root, args.execute)

    gated = [d for d in items if d.gated]
    if gated:
        print("\n=== ACTION REQUIRED: gated datasets (sign/register first) ===")
        for d in gated:
            print(f"  - {d.paper}/{d.key}: {d.url}")
    print("\nDone (plan only)." if not args.execute else "\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
