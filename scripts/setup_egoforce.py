"""Fetch the pinned official EgoForce 2D inference sources and public checkpoint.

Does not run the upstream installation script or install system packages. The
large checkpoint is opt-in with --download. License: upstream CC-BY-NC-4.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

SOURCE_REVISION = "480ffb358516d0d7f971ec9da99b7bb07635731e"
SOURCE_FILES = (
    "models/halo.py", "models/modules.py", "models/mano_mean_params.npz", "demo/demo_hand_arm_loader.py",
    "camera_models/pinhole.py", "utils/augmentation.py", "utils/rotations.py",
    "README.md", "settings.py",
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / "models" / "egoforce"
MODEL_REPO = "chris10/EgoForce"
MODEL_REVISION = "5a57b4026babc5b35cb53627c2e1c709e1276b65"
MODEL_FILENAME = "_DATA/model_weights.pth"
CHECKPOINT_SIZE = 3021671106
CHECKPOINT_SHA256 = "722fe84ba2b6ab3569c3a95000c61a9a77c7b3a41cc76585f891b8bc4fe82870"


def fetch_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "D435i-prototype/1.0"}), timeout=45) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected_size=None, expected_sha256=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if expected_sha256 and sha256(destination) == expected_sha256:
            print(f"verified existing {destination.name}", flush=True)
            return
        if not expected_sha256:
            print(f"existing source {destination.name}", flush=True)
            return
        raise RuntimeError(f"Existing file has a different SHA-256; preserve and inspect it: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "D435i-prototype/1.0"})
    if partial.is_file() and partial.stat().st_size:
        request.add_header("Range", f"bytes={partial.stat().st_size}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        resume = getattr(response, "status", 200) == 206
        count = partial.stat().st_size if resume else 0
        next_report = time.monotonic() + 10
        with partial.open("ab" if resume else "wb") as output:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                count += len(block)
                if time.monotonic() >= next_report:
                    print(f"{destination.name}: {count / 1024**2:.1f} MiB" + (f" / {expected_size / 1024**2:.1f} MiB" if expected_size else ""), flush=True)
                    next_report = time.monotonic() + 10
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete download retained for resume: {partial}")
    if expected_sha256 and sha256(partial) != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch; file was not activated: {partial}")
    partial.replace(destination)
    print(f"verified {destination.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--download", action="store_true", help="Download the approximately 3 GB official checkpoint")
    args = parser.parse_args()
    destination = args.destination.resolve()
    model_info = fetch_json(f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true")
    model_revision = model_info["sha"]
    if model_revision != MODEL_REVISION:
        raise RuntimeError("Official model API returned a different revision; download cancelled")
    entry = next(item for item in model_info["siblings"] if item["rfilename"] == MODEL_FILENAME)
    lfs = entry.get("lfs") or {}
    size = int(lfs.get("size", entry.get("size", 0)))
    digest = lfs.get("sha256") or lfs.get("oid", "").removeprefix("sha256:")
    if size != CHECKPOINT_SIZE or digest != CHECKPOINT_SHA256:
        raise RuntimeError("Official repository checkpoint size or SHA-256 differs from the reviewed revision; download cancelled")
    manifest = {
        "name": "EgoForce HALO 2D", "license": "CC-BY-NC-4.0",
        "source_repository": "https://github.com/dfki-av/EgoForce",
        "source_revision": SOURCE_REVISION, "model_repository": f"https://huggingface.co/{MODEL_REPO}",
        "model_revision": model_revision, "checkpoint": "model_weights.pth",
        "checkpoint_size": size, "checkpoint_sha256": digest,
        "scope": "Official HALO network and 2D heads with shared hand ROI; no official detector or 3D mesh pipeline",
        "sources": {},
    }
    destination.mkdir(parents=True, exist_ok=True)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2), flush=True)
    for relative in SOURCE_FILES:
        target = destination / "source" / relative
        download(f"https://raw.githubusercontent.com/dfki-av/EgoForce/{SOURCE_REVISION}/{relative}", target)
        manifest["sources"][relative] = sha256(target)
    if args.download:
        # mmap loading still needs disk-backed pages; reserve space beyond checkpoint.
        needed = max(0, size - ((destination / "model_weights.pth.part").stat().st_size if (destination / "model_weights.pth.part").exists() else 0))
        if not (destination / "model_weights.pth").exists() and shutil.disk_usage(destination).free < needed + 1024**3:
            raise RuntimeError("Insufficient free disk space for checkpoint and 1 GiB reserve")
        download(f"https://huggingface.co/{MODEL_REPO}/resolve/{model_revision}/{MODEL_FILENAME}", destination / "model_weights.pth", size, digest)
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
