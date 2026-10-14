#!/usr/bin/env python3
"""Build config.json and patched nginx config for the CVAT projection overlay.

    overlay-config.py config --calib calib.json --pcd-dir data/pointcloud --out config.json
    overlay-config.py nginx  --stock stock.conf --out default.conf

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
LOCATIONS = HERE / "nginx-locations.conf"
SCRIPT_TAG = '<script defer src="/d14-overlay/d14-projection-overlay.js"></script></head>'
DEFAULT_CAMERAS = {"image_1": "CAM_P_F"}


class ConfigError(ValueError):
    pass


def read_frames_from_pcd(pcd_dir: Path) -> list[str]:
    """Get frame stems from PCD directory."""
    stems = []
    for f in sorted(pcd_dir.iterdir()):
        if f.suffix.lower() == '.pcd':
            stems.append(f.stem)
    if not stems:
        raise ConfigError(f"no .pcd files in {pcd_dir}")
    return stems


def matrix(value, rows: int, cols: tuple[int, ...], what: str) -> None:
    ok = (isinstance(value, list) and len(value) in (rows,) + ((4,) if rows == 3 and 4 in cols else ())
          and all(isinstance(r, list) and len(r) in cols
                  and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in r)
                  for r in value))
    if not ok:
        raise ConfigError(f"{what} must be a numeric {rows}x{'/'.join(map(str, cols))} matrix")


def check_calibration(calib: dict, cameras: dict[str, str]) -> None:
    frame = str(calib.get("pcd_frame", "lidar")).lower()
    if frame not in ("lidar", "ego"):
        raise ConfigError(f'pcd_frame must be "lidar" or "ego", got {frame!r}')
    if frame == "lidar":
        matrix((calib.get("lidar") or {}).get("sensor2ego"), 3, (4,), "lidar.sensor2ego")
    known = calib.get("cameras") or {}
    for key, name in cameras.items():
        cam = known.get(name)
        if cam is None:
            raise ConfigError(f"camera {name!r} (for {key}) is not in the calibration")
        matrix(cam.get("K"), 3, (3,), f"{name}.K")
        matrix(cam.get("ego2cam"), 3, (4,), f"{name}.ego2cam")
        size = cam.get("image_size")
        if not (isinstance(size, list) and len(size) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)):
            raise ConfigError(f"{name}.image_size must be [width, height] in pixels")
        dist = cam.get("dist", [])
        if not isinstance(dist, list) or len(dist) > 5 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in dist):
            raise ConfigError(f"{name}.dist must hold at most five numbers (k1 k2 p1 p2 k3)")


def build_config(calib_path: Path, pcd_dir: Path, cameras: dict[str, str],
                 task_ids: list[int] | None = None) -> dict:
    raw = calib_path.read_bytes()
    try:
        calib = json.loads(raw)
    except ValueError as e:
        raise ConfigError(f"{calib_path}: not JSON ({e})") from None
    if not isinstance(calib, dict):
        raise ConfigError(f"{calib_path}: expected a JSON object")
    if not cameras:
        raise ConfigError("no context image is mapped to a camera")
    check_calibration(calib, cameras)
    frames = read_frames_from_pcd(pcd_dir)
    entry = {
        "calibration_version": {"file": calib_path.name, "sha256": hashlib.sha256(raw).hexdigest()},
        "frames": frames,
        "cameras": cameras,
        "calibration": calib,
    }
    if task_ids:
        entry["task_ids"] = sorted(set(task_ids))
    return entry


def as_datasets(existing: object) -> list[dict]:
    """Normalise a loaded config (registry, single flat entry, or empty) to a datasets list."""
    if isinstance(existing, dict) and isinstance(existing.get("datasets"), list):
        return [d for d in existing["datasets"] if isinstance(d, dict) and d.get("calibration")]
    if isinstance(existing, dict) and existing.get("calibration"):
        return [existing]
    return []


def merge_entry(existing: object, entry: dict) -> dict:
    """Merge one dataset entry into a registry. Same recording (same calibration sha + same
    panel set) folds its task ids into the new entry instead of duplicating; a task id that
    moves to a different calibration is taken from its old entry; a plain re-`up` without task
    ids replaces the entry with the same panel set."""
    new_tasks = set(entry.get("task_ids", []))
    new_panels = frozenset(entry["cameras"].keys())
    new_sha = (entry.get("calibration_version") or {}).get("sha256")
    kept = []
    for d in as_datasets(existing):
        d_tasks = set(d.get("task_ids", []))
        d_panels = frozenset((d.get("cameras") or {}).keys())
        d_sha = (d.get("calibration_version") or {}).get("sha256")
        if new_sha and d_sha == new_sha and d_panels == new_panels:
            new_tasks |= d_tasks          # same recording: fold tasks, drop the duplicate
            continue
        if new_tasks and (new_tasks & d_tasks):
            continue                       # these tasks now use the new calibration
        if not new_tasks and d_panels == new_panels:
            continue                       # manual re-up without task ids: replace by panels
        kept.append(d)
    if new_tasks:
        entry["task_ids"] = sorted(new_tasks)
    return {"datasets": kept + [entry]}


def block_end(conf: str, open_brace: int) -> int | None:
    depth, i = 0, open_brace
    while i < len(conf):
        c = conf[i]
        if c == "#" and (i == 0 or conf[i - 1] in " \t\n;{}"):
            i = conf.find("\n", i)
            if i < 0:
                return None
        elif c in "\"'":
            i += 1
            while i < len(conf) and conf[i] != c:
                i += 2 if conf[i] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def build_nginx(stock: str, locations: str) -> str:
    roots = list(re.finditer(r"^([ \t]*)location / \{[ \t]*\n", stock, re.M))
    servers = list(re.finditer(r"^[ \t]*server[ \t]*\{", stock, re.M))
    end = block_end(stock, servers[0].end() - 1) if len(servers) == 1 else None
    if (len(roots) != 1 or end is None or not servers[0].end() < roots[0].start() < end
            or not re.fullmatch(r"(\s|#[^\n]*)*", stock[end + 1:])
            or re.search(r"^[ \t]*sub_filter\b", stock, re.M)):
        raise ConfigError("the cvat_ui nginx config has an unexpected shape; overlay not installed")
    indent = roots[0].group(1) + "    "
    inject = f"{indent}sub_filter '</head>' '{SCRIPT_TAG}';\n{indent}sub_filter_once on;\n"
    return (stock[:roots[0].end()] + inject + stock[roots[0].end():end].rstrip(" \t")
            + "\n" + locations + stock[end:])


def write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    def add_build_args(p):
        p.add_argument("--calib", type=Path, required=True)
        p.add_argument("--pcd-dir", type=Path, required=True,
                       help="directory with .pcd files (frame stems)")
        p.add_argument("--cameras", default=json.dumps(DEFAULT_CAMERAS),
                       help='JSON: context image -> calibrated camera, default %(default)s')
        p.add_argument("--task-ids", default="",
                       help="comma-separated CVAT task ids this calibration belongs to")
        p.add_argument("--out", type=Path, required=True)

    c = sub.add_parser("config")
    add_build_args(c)
    r = sub.add_parser("register")  # merge one dataset into an existing registry
    add_build_args(r)
    r.add_argument("--existing", type=Path, help="current config.json to merge into (optional)")
    n = sub.add_parser("nginx")
    n.add_argument("--stock", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    a = parser.parse_args(argv)

    def parse_cameras(s):
        try:
            cameras = json.loads(s)
        except ValueError:
            raise ConfigError(f"--cameras is not JSON: {s!r}") from None
        if not isinstance(cameras, dict) or not all(isinstance(v, str) for v in cameras.values()):
            raise ConfigError('--cameras must map context images to camera names')
        return cameras

    def parse_task_ids(s):
        try:
            return [int(x) for x in s.split(",") if x.strip()]
        except ValueError:
            raise ConfigError(f"--task-ids must be comma-separated integers: {s!r}") from None

    try:
        if a.cmd in ("config", "register"):
            entry = build_config(a.calib, a.pcd_dir, parse_cameras(a.cameras), parse_task_ids(a.task_ids))
            if a.cmd == "register":
                existing = {}
                if a.existing and a.existing.exists() and a.existing.stat().st_size:
                    try:
                        existing = json.loads(a.existing.read_text())
                    except ValueError:
                        existing = {}
                out = merge_entry(existing, entry)
            else:
                out = {"datasets": [entry]}
            write_private(a.out, json.dumps(out))
            v = entry["calibration_version"]
            print(f"{a.cmd}: {len(out['datasets'])} dataset(s); this one {len(entry['frames'])} frames, "
                  f"cameras {entry['cameras']}, tasks {entry.get('task_ids', [])}, "
                  f"calibration {v['file']} sha256 {v['sha256'][:12]}")
        else:
            a.out.write_text(build_nginx(a.stock.read_text(), LOCATIONS.read_text()))
    except (ConfigError, OSError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
