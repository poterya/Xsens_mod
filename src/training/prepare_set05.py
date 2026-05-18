#!/usr/bin/env python3
"""Create datasets/set_05 from set_03 with session-level train/val/test split.

The split is stratified by class directory and copies complete sessions, so
windows from one flight never leak between train/val/test.
"""
from __future__ import annotations

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LABELS: tuple[str, ...] = (
    "normal",
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
)


@dataclass(frozen=True)
class Session:
    label: str
    src_dir: Path
    rel_dst: Path


def _collect_sessions(src: Path) -> dict[str, list[Session]]:
    out: dict[str, list[Session]] = defaultdict(list)

    normal_dir = src / "Normal_mod"
    for csv in sorted(normal_dir.rglob("vibration_log.csv")) if normal_dir.is_dir() else []:
        session_dir = csv.parent
        out["normal"].append(
            Session("normal", session_dir, Path("Normal_mod") / session_dir.name)
        )

    deformed_dir = src / "Deformed_mod"
    for label in LABELS[1:]:
        label_dir = deformed_dir / label
        for csv in sorted(label_dir.rglob("vibration_log.csv")) if label_dir.is_dir() else []:
            session_dir = csv.parent
            out[label].append(
                Session(label, session_dir, Path("Deformed_mod") / label / session_dir.name)
            )
    return out


def _split_items(
    items: list[Session], train_ratio: float, val_ratio: float, rng: random.Random
) -> dict[str, list[Session]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = min(max(1, n_train), n - 2)
        n_val = min(max(1, n_val), n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _copy_session(session: Session, dst_split_root: Path) -> Path:
    dst_dir = dst_split_root / session.rel_dst
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(session.src_dir, dst_dir)
    return dst_dir


def create_set05(
    src: Path,
    dst: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    clean: bool,
) -> None:
    if clean and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    sessions_by_label = _collect_sessions(src)
    rng = random.Random(seed)
    manifest: dict = {
        "source": str(src.resolve()),
        "destination": str(dst.resolve()),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": max(0.0, 1.0 - train_ratio - val_ratio),
        },
        "splits": {"train": [], "val": [], "test": []},
        "counts": {},
    }

    print(f"Источник: {src}")
    print(f"Назначение: {dst}")
    print("Сессий по классам:")
    for label in LABELS:
        print(f"  {label:12s} {len(sessions_by_label.get(label, []))}")

    for label in LABELS:
        split_map = _split_items(
            sessions_by_label.get(label, []), train_ratio, val_ratio, rng
        )
        for split, sessions in split_map.items():
            for session in sessions:
                dst_dir = _copy_session(session, dst / split)
                manifest["splits"][split].append(
                    {
                        "label": session.label,
                        "source": str(session.src_dir.relative_to(src)),
                        "path": str(dst_dir.relative_to(dst)),
                    }
                )

    print("\nИтог set_05:")
    for split in ("train", "val", "test"):
        manifest["counts"][split] = {}
        for label in LABELS:
            n = sum(1 for row in manifest["splits"][split] if row["label"] == label)
            manifest["counts"][split][label] = n
        total = sum(manifest["counts"][split].values())
        print(f"  {split:5s} всего {total}")
        for label, n in manifest["counts"][split].items():
            print(f"    {label:12s} {n}")

    manifest_path = dst / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest: {manifest_path}")


def main() -> None:
    here = _PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Создать datasets/set_05 из set_03 со split train/val/test по сессиям"
    )
    parser.add_argument("--src", type=Path, default=here / "datasets" / "set_03")
    parser.add_argument("--dst", type=Path, default=here / "datasets" / "set_05")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-clean", action="store_true", help="Не очищать set_05 перед сборкой")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("Нужны отношения: train > 0, val >= 0, train + val < 1")
    if not args.src.is_dir():
        raise SystemExit(f"Нет источника: {args.src}")

    create_set05(
        args.src.resolve(),
        args.dst.resolve(),
        args.train_ratio,
        args.val_ratio,
        args.seed,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
