"""Удаление обрубков и случайная перетасовка train/val/test 70/15/15.

Полёты длительностью меньше MIN_DURATION выбрасываются. Остальные
группируются по «логическому полёту» (имя без суффикса _partNN), чтобы
части одного полёта всегда попадали в один сплит. Перетасовка — со
стратификацией по классам и фиксированным random seed.
"""
from __future__ import annotations

import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE / "dataset"
SPLITS = ("train", "val", "test")
CLASSES = ("Normal", "Deformed")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_DURATION = 1.0
RANDOM_SEED = 42

_PART_RE = re.compile(r"^(?P<base>.+?)_part\d+$")


def _logical_id(flight_dir_name: str) -> str:
    m = _PART_RE.match(flight_dir_name)
    return m.group("base") if m else flight_dir_name


def _flight_duration(csv_path: Path) -> float | None:
    try:
        df = pd.read_csv(csv_path, usecols=["time_seconds"])
    except Exception:
        return None
    if len(df) < 2:
        return None
    t = df["time_seconds"].to_numpy(dtype=float)
    return float(t[-1] - t[0])


def collect_flights() -> dict[str, dict[str, list[Path]]]:
    """class -> logical_id -> [path_to_flight_dir, ...]"""
    out: dict[str, dict[str, list[Path]]] = {c: defaultdict(list) for c in CLASSES}
    for split in SPLITS:
        for cls in CLASSES:
            src = ROOT / split / cls
            if not src.is_dir():
                continue
            for flight_dir in sorted(src.iterdir()):
                if not flight_dir.is_dir():
                    continue
                lid = _logical_id(flight_dir.name)
                out[cls][lid].append(flight_dir)
    return out


def group_duration(parts: list[Path]) -> float:
    total = 0.0
    for p in parts:
        csv = p / "vibration_log.csv"
        if not csv.exists():
            continue
        d = _flight_duration(csv)
        if d is not None:
            total += d
    return total


def main() -> None:
    flights = collect_flights()

    drop_log: list[tuple[str, str, float, list[str]]] = []
    keep_groups: dict[str, list[tuple[str, list[Path]]]] = {c: [] for c in CLASSES}
    for cls, by_lid in flights.items():
        for lid, parts in by_lid.items():
            dur = group_duration(parts)
            if dur < MIN_DURATION:
                drop_log.append((cls, lid, dur, [p.name for p in parts]))
            else:
                keep_groups[cls].append((lid, parts))

    print("=== Будут удалены обрубки (< 1.0 с) ===")
    for cls, lid, dur, names in drop_log:
        print(f"   {cls:9s} {lid:42s}  {dur:.2f} с  части={names}")
    print(f"всего удаляется: {len(drop_log)}")

    new_layout: dict[tuple[str, str], list[tuple[str, list[Path]]]] = {}
    rng = random.Random(RANDOM_SEED)
    for cls in CLASSES:
        items = sorted(keep_groups[cls], key=lambda x: x[0])
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * RATIOS["train"]))
        n_val = int(round(n * RATIOS["val"]))
        n_test = n - n_train - n_val
        slices = {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:n_train + n_val + n_test],
        }
        for split, lst in slices.items():
            new_layout[(split, cls)] = lst

    print("\n=== Новое распределение ===")
    for cls in CLASSES:
        line = f"{cls:9s} "
        for split in SPLITS:
            line += f"{split}={len(new_layout[(split, cls)]):>3d}  "
        line += f"всего={sum(len(new_layout[(s, cls)]) for s in SPLITS)}"
        print(line)

    staging = HERE / "_dataset_new"
    if staging.exists():
        shutil.rmtree(staging)
    for split in SPLITS:
        for cls in CLASSES:
            (staging / split / cls).mkdir(parents=True, exist_ok=True)

    moves_done = 0
    for (split, cls), groups in new_layout.items():
        target_dir = staging / split / cls
        for lid, parts in groups:
            for src_dir in parts:
                shutil.move(str(src_dir), str(target_dir / src_dir.name))
                moves_done += 1
    print(f"\nперемещено папок полётов: {moves_done}")

    drop_staging = HERE / "_dataset_dropped"
    if drop_staging.exists():
        shutil.rmtree(drop_staging)
    drop_staging.mkdir(parents=True, exist_ok=True)
    dropped_moved = 0
    for split in SPLITS:
        for cls in CLASSES:
            src = ROOT / split / cls
            if not src.is_dir():
                continue
            for flight_dir in list(src.iterdir()):
                if flight_dir.is_dir():
                    target = drop_staging / cls / flight_dir.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(flight_dir), str(target))
                    dropped_moved += 1
    print(f"перемещено обрубков в _dataset_dropped: {dropped_moved}")

    for split in SPLITS:
        old_split = ROOT / split
        if old_split.exists():
            shutil.rmtree(old_split)
    for split in SPLITS:
        shutil.move(str(staging / split), str(ROOT / split))
    shutil.rmtree(staging)

    print("\nстарый dataset/{train,val,test} заменён на новый.")
    print(f"обрубки сохранены в {drop_staging} на всякий случай.")


if __name__ == "__main__":
    main()
