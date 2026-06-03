"""Сбалансированный датасет: equal-flights, undersample большинства.

Шаги:
- собрать все полёты из текущего dataset/train|val|test;
- объединить части _partNN в группы по логическому имени;
- выбросить группы суммарно короче MIN_DURATION;
- из бо́льшего класса случайно отобрать столько же групп, сколько в
  меньшем классе (undersample);
- случайно перетасовать оставшиеся группы на train/val/test 70/15/15.
Лишние группы кладутся в _dataset_excluded/, обрубки — в _dataset_dropped/.
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
EXCLUDED = HERE / "_dataset_excluded"
DROPPED = HERE / "_dataset_dropped"
SPLITS = ("train", "val", "test")
CLASSES = ("Normal", "Deformed")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_DURATION = 1.0
RANDOM_SEED = 42

_PART_RE = re.compile(r"^(?P<base>.+?)_part\d+$")


def _logical_id(name: str) -> str:
    m = _PART_RE.match(name)
    return m.group("base") if m else name


def _duration(csv_path: Path) -> float | None:
    try:
        df = pd.read_csv(csv_path, usecols=["time_seconds"])
    except Exception:
        return None
    if len(df) < 2:
        return None
    t = df["time_seconds"].to_numpy(dtype=float)
    return float(t[-1] - t[0])


def collect() -> dict[str, dict[str, list[Path]]]:
    out: dict[str, dict[str, list[Path]]] = {c: defaultdict(list) for c in CLASSES}
    for split in SPLITS:
        for cls in CLASSES:
            src = ROOT / split / cls
            if not src.is_dir():
                continue
            for fd in sorted(src.iterdir()):
                if fd.is_dir():
                    out[cls][_logical_id(fd.name)].append(fd)
    return out


def group_dur(parts: list[Path]) -> float:
    total = 0.0
    for p in parts:
        csv = p / "vibration_log.csv"
        if not csv.exists():
            continue
        d = _duration(csv)
        if d is not None:
            total += d
    return total


def _safe_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main() -> None:
    flights = collect()

    keep: dict[str, list[tuple[str, list[Path]]]] = {c: [] for c in CLASSES}
    short_drop: list[tuple[str, str, float]] = []
    for cls in CLASSES:
        for lid, parts in flights[cls].items():
            dur = group_dur(parts)
            if dur < MIN_DURATION:
                short_drop.append((cls, lid, dur))
            else:
                keep[cls].append((lid, parts))

    print("=== Обрубки (< 1.0 с), будут вынесены в _dataset_dropped ===")
    for cls, lid, dur in short_drop:
        print(f"   {cls:9s} {lid:42s}  {dur:.2f} с")
    print(f"всего обрубков: {len(short_drop)}")

    rng = random.Random(RANDOM_SEED)

    n_min = min(len(keep[c]) for c in CLASSES)
    print(f"\nГрупп после очистки: Normal={len(keep['Normal'])}, "
          f"Deformed={len(keep['Deformed'])} → выравниваем до {n_min} в каждом классе")

    balanced: dict[str, list[tuple[str, list[Path]]]] = {}
    excluded: dict[str, list[tuple[str, list[Path]]]] = {}
    for cls in CLASSES:
        items = sorted(keep[cls], key=lambda x: x[0])
        rng.shuffle(items)
        balanced[cls] = items[:n_min]
        excluded[cls] = items[n_min:]

    print("\n=== Лишние группы (выкидываются в _dataset_excluded) ===")
    for cls in CLASSES:
        if not excluded[cls]:
            continue
        print(f"--- {cls}: {len(excluded[cls])} групп ---")
        for lid, parts in excluded[cls]:
            print(f"   {lid:42s}  частей={len(parts)}")

    new_layout: dict[tuple[str, str], list[tuple[str, list[Path]]]] = {}
    for cls in CLASSES:
        items = balanced[cls]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * RATIOS["train"]))
        n_val = int(round(n * RATIOS["val"]))
        new_layout[("train", cls)] = items[:n_train]
        new_layout[("val", cls)] = items[n_train:n_train + n_val]
        new_layout[("test", cls)] = items[n_train + n_val:]

    print("\n=== Новое распределение (групп) ===")
    for cls in CLASSES:
        line = f"{cls:9s} "
        for s in SPLITS:
            line += f"{s}={len(new_layout[(s, cls)]):>3d}  "
        line += f"всего={sum(len(new_layout[(s, cls)]) for s in SPLITS)}"
        print(line)

    staging = HERE / "_dataset_new"
    if staging.exists():
        shutil.rmtree(staging)
    for s in SPLITS:
        for c in CLASSES:
            (staging / s / c).mkdir(parents=True, exist_ok=True)

    if EXCLUDED.exists():
        shutil.rmtree(EXCLUDED)
    EXCLUDED.mkdir(parents=True, exist_ok=True)
    if DROPPED.exists():
        for f in DROPPED.glob("*"):
            pass
    else:
        DROPPED.mkdir(parents=True, exist_ok=True)

    moved_main = 0
    for (split, cls), groups in new_layout.items():
        for _lid, parts in groups:
            for src_dir in parts:
                _safe_move(src_dir, staging / split / cls / src_dir.name)
                moved_main += 1
    print(f"\nперемещено в staging: {moved_main}")

    moved_excl = 0
    for cls in CLASSES:
        for _lid, parts in excluded[cls]:
            for src_dir in parts:
                _safe_move(src_dir, EXCLUDED / cls / src_dir.name)
                moved_excl += 1
    print(f"перемещено в _dataset_excluded: {moved_excl}")

    moved_drop = 0
    for s in SPLITS:
        for c in CLASSES:
            d = ROOT / s / c
            if not d.is_dir():
                continue
            for fd in list(d.iterdir()):
                if fd.is_dir():
                    _safe_move(fd, DROPPED / c / fd.name)
                    moved_drop += 1
    print(f"перемещено в _dataset_dropped: {moved_drop}")

    for s in SPLITS:
        old = ROOT / s
        if old.exists():
            shutil.rmtree(old)
    for s in SPLITS:
        shutil.move(str(staging / s), str(ROOT / s))
    shutil.rmtree(staging)

    print("\ndataset/{train,val,test} обновлён и сбалансирован.")


if __name__ == "__main__":
    main()
