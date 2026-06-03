#!/usr/bin/env python3
"""Ридер набора данных вибраций для задачи "норма / поломка".

Структура датасета (ожидается рядом со скриптом):

    dataset/
        train/
            Normal/   simulation_<ts>/vibration_log.csv
            Deformed/ simulation_<ts>/vibration_log.csv
        val/
            Normal/
            Deformed/
        test/
            Normal/
            Deformed/

Класс определяется по папке: Normal -> 0 (norm), Deformed -> 1 (fault).
Подкаталоги внутри Deformed (small / medium / strong и т.п.) обходятся
рекурсивно — все полёты внутри получают метку fault.

Скрипт ничего не классифицирует: только обходит структуру, проверяет
шапку CSV и печатает сводку. Опциональный флаг --index сохраняет
файл dataset_index.csv со списком всех найденных полётов.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("time_seconds", "total_vibration", "rms_x", "rms_y", "rms_z")
SPLITS = ("train", "val", "test")
CLASS_DIRS = {"Normal": 0, "Deformed": 1}
LABEL_NAME = {0: "normal", 1: "fault"}


@dataclass
class FlightEntry:
    split: str
    label: int
    flight_id: str
    csv_path: Path
    n_rows: int


def _read_header(csv_path: Path) -> list[str] | None:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except OSError:
        return None
    if header is None:
        return None
    return [c.strip() for c in header]


def _count_rows(csv_path: Path) -> int:
    n = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for _ in reader:
            n += 1
    return n


def _validate_csv(csv_path: Path) -> tuple[bool, str, int]:
    header = _read_header(csv_path)
    if header is None:
        return False, "не удалось прочитать заголовок", 0
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        return False, f"в шапке нет колонок: {missing}", 0
    try:
        n = _count_rows(csv_path)
    except OSError as exc:
        return False, f"ошибка чтения строк: {exc}", 0
    if n == 0:
        return False, "файл без данных", 0
    return True, "", n


def scan_dataset(base: Path) -> tuple[list[FlightEntry], list[tuple[Path, str]]]:
    """Возвращает (список найденных полётов, список проблемных файлов)."""
    flights: list[FlightEntry] = []
    issues: list[tuple[Path, str]] = []

    if not base.is_dir():
        raise SystemExit(f"Каталог датасета не найден: {base}")

    for split in SPLITS:
        split_dir = base / split
        if not split_dir.is_dir():
            issues.append((split_dir, "нет папки split"))
            continue
        for class_dir, label in CLASS_DIRS.items():
            cls_root = split_dir / class_dir
            if not cls_root.is_dir():
                issues.append((cls_root, "нет папки класса"))
                continue
            csv_paths = sorted(cls_root.rglob("vibration_log.csv"))
            for csv_path in csv_paths:
                flight_id = csv_path.parent.name
                ok, msg, n = _validate_csv(csv_path)
                if not ok:
                    issues.append((csv_path, msg))
                    continue
                flights.append(
                    FlightEntry(
                        split=split,
                        label=label,
                        flight_id=flight_id,
                        csv_path=csv_path,
                        n_rows=n,
                    )
                )

    return flights, issues


def print_summary(flights: list[FlightEntry], issues: list[tuple[Path, str]]) -> None:
    print("Найденные полёты:")
    counts: dict[tuple[str, int], int] = {}
    rows_total: dict[tuple[str, int], int] = {}
    for f in flights:
        key = (f.split, f.label)
        counts[key] = counts.get(key, 0) + 1
        rows_total[key] = rows_total.get(key, 0) + f.n_rows

    print(f"{'split':<6} {'class':<7} {'flights':>8} {'rows':>10}")
    for split in SPLITS:
        for label in (0, 1):
            n_flights = counts.get((split, label), 0)
            n_rows = rows_total.get((split, label), 0)
            print(f"{split:<6} {LABEL_NAME[label]:<7} {n_flights:>8} {n_rows:>10}")
        sub_total = sum(counts.get((split, l), 0) for l in (0, 1))
        sub_rows = sum(rows_total.get((split, l), 0) for l in (0, 1))
        print(f"{split:<6} {'итого':<7} {sub_total:>8} {sub_rows:>10}")

    print(f"\nВсего полётов: {len(flights)}")
    if issues:
        print(f"\nПроблем при чтении: {len(issues)}")
        for path, msg in issues:
            print(f"  - {path}: {msg}")
    else:
        print("\nПроблем при чтении: нет")


def write_index(flights: list[FlightEntry], index_path: Path) -> None:
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("split", "label", "label_name", "flight_id", "n_rows", "csv_path"))
        for fl in flights:
            writer.writerow(
                (
                    fl.split,
                    fl.label,
                    LABEL_NAME[fl.label],
                    fl.flight_id,
                    fl.n_rows,
                    str(fl.csv_path),
                )
            )
    print(f"\nИндекс сохранён: {index_path}")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Ридер датасета вибраций (норма / поломка)."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=here / "dataset",
        help="Корень датасета (по умолчанию: ./dataset рядом со скриптом)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Сохранить индекс полётов в CSV (например: dataset_index.csv)",
    )
    args = parser.parse_args()

    flights, issues = scan_dataset(args.base.resolve())
    print_summary(flights, issues)
    if args.index is not None:
        write_index(flights, args.index.resolve())

    if not flights:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
