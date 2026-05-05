#!/usr/bin/env python3
"""Собрать метрики из прогонов all_detector_3 и построить график точности vs номер сессии.

Ищет файлы metrics_after_label.txt в nn_v2_binary_detection_logs/nn_v2_binary_* /
сохраняет PNG с кривыми точности и TXT со средним / мин / макс по каждому методу.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Строки в metrics_after_label.txt от all_detector_3 (имена должны совпадать)
METHOD_ORDER = (
    "Калман (бинарно)",
    "NN forest binary, сырой",
    "NN forest binary, сглаж.",
    "NN tree   binary, сырой",
    "NN tree   binary, сглаж.",
)

_LINE_RE = re.compile(
    r"^(.+?):\s*\d+/\d+\s*(?:=\s*([\d.]+)%|\(нет точек\))\s*$", re.MULTILINE
)


@dataclass
class SessionMetrics:
    folder: Path
    order_key: str
    values: dict[str, float | None]


def _parse_metrics_file(path: Path) -> dict[str, float | None]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, float | None] = {}
    for m in _LINE_RE.finditer(text):
        name = m.group(1).strip()
        pct = m.group(2)
        out[name] = float(pct) / 100.0 if pct is not None else None
    return out


def _session_sort_key(folder_name: str) -> tuple:
    # nn_v2_binary_offline_20260505_121541 / nn_v2_binary_live_...
    parts = folder_name.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return (parts[-2], parts[-1])
    return (folder_name,)


def collect_sessions(logs_root: Path) -> list[SessionMetrics]:
    if not logs_root.is_dir():
        return []
    rows: list[SessionMetrics] = []
    for child in sorted(logs_root.iterdir(), key=lambda p: _session_sort_key(p.name)):
        if not child.is_dir():
            continue
        if not child.name.startswith("nn_v2_binary_"):
            continue
        mfile = child / "metrics_after_label.txt"
        if not mfile.is_file():
            continue
        vals = _parse_metrics_file(mfile)
        rows.append(
            SessionMetrics(
                folder=child,
                order_key=child.name,
                values=vals,
            )
        )
    return rows


def _stats_for_method(series: list[float | None]) -> tuple[float | None, float | None, float | None]:
    nums = [x for x in series if x is not None]
    if not nums:
        return None, None, None
    arr = np.array(nums, dtype=float)
    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="График точности по размеченным сессиям all_detector_3"
    )
    ap.add_argument(
        "--logs-root",
        type=Path,
        default=here / "nn_v2_binary_detection_logs",
        help="Каталог с подпапками nn_v2_binary_*",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Куда сохранить график и TXT (по умолчанию: <logs-root>/accuracy_aggregate)",
    )
    args = ap.parse_args()

    logs_root = args.logs_root.resolve()
    out_dir = (args.out_dir or (logs_root / "accuracy_aggregate")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = collect_sessions(logs_root)
    if not sessions:
        raise SystemExit(
            f"Не найдено ни одного metrics_after_label.txt под {logs_root}\n"
            "(нужны размеченные прогоны all_detector_3, не пропускать экспорт в set_03)"
        )

    n = len(sessions)
    x = np.arange(1, n + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    summary_lines: list[str] = [
        f"Источник: {logs_root}",
        f"Число размеченных сессий: {n}",
        "",
    ]

    for label in METHOD_ORDER:
        ys: list[float | None] = []
        for s in sessions:
            ys.append(s.values.get(label))
        y_mean, y_min, y_max = _stats_for_method(ys)
        y_plot = [100.0 * v if v is not None else np.nan for v in ys]
        ax.plot(x, y_plot, marker="o", markersize=4, linewidth=1.2, label=label)

        if y_mean is None:
            summary_lines.append(f"{label}")
            summary_lines.append("  нет валидных точек")
        else:
            summary_lines.append(f"{label}")
            summary_lines.append(
                f"  mean = {100.0 * y_mean:.2f}% | min = {100.0 * y_min:.2f}% | max = {100.0 * y_max:.2f}%"
            )
        summary_lines.append("")

    ax.set_xlabel("номер сессии (по времени сохранения папки)")
    ax.set_ylabel("точность, %")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("all_detector_3: точность по вашей разметке vs количество сессий")
    fig.tight_layout()

    png_path = out_dir / "detector3_accuracy_vs_sessions.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    txt_path = out_dir / "detector3_accuracy_summary.txt"
    txt_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

    print(f"Сессий: {n}")
    print(f"График: {png_path}")
    print(f"Сводка: {txt_path}")


if __name__ == "__main__":
    main()
