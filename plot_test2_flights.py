"""Построение графиков всех полётов из dataset/test_2.

Для каждого CSV строит отдельный ч/б график 4 кривых
(total_vibration, rms_x, rms_y, rms_z) и сохраняет в
images/test_2_flights/{Normal,Deformed}/. Дополнительно строит сводные
обзоры по классам (наложение всех полётов на одну ось time).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": plt.cycler(color=["black", "0.35", "0.55", "0.75"]),
})

HERE = Path(__file__).resolve().parent
SRC = HERE / "dataset" / "test_2"
DST = HERE / "images" / "test_2_flights"

CLASS_TITLE_RU = {"Normal": "норма", "Deformed": "поломка"}
SIGNAL_TITLES = {
    "total_vibration": "Суммарная вибрация",
    "rms_x": "RMS по X",
    "rms_y": "RMS по Y",
    "rms_z": "RMS по Z",
}
SIGNAL_STYLES = {
    "total_vibration": dict(color="black",  linestyle="-",  linewidth=1.4),
    "rms_x":           dict(color="0.30",   linestyle="--", linewidth=1.1),
    "rms_y":           dict(color="0.50",   linestyle="-.", linewidth=1.1),
    "rms_z":           dict(color="0.70",   linestyle=":",  linewidth=1.2),
}


def _plot_flight(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=140)
    t = df["time_seconds"].to_numpy()
    for col in ("total_vibration", "rms_x", "rms_y", "rms_z"):
        ax.plot(t, df[col].to_numpy(), label=SIGNAL_TITLES[col], **SIGNAL_STYLES[col])
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Амплитуда вибрации")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(class_dfs: list[tuple[str, pd.DataFrame]], class_label: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=140, sharex=False)
    cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
    for ax, col in zip(axes.flat, cols):
        for _name, df in class_dfs:
            ax.plot(df["time_seconds"].to_numpy(), df[col].to_numpy(),
                    color="black", linewidth=0.7, alpha=0.45)
        ax.set_title(SIGNAL_TITLES[col])
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Амплитуда")
    fig.suptitle(f"Все полёты test_2 ({CLASS_TITLE_RU[class_label]}, {len(class_dfs)} шт.)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"Нет каталога {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    total = 0
    for class_dir in ("Normal", "Deformed"):
        src_cls = SRC / class_dir
        if not src_cls.is_dir():
            continue
        dst_cls = DST / class_dir
        dst_cls.mkdir(parents=True, exist_ok=True)
        class_dfs: list[tuple[str, pd.DataFrame]] = []
        for csv_path in sorted(src_cls.rglob("vibration_log.csv")):
            flight = csv_path.parent.name
            df = pd.read_csv(csv_path)
            required = {"time_seconds", "total_vibration", "rms_x", "rms_y", "rms_z"}
            if not required.issubset(df.columns):
                print(f"[пропуск] {csv_path}: нет нужных колонок")
                continue
            df = df.dropna(subset=list(required)).reset_index(drop=True)
            if df.empty:
                continue
            title = f"{CLASS_TITLE_RU[class_dir]} | {flight} | {len(df)} отсчётов"
            out = dst_cls / f"{flight}.png"
            _plot_flight(df, title, out)
            class_dfs.append((flight, df))
            total += 1
        if class_dfs:
            _plot_overview(class_dfs, class_dir, DST / f"обзор_{CLASS_TITLE_RU[class_dir]}.png")

    print(f"построено графиков полётов: {total}")
    print(f"сводные обзоры: {DST}/обзор_*.png")


if __name__ == "__main__":
    main()
