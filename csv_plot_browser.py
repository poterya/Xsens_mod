#!/usr/bin/env python3
"""
Просмотр CSV в Normal или Deformed: одно полотно, переключение файлов.
Выделите на графике интервал по оси X (мышь) и сохраните обрезанный CSV.

Для папки Deformed: справа одна группа — позиция винта и мотор X-коптера:
  передний левый (B), передний правый (A), задний левый (A), задний правый (B).
  Каталоги: Deformed_mod/<front_left|front_right|rear_left|rear_right>/...

Для Normal: без подпапок позиции — Normal_mod/simulation_.../...

При --dataset-name: datasets/<name>/Deformed_mod/<позиция>/...

Клик по линии — имя ряда; клик по полю графика — путь к CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, RadioButtons, SpanSelector, TextBox
from matplotlib.lines import Line2D

# Позиция винта: подпись в UI (A/B по схеме моторов) → подпапка в Deformed_mod
_DEFORMED_POSITIONS: tuple[tuple[str, str], ...] = (
    ("передний левый (B)", "front_left"),
    ("передний правый (A)", "front_right"),
    ("задний левый (A)", "rear_left"),
    ("задний правый (B)", "rear_right"),
)
_POSITION_LABELS = tuple(p[0] for p in _DEFORMED_POSITIONS)
_LABEL_TO_SUBDIR = {p[0]: p[1] for p in _DEFORMED_POSITIONS}


def _find_time_column(df: pd.DataFrame) -> str | None:
    for name in ("time_seconds", "time", "Time", "t", "timestamp"):
        if name in df.columns:
            return name
    return None


def _numeric_y_columns(df: pd.DataFrame, x_col: str | None) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c == x_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _load_csv(
    path: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[str], str, str | None] | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    x_col = _find_time_column(df)
    if x_col is None:
        x = np.arange(len(df), dtype=float)
        x_label = "index"
    else:
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        x_label = x_col
    y_cols = _numeric_y_columns(df, x_col)
    if not y_cols:
        return None
    return (df, x, y_cols, x_label, x_col)


def discover_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"))


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Графики всех CSV из подпапки Normal или Deformed (рекурсивно)."
    )
    p.add_argument(
        "folder",
        nargs="?",
        default="Normal",
        help="Имя папки: Normal или Deformed (регистр не важен). По умолчанию: Normal",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=base,
        help=f"Каталог, внутри которого лежит Normal/Deformed (по умолчанию: {base})",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Имя отдельного набора для сохранения _mod, например set_01. "
            "Тогда файлы пойдут в datasets/<name>/Normal_mod или Deformed_mod"
        ),
    )
    p.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help=(
            "Каталог с наборами данных. По умолчанию: <base>/datasets. "
            "Используется вместе с --dataset-name"
        ),
    )
    return p.parse_args()


def resolve_data_root(base: Path, folder_name: str) -> Path:
    key = folder_name.strip().lower()
    if key == "normal":
        sub = "Normal"
    elif key == "deformed":
        sub = "Deformed"
    else:
        raise SystemExit(
            f"Неизвестная папка: {folder_name!r}. Укажите Normal или Deformed."
        )
    root = base / sub
    if not root.is_dir():
        raise SystemExit(f"Каталог не найден: {root}")
    return root


def mod_root_for(base: Path, data_root: Path, dataset_name: str | None, datasets_dir: Path | None) -> Path:
    if dataset_name:
        root = datasets_dir if datasets_dir is not None else (base / "datasets")
        dataset_base = root / dataset_name
        if data_root.name == "Normal":
            return dataset_base / "Normal_mod"
        if data_root.name == "Deformed":
            return dataset_base / "Deformed_mod"
        raise ValueError(f"Неизвестная корневая папка данных: {data_root.name}")

    name = data_root.name
    if name == "Normal":
        return base / "Normal_mod"
    if name == "Deformed":
        return base / "Deformed_mod"
    raise ValueError(f"Неизвестная корневая папка данных: {name}")


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.base, args.folder)
    all_csv = discover_csvs(data_root)
    if not all_csv:
        print(f"CSV не найдены в {data_root}", file=sys.stderr)
        sys.exit(1)

    plottable: list[
        tuple[Path, pd.DataFrame, np.ndarray, list[str], str, str | None]
    ] = []
    skipped = 0
    for p in all_csv:
        loaded = _load_csv(p)
        if loaded is None:
            skipped += 1
            continue
        df, x, y_cols, x_label, x_col = loaded
        plottable.append((p, df, x, y_cols, x_label, x_col))

    if skipped:
        print(
            f"Пропущено файлов без подходящих числовых столбцов: {skipped}",
            file=sys.stderr,
        )

    if not plottable:
        print(
            "Ни один CSV не содержит числовых столбцов для графиков "
            "(проверьте формат).",
            file=sys.stderr,
        )
        sys.exit(1)

    idx = [0]
    region = [None, None]  # xmin, xmax по оси X после выделения SpanSelector

    base_path = args.base.resolve()
    datasets_dir = args.datasets_dir.resolve() if args.datasets_dir is not None else None
    mod_root = mod_root_for(base_path, data_root, args.dataset_name, datasets_dir)
    mod_root.mkdir(parents=True, exist_ok=True)

    if args.dataset_name:
        dataset_base = mod_root.parent
        (dataset_base / "Normal_mod").mkdir(parents=True, exist_ok=True)
        (dataset_base / "Deformed_mod").mkdir(parents=True, exist_ok=True)
        output_hint = f"datasets/{args.dataset_name}"
    else:
        (base_path / "Normal_mod").mkdir(exist_ok=True)
        (base_path / "Deformed_mod").mkdir(exist_ok=True)
        output_hint = mod_root.name

    fig = plt.figure(figsize=(13, 7))
    # Одна область графика; кнопки и радио не пересоздаём при смене файла
    ax_plot = fig.add_axes([0.08, 0.24, 0.62, 0.56])
    ax_plot.set_zorder(1)

    is_deformed = data_root.name == "Deformed"
    prop_subdir: list[str] = [_DEFORMED_POSITIONS[0][1]]
    radio_pos = None
    if is_deformed:
        ax_radio_pos = fig.add_axes([0.72, 0.26, 0.26, 0.54])
        ax_radio_pos.set_title("Позиция винта (мотор)", fontsize=9)
        radio_pos = RadioButtons(ax_radio_pos, _POSITION_LABELS, active=0)

        def _on_position(label: str) -> None:
            prop_subdir[0] = _LABEL_TO_SUBDIR[label]
            status_txt.set_text(f"Сохранение → …/Deformed_mod/{prop_subdir[0]}/")
            fig.canvas.draw_idle()

        radio_pos.on_clicked(_on_position)

    status_txt = fig.text(
        0.02,
        0.01,
        "",
        fontsize=8,
        family="monospace",
        transform=fig.transFigure,
        verticalalignment="bottom",
    )
    title_txt = fig.suptitle("", fontsize=9, y=0.97)

    btn_y, btn_h = 0.07, 0.06
    ax_prev = fig.add_axes([0.08, btn_y, 0.1, btn_h])
    ax_next = fig.add_axes([0.20, btn_y, 0.1, btn_h])
    ax_num = fig.add_axes([0.31, btn_y, 0.11, btn_h])
    ax_save = fig.add_axes([0.43, btn_y, 0.28, btn_h])
    for bax in (ax_prev, ax_next, ax_num, ax_save):
        bax.set_zorder(20)

    btn_prev = Button(ax_prev, "Назад")
    btn_next = Button(ax_next, "Вперёд")
    btn_save = Button(ax_save, f"Сохранить в {output_hint}")

    span_keepalive: list[SpanSelector] = []
    num_tb_ref: list[TextBox | None] = [None]

    def redraw_plot() -> None:
        while span_keepalive:
            sp = span_keepalive.pop()
            sp.disconnect_events()
        region[0] = None
        region[1] = None

        ax_plot.clear()
        path, df, x, y_cols, x_label, _x_col = plottable[idx[0]]
        ax_plot._csv_path = str(path.resolve())  # type: ignore[attr-defined]
        for col in y_cols:
            y = pd.to_numeric(df[col], errors="coerce").to_numpy()
            (line,) = ax_plot.plot(x, y, label=col, linewidth=1.0)
            line.set_picker(8)
        ax_plot.set_xlabel(x_label, fontsize=10)
        ax_plot.set_ylabel("значение", fontsize=10)
        ax_plot.tick_params(labelsize=9)
        ax_plot.grid(True, alpha=0.3)
        if len(y_cols) <= 12:
            ax_plot.legend(fontsize=8, loc="upper right")

        def on_select_span(xmin: float, xmax: float) -> None:
            region[0] = float(min(xmin, xmax))
            region[1] = float(max(xmin, xmax))
            status_txt.set_text(
                f"Интервал X: [{region[0]:.6g} … {region[1]:.6g}] — "
                f"нажмите «Сохранить»"
            )
            fig.canvas.draw_idle()

        span_keepalive.append(
            SpanSelector(
                ax_plot,
                on_select_span,
                direction="horizontal",
                useblit=False,
                props=dict(alpha=0.25, facecolor="tab:green"),
                interactive=True,
                drag_from_anywhere=True,
            )
        )

        pos_hint = ""
        if is_deformed:
            pos_hint = f" Папка: {prop_subdir[0]}."
        title_txt.set_text(
            f"{data_root.name} — файл {idx[0] + 1} из {len(plottable)}: {path.name}\n"
            f"Мышью выделите зону по оси X → «Сохранить». Вывод: {output_hint}.{pos_hint} "
            "Клик по линии — ряд; по полю — путь к файлу. В поле «№» — номер файла, Enter."
        )
        fig.canvas.draw_idle()
        tb = num_tb_ref[0]
        if tb is not None:
            tb.set_val(str(idx[0] + 1))

    def on_num_submit(text: str) -> None:
        s = text.strip()
        try:
            n = int(s)
        except ValueError:
            msg = f"Введите целое число от 1 до {len(plottable)}"
            status_txt.set_text(msg)
            print(msg, file=sys.stderr)
            tb = num_tb_ref[0]
            if tb is not None:
                tb.set_val(str(idx[0] + 1))
            fig.canvas.draw_idle()
            return
        if n < 1 or n > len(plottable):
            msg = f"Номер должен быть от 1 до {len(plottable)}"
            status_txt.set_text(msg)
            print(msg, file=sys.stderr)
            tb = num_tb_ref[0]
            if tb is not None:
                tb.set_val(str(idx[0] + 1))
            fig.canvas.draw_idle()
            return
        idx[0] = n - 1
        redraw_plot()

    tb_num = TextBox(ax_num, "№ ", initial=str(idx[0] + 1))
    tb_num.on_submit(on_num_submit)
    num_tb_ref[0] = tb_num

    def on_prev(_event):
        if idx[0] > 0:
            idx[0] -= 1
            redraw_plot()

    def on_next(_event):
        if idx[0] < len(plottable) - 1:
            idx[0] += 1
            redraw_plot()

    def on_save(_event):
        path_i, df_i, x_i, _yc, _xl, _xc = plottable[idx[0]]
        if region[0] is None or region[1] is None:
            msg = "Сначала выделите интервал на графике (зелёная зона по X)."
            status_txt.set_text(msg)
            print(msg, file=sys.stderr)
            fig.canvas.draw_idle()
            return
        lo, hi = region[0], region[1]
        mask = (x_i >= lo) & (x_i <= hi)
        if not np.any(mask):
            msg = "В выделении нет точек данных — расширьте интервал."
            status_txt.set_text(msg)
            print(msg, file=sys.stderr)
            fig.canvas.draw_idle()
            return
        out_df = df_i[mask].copy()
        rel = path_i.relative_to(data_root)
        if data_root.name == "Deformed":
            dest = mod_root / prop_subdir[0] / rel
        else:
            dest = mod_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_df.to_csv(dest, index=False)
        except OSError as e:
            msg = f"Ошибка записи: {e}"
            status_txt.set_text(msg)
            print(msg, file=sys.stderr)
            fig.canvas.draw_idle()
            return
        msg = f"Сохранено строк: {len(out_df)} →\n{dest}"
        status_txt.set_text(msg)
        print(msg)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_save.on_clicked(on_save)

    def on_pick(event):
        if event.artist is None:
            return
        if isinstance(event.artist, Line2D):
            ax = event.artist.axes
            pth = getattr(ax, "_csv_path", "")
            series = event.artist.get_label()
            msg = f"Ряд: {series}\nФайл: {pth}"
            status_txt.set_text(msg)
            print(msg)
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is None or event.button != 1:
            return
        for line in event.inaxes.get_lines():
            if line.contains(event)[0]:
                return
        pth = getattr(event.inaxes, "_csv_path", None)
        if pth:
            msg = f"Файл CSV:\n{pth}"
            status_txt.set_text(msg)
            print(msg)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("button_press_event", on_click)

    redraw_plot()
    plt.show()


if __name__ == "__main__":
    main()
