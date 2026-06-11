#!/usr/bin/env python3
"""Конвертер обученной MLP из ветки `tests` в PyTorch-весовой файл `.pt`.

Источник: method_mlp.pkl — Pipeline(StandardScaler + sklearn.MLPClassifier)
со скрытыми слоями (16,), ReLU, выходом sigmoid (logistic), 46 входов.

Назначение: получить такой `.pt`, который умеет читать существующий
`export_via_onnx2c.py` (он ждёт класс MLPDetector с state_dict, который
после загрузки `forward` производит логит).

Архитектурная эквивалентность:
  • StandardScaler (mean, scale) превращается в `nn.BatchNorm1d(46, affine=False)`
    с running_mean=mean, running_var=scale**2, eps=0 — это **тождественное**
    воспроизведение `(x - mean) / scale`.
  • Линейный слой (W: 46x16) и bias записываются в `nn.Linear(46, 16)`.
  • Затем ReLU.
  • Линейный слой (W: 16x1) и bias записываются в `nn.Linear(16, 1)`.
  • Выход возвращается как **логит** (без sigmoid); финальный sigmoid
    добавляется в C++-обёртке `nn_detect_model.cpp`.

Запуск:
    python3 export_mlp_to_cpp.py
сохранит файл `propeller_fault_mlp_keras_set05.pt` рядом со скриптом
(имя сохранено ради совместимости с уже существующим
`export_via_onnx2c.py`).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from feature_extraction import extract_features

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PKL = THIS_DIR / "method_mlp.pkl"
DEFAULT_PT = THIS_DIR / "propeller_fault_mlp_keras_set05.pt"

IN_FEATURES = 46
HIDDEN = 16
WINDOW_SIZE = 10
SAMPLE_RATE_HZ = 100.0

DEGENERATE_SCALE = 1e-6


class MLPDetector(nn.Module):
    """Зеркало sklearn.MLPClassifier (StandardScaler + 46->16 ReLU + 16->1)."""

    def __init__(self, in_features: int = IN_FEATURES, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features, affine=False, eps=0.0)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)


def feature_name_order() -> list[str]:
    z = np.zeros(WINDOW_SIZE, dtype=float)
    return list(extract_features(z, z, z, z).keys())


def _build_state_dict_from_pkl(pkl_path: Path, expected_names: list[str]) -> dict[str, torch.Tensor]:
    with pkl_path.open("rb") as f:
        bundle = pickle.load(f)
    pipe = bundle["model"]
    saved_names = bundle["feature_names"]
    if list(saved_names) != list(expected_names):
        raise SystemExit(
            "Имена признаков в method_mlp.pkl не совпадают с feature_extraction.extract_features:\n"
            f"  pkl ({len(saved_names)}): {saved_names}\n"
            f"  py  ({len(expected_names)}): {expected_names}"
        )
    scaler = pipe.named_steps["scaler"]
    mlp = pipe.named_steps["mlp"]
    if mlp.n_features_in_ != IN_FEATURES:
        raise SystemExit(f"Ожидался вход {IN_FEATURES}, получен {mlp.n_features_in_}")
    if tuple(mlp.hidden_layer_sizes) != (HIDDEN,):
        raise SystemExit(f"Ожидался hidden_layer_sizes=({HIDDEN},), получен {mlp.hidden_layer_sizes}")
    if mlp.activation != "relu":
        raise SystemExit(f"Ожидался activation='relu', получен {mlp.activation}")
    if mlp.out_activation_ != "logistic":
        raise SystemExit(f"Ожидался out_activation='logistic', получен {mlp.out_activation_}")


    mean_np = np.asarray(scaler.mean_, dtype=np.float64)
    scale_np = np.asarray(scaler.scale_, dtype=np.float64)
    degenerate = scale_np < DEGENERATE_SCALE
    if degenerate.any():
        idxs = np.flatnonzero(degenerate).tolist()
        names = [expected_names[i] for i in idxs]
        print(f"Подмена scale=1.0 для вырожденных фич: {names}")
        scale_np = np.where(degenerate, 1.0, scale_np)
    mean = torch.from_numpy(mean_np.astype(np.float32))
    scale = torch.from_numpy(scale_np.astype(np.float32))
    var = scale * scale

    w1 = torch.from_numpy(np.asarray(mlp.coefs_[0].T, dtype=np.float32))
    b1 = torch.from_numpy(np.asarray(mlp.intercepts_[0], dtype=np.float32))
    w2 = torch.from_numpy(np.asarray(mlp.coefs_[1].T, dtype=np.float32))
    b2 = torch.from_numpy(np.asarray(mlp.intercepts_[1], dtype=np.float32))

    return {
        "norm.running_mean": mean,
        "norm.running_var": var,
        "norm.num_batches_tracked": torch.tensor(0, dtype=torch.long),
        "fc1.weight": w1,
        "fc1.bias": b1,
        "fc2.weight": w2,
        "fc2.bias": b2,
    }


def _verify(model: nn.Module, pkl_path: Path) -> float:
   
    bundle = pickle.load(pkl_path.open("rb"))
    pipe = bundle["model"]
    csv_dir = THIS_DIR.parent / "CSV_for_tests"
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"Нет CSV для верификации в {csv_dir}")
    import pandas as pd
    max_diff = 0.0
    model.eval()
    with torch.no_grad():
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            cols = ["total_vibration", "rms_x", "rms_y", "rms_z"]
            arr = df[cols].to_numpy(dtype=float)
            if len(arr) < WINDOW_SIZE:
                continue
            for start in range(0, len(arr) - WINDOW_SIZE + 1, max(1, (len(arr) - WINDOW_SIZE) // 4)):
                sl = slice(start, start + WINDOW_SIZE)
                feats = extract_features(arr[sl, 0], arr[sl, 1], arr[sl, 2], arr[sl, 3])
                x = np.array(list(feats.values()), dtype=np.float32).reshape(1, -1)
                py_logit = float(model(torch.from_numpy(x)).item())
                py_p = 1.0 / (1.0 + float(np.exp(-py_logit)))
                sk_p = float(pipe.predict_proba(x.astype(np.float64))[0, 1])
                max_diff = max(max_diff, abs(py_p - sk_p))
    return max_diff


def convert(pkl_path: Path, out_pt: Path) -> None:
    expected = feature_name_order()
    state = _build_state_dict_from_pkl(pkl_path, expected)
    model = MLPDetector(in_features=IN_FEATURES, hidden=HIDDEN)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"Проблема при загрузке весов: missing={missing}, unexpected={unexpected}")
    diff = _verify(model, pkl_path)
    print(f"PyTorch vs sklearn max |P_diff| = {diff:.3e}")
   
    if diff > 5e-3:
        raise SystemExit("Расхождение слишком велико — конвертация неверна")
    torch.save(model.state_dict(), out_pt)
    print(f"Wrote {out_pt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    ap.add_argument("--out", type=Path, default=DEFAULT_PT)
    args = ap.parse_args()
    convert(args.pkl, args.out)


if __name__ == "__main__":
    main()
