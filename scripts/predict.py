# Copyright (c) 2026 Tim DPdp.
# Licensed under the MIT License.
"""Prediksi penyakit daun padi dari satu file gambar.

Dipakai oleh app web (`app.py`) dan bisa dijalankan sendiri:

    <VENV> scripts/predict.py --image path/ke/daun.jpg

Model diambil dari checkpoint terbaik di `runs/` (berdasarkan val macro-F1),
atau dari env var DPDP_CHECKPOINT / --checkpoint. Nama kelas dibaca dari
metadata di dalam checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dpdp_model import eval_transform

PROJECT = Path(__file__).resolve().parent.parent

# Di bawah ambang ini, app menampilkan peringatan keyakinan rendah.
CONFIDENCE_THRESHOLD = 0.5


def find_best_checkpoint() -> Path | None:
    """Checkpoint terbaik di runs/ berdasarkan val macro-F1."""
    best = None
    for m in sorted((PROJECT / "runs").glob("*/metrics.json")):
        try:
            met = json.loads(m.read_text())
        except Exception:
            continue
        if met.get("args", {}).get("max_train_batches", 0):
            continue
        score = met.get("best_val_macro_f1", -1.0)
        if best is None or score > best[0]:
            best = (score, m.parent / "best.pt")
    return best[1] if best else None


def load_predictor(checkpoint: Path):
    """Return (model, class_names, meta). Model sudah mode eval."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = ck.get("arch") or ck.get("args", {}).get("model")
    if not arch:
        raise ValueError(
            f"checkpoint {checkpoint.name} tidak menyimpan nama arsitektur")
    class_names = [str(n) for n in (ck.get("class_names") or [])]
    if not class_names:
        raise ValueError(
            f"checkpoint {checkpoint.name} tidak menyimpan daftar kelas")
    from dpdp_model import build_model
    model = build_model(arch, len(class_names))
    model.load_state_dict(ck["model"])
    model.eval()
    meta = {
        "arch": arch,
        "checkpoint": checkpoint.name,
        "run_dir": checkpoint.parent.name,
        "epoch": ck.get("epoch"),
        "val_macro_f1": ck.get("val_macro_f1"),
        "trained_batches": ck.get("args", {}).get("max_train_batches", 0),
        "class_names": class_names,
        "img_size": ck.get("img_size", 224),
    }
    return model, class_names, meta


def predict(image_path, model, class_names, top_k=3, img_size=224):
    """Prediksi satu gambar -> label, keyakinan, dan top-k."""
    with Image.open(image_path) as im:
        x = eval_transform(img_size)(im.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).numpy()

    order = np.argsort(probs)[::-1]
    best_i = int(order[0])
    return {
        "label": class_names[best_i],
        "confidence": float(probs[best_i]),
        "confident": bool(probs[best_i] >= CONFIDENCE_THRESHOLD),
        "top_k": [{"label": class_names[i], "confidence": float(probs[i])}
                  for i in order[:top_k]],
        "all_probs": {class_names[i]: float(probs[i])
                       for i in range(len(class_names))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif os.environ.get("DPDP_CHECKPOINT"):
        ckpt = Path(os.environ["DPDP_CHECKPOINT"])
    else:
        found = find_best_checkpoint()
        if found is None:
            raise SystemExit("belum ada checkpoint di runs/")
        ckpt = found

    model, class_names, meta = load_predictor(ckpt)
    res = predict(args.image, model, class_names, img_size=meta["img_size"])
    print(json.dumps({"meta": meta, "prediction": res},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
