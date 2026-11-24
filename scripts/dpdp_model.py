# Copyright (c) 2026 Tim DPdp.
# Licensed under the MIT License.
"""Arsitektur model DPdp + transform evaluasi.

Satu sumber kebenaran untuk training (`scripts/train_tl.py`) dan inferensi
(`scripts/predict.py`, `app.py`), supaya keduanya tidak pernah berbeda
arsitektur secara diam-diam.

`pretrained=True` memuat bobot ImageNet. Pelatihan memakainya; inferensi TIDAK
(periksa checkpoint), supaya app tidak pernah mengunduh apa pun.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ARCHITECTURES = ("resnet18_tl", "mobilenet_v2_tl")


def eval_transform(img_size: int = 224):
    """Transform evaluasi. Ukuran harus sama dengan saat model dilatih."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(arch: str, n_classes: int, pretrained: bool = False) -> nn.Module:
    """Bangun arsitektur berdasarkan nama yang tersimpan di checkpoint.

    `pretrained=False` dipakai inferensi: bobot acak lalu ditimpa checkpoint.
    """
    if arch == "resnet18_tl":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model

    if arch == "mobilenet_v2_tl":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        return model

    raise ValueError(f"arsitektur tidak dikenal: {arch!r}. "
                     f"Pilihan: {', '.join(ARCHITECTURES)}")
