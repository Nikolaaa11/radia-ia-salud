"""Genera results/gradcam_ejemplos.png con radiografías de test y su mapa de calor."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import PneumoniaMNIST
from torchvision import transforms

from gradcam import GradCAM, overlay
from train_xray import build_model

ROOT = Path(__file__).resolve().parents[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
thr = json.loads((ROOT / "results" / "xray_metrics.json").read_text(encoding="utf-8"))["test_umbral_validacion"]["umbral"]
model = build_model().to(dev)
model.load_state_dict(torch.load(ROOT / "models" / "xray_resnet18.pt", map_location=dev))
cam = GradCAM(model, model.layer4[-1])
tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                         transforms.Resize(224, antialias=True),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds = PneumoniaMNIST(split="test", download=True, size=128, root=str(ROOT / "data"))
labels = ds.labels.ravel()
rng = np.random.default_rng(7)
idx = list(rng.choice(np.where(labels == 0)[0], 3, replace=False)) + list(rng.choice(np.where(labels == 1)[0], 3, replace=False))
fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
for k, i in enumerate(idx):
    img, y = ds[i]
    gray = np.asarray(img, dtype=np.float32) / 255.0
    heat, p = cam(tf(img).unsqueeze(0).to(dev), cls=1)
    pred = "Neumonía" if p >= thr else "Normal"
    axes[0, k].imshow(gray, cmap="gray"); axes[0, k].set_title(f"Real: {'Neumonía' if y[0] else 'Normal'}", fontsize=10)
    axes[1, k].imshow(overlay(gray, heat)); axes[1, k].set_title(f"Modelo: {pred} ({p:.0%})", fontsize=10)
for a in axes.ravel():
    a.axis("off")
fig.suptitle("RadIA — radiografías de test y mapa de calor Grad-CAM (zonas que más pesan en la predicción)")
fig.tight_layout()
fig.savefig(ROOT / "results" / "gradcam_ejemplos.png", dpi=120)
for i in idx[:1] + idx[3:4]:
    ds[i][0].resize((256, 256)).save(ROOT / "results" / f"ejemplo_{'neumonia' if labels[i] else 'normal'}.png")
print("ok", [int(i) for i in idx])
