"""Entrena un clasificador de neumonía en radiografías de tórax (PneumoniaMNIST, MedMNIST v2).

Modelo: ResNet-18 preentrenada en ImageNet, ajustada (fine-tuning) a 2 clases.
Uso:  python src/train_xray.py --epochs 12 --size 128
Salida: models/xray_resnet18.pt, results/xray_metrics.json, results/*.png
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from medmnist import PneumoniaMNIST
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
CLASSES = ["Normal", "Neumonía"]


def loaders(size, batch):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    to3 = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    train_tf = transforms.Compose([
        transforms.ToTensor(), to3,
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.Resize(224, antialias=True), norm])
    eval_tf = transforms.Compose([transforms.ToTensor(), to3, transforms.Resize(224, antialias=True), norm])
    DATA.mkdir(exist_ok=True)
    ds = {s: PneumoniaMNIST(split=s, transform=train_tf if s == "train" else eval_tf,
                            download=True, size=size, root=str(DATA)) for s in ("train", "val", "test")}
    dl = {s: DataLoader(d, batch_size=batch, shuffle=(s == "train"), num_workers=0, pin_memory=True)
          for s, d in ds.items()}
    return ds, dl


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    probs, ys = [], []
    for x, y in dl:
        p = torch.softmax(model(x.to(dev)), 1)[:, 1]
        probs.append(p.cpu().numpy()); ys.append(y.view(-1).numpy())
    return np.concatenate(ys), np.concatenate(probs)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": (tp + tn) / len(y), "sensibilidad": tp / (tp + fn), "especificidad": tn / (tn + fp),
            "vpp": tp / (tp + fp) if tp + fp else 0.0, "auc": roc_auc_score(y, p),
            "matriz": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)}, "umbral": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, dl = loaders(a.size, a.batch)
    labels = ds["train"].labels.ravel()
    weights = torch.tensor(len(labels) / (2 * np.bincount(labels)), dtype=torch.float32, device=dev)
    print(f"device={dev} train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])} clases={np.bincount(labels)}")

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    best_auc, hist = 0.0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for x, y in dl["train"]:
            x, y = x.to(dev), y.view(-1).long().to(dev)
            opt.zero_grad()
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        yv, pv = predict(model, dl["val"], dev)
        mv = metrics(yv, pv)
        hist.append({"epoch": ep, "loss": tot / len(ds["train"]), "val_auc": mv["auc"], "val_acc": mv["accuracy"]})
        print(f"epoch {ep:2d} loss={hist[-1]['loss']:.4f} val_auc={mv['auc']:.4f} val_acc={mv['accuracy']:.4f}")
        if mv["auc"] > best_auc:
            best_auc = mv["auc"]
            torch.save(model.state_dict(), MODELS / "xray_resnet18.pt")

    # Umbral elegido en validación (máximo índice de Youden); el test se mide una sola vez con ese umbral.
    model.load_state_dict(torch.load(MODELS / "xray_resnet18.pt", map_location=dev))
    yv, pv = predict(model, dl["val"], dev)
    fpr, tpr, thr = roc_curve(yv, pv)
    best_thr = float(thr[np.argmax(tpr - fpr)])
    yt, pt = predict(model, dl["test"], dev)
    out = {"dataset": "PneumoniaMNIST (MedMNIST v2), radiografías pediátricas de tórax", "imagen": a.size,
           "epochs": a.epochs, "minutos_entrenamiento": round((time.time() - t0) / 60, 1), "device": dev,
           "test_umbral_0_5": metrics(yt, pt, 0.5), "test_umbral_validacion": metrics(yt, pt, best_thr), "historial": hist}
    (RESULTS / "xray_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    m = out["test_umbral_validacion"]["matriz"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow([[m["vn"], m["fp"]], [m["fn"], m["vp"]]], cmap="Blues")
    for (i, j), v in np.ndenumerate([[m["vn"], m["fp"]], [m["fn"], m["vp"]]]):
        ax[0].text(j, i, v, ha="center", va="center", fontsize=14)
    ax[0].set_xticks([0, 1], CLASSES); ax[0].set_yticks([0, 1], CLASSES)
    ax[0].set_xlabel("Predicción"); ax[0].set_ylabel("Real"); ax[0].set_title("Matriz de confusión (test)")
    f, t, _ = roc_curve(yt, pt)
    ax[1].plot(f, t, label=f"AUC = {roc_auc_score(yt, pt):.3f}"); ax[1].plot([0, 1], [0, 1], "--", c="gray")
    ax[1].set_xlabel("1 - especificidad"); ax[1].set_ylabel("Sensibilidad"); ax[1].set_title("Curva ROC (test)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS / "xray_confusion_roc.png", dpi=130)
    print(json.dumps({k: out[k] for k in ("test_umbral_0_5", "test_umbral_validacion")}, indent=2, default=float))


if __name__ == "__main__":
    main()
