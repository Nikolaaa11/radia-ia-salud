"""RadIA — demo local (Gradio).

Pestaña 1: radiografía de tórax -> probabilidad de neumonía + mapa de calor (Grad-CAM).
Pestaña 2: síntomas en texto libre -> 3 diagnósticos orientativos más probables.
Uso educativo. No es un dispositivo médico ni reemplaza la evaluación de un profesional de salud.
Ejecutar:  python app.py   (abre http://127.0.0.1:7860)
"""
import json
import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.gradcam import GradCAM, overlay
from src.train_xray import build_model

ROOT = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AVISO = "⚠️ Uso educativo. No es un dispositivo médico ni reemplaza el diagnóstico de un profesional de salud."

model = build_model().to(DEV)
model.load_state_dict(torch.load(ROOT / "models" / "xray_resnet18.pt", map_location=DEV))
model.eval()
cam = GradCAM(model, model.layer4[-1])
metrics = json.loads((ROOT / "results" / "xray_metrics.json").read_text(encoding="utf-8"))
THR = metrics["test_umbral_validacion"]["umbral"]
TF = transforms.Compose([
    transforms.Grayscale(), transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

symptoms = joblib.load(ROOT / "models" / "symptoms.joblib")
# Glosario mínimo español -> inglés (el dataset de síntomas está en inglés).
GLOSARIO = {
    "fiebre": "fever", "tos": "cough", "dolor de cabeza": "headache", "cefalea": "headache",
    "dolor de garganta": "sore throat", "picazón": "itching", "picazon": "itching", "comezón": "itching",
    "sarpullido": "rash", "erupción": "rash", "manchas en la piel": "skin patches", "náuseas": "nausea",
    "nauseas": "nausea", "vómitos": "vomiting", "vomitos": "vomiting", "diarrea": "diarrhea",
    "dolor abdominal": "abdominal pain", "dolor de estómago": "stomach pain", "dolor de estomago": "stomach pain",
    "cansancio": "fatigue", "fatiga": "fatigue", "falta de aire": "shortness of breath", "ahogo": "shortness of breath",
    "dolor de pecho": "chest pain", "dolor en el pecho": "chest pain", "mareo": "dizziness", "escalofríos": "chills",
    "escalofrios": "chills", "dolor articular": "joint pain", "dolor de articulaciones": "joint pain",
    "dolor muscular": "muscle pain", "estornudos": "sneezing", "congestión nasal": "runny nose congestion",
    "congestion nasal": "runny nose congestion", "ojos rojos": "red eyes", "orina frecuente": "frequent urination",
    "ardor al orinar": "burning urination", "sed": "thirst", "pérdida de peso": "weight loss",
    "perdida de peso": "weight loss", "visión borrosa": "blurred vision", "vision borrosa": "blurred vision",
    "dolor de espalda": "back pain", "cuello rígido": "stiff neck", "ojos amarillos": "yellow eyes",
    "piel amarilla": "yellow skin", "acidez": "acid reflux heartburn", "estreñimiento": "constipation",
    "dolor de oído": "ear pain", "sangrado": "bleeding", "hinchazón": "swelling", "hinchazon": "swelling",
    "sudoración": "sweating", "sudoracion": "sweating", "palpitaciones": "palpitations", "flema": "phlegm mucus",
    "mocos": "mucus", "dolor al tragar": "pain swallowing", "pérdida de apetito": "loss of appetite",
}


def a_ingles(texto):
    t = texto.lower()
    for es, en in sorted(GLOSARIO.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{re.escape(es)}\b", en, t)
    return t


def radiografia(img):
    if img is None:
        return None, "Sube una radiografía de tórax."
    pil = Image.fromarray(img).convert("L")
    x = TF(pil).unsqueeze(0).to(DEV)
    heat, p = cam(x, cls=1)
    gray = np.asarray(pil.resize((224, 224)), dtype=np.float32) / 255.0
    etiqueta = "Compatible con neumonía" if p >= THR else "Sin hallazgos compatibles con neumonía"
    return overlay(gray, heat), f"**{etiqueta}** — probabilidad de neumonía: {p:.1%} (umbral {THR:.2f})\n\n{AVISO}"


def sintomas(texto):
    if not texto or len(texto.split()) < 3:
        return "Describe al menos 3 o 4 síntomas."
    probs = symptoms.predict_proba([a_ingles(texto)])[0]
    top = np.argsort(probs)[::-1][:3]
    filas = "\n".join(f"{i + 1}. **{symptoms.classes_[k]}** — {probs[k]:.0%}" for i, k in enumerate(top))
    return f"Diagnósticos orientativos más probables:\n\n{filas}\n\n{AVISO}"


with gr.Blocks(title="RadIA") as demo:
    gr.Markdown("# RadIA — IA aplicada a salud\nDetección de neumonía en radiografías de tórax y orientación por síntomas.")
    with gr.Tab("Radiografía de tórax"):
        with gr.Row():
            inp = gr.Image(label="Radiografía (PNG/JPG)")
            out = gr.Image(label="Mapa de calor (Grad-CAM)")
        txt = gr.Markdown()
        gr.Button("Analizar").click(radiografia, inp, [out, txt])
    with gr.Tab("Síntomas"):
        s = gr.Textbox(label="Describe los síntomas", lines=4,
                       placeholder="Ej.: fiebre alta, tos con flema, falta de aire y dolor de pecho")
        r = gr.Markdown()
        gr.Button("Orientar").click(sintomas, s, r)

if __name__ == "__main__":
    demo.launch()
