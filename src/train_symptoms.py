"""Orientación diagnóstica por síntomas: texto libre -> 22 diagnósticos (dataset público Symptom2Disease).

Modelo: TF-IDF (palabras y n-gramas de caracteres) + regresión logística.
Los textos del dataset están en inglés; la app traduce los síntomas en español con un glosario simple
antes de clasificar (limitación documentada en el README).
Uso:  python src/train_symptoms.py
Salida: models/symptoms.joblib, results/symptoms_metrics.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
REPO = "gretelai/symptom_to_diagnosis"


def load(split):
    path = hf_hub_download(REPO, f"{split}.jsonl", repo_type="dataset", cache_dir=str(ROOT / "data" / "hf"))
    return pd.read_json(path, lines=True)


def main():
    tr, te = load("train"), load("test")
    print(f"train={len(tr)} test={len(te)} clases={tr['output_text'].nunique()}")
    pipe = Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=3000, C=8.0)),
    ])
    pipe.fit(tr["input_text"], tr["output_text"])
    pred = pipe.predict(te["input_text"])
    out = {"dataset": f"{REPO} (Symptom2Disease)", "train": len(tr), "test": len(te),
           "clases": sorted(tr["output_text"].unique()),
           "accuracy": accuracy_score(te["output_text"], pred),
           "f1_macro": f1_score(te["output_text"], pred, average="macro"),
           "por_clase": classification_report(te["output_text"], pred, output_dict=True, zero_division=0)}
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "symptoms.joblib")
    (ROOT / "results" / "symptoms_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"accuracy={out['accuracy']:.4f} f1_macro={out['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
