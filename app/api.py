"""Servicio REST educativo para el clasificador de cultivos."""
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
FEATURES = CONFIG["feature_columns"]
THRESHOLD = float(CONFIG.get("threshold", 0.5))
MODE = CONFIG.get("mode", "binary")
ONNX_FILE = ROOT / "models" / "modelo.onnx"
BUNDLE_FILE = ROOT / CONFIG.get("model_bundle_path", "models/rf_final_b2_pipeline.pkl")
_session = None
_model = None
_input_name = None

class InputData(BaseModel):
    """Mediciones de entrada con límites plausibles del dominio Crop Recommendation."""
    N: float = Field(ge=0, le=140, description="Nitrógeno del suelo")
    P: float = Field(ge=5, le=145, description="Fósforo del suelo")
    K: float = Field(ge=5, le=205, description="Potasio del suelo")
    temperature: float = Field(ge=8, le=45, description="Temperatura en °C")
    humidity: float = Field(ge=14, le=100, description="Humedad relativa (%)")
    ph: float = Field(ge=3, le=10, description="pH del suelo")
    rainfall: float = Field(ge=20, le=300, description="Lluvia en mm")

app = FastAPI(title="Crop Recommendation - Clase 15", version="1.0.0")

@app.on_event("startup")
def load_model():
    """Carga ONNX Runtime cuando está disponible; usa el bundle de sklearn como respaldo."""
    global _session, _model, _input_name
    if ONNX_FILE.exists():
        try:
            import onnxruntime as ort
        except ImportError:
            import joblib
            _model = joblib.load(BUNDLE_FILE)
        else:
            _session = ort.InferenceSession(
                str(ONNX_FILE), providers=["CPUExecutionProvider"]
            )
            _input_name = _session.get_inputs()[0].name
    else:
        import joblib
        _model = joblib.load(BUNDLE_FILE)

def predict_one(values):
    """Devuelve clase y probabilidad usando el runtime activo."""
    row = np.asarray([[values[name] for name in FEATURES]], dtype=np.float32)
    if _session is not None:
        outputs = _session.run(None, {_input_name: row})
        label = np.asarray(outputs[0]).reshape(-1)[0].item()
        probability_output = outputs[1]
        if isinstance(probability_output, list) and probability_output and isinstance(probability_output[0], dict):
            probability_map = probability_output[0]
            if MODE == "binary":
                positive_probability = float(probability_map.get(1, probability_map.get("1", 0.0)))
                label = 1 if positive_probability >= THRESHOLD else 0
                confidence = positive_probability if label == 1 else 1.0 - positive_probability
            else:
                confidence = float(max(probability_map.values()))
        else:
            probabilities = np.asarray(probability_output)
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(1, -1)
            if MODE == "binary":
                positive_probability = float(probabilities[0, 1])
                label = 1 if positive_probability >= THRESHOLD else 0
                confidence = positive_probability if label == 1 else 1.0 - positive_probability
            else:
                confidence = float(np.max(probabilities[0]))
    else:
        probabilities = _model.predict_proba(row)[0]
        classes = list(_model.classes_)
        if MODE == "binary":
            positive_probability = float(probabilities[classes.index(1)])
            label = 1 if positive_probability >= THRESHOLD else 0
            confidence = positive_probability if label == 1 else 1.0 - positive_probability
        else:
            index = int(np.argmax(probabilities))
            label = classes[index]
            confidence = float(probabilities[index])
    if MODE == "binary":
        prediction = "rice" if int(label) == 1 else "no rice"
    else:
        prediction = str(label)
    return {"prediction": prediction, "class_id": int(label), "probability": confidence, "confidence": confidence}

@app.get("/health")
def health():
    """Comprueba que el servicio y el artefacto de inferencia estén cargados."""
    return {"status": "ok", "mode": MODE, "runtime": "onnx" if _session is not None else "scikit-learn"}

@app.post("/predict")
def predict(data: InputData):
    """Valida siete features y retorna la clase y su confianza."""
    return predict_one(data.model_dump())
