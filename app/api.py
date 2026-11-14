"""Servicio REST para el clasificador existente de recomendación de cultivos."""

import json
import logging
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
FEATURES = CONFIG["feature_columns"]
THRESHOLD = float(CONFIG.get("threshold", 0.5))
MODE = str(os.environ.get("CROP_MODEL_MODE", CONFIG.get("mode", "binary"))).lower()
ONNX_FILE = ROOT / "models" / (
    "random_forest.onnx" if MODE == "multiclass" else "rf_final_b2_pipeline.onnx"
)
DEFAULT_BUNDLE = (
    "models/random_forest.pkl"
    if MODE == "multiclass" and "CROP_MODEL_MODE" in os.environ
    else CONFIG.get("model_bundle_path", "models/rf_final_b2_pipeline.pkl")
)
BUNDLE_FILE = ROOT / os.environ.get("CROP_MODEL_BUNDLE", DEFAULT_BUNDLE)
LABEL_ENCODER_FILE = ROOT / "models" / "label_encoder.pkl"
logger = logging.getLogger(__name__)

_session: Any = None
_model: Any = None
_label_encoder: Any = None
_input_name: str | None = None
_runtime = "not_loaded"


class InputData(BaseModel):
    """Las siete mediciones de entrada, limitadas a rangos plausibles."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    N: float = Field(ge=0, le=140, description="Nitrógeno del suelo")
    P: float = Field(ge=5, le=145, description="Fósforo del suelo")
    K: float = Field(ge=5, le=205, description="Potasio del suelo")
    temperature: float = Field(ge=8, le=45, description="Temperatura en °C")
    humidity: float = Field(ge=14, le=100, description="Humedad relativa (%)")
    ph: float = Field(ge=3, le=10, description="pH del suelo")
    rainfall: float = Field(ge=20, le=300, description="Lluvia en mm")


class PredictionResponse(BaseModel):
    prediction: str
    class_id: int | str
    probability: float
    confidence: float
    mode: str


def load_model() -> None:
    """Carga el bundle de sklearn y, si existe, su exportación ONNX dedicada."""
    global _input_name, _label_encoder, _model, _runtime, _session

    if len(FEATURES) != 7:
        raise ValueError(f"Se esperaban siete variables de entrada; config contiene {len(FEATURES)}.")
    if MODE not in {"binary", "multiclass"}:
        raise ValueError(f"Modo de inferencia no soportado: {MODE!r}.")

    import joblib

    _model = joblib.load(BUNDLE_FILE)
    if not hasattr(_model, "predict_proba") or not hasattr(_model, "classes_"):
        raise TypeError("El bundle debe exponer predict_proba y classes_.")
    if len(_model.classes_) < 2:
        raise ValueError("El clasificador cargado debe tener al menos dos clases.")

    if MODE == "binary" and set(_model.classes_) != {0, 1}:
        raise ValueError("El modelo binario debe tener las clases 0 (no rice) y 1 (rice).")

    if MODE == "multiclass":
        if len(_model.classes_) < 3:
            raise ValueError("El modo multiclase requiere al menos tres clases.")
        numeric_classes = all(
            isinstance(value, (int, np.integer)) for value in _model.classes_
        )
        if LABEL_ENCODER_FILE.exists():
            _label_encoder = joblib.load(LABEL_ENCODER_FILE)
            if len(_model.classes_) != len(_label_encoder.classes_):
                raise ValueError(
                    "El clasificador multiclase y LabelEncoder no tienen la misma "
                    "cantidad de clases."
                )
            if numeric_classes and set(map(int, _model.classes_)) != set(
                range(len(_label_encoder.classes_))
            ):
                raise ValueError("Los códigos del clasificador no coinciden con LabelEncoder.")
        elif numeric_classes:
            raise FileNotFoundError(
                f"El modelo multiclase devuelve códigos numéricos; falta {LABEL_ENCODER_FILE}."
            )

    _session = None
    _input_name = None
    _runtime = "scikit-learn"
    if ONNX_FILE.exists():
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "Existe %s, pero falta onnxruntime; se usará el bundle scikit-learn.",
                ONNX_FILE,
            )
        else:
            session = ort.InferenceSession(
                str(ONNX_FILE), providers=["CPUExecutionProvider"]
            )
            input_info = session.get_inputs()[0]
            if len(input_info.shape) != 2 or input_info.shape[1] != len(FEATURES):
                raise ValueError(
                    f"ONNX espera forma {input_info.shape}; se requieren siete variables."
                )
            _session = session
            _input_name = input_info.name
            _runtime = "onnx"


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _probability_vector(output: Any) -> np.ndarray:
    """Normaliza salidas ONNX ZipMap y tensores a orden de clases de sklearn."""
    classes = list(_model.classes_)
    if isinstance(output, list) and output and isinstance(output[0], dict):
        mapping = output[0]
        values = []
        for class_value in classes:
            value = mapping.get(class_value, mapping.get(str(class_value)))
            if value is None:
                raise ValueError(f"ONNX no devolvió probabilidad para clase {class_value!r}.")
            values.append(float(value))
        return np.asarray(values, dtype=float)

    probabilities = np.asarray(output, dtype=float)
    if probabilities.ndim == 2:
        probabilities = probabilities[0]
    if probabilities.ndim != 1 or len(probabilities) != len(classes):
        raise ValueError(
            f"Forma de probabilidades incompatible: {probabilities.shape}; "
            f"se esperaban {len(classes)} clases."
        )
    return probabilities


def _display_class(class_value: Any) -> str:
    value = _python_scalar(class_value)
    if MODE == "binary":
        return "rice" if value == 1 else "no rice"
    if _label_encoder is not None and isinstance(value, (int, np.integer)):
        return str(_label_encoder.inverse_transform([int(value)])[0])
    return str(value)


def predict_one(values: dict[str, float]) -> dict[str, Any]:
    """Predice con el motor cargado y devuelve la clase más su confianza."""
    if _model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado.")

    row = np.asarray([[values[name] for name in FEATURES]], dtype=np.float32)
    if _session is not None:
        outputs = _session.run(None, {_input_name: row})
        probabilities = _probability_vector(outputs[1])
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            probabilities = np.asarray(_model.predict_proba(row)[0], dtype=float)

    classes = list(_model.classes_)
    if MODE == "binary":
        positive_index = classes.index(1)
        positive_probability = float(probabilities[positive_index])
        class_index = positive_index if positive_probability >= THRESHOLD else 1 - positive_index
        confidence = (
            positive_probability
            if class_index == positive_index
            else 1.0 - positive_probability
        )
    else:
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

    class_value = _python_scalar(classes[class_index])
    return {
        "prediction": _display_class(class_value),
        "class_id": class_value,
        "probability": confidence,
        "confidence": confidence,
        "mode": MODE,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Crop Recommendation - Clase 15",
    description=(
        "Clasificador de cultivos con validación de siete mediciones. "
        "CROP_MODEL_MODE=multiclass selecciona el Random Forest de 22 cultivos."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Indica si el modelo está listo y qué motor de inferencia se usa."""
    if _model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado.")
    return {
        "status": "ok",
        "mode": MODE,
        "runtime": _runtime,
        "classes": len(_model.classes_),
        "features": len(FEATURES),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData) -> dict[str, Any]:
    """Valida las siete variables y devuelve una predicción real del modelo."""
    return predict_one(data.model_dump())
