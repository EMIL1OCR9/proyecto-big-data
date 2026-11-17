"""Extensión opcional de la API local de LungAI: baseline de riesgo a 12 meses.

Se inicia con:
    python -m uvicorn backend.api.riesgo_experimental_app:app --host 127.0.0.1 --port 8081

No entrena modelos ni lee imágenes CT. No es un diagnóstico.
Requiere el baseline entrenado con scr_res0, Parquet y scikit-learn.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException

from backend.api.main import app  # Conserva las rutas y CORS ya existentes.

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/monai_dataset/cohorte_cancer_12m.parquet"
ARTIFACT = ROOT / "outputs/cancer_risk_baseline/modelo_riesgo_12m.joblib"
METRICS = ROOT / "outputs/cancer_risk_baseline/metricas.json"
PATIENT_ID = re.compile(r"[0-9]+\Z")

ADVERTENCIA = (
    "Modelo de investigación sin validación clínica. Usa únicamente scr_res0 "
    "(resultado del cribado inicial), no los píxeles de la tomografía ni "
    "las detecciones de nódulos. No diagnostica ni descarta cáncer."
)


@lru_cache(maxsize=1)
def load_resources():
    """Lee únicamente el modelo local de confianza y columnas mínimas de la muestra."""
    if not ARTIFACT.is_file() or not COHORT.is_file():
        raise FileNotFoundError("Faltan los artefactos del baseline o la cohorte.")

    # ADVERTENCIA: joblib ejecuta pickle. Cargar solo archivos creados localmente
    # por el propio entrenamiento, nunca uno recibido de terceros.
    saved = joblib.load(ARTIFACT)
    if not isinstance(saved, dict) or saved.get("variables") != ["scr_res0"]:
        raise ValueError("El archivo no corresponde al baseline esperado.")
    model = saved.get("modelo")
    threshold = float(saved.get("umbral"))
    if model is None or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("El baseline tiene un umbral o modelo inválido.")

    data = pd.read_parquet(
        COHORT, columns=["PatientID", "scr_res0", "split", "en_muestra"]
    )
    if data["PatientID"].isna().any() or data["PatientID"].duplicated().any():
        raise ValueError("La cohorte tiene IDs duplicados o faltantes.")
    sampled = data.loc[data["en_muestra"].eq(True)].copy()
    sampled["PatientID"] = sampled["PatientID"].astype(str).str.strip()
    sampled = sampled.set_index("PatientID", verify_integrity=True)
    return model, threshold, sampled


@app.get("/pacientes/{patient_id}/riesgo-experimental", tags=["Riesgo experimental"])
def patient_risk(patient_id: str):
    if not PATIENT_ID.fullmatch(patient_id):
        raise HTTPException(status_code=400, detail="Identificador inválido.")

    try:
        model, threshold, patients = load_resources()
    except (OSError, ValueError, KeyError, ImportError) as exc:
        # Evitar exponer rutas del sistema o información clínica en errores HTTP.
        raise HTTPException(
            status_code=503, detail="Modelo de investigación no disponible."
        ) from exc

    if patient_id not in patients.index:
        raise HTTPException(status_code=404, detail="Paciente no encontrado en la muestra.")

    raw = patients.at[patient_id, "scr_res0"]
    if pd.isna(raw):
        raise HTTPException(status_code=422, detail="Falta el resultado del cribado inicial.")

    # Reproduce EXACTAMENTE la transformación de entrenar_riesgo_cancer_baseline.py.
    feature = pd.to_numeric(pd.Series([raw]), errors="coerce").astype("Int64").astype(str)
    if feature.iloc[0] == "<NA>":
        raise HTTPException(status_code=422, detail="Resultado del cribado inválido.")
    input_data = pd.DataFrame({"scr_res0": feature})

    try:
        score = float(model.predict_proba(input_data)[0, 1])
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=503, detail="No se pudo evaluar el baseline.") from exc

    if not math.isfinite(score) or not 0 <= score <= 1:
        raise HTTPException(status_code=503, detail="El modelo devolvió un resultado inválido.")

    return {
        "patient_id": patient_id,
        "modelo": "baseline_scr_res0",
        "puntaje_modelo": round(score, 6),
        "umbral_exploratorio": round(threshold, 6),
        "clasificacion_exploratoria": (
            "sobre_umbral" if score >= threshold else "bajo_umbral"
        ),
        "split": str(patients.at[patient_id, "split"]),
        "usa_imagen_ct": False,
        "diagnostico_clinico": False,
        "advertencia": ADVERTENCIA,
    }


@app.get("/riesgo-experimental/metricas", tags=["Riesgo experimental"])
def risk_metrics():
    if not METRICS.is_file():
        raise HTTPException(status_code=503, detail="Métricas del baseline no disponibles.")
    try:
        stats = json.loads(METRICS.read_text(encoding="utf-8"))
        return {
            "modelo": "baseline_scr_res0",
            "predictores": ["scr_res0"],
            "test": stats["prueba_no_utilizada_para_entrenar"],
            "usa_imagen_ct": False,
            "diagnostico_clinico": False,
            "advertencia": ADVERTENCIA,
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Métricas del baseline inválidas.") from exc
