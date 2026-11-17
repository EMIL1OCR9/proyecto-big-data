"""LungAI: baseline experimental de riesgo de diagnóstico de cáncer a 12 meses.

IMPORTANTE: utiliza SOLO el resultado de cribado T0 (scr_res0); NO analiza
los píxeles de la CT y NO predice la malignidad de un nódulo. Requiere el
Parquet completo con splits por paciente ya existentes. No guarda IDs.

Ejecutar desde la raíz del proyecto:
 python backend/training/entrenar_riesgo_cancer_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ARCHIVO = Path("outputs/monai_dataset/cohorte_cancer_12m.parquet")
CARPETA_SALIDA = Path("outputs/cancer_risk_baseline")
OBJETIVO = "etiqueta_cancer_12m"
VARIABLES = ["scr_res0"]
SPLITS = ("train", "validation", "test")


def variables_de_entrada(tabla: pd.DataFrame) -> pd.DataFrame:
    # Es un código categórico de lectura de cribado, no una escala numérica.
    valores = pd.to_numeric(tabla["scr_res0"], errors="coerce")
    return pd.DataFrame({"scr_res0": valores.astype("Int64").astype(str)}, index=tabla.index)


def evaluar(y: np.ndarray, prob: np.ndarray, umbral: float) -> dict:
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ejemplos": int(y.size),
        "positivos": int(y.sum()),
        "prevalencia": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision_pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "umbral": float(umbral),
        "sensibilidad_recall": float(recall_score(y, pred, zero_division=0)),
        "precision_valor_predictivo_positivo": float(precision_score(y, pred, zero_division=0)),
        "especificidad": float(tn / (tn + fp)) if tn + fp else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main() -> None:
    if not ARCHIVO.is_file():
        raise SystemExit(f"No se encontró: {ARCHIVO}. Ejecuta desde la raíz del repo.")

    tabla = pd.read_parquet(ARCHIVO)
    necesarias = {"PatientID", OBJETIVO, "split", *VARIABLES}
    faltantes = sorted(necesarias - set(tabla.columns))
    if faltantes:
        raise SystemExit(f"El Parquet carece de columnas necesarias: {faltantes}")

    # Una fila por paciente, y exclusión de etiquetas desconocidas.
    if tabla["PatientID"].isna().any() or tabla["PatientID"].duplicated().any():
        raise SystemExit("La cohorte contiene IDs faltantes o pacientes duplicados.")
    tabla = tabla[tabla[OBJETIVO].notna()].copy()
    tabla[OBJETIVO] = pd.to_numeric(tabla[OBJETIVO], errors="raise").astype(int)
    if not tabla[OBJETIVO].isin([0, 1]).all():
        raise SystemExit("Etiquetas fuera de 0 y 1.")
    if not tabla["split"].isin(SPLITS).all():
        raise SystemExit("Hay filas etiquetadas sin split válido.")

    partes = {nombre: tabla.loc[tabla["split"].eq(nombre)].copy() for nombre in SPLITS}
    for nombre, parte in partes.items():
        num_pos = int(parte[OBJETIVO].sum())
        print(f"{nombre}: {len(parte)} registros, {num_pos} positivos ({parte[OBJETIVO].mean():.2%})")
        if not 0 < num_pos < len(parte):
            raise SystemExit(f"El conjunto {nombre} necesita ambas clases.")

    # No se usan IDs, fecha de diagnóstico, días sin cáncer, ni etiqueta
    # como predictores: evitar fuga de información del futuro.
    preproceso = ColumnTransformer(
        [("cribado", OneHotEncoder(handle_unknown="ignore"), VARIABLES)],
        remainder="drop",
    )
    modelo = Pipeline([
        ("preproceso", preproceso),
        ("clasificador", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    entrenamiento = partes["train"]
    modelo.fit(variables_de_entrada(entrenamiento), entrenamiento[OBJETIVO].to_numpy())

    validacion = partes["validation"]
    y_val = validacion[OBJETIVO].to_numpy()
    prob_val = modelo.predict_proba(variables_de_entrada(validacion))[:, 1]

    # Elegimos umbral SOLO con validation: buscamos >=80% de sensibilidad
    # y, entre esos umbrales, el menor número relativo de falsos positivos.
    fpr, tpr, umbrales = roc_curve(y_val, prob_val)
    candidatos = np.flatnonzero(tpr >= 0.80)
    idx = int(candidatos[np.argmin(fpr[candidatos])])
    umbral = float(umbrales[idx])

    prueba = partes["test"]
    y_test = prueba[OBJETIVO].to_numpy()
    prob_test = modelo.predict_proba(variables_de_entrada(prueba))[:, 1]

    metricas = {
        "descripcion": "Baseline de riesgo a 12 meses DESPUES del resultado del cribado T0; no analiza CT.",
        "algoritmo": "Regresión logística / OneHot(scr_res0)",
        "predictores": VARIABLES,
        "fuente": str(ARCHIVO),
        "entrenamiento_registros": int(len(entrenamiento)),
        "seleccion_umbral": "Validation: sensibilidad >= 0.80; mínimo FPR posible",
        "validacion": evaluar(y_val, prob_val, umbral),
        "prueba_no_utilizada_para_entrenar": evaluar(y_test, prob_test, umbral),
        "advertencia": (
            "Investigación solamente. No equivale a diagnóstico, no usa los píxeles CT, "
            "ni confirma malignidad de los candidatos a nódulo."
        ),
    }

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    salida_modelo = CARPETA_SALIDA / "modelo_riesgo_12m.joblib"
    salida_metricas = CARPETA_SALIDA / "metricas.json"
    if salida_modelo.exists() or salida_metricas.exists():
        raise SystemExit("Ya existen archivos de salida; no se sobrescribió nada.")

    joblib.dump({"modelo": modelo, "variables": VARIABLES, "umbral": umbral}, salida_modelo)
    salida_metricas.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nEVALUACIÓN FINAL (TEST, sin entrenar ni ajustar umbrales con TEST):")
    finales = metricas["prueba_no_utilizada_para_entrenar"]
    for clave in ["prevalencia", "roc_auc", "average_precision_pr_auc", "sensibilidad_recall", "precision_valor_predictivo_positivo", "especificidad", "tp", "fp", "tn", "fn"]:
        print(f"  {clave}: {finales[clave]:.4f}" if isinstance(finales[clave], float) else f"  {clave}: {finales[clave]}")
    print(f"\nModelo guardado: {salida_modelo}")
    print(f"Métricas agregadas: {salida_metricas}")
    print("NOTA: este baseline NO inspecciona la tomografía, solo scr_res0.")


if __name__ == "__main__":
    main()
