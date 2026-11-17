"""LungAI: auditar SOLO la validacion del modelo MONAI CT ampliado.

Uso desde raiz del repo con ~/lungai-venv:
  python backend/training/revisar_validacion_ct_monai.py

No entrena, no utiliza imagenes ni etiquetas del split test y no escribe archivos.
Los puntajes del modelo NO son probabilidades clinicas.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import resnet10

# Mismo directorio que el entrenador original: reutiliza su preprocesado exacto.
from entrenar_ct_monai_ampliado import preparar_ct, revisar_lote

MANIFIESTO = Path("outputs/monai_dataset/lote_ct_3d_ampliado.parquet")
DATOS = Path("data/monai")
PESOS = Path("outputs/monai_ct_ampliado/modelo_resnet10_ampliado.pt")
REPORTE = Path("outputs/monai_ct_ampliado/entrenamiento.json")


def main() -> None:
    if not PESOS.is_file() or not REPORTE.is_file():
        raise SystemExit("No se encontraron el modelo y el reporte del entrenamiento ampliado.")

    reporte = json.loads(REPORTE.read_text(encoding="utf-8"))
    actual = hashlib.sha256(MANIFIESTO.read_bytes()).hexdigest()
    if reporte.get("manifest_sha256") != actual:
        raise SystemExit("El manifiesto cambio desde el entrenamiento: se cancela la auditoria.")

    grupos = revisar_lote(MANIFIESTO, DATOS)
    validacion = grupos["validation"]
    if len(validacion) != 12 or sum(etiqueta for _, etiqueta in validacion) != 6:
        raise SystemExit("La validacion no contiene las 12 CT esperadas (6 por clase).")

    # Archivo .pt generado localmente por nuestro entrenador. Se cargan solo tensores y tipos simples.
    checkpoint = torch.load(PESOS, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "monai.resnet10_3d":
        raise SystemExit("Arquitectura no reconocida en el checkpoint.")
    tamano = tuple(int(v) for v in checkpoint["size_xyz"])
    if len(tamano) != 3 or any(v < 32 or v > 256 for v in tamano):
        raise SystemExit("Tamano de entrada inesperado.")
    epoca = int(checkpoint["best_epoch"])
    if epoca != int(reporte["mejor_epoca_por_val_loss"]):
        raise SystemExit("La epoca del checkpoint no coincide con el reporte.")

    dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    red = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2)
    red.load_state_dict(checkpoint["state_dict"], strict=True)
    red = red.to(dispositivo).eval()

    reales, predichos, puntajes = [], [], []
    print("\nAUDITORIA DEL MODELO 3D: SOLO VALIDACION", flush=True)
    print(f"Mejor epoca guardada: {epoca}; dispositivo: {dispositivo}", flush=True)
    print("Test sigue reservado: no se leen sus imagenes ni etiquetas para evaluar.", flush=True)

    with torch.inference_mode():
        for numero, (ruta, etiqueta) in enumerate(validacion, start=1):
            x = preparar_ct(ruta, tamano)
            tensor = torch.from_numpy(x).unsqueeze(0).to(dispositivo)
            salida = red(tensor)
            # Puntaje numerico de clase positiva: NO es probabilidad clinica calibrada.
            puntuacion = float(torch.softmax(salida, dim=1)[0, 1].item())
            prediccion = int(salida.argmax(dim=1).item())
            reales.append(int(etiqueta))
            predichos.append(prediccion)
            puntajes.append(puntuacion)
            print(f"Evaluada CT de validacion {numero}/{len(validacion)}", flush=True)

    y = np.asarray(reales, dtype=np.int64)
    p = np.asarray(predichos, dtype=np.int64)
    s = np.asarray(puntajes, dtype=np.float64)
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    sensibilidad = tp / (tp + fn) if tp + fn else float("nan")
    especificidad = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    positivos = s[y == 1]
    negativos = s[y == 0]
    comparaciones = (positivos[:, None] > negativos[None, :]).astype(float)
    empates = (positivos[:, None] == negativos[None, :]).astype(float)
    auc = float((comparaciones + 0.5 * empates).mean())

    print("\nRESULTADOS EN VALIDACION (12 CT; exploratorios)")
    print(f"Predicciones clase 0: {int(np.sum(p == 0))}; clase 1: {int(np.sum(p == 1))}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Exactitud: {(tp + tn) / len(y):.3f}")
    print(f"Sensibilidad: {sensibilidad:.3f}")
    print(f"Especificidad: {especificidad:.3f}")
    print(f"Precision positiva: {precision:.3f}")
    print(f"Exactitud balanceada: {(sensibilidad + especificidad) / 2:.3f}")
    print(f"ROC-AUC exploratorio: {auc:.3f}")
    print(f"Puntaje medio en CT positivas: {float(positivos.mean()):.3f}")
    print(f"Puntaje medio en CT negativas: {float(negativos.mean()):.3f}")
    print("\nSin escritura de archivos. Sin uso del test. No es diagnostico clinico.")


if __name__ == "__main__":
    main()
