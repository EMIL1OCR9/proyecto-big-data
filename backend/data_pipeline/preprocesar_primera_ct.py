"""LungAI - primera prueba de preprocesamiento de una serie CT T0 del NLST.

Objetivo: verificar enlace paciente/estudio/serie y convertir UNA serie DICOM
completa a volumen HU (Z, Y, X), con resumen de calidad y vista previa.

Desde la raíz del proyecto:
    uv run python backend/data_pipeline/preprocesar_primera_ct.py

Opcional: escoger un caso con etiqueta negativa a 12 meses:
    uv run python backend/data_pipeline/preprocesar_primera_ct.py --etiqueta 0

Dependencias ya empleadas en el proyecto: pandas, numpy, pydicom, matplotlib.
ATENCIÓN: una etiqueta de cáncer en 12 meses NO localiza un tumor en esta CT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Guarda imagen sin depender de una ventana interactiva.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cargar_vinculos(csv_series: Path, csv_cohorte: Path) -> pd.DataFrame:
    local = pd.read_csv(csv_series, dtype="string")
    cohorte = pd.read_csv(
        csv_cohorte,
        dtype="string",
        usecols=["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "etiqueta_cancer_12m"],
    )
    for df in (local, cohorte):
        for campo in ("PatientID", "StudyInstanceUID", "SeriesInstanceUID"):
            df[campo] = df[campo].str.strip()
    local = local[local["study_yr"].str.upper().eq("T0")].copy()
    if local.empty:
        raise ValueError("No hay ninguna serie T0 en el reporte de descargas.")

    vinculos = local.merge(
        cohorte,
        on="PatientID",
        how="left",
        suffixes=("_local", "_cohorte"),
        validate="many_to_one",
        indicator=True,
    )
    vinculos["mismo_estudio"] = (
        vinculos["_merge"].eq("both")
        & vinculos["StudyInstanceUID_local"].eq(vinculos["StudyInstanceUID_cohorte"])
    )
    vinculos["misma_serie"] = (
        vinculos["mismo_estudio"]
        & vinculos["SeriesInstanceUID_local"].eq(vinculos["SeriesInstanceUID_cohorte"])
    )
    print("\n=== AUDITORÍA LOCAL T0 ===")
    print("Series T0 locales:", len(vinculos))
    print("Pacientes encontrados en cohorte:", int(vinculos["_merge"].eq("both").sum()))
    print("Series exactas (paciente + estudio + serie):", int(vinculos["misma_serie"].sum()))
    print("Serie distinta, mismo estudio:", int((vinculos["mismo_estudio"] & ~vinculos["misma_serie"]).sum()))
    print("Paciente presente, estudio diferente:", int((vinculos["_merge"].eq("both") & ~vinculos["mismo_estudio"]).sum()))
    print("Paciente no encontrado:", int(vinculos["_merge"].eq("left_only").sum()))
    return vinculos


def _metadatos_de_corte(archivo: Path):
    import pydicom

    ds = pydicom.dcmread(str(archivo), stop_before_pixels=True)
    for tag in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "SeriesInstanceUID"):
        if tag not in ds:
            raise ValueError(f"Falta etiqueta DICOM requerida: {tag}.")
    orientacion = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    posicion = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    espaciado = np.asarray(ds.PixelSpacing, dtype=np.float64)
    if orientacion.shape != (6,) or posicion.shape != (3,) or espaciado.shape != (2,):
        raise ValueError("Geometría DICOM con dimensiones inesperadas.")
    if not np.all(np.isfinite(np.r_[orientacion, posicion, espaciado])) or np.any(espaciado <= 0):
        raise ValueError("Geometría DICOM no finita o espaciado no positivo.")
    return ds, orientacion, posicion, espaciado


def cargar_volumen_hu(carpeta: Path, uid_esperado: str, cortes_esperados: int | None):
    import pydicom

    archivos = sorted(carpeta.glob("*.dcm"))
    if len(archivos) < 2:
        raise ValueError(f"Se necesitan varios cortes DICOM; se encontraron {len(archivos)}.")
    if cortes_esperados is not None and len(archivos) != cortes_esperados:
        raise ValueError(f"Serie incompleta: {len(archivos)} archivos, se esperaban {cortes_esperados}.")

    registros = []
    orientacion_ref = None
    espaciado_ref = None
    normal = None
    estudio_ref = None
    filas_columnas = None

    for archivo in archivos:
        ds, orientacion, posicion, espaciado = _metadatos_de_corte(archivo)
        if str(ds.SeriesInstanceUID) != uid_esperado:
            raise ValueError("Hay un archivo perteneciente a otra serie.")
        estudio = str(ds.StudyInstanceUID)
        if estudio_ref is None:
            estudio_ref = estudio
        elif estudio != estudio_ref:
            raise ValueError("La carpeta mezcla diferentes estudios DICOM.")
        if str(ds.get("Modality", "")) != "CT":
            raise ValueError("La serie contiene un archivo que no es CT.")
        dimensiones = (int(ds.Rows), int(ds.Columns))
        if filas_columnas is None:
            filas_columnas = dimensiones
        elif dimensiones != filas_columnas:
            raise ValueError("Los cortes tienen tamaños diferentes.")
        if orientacion_ref is None:
            orientacion_ref = orientacion
            espaciado_ref = espaciado
            normal = np.cross(orientacion_ref[:3], orientacion_ref[3:])
            norma = np.linalg.norm(normal)
            if norma < 0.99 or norma > 1.01:
                raise ValueError("Vectores de orientación DICOM inválidos.")
            normal /= norma
        elif not np.allclose(orientacion, orientacion_ref, atol=1e-3, rtol=0):
            raise ValueError("Cambios de orientación dentro de la misma serie.")
        elif not np.allclose(espaciado, espaciado_ref, atol=1e-3, rtol=0):
            raise ValueError("Cambios de PixelSpacing dentro de la misma serie.")
        registros.append((float(np.dot(posicion, normal)), archivo))

    registros.sort(key=lambda x: x[0])
    posiciones = np.array([r[0] for r in registros], dtype=np.float64)
    diferencias = np.diff(posiciones)
    if np.any(diferencias < 0.01):
        raise ValueError("Hay cortes con posición espacial duplicada o indistinguible.")
    separacion = float(np.median(diferencias))
    aviso_espaciado = bool(np.any(abs(diferencias - separacion) > 0.25 * separacion))

    imagenes_hu = []
    for _, archivo in registros:
        ds = pydicom.dcmread(str(archivo))
        pixels = ds.pixel_array.astype(np.float32)
        pendiente = float(ds.get("RescaleSlope", 1.0))
        intercepto = float(ds.get("RescaleIntercept", 0.0))
        imagenes_hu.append(pixels * pendiente + intercepto)

    volumen = np.stack(imagenes_hu, axis=0)
    if not np.isfinite(volumen).all():
        raise ValueError("El volumen contiene intensidades no finitas.")
    espacio_zyx = np.array([separacion, *espaciado_ref], dtype=np.float32)
    auditoria = {
        "numero_cortes": int(volumen.shape[0]),
        "dimensiones_zyx": list(map(int, volumen.shape)),
        "espaciado_mm_zyx": [round(float(x), 4) for x in espacio_zyx],
        "valor_minimo_hu": round(float(volumen.min()), 2),
        "valor_maximo_hu": round(float(volumen.max()), 2),
        "espaciado_irregular_advertencia": aviso_espaciado,
        "formato": "Volumen en HU, orden Z-Y-X; NO se ha remuestreado ni normalizado para una red neuronal",
    }
    return volumen, espacio_zyx, auditoria


def guardar_vista_previa(volumen: np.ndarray, ruta: Path) -> None:
    cortes = [round((len(volumen) - 1) * x) for x in (0.25, 0.5, 0.75)]
    fig, ejes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for indice, posicion in enumerate(cortes):
        imagen = volumen[posicion]
        for fila, (centro, ancho, etiqueta) in enumerate(((-600, 1500, "Pulmonar"), (40, 400, "Mediastino"))):
            bajo = centro - ancho / 2
            alto = centro + ancho / 2
            ejes[fila, indice].imshow(imagen, cmap="gray", vmin=bajo, vmax=alto)
            ejes[fila, indice].set_title(f"{etiqueta} | Corte {posicion + 1}/{len(volumen)}")
            ejes[fila, indice].axis("off")
    fig.suptitle("LungAI | Primera CT procesada (previsualización, sin inferencia clínica)")
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etiqueta", type=int, choices=(0, 1), default=1)
    parser.add_argument("--reporte", type=Path, default=Path("data/NLST_DICOM/series_descargadas.csv"))
    parser.add_argument("--cohorte", type=Path, default=Path("outputs/quality_control/cohorte_variables_transformadas.csv"))
    parser.add_argument("--dicom-dir", type=Path, default=Path("data/NLST_DICOM"))
    parser.add_argument("--salida", type=Path, default=Path("outputs/preprocesamiento/primer_ct"))
    args = parser.parse_args()

    relaciones = cargar_vinculos(args.reporte, args.cohorte)
    candidatos = relaciones[relaciones["misma_serie"]].copy()
    candidatos["etiqueta_num"] = pd.to_numeric(candidatos["etiqueta_cancer_12m"], errors="coerce")
    candidatos = candidatos[candidatos["etiqueta_num"].eq(args.etiqueta)]
    if candidatos.empty:
        raise ValueError(f"No existe ninguna CT T0 exacta con etiqueta {args.etiqueta}.")
    fila = candidatos.iloc[0]
    carpeta = args.dicom_dir / str(fila["PatientID"]) / str(fila["StudyInstanceUID_local"]) / str(fila["SeriesInstanceUID_local"])
    if not carpeta.is_dir():
        raise FileNotFoundError("No se encontró en disco la carpeta de la serie seleccionada.")
    esperado = int(fila["ImageCount"]) if pd.notna(fila["ImageCount"]) else None
    volumen, espaciado, auditoria = cargar_volumen_hu(
        carpeta, str(fila["SeriesInstanceUID_local"]), esperado
    )
    auditoria["etiqueta_cancer_12m"] = args.etiqueta
    auditoria["advertencia_etiqueta"] = (
        "La etiqueta representa diagnóstico clínico en 12 meses, no tumor visible ni diagnóstico de la imagen."
    )
    args.salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.salida / "volumen_hu.npz", volumen_hu=volumen, espaciado_mm_zyx=espaciado)
    guardar_vista_previa(volumen, args.salida / "vista_previa.png")
    (args.salida / "resumen_preprocesamiento.json").write_text(
        json.dumps(auditoria, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== PRIMERA TOMOGRAFÍA PROCESADA ===")
    print(f"Forma (cortes, alto, ancho): {volumen.shape}")
    print("Espaciado (mm; Z, Y, X):", tuple(np.round(espaciado, 3)))
    print("Rango HU:", auditoria["valor_minimo_hu"], "a", auditoria["valor_maximo_hu"])
    print("¿Espaciado irregular?:", auditoria["espaciado_irregular_advertencia"])
    print("Etiqueta clínica 12 meses:", args.etiqueta)
    print("Archivos generados:", args.salida / "volumen_hu.npz", ", vista_previa.png, resumen_preprocesamiento.json")
    print("Esto todavía NO es inferencia ni entrenamiento de IA.")


if __name__ == "__main__":
    main()
