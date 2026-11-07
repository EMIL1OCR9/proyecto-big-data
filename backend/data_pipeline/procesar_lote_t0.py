"""LungAI - Paso 11: procesar todas las CT T0 locales, con auditoria de padding.

Ejecutar desde la raiz del proyecto:
    uv run python backend/data_pipeline/procesar_lote_t0.py

Requiere pandas, numpy, pydicom, scipy, matplotlib (instalados en etapas previas).

Advertencias:
- Uso de investigacion: este codigo NO entrena ni diagnostica cancer.
- El padding declarado se identifica ANTES de aplicar RescaleSlope/Intercept.
- Donde no existe PixelPaddingValue se usa una heuristica HU < -1024,
  y la incertidumbre se registra en cada informe.
- Las etiquetas de cancer a 12 meses son solo para vinculo externo, NO entran al NPZ.
- Los identificadores de investigacion solo se guardan en el manifiesto PRIVADO
  ubicado bajo data/, no en los archivos resumen de outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

HU_MIN = -1000.0
HU_MAX = 400.0
HU_AIRE = -1000.0
UMBRAL_SOSPECHOSO = -1024.0


def normalizar_padding(valor: int, bits: int, representacion: int) -> int:
    """Reconstruye valor entero de pixel respetando BitsStored y PixelRepresentation.

    Algunos DICOM presentan 63536 en el tag para un pixel firmado de -2000.
    NO convierte directamente a HU, ni presupone siempre 16 bits con signo.
    """
    if not 1 <= bits <= 32 or representacion not in (0, 1):
        raise ValueError(f"BitsStored/PixelRepresentation invalidos: {bits}/{representacion}")
    modulo = 1 << bits
    valor = int(valor) % modulo
    if representacion == 1 and valor >= (modulo >> 1):
        valor -= modulo
    return valor


def obtener_cohorte(local_path: Path, cohorte_path: Path) -> pd.DataFrame:
    local = pd.read_csv(local_path, dtype="string")
    cohorte = pd.read_csv(
        cohorte_path,
        dtype="string",
        usecols=["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "etiqueta_cancer_12m"],
    )
    for df in (local, cohorte):
        for col in ("PatientID", "StudyInstanceUID", "SeriesInstanceUID"):
            df[col] = df[col].str.strip()
    t0 = local.loc[local["study_yr"].str.upper().eq("T0")].copy()
    if t0.empty:
        raise ValueError("El reporte de descargas no contiene series T0.")
    if t0["PatientID"].duplicated().any():
        raise ValueError("Hay mas de una serie T0 descargada para un paciente. Revisar seleccion.")
    if cohorte["PatientID"].duplicated().any():
        raise ValueError("La cohorte no tiene un registro unico por paciente.")
    vinculada = t0.merge(
        cohorte, on="PatientID", how="left", suffixes=("_local", "_cohorte"),
        indicator=True, validate="one_to_one",
    )
    if (vinculada["_merge"] != "both").any():
        raise ValueError("Hay pacientes T0 locales sin correspondencia en la cohorte.")
    if not vinculada["StudyInstanceUID_local"].eq(vinculada["StudyInstanceUID_cohorte"]).all():
        raise ValueError("Hay T0 locales cuyo estudio no coincide con la cohorte: revisar manualmente.")
    vinculada["etiqueta_numerica"] = pd.to_numeric(vinculada["etiqueta_cancer_12m"], errors="coerce")
    return vinculada.sort_values("PatientID").reset_index(drop=True)


def leer_serie(carpeta: Path, serie_uid: str, estudio_uid: str, n_esperados: int | None):
    import pydicom

    archivos = list(carpeta.glob("*.dcm"))
    if len(archivos) < 2:
        raise ValueError("No hay cortes suficientes en la carpeta.")
    if n_esperados is not None and len(archivos) != n_esperados:
        raise ValueError(f"Serie incompleta: {len(archivos)} cortes vs {n_esperados} esperados.")

    registros = []
    ref_orientacion = ref_espaciado = ref_normal = ref_dimensiones = None
    for archivo in archivos:
        ds = pydicom.dcmread(archivo, stop_before_pixels=True)
        if str(ds.SeriesInstanceUID) != serie_uid or str(ds.StudyInstanceUID) != estudio_uid:
            raise ValueError("UID DICOM diferente al del reporte de descarga.")
        if str(ds.get("Modality", "")) != "CT":
            raise ValueError("Archivo de modalidad distinta de CT.")
        ori = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        pos = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        esp = np.asarray(ds.PixelSpacing, dtype=np.float64)
        dims = (int(ds.Rows), int(ds.Columns))
        if ori.shape != (6,) or pos.shape != (3,) or esp.shape != (2,):
            raise ValueError("Metadatos geometricos DICOM incompletos.")
        if not np.all(np.isfinite(np.r_[ori, pos, esp])) or np.any(esp <= 0):
            raise ValueError("Geometria DICOM invalida.")
        if ref_orientacion is None:
            ref_orientacion, ref_espaciado, ref_dimensiones = ori, esp, dims
            ref_normal = np.cross(ori[:3], ori[3:])
            norma = np.linalg.norm(ref_normal)
            if not 0.99 <= norma <= 1.01:
                raise ValueError("ImageOrientationPatient no es ortonormal.")
            ref_normal = ref_normal / norma
        elif (not np.allclose(ori, ref_orientacion, atol=1e-3, rtol=0)
              or not np.allclose(esp, ref_espaciado, atol=1e-3, rtol=0)
              or dims != ref_dimensiones):
            raise ValueError("La serie combina geometria, orientacion o tamanos diferentes.")
        registros.append((float(np.dot(pos, ref_normal)), archivo))

    registros.sort(key=lambda r: r[0])
    diferencias = np.diff([r[0] for r in registros])
    if np.any(diferencias <= 0.01):
        raise ValueError("Posiciones de cortes duplicadas o mal ordenadas.")
    z_mm = float(np.median(diferencias))
    if np.any(np.abs(diferencias - z_mm) > max(0.25 * z_mm, 0.2)):
        raise ValueError("Espaciado Z irregular: se detiene el remuestreo automatico.")

    alto, ancho = ref_dimensiones
    datos = np.empty((len(registros), alto, ancho), dtype=np.float32)
    mascara = np.empty((len(registros), alto, ancho), dtype=np.uint8)
    num_tag, cuenta_explicita, cuenta_heuristica = 0, 0, 0
    minimo_hu, maximo_hu = float("inf"), -float("inf")
    ejemplos_padding = set()
    avisos = set()

    for i, (_, archivo) in enumerate(registros):
        ds = pydicom.dcmread(archivo)
        pix = np.asarray(ds.pixel_array)
        if pix.shape != (alto, ancho):
            raise ValueError("Matriz de pixeles con dimensiones inesperadas.")
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        if not np.isfinite(slope) or not np.isfinite(intercept) or slope == 0:
            raise ValueError("Transformacion Modality LUT no valida.")
        hu = pix.astype(np.float32) * slope + intercept
        minimo_hu = min(minimo_hu, float(hu.min()))
        maximo_hu = max(maximo_hu, float(hu.max()))

        pad = ds.get("PixelPaddingValue", None)
        rango = ds.get("PixelPaddingRangeLimit", None)
        if pad is not None:
            bits = int(ds.BitsStored)
            rep = int(ds.PixelRepresentation)
            valor = normalizar_padding(pad, bits, rep)
            valor_limite = normalizar_padding(rango, bits, rep) if rango is not None else valor
            pix_entero = pix.astype(np.int64, copy=False)
            m_explicita = (pix_entero >= min(valor, valor_limite)) & (pix_entero <= max(valor, valor_limite))
            num_tag += 1
            cuenta_explicita += int(m_explicita.sum())
            ejemplos_padding.add((int(pad), valor, round(slope, 5), round(intercept, 5)))
            m_heuristica = np.zeros_like(m_explicita)
            sin_tag = False
            if np.any((hu < UMBRAL_SOSPECHOSO) & ~m_explicita):
                avisos.add("Hay HU < -1024 fuera del padding declarado; comprobar manualmente.")
        else:
            if rango is not None:
                raise ValueError("PixelPaddingRangeLimit existe sin PixelPaddingValue.")
            m_explicita = np.zeros(pix.shape, dtype=bool)
            m_heuristica = hu < UMBRAL_SOSPECHOSO
            cuenta_heuristica += int(m_heuristica.sum())
            sin_tag = True
        m = m_explicita | m_heuristica
        if sin_tag and np.any(m_heuristica):
            avisos.add("Se aplico mascara heuristica HU < -1024 en cortes sin tag DICOM.")
        hu[m] = HU_AIRE
        np.clip(hu, HU_MIN, HU_MAX, out=hu)
        hu -= HU_MIN
        hu /= (HU_MAX - HU_MIN)
        datos[i] = hu
        mascara[i] = m

    espaciado = np.asarray([z_mm, *ref_espaciado], dtype=np.float64)
    stats = {
        "forma_original_zyx": list(datos.shape),
        "espaciado_original_mm_zyx": espaciado.tolist(),
        "hu_min_original": minimo_hu,
        "hu_max_original": maximo_hu,
        "numero_cortes_con_pixel_padding_value": num_tag,
        "numero_cortes_sin_pixel_padding_value": len(registros) - num_tag,
        "voxeles_padding_explicito": cuenta_explicita,
        "voxeles_mascara_heuristica": cuenta_heuristica,
        "porcentaje_total_enmascarado": round(100 * (cuenta_explicita + cuenta_heuristica) / datos.size, 3),
        "ejemplos_padding": sorted(ejemplos_padding),
        "advertencias": sorted(avisos),
    }
    return datos, mascara, espaciado, stats


def remuestrear(datos, mascara, espaciado, objetivo_yx_mm):
    forma = np.asarray(datos.shape, dtype=np.int64)
    nueva_yx = np.rint((forma[1:] - 1) * espaciado[1:] / objetivo_yx_mm).astype(np.int64) + 1
    if np.any(nueva_yx < 2):
        raise ValueError("Tamano de salida no valido.")
    nueva = (int(forma[0]), int(nueva_yx[0]), int(nueva_yx[1]))
    if np.prod(nueva) > 100_000_000:
        raise ValueError("Salida demasiado grande para esta prueba.")
    factor = np.asarray(nueva, dtype=np.float64) / forma
    if tuple(forma) == nueva:
        img = datos
        mask = mascara
    else:
        img = zoom(datos, factor, order=1, mode="nearest", prefilter=False, grid_mode=False)
        mask = zoom(mascara, factor, order=0, mode="nearest", prefilter=False, grid_mode=False)
    if img.shape != nueva or mask.shape != nueva or not np.isfinite(img).all():
        raise ValueError("Remuestreo invalido.")
    if img.min() < -1e-5 or img.max() > 1.00001:
        raise ValueError("Valores normalizados fuera de rango.")
    logrado = [float(espaciado[0]),
               float((forma[1] - 1) * espaciado[1] / (nueva[1] - 1)),
               float((forma[2] - 1) * espaciado[2] / (nueva[2] - 1))]
    return img.astype(np.float32), mask.astype(np.uint8), logrado


def previsualizar(datos, mascara, ruta: Path):
    medio = len(datos) // 2
    fig, ejes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    ejes[0].imshow(datos[medio], cmap="gray", vmin=0, vmax=1)
    ejes[0].set_title("CT normalizada [0,1]")
    ejes[1].imshow(mascara[medio], cmap="gray", vmin=0, vmax=1)
    ejes[1].set_title("Mascara de relleno / sospechosos")
    for eje in ejes:
        eje.axis("off")
    fig.suptitle("LungAI | Preparacion de imagenes, sin diagnostico")
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def ejecutar(args):
    vinculadas = obtener_cohorte(args.reporte, args.cohorte)
    args.destino.mkdir(parents=True, exist_ok=True)
    args.resumen.mkdir(parents=True, exist_ok=True)
    registros_publicos, registros_privados = [], []
    for i, (_, fila) in enumerate(vinculadas.iterrows(), start=1):
        caso = f"ct_{i:03d}"
        carpeta = (args.dicom_dir / str(fila["PatientID"])
                   / str(fila["StudyInstanceUID_local"])
                   / str(fila["SeriesInstanceUID_local"]))
        etiqueta = fila["etiqueta_numerica"]
        etiqueta_valida = pd.notna(etiqueta) and float(etiqueta) in (0.0, 1.0)
        coincidencia = "exacta" if (fila["SeriesInstanceUID_local"] == fila["SeriesInstanceUID_cohorte"]) else "misma_estudio_serie_distinta"
        registro = {"caso": caso, "estado": "error", "coincidencia": coincidencia,
                    "etiqueta_cancer_12m": int(etiqueta) if etiqueta_valida else "sin_etiqueta",
                    "n_cortes": "", "porcentaje_mascara": "", "cortes_con_tag": "", "advertencias": ""}
        try:
            esperado = int(fila["ImageCount"]) if pd.notna(fila["ImageCount"]) else None
            datos, mascara, esp, stats = leer_serie(
                carpeta, str(fila["SeriesInstanceUID_local"]),
                str(fila["StudyInstanceUID_local"]), esperado,
            )
            procesado, mascara_proc, esp_logrado = remuestrear(datos, mascara, esp, args.espaciado_yx)
            stats.update({"estado": "procesado; NO entrenado", "forma_salida_zyx": list(procesado.shape),
                          "espaciado_logrado_mm_zyx": esp_logrado,
                          "ventana_hu": [HU_MIN, HU_MAX], "orden_ejes": "Z,Y,X",
                          "tipo_datos": "float32", "origen_mascara": "tag DICOM o heuristica registrada por corte",
                          "nota": "Etiqueta a 12 meses en manifiesto privado; nunca como canal de entrada."})
            destino = args.destino / caso
            destino.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destino / "volumen_normalizado.npz",
                                volumen_normalizado=procesado,
                                espaciado_mm_zyx=np.asarray(esp_logrado, dtype=np.float32),
                                mascara_relleno=mascara_proc)
            (destino / "resumen.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
            previsualizar(procesado, mascara_proc, destino / "vista_previa.png")
            registro.update({"estado": "procesado", "n_cortes": stats["forma_original_zyx"][0],
                             "porcentaje_mascara": stats["porcentaje_total_enmascarado"],
                             "cortes_con_tag": stats["numero_cortes_con_pixel_padding_value"],
                             "advertencias": "; ".join(stats["advertencias"])})
        except Exception as exc:
            registro["advertencias"] = f"{type(exc).__name__}: {exc}"
        registros_publicos.append(registro)
        registros_privados.append({"caso": caso, "PatientID": str(fila["PatientID"]),
                                   "StudyInstanceUID": str(fila["StudyInstanceUID_local"]),
                                   "SeriesInstanceUID": str(fila["SeriesInstanceUID_local"]),
                                   "etiqueta_cancer_12m": registro["etiqueta_cancer_12m"],
                                   "coincidencia": coincidencia,
                                   "carpeta_procesada": str(args.destino / caso)})
        print(f"{caso}: {registro['estado']} | etiqueta={registro['etiqueta_cancer_12m']} | "
              f"cortes={registro['n_cortes']} | tag={registro['cortes_con_tag']} | "
              f"mascara={registro['porcentaje_mascara']}%")
        if registro["estado"] != "procesado":
            print("  ERROR:", registro["advertencias"])
    pd.DataFrame(registros_publicos).to_csv(args.resumen / "resumen_lote.csv", index=False)
    # Este manifiesto contiene identificadores de investigacion: dejar privado bajo data/.
    pd.DataFrame(registros_privados).to_csv(args.destino / "manifest_local_privado.csv", index=False)
    ok = sum(r["estado"] == "procesado" for r in registros_publicos)
    print(f"\nResultado: {ok}/{len(registros_publicos)} CT T0 procesadas")
    print("Resumen sin IDs:", args.resumen / "resumen_lote.csv")
    print("Datos y manifiesto privados:", args.destino)
    print("Esto NO es entrenamiento ni diagnostico de cancer.")
    if ok != len(registros_publicos):
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reporte", type=Path, default=Path("data/NLST_DICOM/series_descargadas.csv"))
    parser.add_argument("--cohorte", type=Path, default=Path("outputs/quality_control/cohorte_variables_transformadas.csv"))
    parser.add_argument("--dicom-dir", type=Path, default=Path("data/NLST_DICOM"))
    parser.add_argument("--destino", type=Path, default=Path("data/CT_T0_procesadas"))
    parser.add_argument("--resumen", type=Path, default=Path("outputs/preprocesamiento/lote_t0"))
    parser.add_argument("--espaciado-yx", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.espaciado_yx) or args.espaciado_yx <= 0:
        parser.error("--espaciado-yx debe ser un numero positivo.")
    ejecutar(args)


if __name__ == "__main__":
    main()
