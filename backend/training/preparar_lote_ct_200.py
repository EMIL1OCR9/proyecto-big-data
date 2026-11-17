"""LungAI: ampliar manifiesto a 200 CT sin tocar CT ni modelos existentes.

Cuotas: 140 train (70+70), 30 validation (15+15), 30 test (15+15).
Conserva las 61 CT válidas ya descargadas, excluye fallos de conversión
registrados y reserva test para una sola evaluación final futura.
NO descarga imágenes. Selección balanceada: NO representa prevalencia de NLST.

Ejecutar desde raíz del proyecto con ~/lungai-risk-venv:
    python backend/training/preparar_lote_ct_200.py
"""
from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd

ORIGEN = Path("outputs/monai_dataset/cohorte_cancer_12m.parquet")
ANTERIOR = Path("outputs/monai_dataset/lote_ct_3d_ampliado.parquet")
DESTINO = Path("outputs/monai_dataset/lote_ct_3d_200.parquet")
FALLAS = Path("data/monai/conversion_fallidas.csv")
BASE = Path("data/monai").resolve()
CUOTAS = {
    "train": {1: 70, 0: 70},
    "validation": {1: 15, 0: 15},
    "test": {1: 15, 0: 15},
}
CAMPOS = {
    "PatientID", "split", "etiqueta_cancer_12m", "en_muestra",
    "ruta_imagen", "SeriesDescription", "series_size_MB",
}


def local(ruta: object) -> bool:
    relativa = Path(str(ruta).replace("\\", "/"))
    if relativa.is_absolute() or ".." in relativa.parts:
        raise ValueError("Ruta de CT no segura")
    archivo = (BASE / relativa).resolve()
    return archivo.is_relative_to(BASE) and archivo.is_file() and archivo.stat().st_size > 0


def main() -> None:
    if DESTINO.exists():
        raise SystemExit(f"Ya existe {DESTINO}. No se sobrescribe.")
    if not ORIGEN.is_file() or not ANTERIOR.is_file():
        raise SystemExit("Falta la cohorte o el manifiesto de 64 CT.")
    cohorte = pd.read_parquet(ORIGEN)
    anterior = pd.read_parquet(ANTERIOR)
    faltantes = CAMPOS.difference(cohorte.columns)
    if faltantes:
        raise SystemExit(f"Faltan columnas en la cohorte: {sorted(faltantes)}")
    cohorte = cohorte[cohorte["en_muestra"].eq(True)].copy()
    cohorte["PatientID"] = cohorte["PatientID"].astype(str).str.strip()
    cohorte = cohorte[
        cohorte["SeriesDescription"].astype(str).str.split(",").str[0].eq("0")
    ].copy()
    anterior["PatientID"] = anterior["PatientID"].astype(str).str.strip()
    if (cohorte["PatientID"].duplicated().any()
            or anterior["PatientID"].duplicated().any()):
        raise SystemExit("Identificadores duplicados detectados. No se escribe nada.")
    indice = cohorte.set_index("PatientID")
    for _, viejo in anterior.iterrows():
        identificador = viejo["PatientID"]
        if identificador not in indice.index:
            raise SystemExit("Un paciente del lote de 64 no está en cohorte.")
        nuevo = indice.loc[identificador]
        if (nuevo["split"] != viejo["split"]
                or int(nuevo["etiqueta_cancer_12m"]) != int(viejo["etiqueta_cancer_12m"])):
            raise SystemExit("Se detectó un cambio de etiqueta o split.")
    # Descartar solo registros que fallaron y siguen sin NIfTI válido.
    ids_fallidos = set()
    if FALLAS.is_file():
        fallas = pd.read_csv(FALLAS, dtype=str)
        if "PatientID" not in fallas.columns:
            raise SystemExit("Reporte de fallos sin PatientID: revisar manualmente.")
        ids_fallidos = set(fallas["PatientID"].dropna().str.strip())
    excluidos = set()
    if ids_fallidos:
        serie_fallida = cohorte[cohorte["PatientID"].isin(ids_fallidos)]
        excluidos = set(serie_fallida.loc[
            ~serie_fallida["ruta_imagen"].map(local), "PatientID"
        ])
    cohorte = cohorte[~cohorte["PatientID"].isin(excluidos)].copy()
    anterior = anterior[~anterior["PatientID"].isin(excluidos)].copy()

    # Reusar toda CT del lote anterior si está local; nuevas filas elegidas al azar
    # con semilla fija, manteniendo el split original y sin releer el test.
    anteriores_locales = anterior[anterior["ruta_imagen"].map(local)].copy()
    if len(anteriores_locales) < 55:
        raise SystemExit("Quedan menos de 55 CT locales del lote anterior. Revisar antes de seguir.")
    partes = []
    for orden, (split, clases) in enumerate(CUOTAS.items()):
        for etiqueta, cuota in clases.items():
            grupo = cohorte[
                cohorte["split"].eq(split) &
                cohorte["etiqueta_cancer_12m"].eq(etiqueta)
            ]
            conservar = anteriores_locales[
                anteriores_locales["split"].eq(split) &
                anteriores_locales["etiqueta_cancer_12m"].eq(etiqueta)
            ]
            if len(conservar) > cuota:
                raise SystemExit("La cuota es menor a las CT locales anteriores.")
            restantes = grupo[~grupo["PatientID"].isin(conservar["PatientID"])].copy()
            adicionales = cuota - len(conservar)
            if len(restantes) < adicionales:
                raise SystemExit(f"Faltan candidatos para {split}, clase {etiqueta}.")
            # Priorizar NIfTI ya disponibles fuera del lote, luego muestreo reproducible.
            restantes["_local"] = restantes["ruta_imagen"].map(local)
            locales = restantes[restantes["_local"]].drop(columns="_local")
            remotos = restantes[~restantes["_local"]].drop(columns="_local")
            remotos = remotos.sample(frac=1, random_state=2026 + orden*10 + etiqueta)
            elegidos = pd.concat([locales, remotos], ignore_index=True).head(adicionales)
            if len(elegidos) != adicionales:
                raise SystemExit("No se pudo completar cuota")
            partes.extend([conservar, elegidos])
    nuevo = pd.concat(partes, ignore_index=True)
    if len(nuevo) != 200 or nuevo["PatientID"].duplicated().any():
        raise SystemExit("El manifiesto nuevo no tiene 200 pacientes únicos.")
    if not set(anteriores_locales["PatientID"]).issubset(set(nuevo["PatientID"])):
        raise SystemExit("Se perdieron imágenes válidas del lote anterior.")
    for split, clases in CUOTAS.items():
        grupo = nuevo[nuevo["split"].eq(split)]
        for etiqueta, cuota in clases.items():
            if int(grupo["etiqueta_cancer_12m"].eq(etiqueta).sum()) != cuota:
                raise SystemExit("La cuota por clase no coincide.")
    disponibles = nuevo["ruta_imagen"].map(local)
    pendientes = nuevo[~disponibles]
    gb = pd.to_numeric(pendientes["series_size_MB"], errors="coerce").sum() / 1024
    libre = shutil.disk_usage(BASE).free / (1024**3)

    print("LOTE CT 3D DE 200 — PLAN SIN DESCARGAS")
    for split in CUOTAS:
        grupo = nuevo["split"].eq(split)
        print(f"{split}: {int(grupo.sum())} CT; locales: {int((disponibles & grupo).sum())}")
    print("CT locales conservadas:", len(anteriores_locales))
    print("Series fallidas excluidas:", len(excluidos))
    print("CT pendientes:", len(pendientes))
    print("Descarga DICOM estimada: %.2f GB" % gb)
    print("Espacio libre: %.1f GB" % libre)
    print("Test sigue reservado: nunca usar sus imágenes para ajustar el modelo.")
    print("ADVERTENCIA: muestreo 50/50, no refleja prevalencia real ni desempeño clínico.")
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    nuevo.to_parquet(DESTINO, index=False)
    print("Nuevo manifiesto:", DESTINO)
    print("No se descargaron imágenes ni se modificaron los modelos.")


if __name__ == "__main__":
    main()
