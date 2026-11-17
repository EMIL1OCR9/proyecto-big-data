"""Prepara un piloto ampliado de LungAI (40 train, 12 validation, 12 test).

Solo crea un nuevo manifiesto Parquet; no descarga, entrena, ni modifica CT.
Conserva las 12 CT del piloto anterior y reserva test para evaluacion final.
Ejecutar desde la raiz de proyecto-big-data con lungai-risk-venv.

IMPORTANTE: seleccion balanceada artificialmente, sin representatividad clinica.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ORIGEN = Path("outputs/monai_dataset/cohorte_cancer_12m.parquet")
PILOTO = Path("outputs/monai_dataset/piloto_ct_3d.parquet")
DESTINO = Path("outputs/monai_dataset/lote_ct_3d_ampliado.parquet")
BASE = Path("data/monai").resolve()
CUOTAS = {"train": {1: 20, 0: 20}, "validation": {1: 6, 0: 6}, "test": {1: 6, 0: 6}}
CAMPOS = {"PatientID", "split", "etiqueta_cancer_12m", "en_muestra", "ruta_imagen", "SeriesDescription", "series_size_MB"}


def presente(ruta: object) -> bool:
    relativa = Path(str(ruta).replace("\\", "/"))
    if relativa.is_absolute() or ".." in relativa.parts:
        raise ValueError("Ruta de imagen invalida en cohorte")
    imagen = (BASE / relativa).resolve()
    return imagen.is_relative_to(BASE) and imagen.is_file() and imagen.stat().st_size > 0


def main() -> None:
    if DESTINO.exists():
        raise SystemExit("El manifiesto ampliado ya existe. No se sobrescribio.")
    if not ORIGEN.is_file() or not PILOTO.is_file():
        raise SystemExit("Falta la cohorte o el piloto original. No se escribio nada.")

    cohorte = pd.read_parquet(ORIGEN)
    piloto = pd.read_parquet(PILOTO)
    faltan = CAMPOS.difference(cohorte.columns)
    if faltan:
        raise SystemExit(f"Faltan columnas: {sorted(faltan)}")
    if not {"PatientID", "split", "etiqueta_cancer_12m"}.issubset(piloto.columns):
        raise SystemExit("El piloto original no tiene las columnas esperadas.")

    cohorte = cohorte[cohorte["en_muestra"].eq(True)].copy()
    cohorte["PatientID"] = cohorte["PatientID"].astype(str)
    cohorte = cohorte[
        cohorte["SeriesDescription"].astype(str).str.split(",").str[0].eq("0")
    ].copy()
    piloto["PatientID"] = piloto["PatientID"].astype(str)

    if cohorte["PatientID"].duplicated().any():
        raise SystemExit("Hay pacientes duplicados entre registros de la cohorte filtrada.")
    if len(piloto) != 12 or piloto["PatientID"].duplicated().any():
        raise SystemExit("El piloto anterior no contiene doce pacientes distintos.")

    indice = cohorte.set_index("PatientID")
    for registro in piloto.itertuples(index=False):
        if registro.PatientID not in indice.index:
            raise SystemExit("Falta un paciente del piloto anterior en cohorte filtrada.")
        actual = indice.loc[registro.PatientID]
        if actual["split"] != registro.split or int(actual["etiqueta_cancer_12m"]) != int(registro.etiqueta_cancer_12m):
            raise SystemExit("Una etiqueta o split no coincide con el piloto anterior.")

    partes = []
    ids_originales = set(piloto["PatientID"])
    for pos_split, (split, clases) in enumerate(CUOTAS.items()):
        for etiqueta, cuota in clases.items():
            grupo = cohorte[
                cohorte["split"].eq(split) & cohorte["etiqueta_cancer_12m"].eq(etiqueta)
            ].copy()
            antiguos = grupo[grupo["PatientID"].isin(ids_originales)]
            if len(antiguos) > cuota:
                raise SystemExit("La cuota no puede contener todos los pacientes anteriores.")
            resto = grupo[~grupo["PatientID"].isin(ids_originales)]
            adicionales = cuota - len(antiguos)
            if len(resto) < adicionales:
                raise SystemExit(f"Insuficientes registros en {split}, clase {etiqueta}.")
            nuevos = resto.sample(n=adicionales, random_state=2026 + pos_split * 10 + etiqueta)
            partes.append(pd.concat([antiguos, nuevos]))

    lote = pd.concat(partes, ignore_index=True)
    if len(lote) != 64 or lote["PatientID"].duplicated().any():
        raise SystemExit("Error al validar los 64 pacientes seleccionados.")
    if not ids_originales.issubset(set(lote["PatientID"])):
        raise SystemExit("Se perdio un paciente del piloto anterior.")

    disponibles = lote["ruta_imagen"].map(presente)
    faltantes = lote.loc[~disponibles]
    mb = pd.to_numeric(faltantes["series_size_MB"], errors="coerce").sum()

    print("LOTE CT 3D AMPLIADO — PLAN SIN DESCARGAS")
    for split in CUOTAS:
        mask = lote["split"].eq(split)
        n = int(mask.sum())
        positivos = int(lote.loc[mask, "etiqueta_cancer_12m"].sum())
        existentes = int((disponibles & mask).sum())
        print(f"{split}: {n} CT ({positivos} positivas, {n-positivos} negativas); locales: {existentes}")
    print(f"CT pendientes: {len(faltantes)}")
    print(f"Descarga DICOM estimada: {mb/1024:.2f} GB")
    print("Atencion: test queda reservado; no usarlo para ajustar hiperparametros.")
    print("Atencion: seleccion balanceada NO representa la prevalencia real de NLST.")

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    lote.to_parquet(DESTINO, index=False)
    print(f"Manifiesto creado: {DESTINO}")
    print("No se modificaron tomografias, modelos ni el piloto anterior.")


if __name__ == "__main__":
    main()
