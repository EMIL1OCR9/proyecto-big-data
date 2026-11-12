"""
LungAI — etiqueta cada candidato como maligno o benigno con las tablas del NLST.

Cruza los JSON de procesar_lotes.py con lo que anoto el radiologo (nlst_ctab) y con el
diagnostico y seguimiento de cada paciente (nlst_prsn). No descarga imagenes ni necesita
que nadie revise las CT. El resultado es la tabla con la que se entrena el clasificador.

Uso:
    python backend/inference/etiquetar_nodulos.py

Reglas:
- Un candidato es el nodulo que reporto el radiologo si esta a +-5 cortes, en el mismo
  lobulo y su caja mide entre la mitad y el doble del diametro reportado.
- maligno: es un nodulo reportado y el paciente tuvo cancer confirmado en ese lobulo.
- benigno: el paciente nunca tuvo cancer y se le siguio al menos 2 anos. Incluye los
  candidatos que el radiologo no reporto (nodulos chicos, vasos, cicatrices).
- sin etiqueta: los demas candidatos de pacientes con cancer, o con poco seguimiento.

Importante:
- "maligno" es aproximado: se sabe que hubo cancer en ese lobulo, no que salio de ese
  nodulo exacto. Es una etiqueta para entrenar, NO un diagnostico.
- Solo usa los JSON que traen corte_dicom (los de procesar_lotes.py).
- outputs/ esta en .gitignore. No subir tablas con identificadores a GitHub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SLICE_TOLERANCE = 5  # cortes DICOM
MIN_FOLLOW_UP_DAYS = 730
# Lobulo segun el NLST -> lobulo segun lungmask. La lingula (5) es parte del superior izquierdo.
REPORTED_LOBES = {
    "1": "Superior derecho", "2": "Medio derecho", "3": "Inferior derecho",
    "4": "Superior izquierdo", "5": "Superior izquierdo", "6": "Inferior izquierdo",
}
CANCER_LOBES = {
    "locrup": "Superior derecho", "locrmid": "Medio derecho", "locrlow": "Inferior derecho",
    "loclup": "Superior izquierdo", "loclin": "Superior izquierdo", "locllow": "Inferior izquierdo",
}


def load_candidates(nodules_dir: Path) -> pd.DataFrame:
    """Lee los JSON y regresa una fila por candidato."""
    rows, skipped = [], 0
    for path in sorted(nodules_dir.glob("*/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        groups = [(True, report["nodulos"]), (False, report["otros_candidatos"])]
        if any("corte_dicom" not in c for _, group in groups for c in group):
            skipped += 1
            continue
        for is_nodule, group in groups:
            for candidate in group:
                rows.append({
                    "PatientID": report["patient_id"],
                    "SeriesInstanceUID": report["series_uid"],
                    "puntaje": candidate["puntaje"],
                    "es_nodulo": is_nodule,
                    "lobulo": candidate["lobulo"],
                    "tamano_mm": max(candidate["tamano_mm"][:2]),
                    "corte_dicom": candidate["corte_dicom"],
                    "en_pulmon": candidate["en_pulmon"],
                    "recorte": candidate.get("recorte"),
                })
    if skipped:
        print(f"Se omiten {skipped} JSON sin corte_dicom (no vienen de procesar_lotes.py).")
    candidates = pd.DataFrame(rows)
    if len(candidates):
        candidates["recorte"] = candidates["recorte"].astype("Int64")
    return candidates


def reported_nodules(ctab: pd.DataFrame, patients: set[str]) -> pd.DataFrame:
    """Nodulos no calcificados de 4 mm o mas que el radiologo anoto en la ronda T0."""
    nodules = ctab[(ctab["sct_ab_desc"] == 51) & (ctab["study_yr"] == 0)]
    nodules = nodules[nodules["dicom_patient_id"].astype("string").isin(patients)]
    slices = pd.to_numeric(nodules["sct_slice_num"], errors="coerce")
    reported = pd.DataFrame({
        "PatientID": nodules["dicom_patient_id"].astype("string"),
        "lobulo": nodules["sct_epi_loc"].astype("string").map(REPORTED_LOBES),
        "diametro_mm": pd.to_numeric(nodules["sct_long_dia"], errors="coerce"),
        "corte_dicom": slices.where(slices.between(1, 998)),  # 999 = dato faltante
    })
    return reported.dropna(subset=["corte_dicom"]).reset_index(drop=True)


def patient_status(prsn: pd.DataFrame, patients: set[str]) -> pd.DataFrame:
    """Diagnostico, lobulos con cancer y seguimiento de cada paciente."""
    table = prsn[prsn["dicom_patient_id"].astype("string").isin(patients)]
    numeric = table[["can_scr", "candx_days", "canc_free_days", "scr_days0", *CANCER_LOBES]].apply(
        pd.to_numeric, errors="coerce"
    )
    return pd.DataFrame({
        "PatientID": table["dicom_patient_id"].astype("string"),
        "tiene_cancer": numeric["can_scr"].gt(0),
        "lobulos_cancer": [
            {lobe for column, lobe in CANCER_LOBES.items() if row[column] == 1}
            for row in numeric.to_dict("records")
        ],
        "dias_al_diagnostico": numeric["candx_days"] - numeric["scr_days0"],
        "dias_sin_cancer": numeric["canc_free_days"] - numeric["scr_days0"],
    }).set_index("PatientID")


def match_reported(candidates: pd.DataFrame, reported: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Busca, para cada nodulo reportado, el candidato de mayor puntaje que le corresponde."""
    candidates = candidates.assign(reportado=False, lobulo_reportado="", diametro_reportado_mm=np.nan)
    reported = reported.assign(detectado="no", puntaje=np.nan)

    for index, nodule in reported.iterrows():
        match = (
            candidates["PatientID"].eq(nodule["PatientID"])
            & ~candidates["reportado"]
            & (candidates["corte_dicom"] - nodule["corte_dicom"]).abs().le(SLICE_TOLERANCE)
        )
        if pd.notna(nodule["lobulo"]):
            match &= candidates["lobulo"].eq(nodule["lobulo"])
        if pd.notna(nodule["diametro_mm"]):
            match &= candidates["tamano_mm"].between(nodule["diametro_mm"] / 2, nodule["diametro_mm"] * 2)
        if not match.any():
            continue

        best = candidates.loc[match, "puntaje"].idxmax()
        candidates.loc[best, "reportado"] = True
        candidates.loc[best, "lobulo_reportado"] = nodule["lobulo"] if pd.notna(nodule["lobulo"]) else ""
        candidates.loc[best, "diametro_reportado_mm"] = nodule["diametro_mm"]
        reported.loc[index, "detectado"] = "si" if candidates.loc[best, "es_nodulo"] else "bajo umbral"
        reported.loc[index, "puntaje"] = candidates.loc[best, "puntaje"]
    return candidates, reported


def assign_labels(candidates: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """Agrega la etiqueta (maligno, benigno o vacia) y los dias al diagnostico."""
    patient = status.reindex(candidates["PatientID"])
    has_cancer = patient["tiene_cancer"].to_numpy(dtype=bool)
    same_lobe = np.array([
        lobe in lobes for lobe, lobes in zip(candidates["lobulo_reportado"], patient["lobulos_cancer"])
    ], dtype=bool)

    labeled = candidates.copy()
    labeled["etiqueta"] = np.select(
        [
            has_cancer & candidates["reportado"].to_numpy() & same_lobe,
            ~has_cancer & patient["dias_sin_cancer"].ge(MIN_FOLLOW_UP_DAYS).to_numpy(),
        ],
        ["maligno", "benigno"],
        default="",
    )
    labeled["dias_al_diagnostico"] = patient["dias_al_diagnostico"].to_numpy()
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodules-dir", type=Path, default=Path("outputs/nodulos"))
    parser.add_argument("--cohort", type=Path, default=Path("outputs/monai_dataset/cohorte_cancer_12m.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/etiquetas"))
    args = parser.parse_args()

    candidates = load_candidates(args.nodules_dir)
    if candidates.empty:
        raise SystemExit("No hay JSON con corte_dicom. Corre antes procesar_lotes.py.")
    patients = set(candidates["PatientID"])

    from idc_index import IDCClient

    client = IDCClient()
    client.fetch_index("clinical_index")
    reported = reported_nodules(client.get_clinical_table("nlst_ctab"), patients)
    status = patient_status(client.get_clinical_table("nlst_prsn"), patients)

    candidates, reported = match_reported(candidates, reported)
    candidates = assign_labels(candidates, status)

    splits = pd.read_parquet(args.cohort, columns=["PatientID", "split"]).drop_duplicates("PatientID")
    candidates = candidates.merge(splits, on="PatientID", how="left", validate="many_to_one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.output_dir / "candidatos_etiquetados.parquet", index=False)
    reported.to_parquet(args.output_dir / "nodulos_reportados.parquet", index=False)

    with_crop = candidates[candidates["recorte"].notna() & candidates["etiqueta"].ne("")]
    summary = pd.crosstab(
        [with_crop["etiqueta"], with_crop["reportado"]], with_crop["split"].fillna("sin split"),
        margins=True, margins_name="total",
    )
    summary.to_csv(args.output_dir / "resumen_etiquetas.csv")

    print(f"Series: {candidates['SeriesInstanceUID'].nunique()} | candidatos: {len(candidates)}")
    print("\nRecortes con etiqueta (reportado = lo anoto el radiologo):")
    print(summary.to_string())
    print("\nNodulos reportados por el radiologo y si el detector los encontro:")
    print(reported["detectado"].value_counts().reindex(["si", "bajo umbral", "no"], fill_value=0).to_string())
    print(f"\nArchivos guardados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
