
"""
LungAI — vinculación provisional entre metadatos de CT y rondas de cribado NLST.

Este módulo NO descarga los píxeles DICOM. Consulta metadatos del índice IDC,
relaciona estudios con rondas clínicas mediante una regla conservadora y exporta
tablas de auditoría. El mapeo es provisional: debe validarse antes de usarlo para
entrenar o evaluar un modelo.

Uso:
    python backend/data_pipeline/link_nlst_ct.py --output-dir outputs/link_audit

Dependencias:
    pip install idc-index pandas numpy

Importante:
- No se deben subir tomografías ni datos clínicos sensibles a GitHub.
- La cohorte clínica se restringe a IDs presentes en nlst_screen porque la tabla
  pública nlst_prsn puede no incluir rndgroup. Esto no valida por sí solo las rondas.
- Las fechas DICOM pueden estar anonimizadas. La comparación por intervalos solo
  es una hipótesis de vinculación, no una garantía de identidad de la ronda.
- Los códigos de resultado deben verificarse contra el diccionario oficial NLST.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("lungai.link_nlst_ct")

IMAGE_COLUMNS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "StudyDate",
    "Modality",
    "instanceCount",
    "series_size_MB",
    "SeriesDescription",
]

CLINICAL_COLUMNS = [
    "dicom_patient_id",
    "scr_days0",
    "scr_days1",
    "scr_days2",
    "scr_res0",
    "scr_res1",
    "scr_res2",
    "candx_days",
]

CT_SCREEN_COLUMNS = ["dicom_patient_id"]


def normalize_id(values: pd.Series) -> pd.Series:
    """Normaliza IDs como texto sin convertirlos a enteros."""
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def parse_dicom_date(values: pd.Series) -> pd.Series:
    """Convierte fechas DICOM YYYYMMDD a datetime."""
    digits = values.astype("string").str.replace(r"[^0-9]", "", regex=True)
    digits = digits.where(digits.str.len() == 8)
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")


def require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas en {name}: {missing}")


def prepare_image_index(image_index: pd.DataFrame) -> pd.DataFrame:
    """Filtra series CT de NLST y prepara fechas y medidas."""
    require_columns(
        image_index,
        ["collection_id", "Modality", *IMAGE_COLUMNS],
        "índice de imágenes IDC",
    )

    mask = image_index["collection_id"].astype("string").str.lower().eq("nlst")
    mask &= image_index["Modality"].astype("string").str.upper().eq("CT")

    if "analysis_result_id" in image_index.columns:
        mask &= image_index["analysis_result_id"].isna()

    images = image_index.loc[mask, IMAGE_COLUMNS].copy()
    images["PatientID"] = normalize_id(images["PatientID"])
    images["StudyInstanceUID"] = (
        images["StudyInstanceUID"].astype("string").str.strip()
    )
    images["SeriesInstanceUID"] = (
        images["SeriesInstanceUID"].astype("string").str.strip()
    )

    for column in ("instanceCount", "series_size_MB"):
        images[column] = pd.to_numeric(images[column], errors="coerce")

    images = images.dropna(
        subset=["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]
    )
    images = images.drop_duplicates(subset=["SeriesInstanceUID"]).copy()
    images["fecha_dicom"] = parse_dicom_date(images["StudyDate"])

    LOGGER.info("Series CT NLST únicas: %s", len(images))
    LOGGER.info(
        "Series con fecha DICOM válida: %s",
        images["fecha_dicom"].notna().sum(),
    )
    return images


def prepare_clinical_table(
    participants: pd.DataFrame,
    ct_screen: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara nlst_prsn y filtra participantes presentes en nlst_screen.
    No depende de la columna rndgroup.
    """
    require_columns(participants, CLINICAL_COLUMNS, "nlst_prsn")
    require_columns(ct_screen, CT_SCREEN_COLUMNS, "nlst_screen")

    clinical = participants[CLINICAL_COLUMNS].copy()
    clinical = clinical.rename(
        columns={"dicom_patient_id": "PatientID"}
    )
    clinical["PatientID"] = normalize_id(clinical["PatientID"])

    ct_ids = set(
        normalize_id(ct_screen["dicom_patient_id"]).dropna()
    )

    if not ct_ids:
        raise ValueError(
            "nlst_screen no contiene dicom_patient_id válidos."
        )

    clinical = clinical[
        clinical["PatientID"].isin(ct_ids)
    ].copy()

    if clinical.empty:
        raise ValueError(
            "No se encontraron participantes compartidos entre "
            "nlst_prsn y nlst_screen. Revisa los identificadores."
        )

    LOGGER.info(
        "Participantes de nlst_prsn presentes en nlst_screen: %s",
        clinical["PatientID"].nunique(),
    )

    numeric_columns = [
        "scr_days0",
        "scr_days1",
        "scr_days2",
        "scr_res0",
        "scr_res1",
        "scr_res2",
        "candx_days",
    ]

    for column in numeric_columns:
        clinical[column] = pd.to_numeric(
            clinical[column],
            errors="coerce",
        )

    # Resolver duplicados sin elegir arbitrariamente entre datos en conflicto.
    duplicated = clinical["PatientID"].duplicated(keep=False)

    if duplicated.any():
        value_columns = [
            column
            for column in clinical.columns
            if column != "PatientID"
        ]

        distinct_counts = clinical.groupby("PatientID")[
            value_columns
        ].nunique(dropna=True)

        conflict_ids = distinct_counts.index[
            (distinct_counts > 1).any(axis=1)
        ]

        LOGGER.warning(
            "Se excluirán %s pacientes con filas clínicas en conflicto.",
            len(conflict_ids),
        )

        clinical = clinical[
            ~clinical["PatientID"].isin(conflict_ids)
        ].copy()

        clinical = clinical.drop_duplicates(
            subset=["PatientID"]
        )

    return clinical


def summarize_studies(images: pd.DataFrame) -> pd.DataFrame:
    """Resume las series a nivel de estudio y conserva fechas válidas."""
    studies = (
        images.groupby(["PatientID", "StudyInstanceUID"], dropna=False)
        .agg(
            n_series=("SeriesInstanceUID", "nunique"),
            num_fechas=("fecha_dicom", lambda s: s.dropna().nunique()),
            fecha_dicom=("fecha_dicom", "min"),
        )
        .reset_index()
    )

    valid = studies[
        (studies["num_fechas"] == 1)
        & studies["fecha_dicom"].notna()
    ].copy()

    LOGGER.info("Estudios con fecha única y válida: %s", len(valid))
    return valid


def map_screening_rounds(
    studies: pd.DataFrame,
    clinical: pd.DataFrame,
    tolerance_days: int = 35,
    minimum_interval_days: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Propone un mapeo provisional T0/T1/T2.

    Requiere exactamente tres estudios con fechas distintas y tres días
    clínicos disponibles. Los intervalos DICOM y clínicos deben concordar
    dentro de la tolerancia indicada.
    """
    require_columns(
        studies,
        ["PatientID", "StudyInstanceUID", "fecha_dicom"],
        "estudios",
    )
    require_columns(
        clinical,
        ["PatientID", "scr_days0", "scr_days1", "scr_days2"],
        "tabla clínica",
    )

    counts = studies.groupby("PatientID").agg(
        n_estudios=("StudyInstanceUID", "nunique"),
        n_fechas=("fecha_dicom", "nunique"),
    )

    eligible_ids = counts.index[
        (counts["n_estudios"] == 3)
        & (counts["n_fechas"] == 3)
    ]

    ordered = studies[
        studies["PatientID"].isin(eligible_ids)
    ].copy()
    ordered = ordered.sort_values(["PatientID", "fecha_dicom"])
    ordered["study_yr"] = ordered.groupby("PatientID").cumcount()

    date_pivot = ordered.pivot(
        index="PatientID",
        columns="study_yr",
        values="fecha_dicom",
    )
    date_pivot = date_pivot.rename(
        columns={0: "fecha0", 1: "fecha1", 2: "fecha2"}
    ).reset_index()

    audit = date_pivot.merge(
        clinical[["PatientID", "scr_days0", "scr_days1", "scr_days2"]],
        on="PatientID",
        how="inner",
        validate="one_to_one",
    )

    audit["gap_dicom_01"] = (
        audit["fecha1"] - audit["fecha0"]
    ).dt.days
    audit["gap_dicom_12"] = (
        audit["fecha2"] - audit["fecha1"]
    ).dt.days

    audit["gap_clin_01"] = (
        audit["scr_days1"] - audit["scr_days0"]
    )
    audit["gap_clin_12"] = (
        audit["scr_days2"] - audit["scr_days1"]
    )

    audit["error_gap_01"] = (
        audit["gap_dicom_01"] - audit["gap_clin_01"]
    ).abs()
    audit["error_gap_12"] = (
        audit["gap_dicom_12"] - audit["gap_clin_12"]
    ).abs()

    complete_days = audit[
        ["scr_days0", "scr_days1", "scr_days2"]
    ].notna().all(axis=1)

    consistent = (
        complete_days
        & audit["gap_clin_01"].ge(minimum_interval_days)
        & audit["gap_clin_12"].ge(minimum_interval_days)
        & audit["gap_dicom_01"].ge(minimum_interval_days)
        & audit["gap_dicom_12"].ge(minimum_interval_days)
        & audit["error_gap_01"].le(tolerance_days)
        & audit["error_gap_12"].le(tolerance_days)
    )

    audit["consistente"] = consistent

    accepted_ids = set(
        audit.loc[consistent, "PatientID"].dropna()
    )

    mapping = ordered[
        ordered["PatientID"].isin(accepted_ids)
    ][
        ["PatientID", "StudyInstanceUID", "study_yr", "fecha_dicom"]
    ].copy()

    mapping["metodo_mapeo"] = (
        f"provisional_3_estudios_intervalos_{tolerance_days}d"
    )

    LOGGER.info("Pacientes evaluados por la regla: %s", len(audit))
    LOGGER.info(
        "Pacientes con intervalos consistentes: %s",
        len(accepted_ids),
    )
    LOGGER.info("Estudios con ronda provisional: %s", len(mapping))

    return mapping, audit


def build_screening_cohort(
    images: pd.DataFrame,
    clinical: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une series de imagen con la ronda provisional y añade el resultado T0.

    Codificación provisional:
      códigos 1, 2, 3 -> negativo (0)
      códigos 4, 5, 6 -> positivo (1)
      otros -> sin etiqueta (NaN)

    Un cribado positivo NO equivale a un diagnóstico confirmado de cáncer.
    """
    require_columns(
        mapping,
        ["PatientID", "StudyInstanceUID", "study_yr", "metodo_mapeo"],
        "mapa de rondas",
    )

    cohort = images.merge(
        mapping[
            ["PatientID", "StudyInstanceUID", "study_yr", "metodo_mapeo"]
        ],
        on=["PatientID", "StudyInstanceUID"],
        how="inner",
        validate="many_to_one",
    )

    cohort = cohort.merge(
        clinical,
        on="PatientID",
        how="left",
        validate="many_to_one",
    )

    t0 = cohort[cohort["study_yr"].eq(0)].copy()
    t0["resultado_cribado_t0"] = t0["scr_res0"]

    t0["etiqueta_cribado_provisional"] = np.select(
        [
            t0["resultado_cribado_t0"].isin([1, 2, 3]),
            t0["resultado_cribado_t0"].isin([4, 5, 6]),
        ],
        [0, 1],
        default=np.nan,
    )

    # Filtro orientativo para seleccionar una reconstrucción manejable.
    t0 = t0[
        t0["instanceCount"].between(60, 600)
        & t0["series_size_MB"].between(15, 300)
    ].copy()

    t0 = t0.sort_values(
        ["PatientID", "instanceCount", "series_size_MB"],
        ascending=[True, False, False],
    )
    t0 = t0.drop_duplicates("PatientID")

    return t0


def run_pipeline(
    output_dir: Path,
    tolerance_days: int = 35,
) -> None:
    """Consulta IDC, crea el mapeo y guarda tablas de auditoría."""
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise RuntimeError(
            "Falta idc-index. Instala dependencias con: "
            "pip install idc-index pandas numpy"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    client = IDCClient()
    client.fetch_index("clinical_index")

    participants = client.get_clinical_table("nlst_prsn")

    if participants is None:
        raise RuntimeError(
            "No se pudo cargar la tabla clínica nlst_prsn."
        )

    ct_screen = client.get_clinical_table("nlst_screen")

    if ct_screen is None:
        raise RuntimeError(
            "No se pudo cargar la tabla clínica nlst_screen."
        )

    images = prepare_image_index(client.index)

    clinical = prepare_clinical_table(
        participants,
        ct_screen,
    )

    studies = summarize_studies(images)

    mapping, audit = map_screening_rounds(
        studies,
        clinical,
        tolerance_days=tolerance_days,
    )

    cohort = build_screening_cohort(images, clinical, mapping)

    # Estos CSV contienen identificadores de investigación.
    # No subirlos a repositorios públicos.
    images.to_csv(
        output_dir / "ct_series_index_audit.csv",
        index=False,
    )
    studies.to_csv(
        output_dir / "ct_studies_audit.csv",
        index=False,
    )
    audit.to_csv(
        output_dir / "round_mapping_patient_audit.csv",
        index=False,
    )
    mapping.to_csv(
        output_dir / "round_mapping_provisional.csv",
        index=False,
    )
    cohort.to_csv(
        output_dir / "t0_screening_cohort_provisional.csv",
        index=False,
    )

    summary = {
        "ct_series": len(images),
        "ct_studies_valid_date": len(studies),
        "clinical_participants_in_nlst_screen": int(
            clinical["PatientID"].nunique()
        ),
        "patients_round_mapping_audited": len(audit),
        "patients_round_mapping_consistent": int(
            audit["consistente"].sum()
        ) if "consistente" in audit else 0,
        "series_with_provisional_round": len(mapping),
        "t0_series_after_size_filter": len(cohort),
        "positive_provisional": int(
            (cohort["etiqueta_cribado_provisional"] == 1).sum()
        ),
        "negative_provisional": int(
            (cohort["etiqueta_cribado_provisional"] == 0).sum()
        ),
        "unlabeled_provisional": int(
            cohort["etiqueta_cribado_provisional"].isna().sum()
        ),
        "warning": (
            "Vinculación provisional. Confirmar fechas, rondas y etiquetas "
            "antes de utilizar estos datos en machine learning."
        ),
    }

    pd.Series(summary, name="value").to_csv(
        output_dir / "summary.csv",
        header=True,
    )

    LOGGER.info("Archivos guardados en: %s", output_dir.resolve())
    for key, value in summary.items():
        LOGGER.info("%s: %s", key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/link_audit"),
        help="Directorio local para los CSV de auditoría.",
    )

    parser.add_argument(
        "--tolerance-days",
        type=int,
        default=35,
        help="Tolerancia entre intervalos DICOM y clínicos.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar logs detallados.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    run_pipeline(
        args.output_dir,
        tolerance_days=args.tolerance_days,
    )


if __name__ == "__main__":
    main()
