"""
LungAI — cohorte T0 con etiqueta de cáncer confirmado a 12 meses.

Este módulo NO descarga los píxeles DICOM. Toma una serie CT de la ronda T0 por
paciente y le asigna una etiqueta según el diagnóstico y el seguimiento clínico.

Uso:
    python backend/data_pipeline/build_cancer_cohort.py --output-dir outputs/cancer_cohort

Dependencias:
    pip install idc-index pandas numpy

Importante:
- La ronda sale del primer campo de SeriesDescription ("0,OPA,..."). StudyDate solo
  tiene una fecha anonimizada por ronda, así que no sirve para validar intervalos.
- La etiqueta indica un diagnóstico en los 365 días posteriores a la CT. No confirma
  que el cáncer sea visible en esa CT.
- Un valor ausente en candx_days no se interpreta como negativo: el negativo exige
  que canc_free_days cubra todo el horizonte.
- Los CSV contienen identificadores de investigación. No subirlos a GitHub.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from link_nlst_ct import normalize_id, prepare_image_index, require_columns

LOGGER = logging.getLogger("lungai.build_cancer_cohort")

CLINICAL_COLUMNS = [
    "dicom_patient_id",
    "scr_days0",
    "scr_res0",
    "candx_days",
    "canc_free_days",
    "can_scr",
    "cancyr",
]


def select_t0_series(images: pd.DataFrame) -> pd.DataFrame:
    """Elige una serie axial original (OPA) de T0 por paciente."""
    fields = images["SeriesDescription"].astype("string").str.split(",")
    t0 = images[fields.str[0].eq("0") & fields.str[1].eq("OPA")].copy()

    # Mismo filtro orientativo que link_nlst_ct.build_screening_cohort.
    t0 = t0[
        t0["instanceCount"].between(60, 600)
        & t0["series_size_MB"].between(15, 300)
    ].copy()

    t0 = t0.sort_values(
        ["PatientID", "instanceCount", "series_size_MB"],
        ascending=[True, False, False],
    )
    return t0.drop_duplicates("PatientID")


def prepare_clinical_table(
    participants: pd.DataFrame,
    cancer: pd.DataFrame,
) -> pd.DataFrame:
    """Prepara nlst_prsn y excluye registros de cáncer contradictorios."""
    require_columns(participants, CLINICAL_COLUMNS, "nlst_prsn")
    require_columns(cancer, ["dicom_patient_id"], "nlst_canc")

    clinical = participants[CLINICAL_COLUMNS].rename(
        columns={"dicom_patient_id": "PatientID"}
    )
    clinical["PatientID"] = normalize_id(clinical["PatientID"])
    clinical = clinical.dropna(subset=["PatientID"])

    if clinical["PatientID"].duplicated().any():
        raise ValueError("nlst_prsn tiene pacientes repetidos.")

    for column in CLINICAL_COLUMNS[1:]:
        clinical[column] = pd.to_numeric(clinical[column], errors="coerce")

    cancer_ids = set(normalize_id(cancer["dicom_patient_id"]).dropna())
    clinical["en_nlst_canc"] = clinical["PatientID"].isin(cancer_ids)
    has_diagnosis = clinical["candx_days"].notna()

    consistent = (
        has_diagnosis.eq(clinical["can_scr"].gt(0))
        & has_diagnosis.eq(clinical["en_nlst_canc"])
    )
    LOGGER.info(
        "Pacientes excluidos por registros de cáncer contradictorios: %s",
        int((~consistent).sum()),
    )
    return clinical[consistent].copy()


def label_cancer_12m(
    cohort: pd.DataFrame,
    horizon_days: int = 365,
) -> pd.DataFrame:
    """Asigna etiqueta_cancer_12m (1, 0 o NaN) y el motivo de cada caso."""
    exam = cohort["scr_days0"]
    diagnosis = cohort["candx_days"]
    limit = exam + horizon_days

    conditions = [
        exam.isna(),
        diagnosis.le(exam),
        diagnosis.le(limit),
        diagnosis.gt(limit),
        cohort["canc_free_days"].ge(limit),
    ]
    reasons = [
        "sin_dia_de_examen",
        "diagnostico_previo_o_mismo_dia",
        "diagnostico_en_horizonte",
        "diagnostico_despues_del_horizonte",
        "seguimiento_sin_cancer_suficiente",
    ]
    labels = [np.nan, np.nan, 1.0, 0.0, 0.0]

    labeled = cohort.copy()
    labeled["etiqueta_cancer_12m"] = np.select(
        conditions,
        labels,
        default=np.nan,
    )
    labeled["motivo_etiqueta"] = np.select(
        conditions,
        reasons,
        default="seguimiento_insuficiente",
    )
    return labeled


def run_pipeline(output_dir: Path, horizon_days: int = 365) -> None:
    """Consulta IDC, construye la cohorte y guarda los CSV."""
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
    cancer = client.get_clinical_table("nlst_canc")

    if participants is None or cancer is None:
        raise RuntimeError("No se pudieron cargar nlst_prsn y nlst_canc.")

    images = prepare_image_index(client.index)
    clinical = prepare_clinical_table(participants, cancer)

    cohort = select_t0_series(images).merge(
        clinical,
        on="PatientID",
        how="inner",
        validate="one_to_one",
    )
    cohort = label_cancer_12m(cohort, horizon_days)

    summary = cohort["motivo_etiqueta"].value_counts().to_dict()
    summary.update(
        {
            "series_t0": len(cohort),
            "positivos": int(cohort["etiqueta_cancer_12m"].eq(1).sum()),
            "negativos": int(cohort["etiqueta_cancer_12m"].eq(0).sum()),
            "sin_etiqueta": int(cohort["etiqueta_cancer_12m"].isna().sum()),
        }
    )

    cohort.to_csv(output_dir / "cohorte_cancer_12m.csv", index=False)
    pd.Series(summary, name="value").to_csv(
        output_dir / "resumen_cohorte.csv",
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
        default=Path("outputs/cancer_cohort"),
        help="Directorio local para los CSV de la cohorte.",
    )

    parser.add_argument(
        "--horizon-days",
        type=int,
        default=365,
        help="Días posteriores a la CT que cubre la etiqueta.",
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

    run_pipeline(args.output_dir, args.horizon_days)


if __name__ == "__main__":
    main()
