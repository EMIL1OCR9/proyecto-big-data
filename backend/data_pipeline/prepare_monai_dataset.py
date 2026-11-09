"""
LungAI — division de la cohorte y lista de datos para MONAI.

Este modulo NO descarga imagenes. Toma la cohorte de build_cancer_cohort.py, asigna
train/validation/test por paciente, elige la muestra a descargar y exporta la tabla
en Parquet junto con el datalist JSON que lee MONAI.

Uso:
    python backend/data_pipeline/prepare_monai_dataset.py

Importante:
- Cada paciente cae en un solo conjunto, para evitar fuga de datos.
- Todos los positivos entran a la muestra; los negativos se submuestrean. Al evaluar
  hay que usar peso_muestreo para reportar con la prevalencia real.
- La semilla hace la division reproducible. Si cambia, cambia todo el experimento.
- Los archivos contienen identificadores de investigacion. No subirlos a GitHub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")
DATALIST_KEYS = {"train": "training", "validation": "validation", "test": "test"}
STRATA = ["etiqueta_cancer_12m", "cribado_positivo", "fabricante"]
TEXT_COLUMNS = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]


def load_cohort(path: Path) -> pd.DataFrame:
    """Lee la cohorte y agrega las variables para estratificar."""
    cohort = pd.read_csv(path, dtype={c: "string" for c in TEXT_COLUMNS}, low_memory=False)
    if cohort["PatientID"].duplicated().any():
        raise ValueError("La cohorte debe tener una sola serie por paciente.")

    cohort["fabricante"] = cohort["SeriesDescription"].str.split(",").str[2]
    cohort["cribado_positivo"] = cohort["scr_res0"].isin([4, 5, 6])
    cohort["ruta_imagen"] = (
        "images/" + cohort["PatientID"] + "/" + cohort["SeriesInstanceUID"] + ".nii.gz"
    )
    # Orden fijo: asi la misma semilla siempre da la misma division.
    return cohort.sort_values("PatientID", ignore_index=True)


def assign_splits(
    labeled: pd.DataFrame,
    train: float,
    validation: float,
    seed: int,
) -> pd.DataFrame:
    """Baraja dentro de cada estrato y reparte por proporcion."""
    shuffled = labeled.sample(frac=1, random_state=seed)
    groups = shuffled.groupby(STRATA)
    position = (groups.cumcount() + 0.5) / groups["PatientID"].transform("size")

    shuffled["split"] = np.select(
        [position < train, position < train + validation],
        ["train", "validation"],
        default="test",
    )
    return shuffled


def sample_negatives(
    labeled: pd.DataFrame,
    negatives_per_positive: float,
    seed: int,
) -> pd.DataFrame:
    """Conserva los positivos y elige negativos con las mismas proporciones."""
    sampled = labeled.copy()
    positive = sampled["etiqueta_cancer_12m"].eq(1)
    sampled["en_muestra"] = positive
    sampled["peso_muestreo"] = 1.0

    for split in SPLITS:
        negatives = sampled[sampled["split"].eq(split) & ~positive]
        negatives = negatives.sample(frac=1, random_state=seed)
        wanted = negatives_per_positive * (positive & sampled["split"].eq(split)).sum()
        fraction = min(1.0, wanted / len(negatives))

        cells = negatives.groupby(STRATA[1:])
        size = cells["PatientID"].transform("size")
        take = np.ceil(size * fraction)
        chosen = negatives.index[cells.cumcount() < take]

        sampled.loc[chosen, "en_muestra"] = True
        sampled.loc[chosen, "peso_muestreo"] = (size / take).loc[chosen]

    return sampled


def build_datalist(cohort: pd.DataFrame) -> dict[str, list[dict]]:
    """Arma el datalist estilo Decathlon que lee load_decathlon_datalist."""
    sample = cohort[cohort["en_muestra"]]
    return {
        DATALIST_KEYS[split]: [
            {
                "image": row.ruta_imagen,
                "label": int(row.etiqueta_cancer_12m),
                "patient_id": row.PatientID,
                "peso_muestreo": round(row.peso_muestreo, 4),
            }
            for row in sample[sample["split"].eq(split)].itertuples()
        ]
        for split in SPLITS
    }


def summarize(cohort: pd.DataFrame) -> pd.DataFrame:
    """Compara los conjuntos completos y la muestra a descargar."""
    rows = []
    for split in SPLITS:
        members = cohort[cohort["split"].eq(split)]
        for name, group in (("completo", members), ("muestra", members[members["en_muestra"]])):
            makers = group["fabricante"].value_counts(normalize=True).mul(100).round(1)
            rows.append(
                {
                    "conjunto": split,
                    "grupo": name,
                    "series": len(group),
                    "positivos": int(group["etiqueta_cancer_12m"].eq(1).sum()),
                    "prevalencia_%": round(group["etiqueta_cancer_12m"].mean() * 100, 2),
                    "cribado_positivo_%": round(group["cribado_positivo"].mean() * 100, 1),
                    **{f"{maker}_%": value for maker, value in makers.sort_index().items()},
                    "GB": round(group["series_size_MB"].sum() / 1024, 1),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/cancer_cohort/cohorte_cancer_12m.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/monai_dataset"))
    parser.add_argument("--train", type=float, default=0.70, help="Fraccion para train.")
    parser.add_argument("--validation", type=float, default=0.15, help="Fraccion para validation.")
    parser.add_argument("--negatives-per-positive", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train <= 0 or args.validation <= 0 or args.train + args.validation >= 1:
        raise ValueError("train y validation deben dejar una fraccion para test.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(args.input)
    has_label = cohort["etiqueta_cancer_12m"].notna()

    labeled = assign_splits(cohort[has_label], args.train, args.validation, args.seed)
    labeled = sample_negatives(labeled, args.negatives_per_positive, args.seed)

    unlabeled = cohort[~has_label].assign(split=pd.NA, en_muestra=False, peso_muestreo=np.nan)
    result = pd.concat([labeled, unlabeled]).sort_values("PatientID", ignore_index=True)

    summary = summarize(result)

    result.to_parquet(args.output_dir / "cohorte_cancer_12m.parquet", index=False)
    summary.to_csv(args.output_dir / "resumen_division.csv", index=False)
    (args.output_dir / "datalist.json").write_text(
        json.dumps(build_datalist(result), indent=2),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"\nArchivos guardados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
