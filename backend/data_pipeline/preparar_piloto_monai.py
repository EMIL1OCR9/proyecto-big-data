"""LungAI - selecciona un piloto mínimo de CT para probar un ciclo de entrenamiento.

Elige una CT negativa ya convertida en training si existe, una positiva de
training y una de cada clase en validation. No incluye test ni cambia el
Parquet original. La salida contiene IDs de investigación: mantener local.

Uso desde la raíz del proyecto:
    uv run python backend/data_pipeline/preparar_piloto_monai.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("outputs/monai_dataset/cohorte_cancer_12m.parquet"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/monai"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/monai_dataset/piloto_monai.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    required = {"split", "en_muestra", "etiqueta_cancer_12m", "ruta_imagen",
                "PatientID", "SeriesInstanceUID", "series_size_MB"}
    missing = required - set(df.columns)
    if missing:
        parser.error(f"Faltan columnas: {sorted(missing)}")

    df = df.loc[df["en_muestra"].fillna(False).astype(bool)].copy()
    df["etiqueta_cancer_12m"] = pd.to_numeric(df["etiqueta_cancer_12m"], errors="coerce")
    df["series_size_MB"] = pd.to_numeric(df["series_size_MB"], errors="coerce")
    df["archivo_existe"] = df["ruta_imagen"].map(
        lambda p: isinstance(p, str) and (args.data_dir / p).is_file()
    )

    rows = []
    for split, label in (("train", 0), ("train", 1),
                         ("validation", 0), ("validation", 1)):
        candidates = df.loc[df["split"].eq(split)
                            & df["etiqueta_cancer_12m"].eq(label)].copy()
        if candidates.empty:
            parser.error(f"No hay casos disponibles para {split}, etiqueta {label}")
        # Prioriza imágenes ya convertidas y después descargas pequeñas.
        candidates = candidates.sort_values(
            ["archivo_existe", "series_size_MB", "PatientID"],
            ascending=[False, True, True], kind="stable")
        rows.append(candidates.iloc[[0]])

    sample = pd.concat(rows, ignore_index=True)
    if sample["PatientID"].nunique() != len(sample):
        raise RuntimeError("Mismo paciente seleccionado más de una vez: no continuar")

    already = int(sample["archivo_existe"].sum())
    needed_mb = float(sample.loc[~sample["archivo_existe"], "series_size_MB"].sum())
    sample = sample.drop(columns="archivo_existe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output, index=False)

    print("=== LUNGAI | PILOTO MONAI (SOLO PRUEBA TECNICA) ===")
    print("Training: 1 negativo y 1 positivo")
    print("Validation: 1 negativo y 1 positivo")
    print("Test: no incluido (se conserva reservado)")
    print(f"NIfTI ya presentes: {already}/4")
    print(f"Descarga adicional estimada: {needed_mb / 1024:.2f} GB DICOM")
    print(f"Parquet local creado: {args.output}")
    print("ADVERTENCIA: 4 pacientes NO permiten medir rendimiento clínico ni generalización.")


if __name__ == "__main__":
    main()
