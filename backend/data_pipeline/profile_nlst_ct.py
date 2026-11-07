from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Perfil de variables de la cohorte LungAI.")
    parser.add_argument("--input", type=Path, default=Path("outputs/quality_control/cohorte_revisada.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quality_control"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, low_memory=False)

    perfil = pd.DataFrame({
        "variable": df.columns,
        "tipo_dato": [str(df[c].dtype) for c in df.columns],
        "valores_faltantes": [int(df[c].isna().sum()) for c in df.columns],
        "porcentaje_faltante": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
        "valores_unicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    perfil.to_csv(args.output_dir / "perfil_variables.csv", index=False)

    print("Dimensiones:", df.shape)
    print("Perfil guardado en:", args.output_dir / "perfil_variables.csv")
    if "etiqueta_cribado_provisional" in df.columns:
        print("\nDistribución de la etiqueta provisional:")
        print(df["etiqueta_cribado_provisional"].value_counts(dropna=False))
    print("\nFaltantes por variable:")
    print(df.isna().sum().to_string())

if __name__ == "__main__":
    main()