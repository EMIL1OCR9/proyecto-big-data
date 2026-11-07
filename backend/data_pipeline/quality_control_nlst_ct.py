from pathlib import Path
import argparse
import logging
import pandas as pd

def normalizar_id(columna):
    """Conserva los identificadores como texto."""
    return columna.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def convertir_fecha_dicom(columna):
    """Convierte fechas DICOM YYYYMMDD a fecha; los valores inválidos quedan como NaT."""
    texto = columna.astype("string").str.replace(r"[^0-9]", "", regex=True)
    texto = texto.where(texto.str.len() == 8)
    return pd.to_datetime(texto, format="%Y%m%d", errors="coerce")

def ejecutar_control_calidad(entrada, salida):
    salida.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(entrada, low_memory=False)
    filas_originales = len(df)

    if df.empty:
        raise ValueError("El CSV está vacío. Revisa la etapa de integración.")

    requeridas = ["PatientID", "SeriesInstanceUID"]
    faltantes_cols = [c for c in requeridas if c not in df.columns]
    if faltantes_cols:
        raise ValueError(f"Faltan columnas necesarias: {faltantes_cols}")

    for col in ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]:
        if col in df.columns:
            df[col] = normalizar_id(df[col])

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    if "StudyDate" in df.columns:
        df["fecha_dicom_qc"] = convertir_fecha_dicom(df["StudyDate"])
    if "fecha_dicom" in df.columns:
        df["fecha_dicom"] = pd.to_datetime(df["fecha_dicom"], errors="coerce")

    numericas = [
        "instanceCount", "series_size_MB", "study_yr",
        "scr_days0", "scr_days1", "scr_days2",
        "scr_res0", "scr_res1", "scr_res2", "candx_days",
        "resultado_cribado_t0", "etiqueta_cribado_provisional",
    ]
    for col in numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Auditoría de SeriesInstanceUID repetidos; no se elige una fila arbitrariamente.
    duplicados = df[
        df["SeriesInstanceUID"].notna()
        & df["SeriesInstanceUID"].duplicated(keep=False)
    ].copy()
    duplicados.to_csv(salida / "duplicados_por_serie.csv", index=False)

    duplicados_exactos = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df["qc_id_faltante"] = df["PatientID"].isna() | df["SeriesInstanceUID"].isna()
    df["qc_instanceCount_invalido"] = False
    if "instanceCount" in df.columns:
        df["qc_instanceCount_invalido"] = df["instanceCount"].notna() & (df["instanceCount"] <= 0)
    df["qc_tamano_invalido"] = False
    if "series_size_MB" in df.columns:
        df["qc_tamano_invalido"] = df["series_size_MB"].notna() & (df["series_size_MB"] <= 0)

    reporte = pd.DataFrame({
        "columna": df.columns,
        "faltantes": [int(df[c].isna().sum()) for c in df.columns],
        "porcentaje_faltante": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
        "tipo_dato": [str(df[c].dtype) for c in df.columns],
    })
    reporte.to_csv(salida / "reporte_faltantes.csv", index=False)

    resumen = {
        "filas_entrada": filas_originales,
        "filas_salida": len(df),
        "filas_duplicadas_exactas_eliminadas": duplicados_exactos,
        "filas_con_id_faltante": int(df["qc_id_faltante"].sum()),
        "filas_con_numero_cortes_invalido": int(df["qc_instanceCount_invalido"].sum()),
        "filas_con_tamano_invalido": int(df["qc_tamano_invalido"].sum()),
        "series_en_grupos_duplicados": int(duplicados["SeriesInstanceUID"].nunique()),
    }
    pd.Series(resumen, name="valor").to_csv(salida / "resumen_qc.csv", header=True)
    df.to_csv(salida / "cohorte_revisada.csv", index=False)

    logging.info("Control de calidad terminado.")
    print(pd.Series(resumen).to_string())
    print(f"\nArchivos guardados en: {salida.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Control de calidad de metadatos CT/NLST.")
    parser.add_argument("--input", type=Path, required=True, help="CSV de entrada de la integración.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quality_control"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ejecutar_control_calidad(args.input, args.output_dir)

if __name__ == "__main__":
    main()