from pathlib import Path
import pandas as pd

ENTRADA = Path("outputs/quality_control/cohorte_revisada.csv")
SALIDA = Path("outputs/quality_control")
SALIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ENTRADA, low_memory=False)
transformado = df.copy()

# Identificadores: conservar como texto; no son predictores.
for col in ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]:
    if col in transformado.columns:
        transformado[col] = transformado[col].astype("string").str.strip()

# Variables categóricas: los códigos se conservan; no se inventan etiquetas.
for col in ["scr_res0", "scr_res1", "scr_res2"]:
    if col in transformado.columns:
        transformado[col] = pd.to_numeric(transformado[col], errors="coerce").astype("Int64").astype("string")
        transformado[col] = transformado[col].astype("category")

# Variables numéricas/temporales.
for col in ["instanceCount", "series_size_MB", "scr_days0", "scr_days1", "scr_days2", "candx_days"]:
    if col in transformado.columns:
        transformado[col] = pd.to_numeric(transformado[col], errors="coerce")

# Auditoría de variables: rango, faltantes, tipos y valores únicos.
filas = []
for col in transformado.columns:
    serie = transformado[col]
    numerica = pd.to_numeric(serie, errors="coerce")
    es_numerica = pd.api.types.is_numeric_dtype(serie)
    filas.append({
        "variable": col,
        "tipo_dato": str(serie.dtype),
        "faltantes": int(serie.isna().sum()),
        "porcentaje_faltante": round(serie.isna().mean() * 100, 2),
        "valores_unicos": int(serie.nunique(dropna=True)),
        "minimo": numerica.min() if es_numerica and numerica.notna().any() else None,
        "maximo": numerica.max() if es_numerica and numerica.notna().any() else None,
    })
reporte = pd.DataFrame(filas)

# CSV no conserva el dtype category al volver a leerse, pero conserva los códigos.
transformado.to_csv(SALIDA / "cohorte_variables_transformadas.csv", index=False)
reporte.to_csv(SALIDA / "reporte_transformacion_final.csv", index=False)

print("Dimensiones:", transformado.shape)
print("\nVariables clínicas:")
for col in ["scr_res0", "scr_res1", "scr_res2", "scr_days0", "scr_days1", "scr_days2", "candx_days"]:
    if col in transformado.columns:
        print(f"- {col}: {transformado[col].dtype}; faltantes={transformado[col].isna().sum()}")

print("\nAdvertencias a documentar:")
if "candx_days" in transformado.columns:
    p = transformado["candx_days"].isna().mean() * 100
    if p > 50:
        print(f"- candx_days tiene {p:.2f}% de faltantes; no se imputó.")
for col in ["StudyDate", "fecha_dicom", "fecha_dicom_qc"]:
    if col in transformado.columns and transformado[col].nunique(dropna=True) == 1:
        print(f"- {col} tiene un solo valor no nulo; revisar su procedencia.")
print("\nArchivos generados:")
print(SALIDA / "cohorte_variables_transformadas.csv")
print(SALIDA / "reporte_transformacion_final.csv")