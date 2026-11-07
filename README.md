# LungAI

Proyecto escolar de big data sobre tomografias de torax (CT) del [NLST](https://cdas.cancer.gov/datasets/nlst/), el ensayo de tamizaje de cancer de pulmon de Estados Unidos. El objetivo es relacionar cada CT con sus datos clinicos, preparar etiquetas confiables y probar modelos que detecten nodulos y estimen el riesgo de cancer.

**Es un proyecto academico: no es una herramienta diagnostica ni esta validado para uso clinico.**

## Estado

| Parte | Estado |
|---|---|
| Descarga de una muestra de CT | Listo (`nlst/`) |
| Union de CT y datos clinicos | Listo, con mapeo de rondas por corregir |
| Limpieza y transformacion de metadatos | Listo |
| Cohorte con cancer confirmado a 12 meses | Listo |
| Division train/validation/test, Parquet y datalist MONAI | Listo |
| Conversion de imagenes a NIfTI | Listo, probado con 7 series |
| Inferencia (pulmones y nodulos) | Probado en un notebook de Colab, pendiente en el repo |
| Medicion, analisis longitudinal y evaluacion clinica | Pendiente |
| Web (React, backend y base de datos) y agente | Pendiente |

## Estructura

```text
backend/data_pipeline/
    link_nlst_ct.py             Une CT y datos clinicos, y arma la cohorte T0 de cribado
    quality_control_nlst_ct.py  Revisa duplicados, IDs faltantes y valores invalidos
    profile_nlst_ct.py          Resume cada variable: tipo, faltantes y valores unicos
    transform_nlst_ct.py        Normaliza tipos y genera el reporte final
    build_cancer_cohort.py      Arma la cohorte T0 con cancer confirmado a 12 meses
    prepare_monai_dataset.py    Divide la cohorte, elige la muestra y exporta Parquet y datalist
    convert_to_nifti.py         Descarga la muestra y la guarda en NIfTI
nlst/
    descargar_nlst.py           Descarga CT con limite de GB
    ver_dicom.py                Muestra un corte de una CT
    README.md                   Detalles de la descarga y de las tablas clinicas
notebooks/
    01_exploracion.py           Notebook marimo de prueba
src/proyecto_big_data/          Paquete base creado por uv
```

Los scripts escriben en `data/` y `outputs/`. Git ignora ambas carpetas: **no subir CT ni CSV con identificadores a GitHub.**

## Instalacion

Necesitas [uv](https://docs.astral.sh/uv/). Python 3.13 lo instala uv si hace falta.

```bash
git clone https://github.com/EMIL1OCR9/proyecto-big-data.git
cd proyecto-big-data
uv sync
```

## Uso

Corre los comandos desde la raiz del repo.

**Cohorte de cancer confirmado:**

```bash
uv run python backend/data_pipeline/build_cancer_cohort.py
```

Genera `outputs/cancer_cohort/cohorte_cancer_12m.csv` y `resumen_cohorte.csv`.

**Cohorte de cribado y control de calidad:**

```bash
uv run python backend/data_pipeline/link_nlst_ct.py
uv run python backend/data_pipeline/quality_control_nlst_ct.py --input outputs/link_audit/t0_screening_cohort_provisional.csv
uv run python backend/data_pipeline/profile_nlst_ct.py
uv run python backend/data_pipeline/transform_nlst_ct.py
```

**Dataset para MONAI:**

```bash
uv run python backend/data_pipeline/prepare_monai_dataset.py
uv run python backend/data_pipeline/convert_to_nifti.py --split test --limit 10 --workers 1
```

Genera en `outputs/monai_dataset/` la tabla `cohorte_cancer_12m.parquet`, `datalist.json` y `resumen_division.csv`. Antes de remuestrear se revisa la geometria de cada serie y el relleno fuera del campo de vision (`PixelPaddingValue` y todo lo que esta debajo de -1024 HU) se cambia por aire; esta parte viene del preprocesamiento de Iris. Las imagenes quedan en `data/monai/images/`, en HU y remuestreadas a 0.703 x 0.703 x 1.25 mm; cada serie ocupa cerca de 50 MB. Usa `--workers 1` en una PC y sube el valor en los servidores de la escuela, donde se hara la descarga completa. Cada worker usa cerca de 2 GB de RAM. `--limit` es obligatorio. Volver a correr el script salta lo ya convertido y reintenta lo que fallo; los fallos quedan en `data/monai/conversion_fallidas.csv` y el resumen de calidad de cada serie en `data/monai/conversion_resumen.csv`.

**Descargar una muestra de CT y ver un corte:** sigue [nlst/README.md](nlst/README.md).

Las consultas usan metadatos publicos de [IDC](https://portal.imaging.datacommons.cancer.gov/collections/nlst/); solo `descargar_nlst.py` baja imagenes.

## Cohorte de cancer a 12 meses

Usa una serie axial original (OPA) de la ronda T0 por paciente. La ronda sale del primer campo de `SeriesDescription`, y el dia de la CT sale de `scr_days0`.

| Etiqueta | Regla | Series |
|---|---|---|
| 1 | Diagnostico en los 365 dias despues de la CT | 283 |
| 0 | Diagnostico despues de 365 dias | 758 |
| 0 | Sin diagnostico, y `canc_free_days` cubre los 365 dias | 24,395 |
| Sin etiqueta | Seguimiento insuficiente | 270 |

Total: 25,706 series.

- Un cribado positivo no es cancer: la etiqueta usa el diagnostico (`candx_days`), no `scr_res0`.
- Si falta `candx_days` no se asume negativo: se exige seguimiento suficiente.
- La etiqueta indica un diagnostico en el ano siguiente; no confirma que el cancer se vea en esa CT.
- Solo 1.1 % de las series etiquetadas es positiva. Para entrenar hay que separar por paciente y reportar con la prevalencia real.

## Division de datos

Con la semilla 42 cada paciente cae en un solo conjunto (70 % train, 15 % validation, 15 % test), estratificado por etiqueta, cribado positivo y fabricante del tomografo. Se descargan todos los positivos y 9 negativos por positivo, para cumplir el minimo de 250 GB de imagenes del proyecto.

| Conjunto | Pacientes con etiqueta | Positivos | Series a descargar | Descarga (GB) |
|---|---|---|---|---|
| train | 17,804 | 198 | 1,984 | 183.9 |
| validation | 3,818 | 43 | 433 | 38.3 |
| test | 3,814 | 42 | 424 | 38.1 |

La muestra tiene cerca de 10 % de positivos; la prevalencia real es 1.1 %. Para reportar metricas de validation y test hay que ponderar con `peso_muestreo`.

## Limitaciones conocidas

- `link_nlst_ct.py` valida rondas comparando intervalos de fechas, pero `StudyDate` esta anonimizada: solo tiene una fecha por ronda. Ese filtro deja 11,561 series T0; hay que cambiarlo por `SeriesDescription`.
- Las anotaciones publicas pueden traer errores. Por ejemplo, en un paciente la tabla reporta 53 mm y en la imagen el nodulo mide cerca de 4 mm.
- El notebook de Colab con los modelos todavia no esta en el repo.

## Datos y licencias

Las imagenes y tablas clinicas son publicas y anonimizadas. Revisa las condiciones de [IDC](https://portal.imaging.datacommons.cancer.gov/collections/nlst/) y [TCIA](https://www.cancerimagingarchive.net/) antes de redistribuirlas. El conjunto clinico completo del NLST requiere solicitud en [CDAS](https://cdas.cancer.gov/datasets/nlst/).

## Equipo

Emilio, Iris y Sammy.
