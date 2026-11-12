# LungAI

Proyecto escolar de big data sobre tomografias de torax (CT) del [NLST](https://cdas.cancer.gov/datasets/nlst/), el ensayo de tamizaje de cancer de pulmon de Estados Unidos. El objetivo es relacionar cada CT con sus datos clinicos, preparar etiquetas confiables y probar modelos que detecten nodulos y estimen el riesgo de cancer.

**Es un proyecto academico: no es una herramienta diagnostica ni esta validado para uso clinico.**

## Estado

| Parte | Estado |
|---|---|
| Descarga de una muestra de CT | Listo (`nlst/`) |
| Union de CT y datos clinicos | Listo |
| Limpieza y transformacion de metadatos | Listo |
| Cohorte con cancer confirmado a 12 meses | Listo |
| Division train/validation/test, Parquet y datalist MONAI | Listo |
| Conversion de imagenes a NIfTI | Listo, piloto de 54 series sin fallos |
| Deteccion de nodulos | Listo, probado en CPU con 2 CT de referencia; falta la corrida con GPU |
| Procesamiento por lotes sin guardar las CT | Listo, probado en CPU con 2 CT; falta la corrida con GPU |
| Etiquetas de nodulos (benigno o maligno) | Listo, probado con 2 CT; faltan los datos de la corrida completa |
| Clasificador de nodulos | Pendiente: falta el entrenamiento |
| Medicion, analisis longitudinal y evaluacion clinica | Pendiente |
| API (FastAPI) y front (React con NiiVue) | Prototipo |
| Base de datos y agente | Pendiente |

## Estructura

```text
backend/data_pipeline/
    link_nlst_ct.py             Une CT y datos clinicos, y arma la cohorte T0 de cribado
    quality_control_nlst_ct.py  Revisa duplicados, IDs, una serie T0 por paciente y etiquetas
    profile_nlst_ct.py          Resume cada variable: tipo, faltantes y valores unicos
    transform_nlst_ct.py        Normaliza tipos y genera el reporte final
    build_cancer_cohort.py      Arma la cohorte T0 con cancer confirmado a 12 meses
    prepare_monai_dataset.py    Divide la cohorte, elige la muestra y exporta Parquet y datalist
    convert_to_nifti.py         Descarga la muestra y la guarda en NIfTI
    preparar_piloto_monai.py    Elige 4 CT para probar un ciclo de entrenamiento
    probar_modelo_monai.py      Comprueba que una red 3D de MONAI corre con una CT real
backend/inference/
    detectar_nodulos.py         Detecta nodulos en los NIfTI y guarda un JSON por serie
    ver_nodulos.py              Genera una imagen PNG por serie con los nodulos marcados
    procesar_lotes.py           Descarga, convierte, detecta y recorta por lotes, sin guardar las CT
    etiquetar_nodulos.py        Etiqueta cada candidato como maligno o benigno con las tablas del NLST
backend/api/                    API de pacientes y nodulos (ver su README)
frontend/                       Interfaz web con visor de CT (ver su README)
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

**Control de calidad de la cohorte de cancer:**

```bash
uv run python backend/data_pipeline/quality_control_nlst_ct.py --input outputs/cancer_cohort/cohorte_cancer_12m.csv
uv run python backend/data_pipeline/profile_nlst_ct.py
uv run python backend/data_pipeline/transform_nlst_ct.py
```

Revisa duplicados, IDs faltantes, que cada paciente tenga una sola serie T0 axial original y que la etiqueta coincida con los dias al diagnostico. El resumen queda en `outputs/quality_control/resumen_qc.csv`.

**Cohorte de cribado (opcional):**

```bash
uv run python backend/data_pipeline/link_nlst_ct.py
```

Usa las mismas series que la cohorte de cancer, pero con la etiqueta de cribado (`scr_res0`). Un cribado positivo no es un cancer confirmado.

**Dataset para MONAI:**

```bash
uv run python backend/data_pipeline/prepare_monai_dataset.py
uv run python backend/data_pipeline/convert_to_nifti.py --split test --limit 10 --workers 1
```

Genera en `outputs/monai_dataset/` la tabla `cohorte_cancer_12m.parquet`, `datalist.json` y `resumen_division.csv`. Antes de remuestrear se ordenan los cortes por posicion, se quitan los repetidos identicos, se revisa la geometria de cada serie y el relleno fuera del campo de vision (`PixelPaddingValue` y todo lo que esta debajo de -1024 HU) se cambia por aire; esta parte viene del preprocesamiento de Iris. Las imagenes quedan en `data/monai/images/`, en HU y remuestreadas a 0.703 x 0.703 x 1.25 mm; cada serie ocupa cerca de 50 MB. Usa `--workers 1` en una PC y sube el valor en los servidores de la escuela, donde se hara la descarga completa. Cada worker usa cerca de 2 GB de RAM. `--limit` es obligatorio y el script se detiene antes de empezar si no hay disco suficiente. Al inicio y al final muestra cuantas series y GB van descargados, para comprobar el minimo de 250 GB. Volver a correr el script salta lo ya convertido y reintenta lo que fallo; los fallos quedan en `data/monai/conversion_fallidas.csv` y el resumen de calidad de cada serie en `data/monai/conversion_resumen.csv`.

**Deteccion de nodulos:**

```bash
uv run python backend/inference/detectar_nodulos.py --limit 5
```

Lee los NIfTI de `data/monai/images/`, segmenta pulmones y lobulos con lungmask y corre el detector preentrenado de MONAI (`lung_nodule_ct_detection`, entrenado con LUNA16). Guarda un JSON por serie en `outputs/nodulos/<PatientID>/<SeriesInstanceUID>.json`:

- `nodulos`: candidatos con puntaje de 0.5 o mas dentro del pulmon. Cada uno trae `puntaje`, `centro_mm` (coordenadas del paciente, LPS), `centro_voxel` (x, y, z en el NIfTI), `tamano_mm`, `lobulo` y `en_pulmon`.
- `otros_candidatos`: el resto, para evaluar con otros umbrales sin volver a correr el modelo.

Detecta nodulos; no dice si son cancer. La primera vez descarga los modelos a `data/modelos`. Necesita GPU para correr a escala: en CPU tardo 9 y 25 minutos en las 2 CT de prueba. `--limit` es obligatorio. Volver a correrlo salta las series que ya tienen JSON; los fallos quedan en `outputs/nodulos/deteccion_fallidas.csv`.

**Ver los nodulos detectados:**

```bash
uv run python backend/inference/ver_nodulos.py
```

Genera una imagen PNG junto a cada JSON: por cada nodulo, el corte completo con la caja en rojo y un acercamiento de 6 x 6 cm, con su puntaje, lobulo y tamano. Muestra hasta 5 nodulos por serie (`--max`). Las series sin nodulos no generan imagen.

**Procesar la muestra por lotes, sin guardar las CT:**

```bash
uv run python backend/inference/procesar_lotes.py --limit 8
```

Por cada lote descarga las series de IDC, las convierte, detecta nodulos, recorta un cubo alrededor de cada candidato y borra las CT. Asi los 260 GB de la muestra pasan por el programa, pero solo se guarda:

- `outputs/nodulos/`: el mismo JSON por serie, con dos campos mas por candidato: `corte_dicom` (el numero de corte que usa el radiologo) y `recorte` (su posicion en el archivo de recortes).
- `outputs/recortes/<PatientID>/<SeriesInstanceUID>.npz`: un cubo de 64 x 64 x 40 voxeles (unos 45 x 45 x 50 mm, en HU) por cada candidato con puntaje de 0.1 o mas dentro del pulmon. Son la entrada del clasificador; cada uno pesa cerca de 0.2 MB.
- `outputs/lotes_progreso.csv`: una fila por serie procesada, con sus MB de descarga. Al final de cada corrida se muestra el total de series y GB procesados.

Volver a correrlo sigue donde se quedo; los fallos quedan en `outputs/lotes_fallidas.csv`. Con `--keep-images` las CT no se borran, y las que ya existian en `data/monai/` nunca se borran. Necesita GPU para correr a escala.

**Etiquetar los candidatos para el clasificador:**

```bash
uv run python backend/inference/etiquetar_nodulos.py
```

Cruza los JSON de `procesar_lotes.py` con las tablas publicas del NLST, sin descargar imagenes:

- Un candidato es el nodulo que reporto el radiologo si esta a +-5 cortes, en el mismo lobulo y su caja mide entre la mitad y el doble del diametro reportado.
- `maligno`: es un nodulo reportado y el paciente tuvo cancer confirmado en ese lobulo.
- `benigno`: el paciente nunca tuvo cancer y se le siguio al menos 2 anos. Incluye candidatos que el radiologo no reporto.
- Sin etiqueta: los demas candidatos de pacientes con cancer, o con poco seguimiento.

Guarda en `outputs/etiquetas/` la tabla `candidatos_etiquetados.parquet` (una fila por candidato, con su `recorte` y su `split`), `nodulos_reportados.parquet` (si el detector encontro cada nodulo del radiologo) y `resumen_etiquetas.csv`. La etiqueta `maligno` es aproximada: se sabe que hubo cancer en ese lobulo, no que salio de ese nodulo exacto.

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

- `StudyDate` esta anonimizada (una sola fecha por ronda), asi que la ronda se toma de `SeriesDescription`.
- Las anotaciones publicas pueden traer errores. Por ejemplo, en un paciente la tabla reporta 53 mm y en la imagen el nodulo mide cerca de 4 mm.
- La deteccion de nodulos solo se ha probado en CPU con 2 CT; falta medir tiempos con GPU y evaluarla contra las anotaciones del radiologo.

## Datos y licencias

Las imagenes y tablas clinicas son publicas y anonimizadas. Revisa las condiciones de [IDC](https://portal.imaging.datacommons.cancer.gov/collections/nlst/) y [TCIA](https://www.cancerimagingarchive.net/) antes de redistribuirlas. El conjunto clinico completo del NLST requiere solicitud en [CDAS](https://cdas.cancer.gov/datasets/nlst/).

## Equipo

Emilio, Iris y Sammy.
