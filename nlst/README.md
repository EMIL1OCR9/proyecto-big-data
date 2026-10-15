# Proyecto NLST - descarga de muestra de TC

Descarga tomografias completas (todas las rebanadas de cada serie) del NLST desde la API publica NBIA de TCIA, con limite de tamano, reanudacion, verificacion MD5/DICOM y etiquetas clinicas publicas.

## Como correrlo

PowerShell, desde la raiz de PROYECTO BIG DATA (los datos quedan en `data/`, que git ignora):

```powershell
cd "C:\Proyectos de VS\PROYECTO BIG DATA"
uv sync                                       # instala dependencias si faltan
uv run python nlst/descargar_nlst.py          # descarga hasta LIMITE_GB
uv run python nlst/ver_dicom.py "data\NLST_DICOM\<PatientID>\<StudyUID>\<SeriesUID>"   # corte central
```

Sin uv: `.\.venv\Scripts\Activate.ps1` y luego `python nlst/descargar_nlst.py`.

VS Code: abre la carpeta PROYECTO BIG DATA, `Ctrl+Shift+P` > *Python: Select Interpreter* > `.venv\Scripts\python.exe`, abre `nlst/descargar_nlst.py` y dale a *Run Python File* (o F5).

## Ampliar la descarga

Cambia `LIMITE_GB` (10, 50, 250) y vuelve a correr. El plan (`data/NLST_DICOM/plan.csv`) se conserva: lo ya bajado no se repite y solo se agregan pacientes nuevos hasta el nuevo limite. Si quieres *reducir* el limite o cambiar `PROPORCION_CANCER`/`SEMILLA`, borra `plan.csv` primero.

Si se corta la conexion o cierras el programa, vuelve a correrlo: una serie solo aparece en su carpeta final cuando ya se valido completa; las incompletas (`.zip.part`, `.tmp`) se rehacen desde cero.

## Que se descarga

- Servidor: `https://nlst.cancerimagingarchive.net/nbia-api/services/v1` (NLST tiene su propio servidor NBIA). Publico, sin login.
- Por paciente se bajan todos sus estudios de tamizaje (T0, T1, T2) y por estudio **una** serie axial original (`OPA`), preferentemente con kernel suave (`B30f`, `STANDARD`, `FC10`, `C`); si no hay, la de mas cortes. Se descartan localizadores (`OPL`).
- Se meten pacientes completos mientras quepan en el limite, usando `FileSize` de la API (= bytes exactos en disco), asi que la carpeta nunca pasa de `LIMITE_GB`. Durante la descarga se usa ademas espacio temporal para el ZIP de una serie (~70 MB).
- `PROPORCION_CANCER = 0.5`: la mitad de los pacientes tienen cancer pulmonar confirmado. Solo ~4 % de NLST tiene cancer, asi que una muestra aleatoria chica casi no tendria positivos. Ojo: esto **no** es la prevalencia real; para evaluar un modelo hay que tenerlo en cuenta.

Estructura: `data/NLST_DICOM/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`

`data/NLST_DICOM/series_descargadas.csv`: PatientID, StudyInstanceUID, SeriesInstanceUID, study_yr, kernel, ImageCount, bytes_descargados y etiquetas (`can_scr`, `cancyr`, `candx_days`).

## Etiquetas clinicas (data/NLST_clinico/)

Paquete publico de TCIA (CC BY 4.0, sin registro), el script lo baja solo. `pid` = `PatientID` de las imagenes.

| Archivo | Contenido |
|---|---|
| `nlst_780_prsn_*.csv` | 1 fila por participante: demografia, tabaquismo, resultado de cada tamizaje (`scr_res0-2`), cancer confirmado (`can_scr`, `cancyr`, `candx_days`), estadio, histologia |
| `nlst_780_canc_*.csv` | detalle de cada cancer pulmonar confirmado (histologia, estadio TNM, `study_yr`) |
| `nlst_780_ctab_*.csv` | anomalias/nodulos reportados por el radiologo en cada TC (tamano, localizacion, rebanada) |
| `nlst_780_ctabc_*.csv` | comparacion de anomalias con estudios previos (crecimiento, cambio de atenuacion) |
| `nlst_780_screen_*.csv` | parametros tecnicos y calidad de cada TC |

Codigos clave (diccionarios `.rtf` en la misma carpeta):
- `can_scr`: 0 sin cancer, 1 tras tamizaje positivo, 2 tras tamizaje negativo, 3 tamizaje fallido, 4 post-tamizaje.
- `cancyr`: study year (0 = T0 ... 7) del primer cancer confirmado.
- `study_yr` en el CSV de series (T0/T1/T2) sale de la descripcion de la serie y coincide con el tag DICOM `ClinicalTrialTimePointID`.

Las etiquetas son **por participante**, no por imagen ni por rebanada. Un tamizaje positivo (`scr_res` 4-6, o un nodulo en `ctab`) no es cancer: la gran mayoria de positivos en NLST fueron falsos positivos. Para "predecir cancer" usa `can_scr`/`candx_days` (p. ej. cancer en los N anios siguientes a la fecha de la TC), y separa train/test por paciente.

El dataset clinico completo (mortalidad, seguimiento detallado, procedimientos diagnosticos, etc.) esta en CDAS de NCI y requiere propuesta aprobada y acuerdo de uso de datos: https://cdas.cancer.gov/nlst/
