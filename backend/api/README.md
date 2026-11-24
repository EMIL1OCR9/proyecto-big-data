# API de LungAI — piloto local

Consulta la cohorte y analiza estudios que se suben desde la web. Para la cohorte lee:
- `outputs/monai_dataset/cohorte_cancer_12m.parquet` (solo filas `en_muestra=True`).
- `outputs/nodulos/<PatientID>/<SeriesInstanceUID>.json` (si existe).
- `outputs/nodulos/<PatientID>/<SeriesInstanceUID>.png` (si existe).
- `data/monai/<ruta_imagen>` (ej. `images/<PatientID>/<SeriesUID>.nii.gz`).

## Instalar, desde la raiz del repositorio

```powershell
uv sync
```

## Ejecutar (solo local)

```powershell
uv run python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8081
```

La web (`frontend/`) busca la API en el puerto 8081. Visitar `http://127.0.0.1:8081/docs` para probar las rutas.

## Rutas

- `GET /health`: comprobar servidor sin cargar datos.
- `GET /pacientes?limit=20&offset=0&solo_con_resultados=true`: pacientes con resultados JSON disponibles por defecto; con `solo_con_resultados=false`, muestra tambien los pendientes. Admite `split=train|validation|test`.
- `GET /pacientes/{id}`: series, `split`, etiqueta historica a 12 meses y estado de analisis.
- `GET /series/{uid}/nodulos`: JSON generado por el detector; 404 si no se ejecuto.
- `GET /series/{uid}/imagen`: NIfTI `.nii.gz`, si fue convertido.
- `GET /series/{uid}/vista`: PNG generado por `ver_nodulos.py`; 404 para series sin PNG.

### Analizar un estudio subido

- `POST /estudios` (campo `archivo`): un `.zip` con los DICOM de una serie, o un `.nii` / `.nii.gz`. Responde 202 y lo pone en cola.
- `GET /estudios`: estudios subidos, el mas nuevo primero, con su estado (`en_cola`, `procesando`, `listo`, `error`) y el paso en que va.
- `GET /estudios/{id}`: estado y, cuando esta listo, el resultado: nodulos con posicion, diametro, lobulo, puntaje del detector y probabilidad del clasificador.
- `GET /estudios/{id}/imagen` y `/marcas`: la CT convertida y la mascara de los nodulos, para el visor.

El analisis repite el pipeline de Colab en esta computadora: convierte la CT, corre el detector y lungmask, recorta cada nodulo y lo pasa por el clasificador. Va un estudio a la vez; en CPU tarda varios minutos.

- Los estudios se guardan en `data/estudios/<id>/` (en `.gitignore`). El archivo subido se borra al terminar; queda solo la CT convertida.
- El clasificador se lee de `data/modelos/clasificador/modelo.pt`. Copialo ahi desde `lungai/clasificador` en Drive, junto con `metricas.json`, que trae los AUC de test que muestra la web. Sin el, el analisis solo muestra la deteccion.
- La probabilidad viene de un modelo entrenado con 1 maligno por cada 10 benignos: sirve para comparar nodulos entre si, no es el riesgo real. En test no supero a medir solo el tamano.

Distinciones:
- `nodulos_detectados: null` = **no existe JSON** (analisis pendiente).
- `nodulos_detectados: 0` = existe JSON con **cero nodulos por encima del criterio del detector**.
- `nodulo_mayor_mm` = mayor dimension del bounding box (no diagnostico).
- `etiqueta_cancer_12m` = etiqueta historica de investigacion (no prediccion del detector).

Las rutas de la cohorte no hacen inferencia ni crean JSON o PNG; solo `/estudios` corre modelos.
Para probar el backend sin resultados reales, las pruebas automatizadas usan datos sinteticos y un analisis ficticio.

## Pruebas

```powershell
uv run python -m pytest -q tests/test_api.py
```

## Seguridad (muy importante)

**No publicar esta version en internet ni usar `--host 0.0.0.0`.**
No tiene autenticacion/autorizacion. Aunque NLST esta desidentificado, el servicio expone identificadores de investigacion, etiquetas clinicas y estudios. Mantener en localhost. Para servidores: autenticacion por usuario, politicas de acceso, IDs opacos, audit log, TLS, limites de transferencia, almacenamiento protegido y revision de licencias/condiciones de uso. No subir `data/` ni `outputs/`.

La capa `DatasetService` concentra accesos a disco para reemplazarla posteriormente por la base de datos de Emi. Mantiene un indice pequeno de la cohorte en memoria, pero transmite NIfTI mediante `FileResponse`. En una instancia con miles de usuarios o decenas de miles de reportes, pasar la agregacion de reportes a una tabla Parquet / base de datos indexada.

**Este proyecto es academico, no esta validado para uso clinico.**
