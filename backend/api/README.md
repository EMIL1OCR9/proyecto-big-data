# API de LungAI — piloto local

API de solo lectura, separada del detector y de MONAI. Lee:
- `outputs/monai_dataset/cohorte_cancer_12m.parquet` (solo filas `en_muestra=True`).
- `outputs/nodulos/<PatientID>/<SeriesInstanceUID>.json` (si existe).
- `outputs/nodulos/<PatientID>/<SeriesInstanceUID>.png` (si existe).
- `data/monai/<ruta_imagen>` (ej. `images/<PatientID>/<SeriesUID>.nii.gz`).

## Instalar, desde la raiz del repositorio

```powershell
uv add fastapi "uvicorn[standard]"
uv add --dev pytest httpx
```

## Ejecutar (solo local)

```powershell
uv run uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Visitar `http://127.0.0.1:8000/docs` para probar las rutas.

## Rutas

- `GET /health`: comprobar servidor sin cargar datos.
- `GET /pacientes?limit=20&offset=0&solo_con_resultados=true`: pacientes con resultados JSON disponibles por defecto; con `solo_con_resultados=false`, muestra tambien los pendientes. Admite `split=train|validation|test`.
- `GET /pacientes/{id}`: series, `split`, etiqueta historica a 12 meses y estado de analisis.
- `GET /series/{uid}/nodulos`: JSON generado por el detector; 404 si no se ejecuto.
- `GET /series/{uid}/imagen`: NIfTI `.nii.gz`, si fue convertido.
- `GET /series/{uid}/vista`: PNG generado por `ver_nodulos.py`; 404 para series sin PNG.

Distinciones:
- `nodulos_detectados: null` = **no existe JSON** (analisis pendiente).
- `nodulos_detectados: 0` = existe JSON con **cero nodulos por encima del criterio del detector**.
- `nodulo_mayor_mm` = mayor dimension del bounding box (no diagnostico).
- `etiqueta_cancer_12m` = etiqueta historica de investigacion (no prediccion del detector).

El servidor NO hace inferencia, NO crea JSON ni PNG y NO descarga modelos.
Para probar el backend sin resultados reales, las pruebas automatizadas usan datos sinteticos.

## Pruebas

```powershell
uv run pytest -q tests/test_api.py
```

## Seguridad (muy importante)

**No publicar esta version en internet ni usar `--host 0.0.0.0`.**
No tiene autenticacion/autorizacion. Aunque NLST esta desidentificado, el servicio expone identificadores de investigacion, etiquetas clinicas y estudios. Mantener en localhost. Para servidores: autenticacion por usuario, politicas de acceso, IDs opacos, audit log, TLS, limites de transferencia, almacenamiento protegido y revision de licencias/condiciones de uso. No subir `data/` ni `outputs/`.

La capa `DatasetService` concentra accesos a disco para reemplazarla posteriormente por la base de datos de Emi. Mantiene un indice pequeno de la cohorte en memoria, pero transmite NIfTI mediante `FileResponse`. En una instancia con miles de usuarios o decenas de miles de reportes, pasar la agregacion de reportes a una tabla Parquet / base de datos indexada.

**Este proyecto es academico, no esta validado para uso clinico.**
