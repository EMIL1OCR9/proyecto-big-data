"""FastAPI local para LungAI: consulta la cohorte y analiza estudios subidos. No diagnostica."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.api.analisis import AnalysisService, InvalidUpload, StudyNotFound
from backend.api.data_service import DatasetService, DatasetUnavailable, InvalidResult, SeriesRecord
from backend.api.resultados import ResultsService


def create_app(root: Path | None = None, service: DatasetService | None = None,
               analysis: AnalysisService | None = None) -> FastAPI:
    project_root = root or Path(__file__).resolve().parents[2]
    app = FastAPI(
        title="LungAI API (investigacion)",
        description="Hallazgos de un detector de nodulos y de un clasificador experimental; no ofrece diagnosticos.",
        version="0.2.0",
    )
    app.state.service = service
    app.state.analysis = analysis
    app.state.results = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:3000", "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def safety_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(DatasetUnavailable)
    async def dataset_error(request: Request, exc: DatasetUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(InvalidResult)
    async def report_error(request: Request, exc: InvalidResult):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    def get_service(request: Request) -> DatasetService:
        if request.app.state.service is None:
            # Metadatos en memoria; imagenes NIfTI nunca se precargan.
            request.app.state.service = DatasetService(project_root)
        return request.app.state.service

    def get_series(uid: str, svc: DatasetService) -> SeriesRecord:
        rec = svc.series(uid)
        if rec is None:
            raise HTTPException(status_code=404, detail="Serie no encontrada en la muestra")
        return rec

    @app.get("/health", tags=["Sistema"])
    def health():
        return {"estado": "activo", "alcance": "investigacion_local", "diagnostico": False}

    @app.get("/pacientes", tags=["Pacientes"])
    def list_patients(
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        split: Literal["train", "validation", "test"] | None = None,
        solo_con_resultados: bool = True,
        con_cancer: bool | None = Query(None, description="Etiqueta historica del NLST a 12 meses"),
        min_tamano_mm: float | None = Query(None, ge=0, description="Nodulo mayor de al menos estos mm"),
        orden: Literal["id", "nodulos", "tamano"] = "id",
        svc: DatasetService = Depends(get_service),
    ):
        patients = svc.filtered(split, solo_con_resultados, con_cancer, min_tamano_mm, orden)
        return {
            "total": len(patients), "limit": limit, "offset": offset,
            "pacientes": [svc.summarize(p, split=split) for p in patients[offset:offset + limit]],
            "advertencia": "Hallazgos automatizados de investigacion; no son un diagnostico.",
        }

    @app.get("/pacientes/{patient_id}", tags=["Pacientes"])
    def patient_detail(patient_id: str, svc: DatasetService = Depends(get_service)):
        if patient_id not in svc.by_patient:
            raise HTTPException(status_code=404, detail="Paciente no encontrado en la muestra")
        return svc.patient_detail(patient_id)

    @app.get("/series/{uid}/nodulos", tags=["Series"])
    def series_nodules(uid: str, svc: DatasetService = Depends(get_service)):
        rec = get_series(uid, svc)
        report = svc.report(rec)
        if report is None:
            raise HTTPException(status_code=404, detail="Esta serie aun no tiene resultado del detector")
        return report

    @app.get("/series/{uid}/imagen", tags=["Series"])
    def series_image(uid: str, svc: DatasetService = Depends(get_service)):
        rec = get_series(uid, svc)
        path = svc.image_path(rec)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Imagen NIfTI no disponible")
        # FileResponse transmite el archivo sin leerlo por completo en Python.
        return FileResponse(path, media_type="application/gzip")

    @app.get("/series/{uid}/vista", tags=["Series"])
    def series_preview(uid: str, svc: DatasetService = Depends(get_service)):
        rec = get_series(uid, svc)
        if svc.report(rec) is None:
            raise HTTPException(status_code=404, detail="Esta serie aun no tiene resultado del detector")
        path = svc.image_preview_path(rec)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Vista PNG no disponible para esta serie")
        return FileResponse(path, media_type="image/png")

    def get_analysis(request: Request) -> AnalysisService:
        if request.app.state.analysis is None:
            request.app.state.analysis = AnalysisService(project_root)
        return request.app.state.analysis

    # --- Series de la cohorte: la CT se baja de IDC y los nodulos vienen del JSON de Colab ---

    @app.post("/series/{uid}/preparar", status_code=202, tags=["Series"])
    def prepare_series(uid: str, rehacer: bool = False, svc: DatasetService = Depends(get_service),
                       analysis: AnalysisService = Depends(get_analysis)):
        """Baja la CT de IDC y marca y clasifica sus nodulos; no vuelve a correr el detector.

        Con rehacer=true vuelve a marcar y clasificar (por ejemplo, despues de copiar las etiquetas);
        la CT ya bajada se reutiliza.
        """
        rec = get_series(uid, svc)
        report = svc.report(rec)
        if report is None:
            raise HTTPException(status_code=404, detail="Esta serie aun no tiene resultado del detector")
        if rehacer:
            svc.detail_path(rec).unlink(missing_ok=True)
        extra = {"split": rec.split, "etiqueta_cancer_12m": rec.label_12m, "etiquetas": svc.labels(rec)}
        return analysis.prepare(rec.series_uid, report, svc.image_path(rec), svc.marks_path(rec),
                                svc.detail_path(rec), extra)

    @app.get("/series/{uid}/preparacion", tags=["Series"])
    def series_preparation(uid: str, svc: DatasetService = Depends(get_service),
                           analysis: AnalysisService = Depends(get_analysis)):
        rec = get_series(uid, svc)
        detail = svc.detail_path(rec)
        state = analysis.preparation(rec.series_uid, detail)
        result = json.loads(detail.read_text(encoding="utf-8")) if state["estado"] == "listo" else None
        return {**state, "resultado": result}

    @app.get("/series/{uid}/marcas", tags=["Series"])
    def series_marks(uid: str, svc: DatasetService = Depends(get_service)):
        path = svc.marks_path(get_series(uid, svc))
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Todavia no se marcan los nodulos de esta serie")
        return FileResponse(path, media_type="application/gzip")

    # --- Resultados del proyecto (no corre modelos) ---

    def get_results(request: Request) -> ResultsService:
        if request.app.state.results is None:
            request.app.state.results = ResultsService(project_root)
        return request.app.state.results

    @app.get("/resultados", tags=["Resultados"])
    def project_results(actualizar: bool = False, results: ResultsService = Depends(get_results)):
        """Procesamiento, detector, etiquetas y clasificador; se calcula una vez."""
        try:
            return results.summary(refresh=actualizar)
        except FileNotFoundError as exc:
            raise DatasetUnavailable("Falta la cohorte o los resultados de Colab en outputs/.") from exc

    @app.get("/resultados/casos", tags=["Resultados"])
    def classifier_cases(results: ResultsService = Depends(get_results)):
        """Aciertos y errores del clasificador en test (necesita outputs/clasificador/predicciones.csv)."""
        cases = results.cases()
        if cases is None:
            raise HTTPException(status_code=404, detail="Copia clasificador/predicciones.csv de Drive a outputs/clasificador/.")
        return cases

    # --- Estudios subidos: se analizan en esta computadora ---

    @app.exception_handler(StudyNotFound)
    async def study_missing(request: Request, exc: StudyNotFound):
        return JSONResponse(status_code=404, content={"detail": "Estudio no encontrado"})

    @app.exception_handler(InvalidUpload)
    async def bad_upload(request: Request, exc: InvalidUpload):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/estudios", status_code=202, tags=["Estudios"])
    def upload_study(archivo: UploadFile, svc: AnalysisService = Depends(get_analysis)):
        """Recibe un .zip con los DICOM de una serie, o un .nii / .nii.gz, y lo pone en cola."""
        return svc.submit(archivo.filename, archivo.file)

    @app.get("/estudios", tags=["Estudios"])
    def list_studies(svc: AnalysisService = Depends(get_analysis)):
        return {"estudios": svc.studies()}

    @app.get("/estudios/{study_id}", tags=["Estudios"])
    def study_detail(study_id: str, svc: AnalysisService = Depends(get_analysis)):
        return {**svc.state(study_id), "resultado": svc.result(study_id)}

    @app.get("/estudios/{study_id}/{archivo}", tags=["Estudios"])
    def study_file(study_id: str, archivo: Literal["imagen", "marcas"], svc: AnalysisService = Depends(get_analysis)):
        path = svc.file(study_id, "estudio.nii.gz" if archivo == "imagen" else "marcas.nii.gz")
        if path is None:
            raise HTTPException(status_code=404, detail="Todavia no esta lista")
        return FileResponse(path, media_type="application/gzip")

    return app


app = create_app()
