"""FastAPI de consulta local para LungAI; no ejecuta inferencia ni realiza diagnosticos."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.api.data_service import DatasetService, DatasetUnavailable, InvalidResult, SeriesRecord


def create_app(root: Path | None = None, service: DatasetService | None = None) -> FastAPI:
    project_root = root or Path(__file__).resolve().parents[2]
    app = FastAPI(
        title="LungAI API (investigacion)",
        description="Consulta local de hallazgos de un detector de nodulos; no ofrece diagnosticos.",
        version="0.1.0",
    )
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:3000", "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET"],
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
        svc: DatasetService = Depends(get_service),
    ):
        patients = svc.patients(split=split, only_results=solo_con_resultados)
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

    return app


app = create_app()
