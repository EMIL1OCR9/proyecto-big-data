"""Capa unica de lectura: Parquet de la cohorte + JSON/PNG/NIfTI locales.

No ejecuta el detector, no modifica imagenes y NO permite elegir rutas arbitrarias.
Una futura base de datos puede reemplazar este servicio sin cambiar las rutas HTTP.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DICOM_UID = re.compile(r"[0-9]+(?:\.[0-9]+)+\Z")
PATIENT_ID = re.compile(r"[0-9]+\Z")


class DatasetUnavailable(Exception):
    """No se puede cargar la cohorte del proyecto."""


class InvalidResult(Exception):
    """El JSON de salida no coincide con la cohorte o esta dañado."""


@dataclass(frozen=True)
class SeriesRecord:
    patient_id: str
    series_uid: str
    image_relpath: str
    split: str
    label_12m: int | None


class DatasetService:
    """Indice de metadatos en memoria; imagenes se sirven por streaming.

    Solo contiene registros con en_muestra=True. Los JSON se consultan al pedirlos,
    para detectar resultados generados despues de iniciar la API.
    """

    def __init__(self, root: Path, cohort: pd.DataFrame | None = None):
        self.root = Path(root).resolve()
        self.data_root = (self.root / "data" / "monai").resolve()
        self.report_root = (self.root / "outputs" / "nodulos").resolve()
        self.cohort_path = self.root / "outputs" / "monai_dataset" / "cohorte_cancer_12m.parquet"

        if cohort is None:
            if not self.cohort_path.is_file():
                raise DatasetUnavailable("Falta la cohorte Parquet. Ejecuta prepare_monai_dataset.py.")
            try:
                cohort = pd.read_parquet(self.cohort_path, columns=[
                    "PatientID", "SeriesInstanceUID", "ruta_imagen", "split",
                    "en_muestra", "etiqueta_cancer_12m",
                ])
            except Exception as exc:
                raise DatasetUnavailable("No se pudo leer la cohorte Parquet.") from exc

        required = {"PatientID", "SeriesInstanceUID", "ruta_imagen", "split", "en_muestra", "etiqueta_cancer_12m"}
        if not required.issubset(cohort.columns):
            raise DatasetUnavailable("La cohorte no tiene el esquema esperado.")

        sample = cohort.loc[cohort["en_muestra"].eq(True)]
        self.by_series: dict[str, SeriesRecord] = {}
        self.by_patient: dict[str, list[SeriesRecord]] = {}
        for row in sample.to_dict(orient="records"):
            try:
                patient = str(row["PatientID"]).strip()
                uid = str(row["SeriesInstanceUID"]).strip()
                rel = str(row["ruta_imagen"]).strip().replace("\\", "/")
                split = str(row["split"]).strip()
                label_raw = row["etiqueta_cancer_12m"]
                label = None if pd.isna(label_raw) else int(label_raw)
                if not PATIENT_ID.fullmatch(patient) or not DICOM_UID.fullmatch(uid):
                    raise ValueError("IDs con formato inesperado")
                if split not in {"train", "validation", "test"} or label not in {0, 1, None}:
                    raise ValueError("División o etiqueta inesperada")
                # Valida incluso las rutas que todavia no existen.
                self._within(self.data_root, rel, ".nii.gz")
                record = SeriesRecord(patient, uid, rel, split, label)
            except (ValueError, TypeError) as exc:
                raise DatasetUnavailable("La cohorte contiene metadatos invalidos.") from exc
            if uid in self.by_series:
                raise DatasetUnavailable("Hay SeriesInstanceUID duplicados en la muestra.")
            self.by_series[uid] = record
            self.by_patient.setdefault(patient, []).append(record)

    @staticmethod
    def _within(base: Path, rel: str, suffix: str) -> Path:
        """Resuelve rutas solo dentro de un directorio permitido (anti-traversal)."""
        if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts or not rel.endswith(suffix):
            raise ValueError("Ruta de archivo no permitida")
        resolved = (base / rel).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError("Ruta fuera del almacen autorizado")
        return resolved

    def series(self, uid: str) -> SeriesRecord | None:
        return self.by_series.get(uid)

    def patients(self, split: str | None = None, only_results: bool = True) -> list[str]:
        patients = []
        for patient, series in self.by_patient.items():
            candidates = [r for r in series if split is None or r.split == split]
            if not candidates:
                continue
            if only_results and not any(self.report_path(r).is_file() for r in candidates):
                continue
            patients.append(patient)
        return sorted(patients)

    def report_path(self, rec: SeriesRecord) -> Path:
        # Construida a partir de IDs previamente validados, no de parametros HTTP.
        return self._within(self.report_root, f"{rec.patient_id}/{rec.series_uid}.json", ".json")

    def report(self, rec: SeriesRecord) -> dict[str, Any] | None:
        path = self.report_path(rec)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("nodulos"), list):
                raise ValueError("Esquema del JSON invalido")
            if value.get("patient_id") != rec.patient_id or value.get("series_uid") != rec.series_uid:
                raise ValueError("IDs del reporte distintos de la cohorte")
            # No se usa el campo imagen del JSON para servir archivos.
            if value.get("imagen", "").replace("\\", "/") != rec.image_relpath:
                raise ValueError("Imagen del reporte distinta de la cohorte")
            return value
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise InvalidResult("Reporte de deteccion corrupto o inconsistente.") from exc

    def image_path(self, rec: SeriesRecord) -> Path:
        return self._within(self.data_root, rec.image_relpath, ".nii.gz")

    def image_preview_path(self, rec: SeriesRecord) -> Path:
        return self.report_path(rec).with_suffix(".png")

    @staticmethod
    def _largest(report: dict[str, Any]) -> float | None:
        sizes = []
        for item in report.get("nodulos", []):
            if not isinstance(item, dict) or not isinstance(item.get("tamano_mm"), list):
                raise InvalidResult("Caja de nodulo invalida")
            try:
                vals = [float(v) for v in item["tamano_mm"]]
                if len(vals) != 3 or any(v < 0 or not 0 <= v < float("inf") for v in vals):
                    raise ValueError
                sizes.append(max(vals))
            except (ValueError, TypeError) as exc:
                raise InvalidResult("Tamano de nodulo invalido") from exc
        return max(sizes, default=None)

    def summarize(self, patient: str, split: str | None = None) -> dict[str, Any]:
        records = [r for r in self.by_patient[patient] if split is None or r.split == split]
        available = [self.report(r) for r in records]
        reports = [r for r in available if r is not None]
        count = sum(len(r["nodulos"]) for r in reports) if reports else None
        sizes = [s for r in reports if (s := self._largest(r)) is not None]
        return {
            "patient_id": patient,
            "series_total": len(records),
            "series_analizadas": len(reports),
            "nodulos_detectados": count,
            "nodulo_mayor_mm": round(max(sizes), 2) if sizes else None,
            "estado": "pendiente" if not reports else "disponible" if len(reports) == len(records) else "parcial",
        }

    def patient_detail(self, patient: str) -> dict[str, Any]:
        items = []
        for rec in self.by_patient[patient]:
            report = self.report(rec)
            items.append({
                "series_uid": rec.series_uid,
                "split": rec.split,
                "etiqueta_cancer_12m": rec.label_12m,  # Etiqueta clinica historica, NO prediccion.
                "imagen_disponible": self.image_path(rec).is_file(),
                "analisis_disponible": report is not None,
                "nodulos_detectados": len(report["nodulos"]) if report is not None else None,
                "nodulo_mayor_mm": self._largest(report) if report is not None else None,
            })
        return {**self.summarize(patient), "series": items,
                "advertencia": "Resultados experimentales; no son un diagnostico clinico."}
