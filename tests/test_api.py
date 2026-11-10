"""Pruebas con datos ficticios; no usan pacientes ni imagenes reales."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.data_service import DatasetService, DatasetUnavailable
from backend.api.main import create_app


UID_1 = "1.2.840.111"
UID_2 = "1.2.840.222"
UID_3 = "1.2.840.333"


@pytest.fixture
def client(tmp_path: Path):
    frame = pd.DataFrame([
        {"PatientID": "101", "SeriesInstanceUID": UID_1, "ruta_imagen": f"images/101/{UID_1}.nii.gz", "split": "train", "en_muestra": True, "etiqueta_cancer_12m": 0.0},
        {"PatientID": "102", "SeriesInstanceUID": UID_2, "ruta_imagen": f"images/102/{UID_2}.nii.gz", "split": "validation", "en_muestra": True, "etiqueta_cancer_12m": 1.0},
        {"PatientID": "103", "SeriesInstanceUID": UID_3, "ruta_imagen": f"images/103/{UID_3}.nii.gz", "split": "test", "en_muestra": True, "etiqueta_cancer_12m": 0.0},
    ])
    base = tmp_path / "outputs/nodulos/101"
    base.mkdir(parents=True)
    report = {
        "patient_id": "101", "series_uid": UID_1, "imagen": f"images/101/{UID_1}.nii.gz",
        "modelo": {"detector": "prueba_sintetica", "umbral": 0.5},
        "nodulos": [{"puntaje": 0.8, "centro_mm": [0, 1, 2], "centro_voxel": [3, 4, 5],
                     "tamano_mm": [4, 6, 5], "lobulo": "Superior izquierdo", "en_pulmon": True}],
        "otros_candidatos": [],
    }
    (base / f"{UID_1}.json").write_text(json.dumps(report), encoding="utf-8")
    (base / f"{UID_1}.png").write_bytes(b"\x89PNG\r\n\x1a\nFakePNG")
    path = tmp_path / "data/monai/images/101" / f"{UID_1}.nii.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"FAKE_NIFTI_FOR_TEST_ONLY")
    service = DatasetService(tmp_path, cohort=frame)
    return TestClient(create_app(root=tmp_path, service=service))


def test_health(client):
    assert client.get("/health").json()["estado"] == "activo"


def test_list_defaults_only_results(client):
    res = client.get("/pacientes")
    assert res.status_code == 200
    assert res.json()["total"] == 1
    row = res.json()["pacientes"][0]
    assert row["nodulos_detectados"] == 1
    assert row["nodulo_mayor_mm"] == 6


def test_pending_not_misreported_as_zero(client):
    data = client.get("/pacientes?solo_con_resultados=false").json()
    assert data["total"] == 3
    pending = next(x for x in data["pacientes"] if x["patient_id"] == "102")
    assert pending["nodulos_detectados"] is None
    assert pending["estado"] == "pendiente"


def test_patient_detail(client):
    data = client.get("/pacientes/102").json()
    assert data["series"][0]["etiqueta_cancer_12m"] == 1
    assert data["series"][0]["analisis_disponible"] is False


def test_report_and_files(client):
    assert client.get(f"/series/{UID_1}/nodulos").json()["nodulos"][0]["puntaje"] == 0.8
    img = client.get(f"/series/{UID_1}/imagen")
    assert img.status_code == 200 and img.content == b"FAKE_NIFTI_FOR_TEST_ONLY"
    png = client.get(f"/series/{UID_1}/vista")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"


def test_missing_reports_are_404(client):
    assert client.get(f"/series/{UID_2}/nodulos").status_code == 404
    assert client.get(f"/series/{UID_2}/vista").status_code == 404
    assert client.get("/series/1.2.0/nodulos").status_code == 404
    assert client.get("/pacientes/999").status_code == 404


def test_filter_pagination(client):
    assert client.get("/pacientes?split=train&solo_con_resultados=false").json()["total"] == 1
    assert client.get("/pacientes?limit=1&offset=1&solo_con_resultados=false").json()["pacientes"][0]["patient_id"] == "102"
    assert client.get("/pacientes?limit=999").status_code == 422


def test_rejects_traversal_in_metadata(tmp_path):
    frame = pd.DataFrame([{"PatientID": "101", "SeriesInstanceUID": UID_1,
                           "ruta_imagen": "../outside.nii.gz", "split": "train",
                           "en_muestra": True, "etiqueta_cancer_12m": 0}])
    with pytest.raises(DatasetUnavailable):
        DatasetService(tmp_path, cohort=frame)


def test_corrupt_report_is_not_exposed(client, tmp_path):
    # fixture no retorna el directorio; recuperamos el root de la instancia del servicio.
    service = client.app.state.service
    p = service.report_path(service.by_series[UID_1])
    content = json.loads(p.read_text())
    content["imagen"] = "../../secret.nii.gz"
    p.write_text(json.dumps(content))
    assert client.get(f"/series/{UID_1}/nodulos").status_code == 500


def test_existing_empty_report_is_zero_not_pending(client):
    service = client.app.state.service
    rec = service.by_series[UID_2]
    p = service.report_path(rec)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "patient_id": rec.patient_id,
        "series_uid": rec.series_uid,
        "imagen": rec.image_relpath,
        "modelo": {"detector": "prueba_sintetica", "umbral": 0.5},
        "nodulos": [], "otros_candidatos": [],
    }), encoding="utf-8")
    response = client.get("/pacientes?solo_con_resultados=true")
    assert response.json()["total"] == 2
    row = next(v for v in response.json()["pacientes"] if v["patient_id"] == "102")
    assert row["nodulos_detectados"] == 0
    assert row["estado"] == "disponible"
    assert client.get(f"/series/{UID_2}/vista").status_code == 404


def test_missing_cohort_returns_503(tmp_path):
    c = TestClient(create_app(root=tmp_path))
    assert c.get("/health").status_code == 200
    res = c.get("/pacientes")
    assert res.status_code == 503
    assert "Parquet" in res.json()["detail"]
