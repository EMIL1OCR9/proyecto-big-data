"""Pruebas con datos ficticios; no usan pacientes ni imagenes reales."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.analisis import AnalysisService
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


# --- Estudios subidos (sin modelos: el analisis se reemplaza por uno ficticio) ---

def fake_runner(folder, source, step):
    step("Paso de prueba")
    (folder / "estudio.nii.gz").write_bytes(b"FAKE_CT")
    (folder / "marcas.nii.gz").write_bytes(b"FAKE_MARKS")
    return {"nodulos": [{"numero": 1, "diametro_mm": 8.0, "probabilidad_maligno": 0.3}]}


def failing_runner(folder, source, step):
    raise ValueError("espaciado Z irregular")


def studies_client(tmp_path, runner):
    service = AnalysisService(tmp_path, runner=runner)
    return TestClient(create_app(root=tmp_path, analysis=service)), service


def upload(client, name="estudio.zip", content=b"PK"):
    return client.post("/estudios", files={"archivo": (name, io.BytesIO(content))})


def test_upload_is_analyzed_and_served(tmp_path):
    client, service = studies_client(tmp_path, fake_runner)
    res = upload(client)
    assert res.status_code == 202
    study = res.json()["id"]
    service.pool.submit(lambda: None).result()  # espera a que termine la cola

    data = client.get(f"/estudios/{study}").json()
    assert data["estado"] == "listo" and data["archivo"] == "estudio.zip"
    assert data["resultado"]["nodulos"][0]["diametro_mm"] == 8.0
    assert "no un diagnóstico" in data["resultado"]["advertencia"]
    assert client.get(f"/estudios/{study}/imagen").content == b"FAKE_CT"
    assert client.get(f"/estudios/{study}/marcas").content == b"FAKE_MARKS"
    assert not list((tmp_path / "data/estudios" / study).glob("original*"))  # solo queda la CT convertida
    assert [s["id"] for s in client.get("/estudios").json()["estudios"]] == [study]


def test_failed_analysis_reports_error(tmp_path):
    client, service = studies_client(tmp_path, failing_runner)
    study = upload(client, "serie.nii.gz").json()["id"]
    service.pool.submit(lambda: None).result()
    data = client.get(f"/estudios/{study}").json()
    assert data["estado"] == "error" and "irregular" in data["error"]
    assert data["resultado"] is None
    assert client.get(f"/estudios/{study}/imagen").status_code == 404


def test_upload_rejects_other_files(tmp_path):
    client, _ = studies_client(tmp_path, fake_runner)
    res = upload(client, "notas.txt")
    assert res.status_code == 400 and ".zip" in res.json()["detail"]
    assert not (tmp_path / "data/estudios").exists() or not any((tmp_path / "data/estudios").iterdir())


def test_unknown_studies_are_404(tmp_path):
    client, _ = studies_client(tmp_path, fake_runner)
    assert client.get("/estudios/..%2F..%2Fsecret").status_code == 404
    assert client.get(f"/estudios/{'a' * 32}").status_code == 404
    assert client.get(f"/estudios/{'a' * 32}/imagen").status_code == 404
    assert client.get(f"/estudios/{'a' * 32}/otro").status_code == 422


def test_zip_paths_never_reach_disk(tmp_path):
    archive = tmp_path / "subido.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../fuera.dcm", b"a")
        zf.writestr("carpeta/sub/corte.dcm", b"b")
    target = tmp_path / "extraido"
    target.mkdir()
    AnalysisService._unzip(archive, target)
    assert sorted(p.name for p in target.iterdir()) == ["00000.dcm", "00001.dcm"]
    assert not (tmp_path / "fuera.dcm").exists()


def test_interrupted_studies_become_errors(tmp_path):
    folder = tmp_path / "data/estudios" / ("b" * 32)
    folder.mkdir(parents=True)
    (folder / "estado.json").write_text(json.dumps({"id": "b" * 32, "archivo": "x.zip", "estado": "procesando",
                                                    "paso": "Buscando", "error": None, "creado": 1, "actualizado": 1}))
    service = AnalysisService(tmp_path, runner=fake_runner)
    assert service.state("b" * 32)["estado"] == "error"


def test_marks_are_rings_around_each_nodule(tmp_path):
    import SimpleITK as sitk

    from backend.api.analisis import write_marks

    image = sitk.Image(64, 64, 40, sitk.sitkInt16)
    image.SetSpacing((1.0, 1.0, 1.25))
    nodule = {"centro_voxel": [32, 30, 20], "tamano_mm": [8.0, 8.0, 8.0]}
    write_marks(image, [nodule], tmp_path / "marcas.nii.gz")
    marks = sitk.ReadImage(str(tmp_path / "marcas.nii.gz"))
    array = sitk.GetArrayFromImage(marks)
    assert marks.GetSize() == image.GetSize()
    assert array[20, 30, 32] == 0  # el nodulo queda visible por dentro
    assert array[20, 30, 32 + 5] == 1  # el anillo lo rodea (radio 8 / 2 * 1.4 = 5.6 mm)
    assert array[20, 30, 32 + 8] == 0  # y no se extiende de mas


class FakeCohortAnalysis(AnalysisService):
    """Sin IDC ni modelos: la descarga escribe una CT vacia y el clasificador responde 0.7."""

    def download(self, uid, destination):
        import SimpleITK as sitk

        destination.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.Image(16, 16, 12, sitk.sitkInt16), str(destination))

    def classify(self, volume, nodules):
        return [0.7] * len(nodules)

    def model_summary(self):
        return {"umbral": 0.5, "clasificador": True, "metricas_test": None}


def test_cohort_series_is_downloaded_marked_and_labeled(client):
    service = client.app.state.service
    rec = service.by_series[UID_2]
    report_path = service.report_path(rec)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    nodule = {"puntaje": 0.9, "centro_mm": [1, 2, 3], "centro_voxel": [8, 8, 6], "tamano_mm": [4, 4, 4],
              "lobulo": "Medio derecho", "en_pulmon": True, "recorte": 0}
    report_path.write_text(json.dumps({"patient_id": "102", "series_uid": UID_2, "imagen": rec.image_relpath,
                                       "nodulos": [nodule], "otros_candidatos": []}), encoding="utf-8")
    labels = service.root / "outputs/etiquetas/candidatos_etiquetados.parquet"
    labels.parent.mkdir(parents=True)
    pd.DataFrame([{"SeriesInstanceUID": UID_2, "recorte": 0, "etiqueta": "maligno", "reportado": True}]).to_parquet(labels)
    analysis = FakeCohortAnalysis(service.root)
    client.app.state.analysis = analysis

    assert client.get(f"/series/{UID_2}/preparacion").json()["estado"] == "pendiente"
    assert client.post(f"/series/{UID_2}/preparar").status_code == 202
    analysis.pool.submit(lambda: None).result()

    data = client.get(f"/series/{UID_2}/preparacion").json()
    assert data["estado"] == "listo", data
    found = data["resultado"]["nodulos"][0]
    assert found["probabilidad_maligno"] == 0.7 and found["etiqueta_nlst"] == "maligno" and found["reportado_radiologo"]
    assert found["centro_ras_mm"] == [-1, -2, 3] and data["resultado"]["split"] == "validation"
    assert client.get(f"/series/{UID_2}/marcas").status_code == 200
    assert client.get(f"/series/{UID_2}/imagen").status_code == 200
    assert client.post(f"/series/{UID_3}/preparar").status_code == 404  # sin JSON del detector


def test_patient_filters_and_order(client):
    data = client.get("/pacientes?solo_con_resultados=false&con_cancer=true").json()
    assert [p["patient_id"] for p in data["pacientes"]] == ["102"] and data["pacientes"][0]["cancer_12m"] == 1
    data = client.get("/pacientes?solo_con_resultados=false&min_tamano_mm=5").json()
    assert [p["patient_id"] for p in data["pacientes"]] == ["101"]
    data = client.get("/pacientes?solo_con_resultados=false&orden=tamano").json()
    assert data["pacientes"][0]["patient_id"] == "101" and data["total"] == 3
    assert client.get("/pacientes?min_tamano_mm=-1").status_code == 422


def test_project_results_and_cases(tmp_path):
    out = tmp_path / "outputs"
    (out / "monai_dataset").mkdir(parents=True)
    pd.DataFrame([
        {"SeriesInstanceUID": UID_1, "split": "train", "en_muestra": True, "series_size_MB": 1024.0},
        {"SeriesInstanceUID": UID_2, "split": "test", "en_muestra": True, "series_size_MB": 512.0},
    ]).to_parquet(out / "monai_dataset/cohorte_cancer_12m.parquet")
    (out / "nodulos/101").mkdir(parents=True)
    (out / "nodulos/101" / f"{UID_1}.json").write_text(json.dumps({
        "series_uid": UID_1, "nodulos": [{"tamano_mm": [5, 5, 5]}, {"tamano_mm": [16, 14, 15]}]}))
    (out / "lotes").mkdir()
    pd.DataFrame([{"PatientID": "102", "SeriesInstanceUID": UID_2, "error": "espaciado Z irregular | mediana=1"}]) \
        .to_csv(out / "lotes/lotes_fallidas_test.csv", index=False)
    client = TestClient(create_app(root=tmp_path))

    p = client.get("/resultados").json()["procesamiento"]
    assert (p["series_procesadas"], p["series_total"], p["gb_procesados"], p["nodulos"]) == (1, 2, 1.0, 2)
    assert {r["categoria"]: r["nodulos"] for r in p["lung_rads"]} == {"2": 1, "3": 0, "4A": 0, "4B": 1}
    assert p["causas_exclusion"] == [{"causa": "Espaciado irregular entre cortes", "series": 1}]
    assert client.get("/resultados/casos").status_code == 404

    (out / "clasificador").mkdir()
    pd.DataFrame([
        {"PatientID": "1", "SeriesInstanceUID": "a", "recorte": 0, "split": "test", "etiqueta": "maligno", "tamano_mm": 20, "puntaje": 0.9, "probabilidad_maligno": 0.95},
        {"PatientID": "2", "SeriesInstanceUID": "b", "recorte": 1, "split": "test", "etiqueta": "benigno", "tamano_mm": 15, "puntaje": 0.8, "probabilidad_maligno": 0.7},
        {"PatientID": "3", "SeriesInstanceUID": "c", "recorte": 0, "split": "test", "etiqueta": "benigno", "tamano_mm": 4, "puntaje": 0.3, "probabilidad_maligno": 0.9},
        {"PatientID": "4", "SeriesInstanceUID": "d", "recorte": 0, "split": "validation", "etiqueta": "maligno", "tamano_mm": 9, "puntaje": 0.9, "probabilidad_maligno": 0.1},
    ]).to_csv(out / "clasificador/predicciones.csv", index=False)
    cases = client.get("/resultados/casos").json()
    assert (cases["malignos"], cases["benignos"], cases["malignos_detectados"], cases["benignos_marcados"]) == (1, 1, 1, 1)
    assert cases["falsos_positivos"][0]["patient_id"] == "2"  # el de puntaje 0.3 no se ve en el visor
