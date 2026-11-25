"""Analisis de estudios subidos: convierte la CT, detecta nodulos y estima malignidad.

Todo corre en esta computadora, un estudio a la vez. Cada estudio vive en
data/estudios/<id>/ (data/ esta en .gitignore):
- estado.json: en que paso va.
- estudio.nii.gz: la CT al espaciado del detector, para el visor.
- marcas.nii.gz: una mascara con los nodulos, para dibujarlos encima.
- resultado.json: nodulos con posicion, tamano, lobulo y probabilidad del clasificador.

El clasificador se lee de data/modelos/clasificador/modelo.pt (y metricas.json). Si falta,
el analisis sigue y solo muestra la deteccion. La probabilidad NO es un diagnostico.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO, Callable

STUDY_ID = re.compile(r"[0-9a-f]{32}\Z")
SUFFIXES = (".zip", ".nii.gz", ".nii")
MAX_UPLOAD = 2 * 1024**3
MAX_UNZIPPED = 4 * 1024**3
MAX_FILES = 5000
THRESHOLD = 0.5  # el mismo puntaje minimo que se uso al procesar la muestra
WARNING = ("Resultado de investigación, no un diagnóstico. En la prueba final el clasificador no "
           "superó a medir solo el tamaño del nódulo.")

Step = Callable[[str], None]
Runner = Callable[[Path, Path, Step], dict]


class StudyNotFound(Exception):
    """El id no tiene formato valido o no existe."""


class InvalidUpload(Exception):
    """El archivo subido no se puede analizar."""


def write_marks(image, nodules: list[dict], output: Path) -> None:
    """Guarda una mascara con un anillo alrededor de cada nodulo, con la geometria de la CT.

    Es un cascaron, no una esfera llena: al acercarse se sigue viendo el nodulo por dentro.
    """
    import numpy as np
    import SimpleITK as sitk

    shape = image.GetSize()[::-1]  # z, y, x, como el arreglo
    spacing = image.GetSpacing()[::-1]
    marks = np.zeros(shape, dtype=np.uint8)
    for nodule in nodules:
        center = nodule["centro_voxel"][::-1]
        # Un poco mas grande que la caja del detector, para no tapar el borde del nodulo.
        outer = [max(size / 2 * 1.4, 3.0) / sp for size, sp in zip(nodule["tamano_mm"][::-1], spacing)]
        inner = [max(r - 1.5 / sp, 0.5) for r, sp in zip(outer, spacing)]  # anillo de 1.5 mm
        box = tuple(slice(max(c - int(r) - 1, 0), min(c + int(r) + 2, n)) for c, r, n in zip(center, outer, shape))
        grid = np.ogrid[box]
        within = sum(((axis - c) / r) ** 2 for axis, c, r in zip(grid, center, outer)) <= 1
        hollow = sum(((axis - c) / r) ** 2 for axis, c, r in zip(grid, center, inner)) < 1
        marks[box][within & ~hollow] = 1
    mask = sitk.GetImageFromArray(marks)
    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(output), useCompression=True)


def describe(nodules: list[dict], probabilities: list[float | None]) -> None:
    """Agrega numero, diametro, centro en RAS (el que usa NiiVue) y probabilidad a cada nodulo."""
    for number, (nodule, probability) in enumerate(zip(nodules, probabilities), start=1):
        # SimpleITK usa LPS y NiiVue RAS: x e y cambian de signo.
        lx, ly, lz = nodule["centro_mm"]
        nodule.update(numero=number, diametro_mm=max(nodule["tamano_mm"]), centro_ras_mm=[-lx, -ly, lz],
                      probabilidad_maligno=None if probability is None else round(probability, 3))


def write_json(path: Path, value: dict) -> None:
    """Escribe aparte y renombra: quien lo lea a la vez nunca ve un archivo a medias."""
    partial = path.with_suffix(".parcial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    partial.replace(path)


class AnalysisService:
    def __init__(self, root: Path, runner: Runner | None = None):
        self.root = Path(root).resolve()
        self.store = self.root / "data" / "estudios"
        self.model_dir = self.root / "data" / "modelos"
        self.runner = runner or self.run_pipeline
        # Los modelos ocupan varios GB: los estudios se analizan uno por uno, en orden.
        self.pool = ThreadPoolExecutor(1)
        self.models: dict[str, Any] | None = None
        self.classifier: dict[str, Any] | None = None
        self.jobs: dict[str, dict] = {}  # series de la cohorte que se estan preparando
        # Lo que quedo a medias cuando se apago el servidor ya no va a terminar.
        for state in self.studies():
            if state["estado"] in ("en_cola", "procesando"):
                self._update(state["id"], estado="error", paso="",
                             error="El servidor se apagó antes de terminar. Vuelve a subir el estudio.")

    # --- Archivos de cada estudio ---

    def folder(self, study_id: str) -> Path:
        if not STUDY_ID.fullmatch(study_id) or not (self.store / study_id / "estado.json").is_file():
            raise StudyNotFound(study_id)
        return self.store / study_id

    def state(self, study_id: str) -> dict:
        return json.loads((self.folder(study_id) / "estado.json").read_text(encoding="utf-8"))

    def result(self, study_id: str) -> dict | None:
        path = self.folder(study_id) / "resultado.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def file(self, study_id: str, name: str) -> Path | None:
        path = self.folder(study_id) / name
        return path if path.is_file() else None

    def studies(self) -> list[dict]:
        states = [json.loads(p.read_text(encoding="utf-8")) for p in self.store.glob("*/estado.json")
                  if STUDY_ID.fullmatch(p.parent.name)]
        return sorted(states, key=lambda s: s["creado"], reverse=True)

    def _update(self, study_id: str, **changes) -> None:
        path = self.store / study_id / "estado.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        write_json(path, {**state, **changes, "actualizado": time.time()})

    # --- Subir y procesar ---

    def submit(self, filename: str | None, stream: BinaryIO) -> dict:
        name = Path(filename or "").name[:120]
        suffix = next((s for s in SUFFIXES if name.lower().endswith(s)), None)
        if suffix is None:
            raise InvalidUpload("Sube un .zip con los DICOM de una serie, o un .nii / .nii.gz.")

        study_id = uuid.uuid4().hex
        folder = self.store / study_id
        folder.mkdir(parents=True)
        source = folder / f"original{suffix}"
        size = 0
        with source.open("wb") as target:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    target.close()
                    shutil.rmtree(folder)
                    raise InvalidUpload("El archivo pasa de 2 GB.")
                target.write(chunk)

        now = time.time()
        write_json(folder / "estado.json", {"id": study_id, "archivo": name, "estado": "en_cola", "paso": "En cola",
                                            "error": None, "creado": now, "actualizado": now})
        self.pool.submit(self._process, study_id)
        return self.state(study_id)

    def _process(self, study_id: str) -> None:
        folder = self.store / study_id
        source = next(folder.glob("original*"))
        start = time.time()
        try:
            result = self.runner(folder, source, lambda text: self._update(study_id, estado="procesando", paso=text))
            write_json(folder / "resultado.json", {
                "id": study_id, **result, "segundos": round(time.time() - start), "advertencia": WARNING,
            })
            self._update(study_id, estado="listo", paso="Listo")
        except Exception as error:  # El servidor sigue para los demas estudios.
            self._update(study_id, estado="error", paso="", error=str(error) or type(error).__name__)
        finally:
            source.unlink(missing_ok=True)  # Solo se guarda la CT convertida.

    # --- Pasos del analisis real ---

    @staticmethod
    def _import_pipeline() -> None:
        """Los scripts del pipeline se importan como en Colab: desde sus carpetas."""
        backend = Path(__file__).resolve().parents[1]
        for name in ("data_pipeline", "inference", "training"):
            path = str(backend / name)
            if path not in sys.path:
                sys.path.insert(0, path)

    @staticmethod
    def _unzip(source: Path, target: Path) -> None:
        """Extrae solo archivos, con nombres propios: ninguna ruta del ZIP llega al disco."""
        try:
            with zipfile.ZipFile(source) as archive:
                members = [m for m in archive.infolist() if not m.is_dir()]
                if len(members) > MAX_FILES or sum(m.file_size for m in members) > MAX_UNZIPPED:
                    raise InvalidUpload("El ZIP es demasiado grande: sube una sola serie.")
                for number, member in enumerate(members):
                    with archive.open(member) as src, (target / f"{number:05d}.dcm").open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile as error:
            raise InvalidUpload("El archivo no es un ZIP válido.") from error

    def convert(self, source: Path, destination: Path) -> dict:
        """Deja la CT en HU, con el relleno como aire y al espaciado del detector."""
        self._import_pipeline()
        import numpy as np
        import SimpleITK as sitk
        from convert_to_nifti import AIR_HU, SUSPICIOUS_HU, convert_series, resample_to_detector

        if source.suffix == ".zip":
            with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
                self._unzip(source, Path(temporary))
                reader = sitk.ImageSeriesReader()
                series = {uid: len(reader.GetGDCMSeriesFileNames(temporary, uid))
                          for uid in reader.GetGDCMSeriesIDs(temporary)}
                if not series:
                    raise InvalidUpload("El ZIP no tiene una serie DICOM que se pueda leer.")
                # Un estudio puede traer varias reconstrucciones: se usa la de mas cortes.
                uid = max(series, key=series.get)
                info = convert_series({"source": temporary, "uid": uid, "slices": series[uid],
                                       "destination": str(destination)})
            if info["error"]:
                raise InvalidUpload(f"No se pudo armar la CT: {info['error']}")
            return {"tipo": "DICOM", "series_en_el_zip": len(series), "cortes": info["cortes_originales"],
                    "espaciado_z_original_mm": info["espaciado_z_original_mm"]}

        image = sitk.ReadImage(str(source))
        if image.GetDimension() != 3:
            raise InvalidUpload("El NIfTI debe ser un volumen 3D.")
        hu = sitk.GetArrayFromImage(image)
        hu[hu < SUSPICIOUS_HU] = AIR_HU
        clean = sitk.GetImageFromArray(hu.astype(np.int16))
        clean.CopyInformation(image)
        sitk.WriteImage(resample_to_detector(clean), str(destination), useCompression=True)
        return {"tipo": "NIfTI", "cortes": image.GetSize()[2],
                "espaciado_z_original_mm": round(image.GetSpacing()[2], 3)}

    def load_classifier(self) -> dict[str, Any]:
        """El clasificador y sus metricas de test; model es None si no se copio modelo.pt."""
        if self.classifier is None:
            self._import_pipeline()
            import torch
            from entrenar_clasificador import build_model

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            info: dict[str, Any] = {"model": None, "device": device, "metrics": None}
            weights = self.model_dir / "clasificador" / "modelo.pt"
            if weights.is_file():
                model = build_model()
                model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
                info["model"] = model.to(device).eval()
                metrics = weights.with_name("metricas.json")
                if metrics.is_file():
                    info["metrics"] = json.loads(metrics.read_text(encoding="utf-8")).get("test")
            self.classifier = info
        return self.classifier

    def load_detector(self) -> dict[str, Any]:
        """Detector y lungmask, una sola vez; el primer estudio tarda mas."""
        if self.models is None:
            self._import_pipeline()
            from detectar_nodulos import load_detector
            from lungmask import LMInferer

            detector, device = load_detector(self.model_dir)
            self.models = {
                "detector": detector, "device": device,
                "lung": LMInferer(modelname="R231", tqdm_disable=True),
                "lobes": LMInferer(modelname="LTRCLobes", tqdm_disable=True),
            }
        return self.models

    def classify(self, volume, nodules: list[dict]) -> list[float | None]:
        """Probabilidad del clasificador para cada nodulo, con el mismo recorte que en Colab."""
        info = self.load_classifier()
        if info["model"] is None or not nodules:
            return [None] * len(nodules)
        import torch
        from entrenar_clasificador import to_input
        from procesar_lotes import crop_candidate

        batch = torch.stack([to_input(crop_candidate(volume, n["centro_voxel"])) for n in nodules])
        with torch.no_grad():
            logits = info["model"](batch.to(info["device"])).squeeze(1).float().cpu()
        return torch.sigmoid(logits).tolist()

    def model_summary(self) -> dict:
        info = self.load_classifier()
        return {"umbral": THRESHOLD, "clasificador": info["model"] is not None, "metricas_test": info["metrics"]}

    def run_pipeline(self, folder: Path, source: Path, step: Step) -> dict:
        import SimpleITK as sitk

        step("Preparando la tomografía")
        image_path = folder / "estudio.nii.gz"
        info = self.convert(source, image_path)

        step("Cargando los modelos")
        models = self.load_detector()
        from detectar_nodulos import BUNDLE, detect_series

        step("Buscando nódulos y lóbulos (sin tarjeta gráfica tarda varios minutos)")
        candidates = detect_series(image_path, models["detector"], models["device"], models["lung"], models["lobes"])
        nodules = [c for c in candidates if c["puntaje"] >= THRESHOLD and c["en_pulmon"]]

        step("Midiendo y clasificando los nódulos")
        image = sitk.ReadImage(str(image_path))
        describe(nodules, self.classify(sitk.GetArrayFromImage(image), nodules))
        write_marks(image, nodules, folder / "marcas.nii.gz")
        return {
            "fuente": info,
            "modelo": {"detector": BUNDLE, **self.model_summary()},
            "nodulos": nodules,
            "candidatos_descartados": len(candidates) - len(nodules),
        }

    # --- Series de la cohorte, ya analizadas en Colab ---

    def preparation(self, uid: str, detail: Path) -> dict:
        """Estado de una serie de la cohorte: pendiente, en_cola, procesando, listo o error."""
        if detail.is_file():
            return {"estado": "listo", "paso": "Listo", "error": None}
        return self.jobs.get(uid, {"estado": "pendiente", "paso": "", "error": None})

    def prepare(self, uid: str, report: dict, image: Path, marks: Path, detail: Path, extra: dict) -> dict:
        """Baja la CT de IDC (si no esta) y marca y clasifica los nodulos del JSON de Colab.

        No vuelve a correr el detector: los nodulos son los mismos que se procesaron en Colab.
        """
        state = self.preparation(uid, detail)
        if state["estado"] in ("listo", "en_cola", "procesando"):
            return state
        self.jobs[uid] = {"estado": "en_cola", "paso": "En cola", "error": None}
        self.pool.submit(self._prepare, uid, report, image, marks, detail, extra)
        return self.jobs[uid]

    def _prepare(self, uid: str, report: dict, image: Path, marks: Path, detail: Path, extra: dict) -> None:
        def step(text: str) -> None:
            self.jobs[uid] = {"estado": "procesando", "paso": text, "error": None}

        try:
            import SimpleITK as sitk

            if not image.is_file():
                step("Descargando la tomografía de IDC")
                self.download(uid, image)
            step("Marcando y clasificando los nódulos")
            ct = sitk.ReadImage(str(image))
            nodules = [dict(n) for n in report["nodulos"]]
            describe(nodules, self.classify(sitk.GetArrayFromImage(ct), nodules))
            labels = extra.get("etiquetas", {})
            for nodule in nodules:
                label = labels.get(nodule.get("recorte"), {})
                nodule.update(etiqueta_nlst=label.get("etiqueta"), reportado_radiologo=label.get("reportado"))
            marks.parent.mkdir(parents=True, exist_ok=True)
            write_marks(ct, nodules, marks)
            detail.parent.mkdir(parents=True, exist_ok=True)
            write_json(detail, {
                "patient_id": report["patient_id"], "series_uid": uid,
                **{k: v for k, v in extra.items() if k != "etiquetas"},
                "modelo": self.model_summary(), "nodulos": nodules,
                "candidatos_descartados": len(report.get("otros_candidatos", [])),
                "etiquetas_disponibles": bool(labels), "advertencia": WARNING,
            })
            self.jobs.pop(uid, None)
        except Exception as error:  # El servidor sigue para las demas series.
            self.jobs[uid] = {"estado": "error", "paso": "", "error": str(error) or type(error).__name__}

    def download(self, uid: str, destination: Path) -> None:
        """Baja una serie de IDC y la convierte como en Colab; los DICOM se borran al terminar."""
        self._import_pipeline()
        import SimpleITK as sitk
        from convert_to_nifti import convert_series
        from idc_index import IDCClient

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            IDCClient().download_from_selection(downloadDir=temporary, seriesInstanceUID=[uid],
                                                dirTemplate="%SeriesInstanceUID", show_progress_bar=False, quiet=True)
            source = str(Path(temporary) / uid)
            slices = len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(source, uid))
            info = convert_series({"source": source, "uid": uid, "slices": slices, "destination": str(destination)})
        if info["error"]:
            raise ValueError(f"No se pudo armar la CT: {info['error']}")
