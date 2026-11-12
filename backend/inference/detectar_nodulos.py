"""
LungAI — detecta nodulos en las CT ya convertidas y guarda un JSON por serie.

Usa los NIfTI de convert_to_nifti.py. Segmenta pulmones y lobulos con lungmask, corre el
detector preentrenado de MONAI (lung_nodule_ct_detection: RetinaNet 3D entrenada con
LUNA16) y guarda cada candidato con su posicion, tamano, lobulo y puntaje. Volver a
correrlo salta las series que ya tienen JSON y reintenta las que fallaron.

Uso:
    python backend/inference/detectar_nodulos.py --limit 5

Importante:
- Detecta nodulos; NO dice si son cancer ni sirve para diagnostico.
- Para correr a escala necesita GPU: en CPU tarda varios minutos por CT.
- La primera vez descarga los modelos a data/modelos y a la cache de torch.
- outputs/ esta en .gitignore. No subir resultados con identificadores a GitHub.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from lungmask import LMInferer
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToWorldCoordinated,
    ClipBoxToImaged,
    ConvertBoxModed,
)
from monai.bundle import ConfigParser, download
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
)

BUNDLE = "lung_nodule_ct_detection"
LOBES = {
    1: "Superior izquierdo",
    2: "Inferior izquierdo",
    3: "Superior derecho",
    4: "Medio derecho",
    5: "Inferior derecho",
}

# Los NIfTI ya vienen al espaciado del detector: solo falta orientar y escalar los HU.
PREPROCESS = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS", labels=None),
    ScaleIntensityRanged(keys="image", a_min=-1024, a_max=300, b_min=0, b_max=1, clip=True),
])

POSTPROCESS = Compose([
    ClipBoxToImaged(box_keys="box", label_keys=["label", "label_scores"],
                    box_ref_image_keys="image", remove_empty=True),
    AffineBoxToWorldCoordinated(box_keys="box", box_ref_image_keys="image"),
    # Centro (x, y, z) y tamano (ancho, alto, profundidad) en mm.
    ConvertBoxModed(box_keys="box", src_mode="xyzxyz", dst_mode="cccwhd"),
])


def load_detector(model_dir: Path):
    """Descarga el modelo si falta y lo arma con la configuracion con la que se entreno."""
    root = model_dir / BUNDLE
    if not (root / "models" / "model.pt").exists():
        download(name=BUNDLE, bundle_dir=str(model_dir))

    config = ConfigParser()
    config.read_config(str(root / "configs" / "inference.json"))
    config["bundle_root"] = str(root)

    device = config.get_parsed_content("device")
    detector = config.get_parsed_content("detector")
    config.get_parsed_content("detector_ops")  # puntaje minimo, NMS y ventana deslizante

    weights = torch.load(root / "models" / "model.pt", map_location=device, weights_only=True)
    detector.network.load_state_dict(weights)
    detector.eval()
    return detector, device


def detect_series(path: Path, detector, device, lung_model, lobe_model) -> list[dict]:
    """Regresa todos los candidatos de una CT, del puntaje mas alto al mas bajo."""
    image = sitk.ReadImage(str(path))
    lung = lung_model.apply(image) > 0  # R231: incluye zonas densas, como un nodulo grande
    lobes = lobe_model.apply(image)  # LTRCLobes: separa los 5 lobulos, deja fuera lo denso

    data = PREPROCESS({"image": str(path)})
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        prediction = detector([data["image"].to(device)], use_inferer=True)[0]
    result = POSTPROCESS({**prediction, "image": data["image"]})

    limit = np.array(image.GetSize()) - 1
    candidates = []
    for box, score in zip(result["box"].cpu().numpy(), result["label_scores"].cpu().numpy()):
        size = [float(v) for v in box[3:]]
        # MONAI usa coordenadas RAS y SimpleITK usa LPS: x e y cambian de signo.
        center = (-float(box[0]), -float(box[1]), float(box[2]))
        x, y, z = (int(v) for v in np.clip(image.TransformPhysicalPointToIndex(center), 0, limit))

        # El lobulo es el mas frecuente dentro de la caja, por si el centro cae en zona densa.
        hx, hy, hz = (max(1, round(s / sp / 2)) for s, sp in zip(size, image.GetSpacing()))
        region = lobes[max(z - hz, 0):z + hz + 1, max(y - hy, 0):y + hy + 1, max(x - hx, 0):x + hx + 1]
        labels = region[region > 0]

        candidates.append({
            "puntaje": round(float(score), 3),
            "centro_mm": [round(v, 1) for v in center],
            "centro_voxel": [x, y, z],
            "tamano_mm": [round(v, 1) for v in size],
            "lobulo": LOBES[int(np.bincount(labels).argmax())] if labels.size else "",
            "en_pulmon": bool(lung[z, y, x]),
        })
    return sorted(candidates, key=lambda c: -c["puntaje"])


def save_report(row, candidates: list[dict], threshold: float, output: Path) -> dict:
    """Separa los nodulos del resto de candidatos y guarda el JSON de la serie."""
    is_nodule = [c["puntaje"] >= threshold and c["en_pulmon"] for c in candidates]
    report = {
        "patient_id": row.PatientID,
        "series_uid": row.SeriesInstanceUID,
        "imagen": row.ruta_imagen,
        "modelo": {"detector": BUNDLE, "umbral": threshold},
        "nodulos": [c for c, ok in zip(candidates, is_nodule) if ok],
        # Se guardan para poder evaluar con otros umbrales sin volver a correr el modelo.
        "otros_candidatos": [c for c, ok in zip(candidates, is_nodule) if not ok],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Solo aparece con su nombre final si termino de escribirse.
    partial = output.with_suffix(".parcial")
    partial.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    partial.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/monai_dataset/cohorte_cancer_12m.parquet"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/monai"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nodulos"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/modelos"))
    parser.add_argument("--split", choices=["train", "validation", "test"], help="Por defecto, todos.")
    parser.add_argument("--limit", type=int, required=True, help="Maximo de series a procesar.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Puntaje minimo para contar como nodulo.")
    args = parser.parse_args()

    cohort = pd.read_parquet(args.input)
    sample = cohort[cohort["en_muestra"]]
    if args.split:
        sample = sample[sample["split"].eq(args.split)]

    converted = sample[sample["ruta_imagen"].map(lambda p: (args.data_dir / p).exists())]
    outputs = pd.Series(
        [args.output_dir / p / f"{u}.json" for p, u in zip(converted["PatientID"], converted["SeriesInstanceUID"])],
        index=converted.index,
        dtype=object,
    )
    pending = converted[~outputs.map(Path.exists).astype(bool)]
    selected = pending.head(args.limit)
    print(f"Convertidas: {len(converted)} | pendientes: {len(pending)} | ahora: {len(selected)}")
    if selected.empty:
        return

    detector, device = load_detector(args.model_dir)
    lung_model = LMInferer(modelname="R231", tqdm_disable=True)
    lobe_model = LMInferer(modelname="LTRCLobes", tqdm_disable=True)
    print("Dispositivo:", device)

    failures = []
    for number, row in enumerate(selected.itertuples(), start=1):
        start = time.time()
        try:
            candidates = detect_series(args.data_dir / row.ruta_imagen, detector, device, lung_model, lobe_model)
            report = save_report(row, candidates, args.threshold, outputs[row.Index])
            status = f"{len(report['nodulos'])} nodulos de {len(candidates)} candidatos"
        except Exception as error:  # Una serie mala no debe detener las demas.
            failures.append({"PatientID": row.PatientID, "SeriesInstanceUID": row.SeriesInstanceUID, "error": str(error)})
            status = f"fallo: {error}"
        print(f"[{number}/{len(selected)}] {row.PatientID}: {status} | {time.time() - start:.0f} s", flush=True)

    # Solo guarda los fallos de esta corrida: al reintentar, el archivo se actualiza.
    log = args.output_dir / "deteccion_fallidas.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["PatientID", "SeriesInstanceUID", "error"]).to_csv(log, index=False)
    if failures:
        print(f"{len(failures)} series fallaron; detalle en {log}. Vuelve a correr el script para reintentarlas.")


if __name__ == "__main__":
    main()
