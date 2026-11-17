"""LungAI — experimento técnico ResNet-10 3D con 200 CT NLST.

No es un clasificador clínico. Etiqueta: diagnóstico de cáncer a 12 meses,
no confirmación de una lesión visible en la tomografía. Muestreo 50/50.

Uso desde la raíz del proyecto (~/lungai-venv):
  python backend/training/entrenar_ct_monai_200_pulmon_amp.py --check-only
  python backend/training/entrenar_ct_monai_200_pulmon_amp.py --prepare --limit 2
  python backend/training/entrenar_ct_monai_200_pulmon_amp.py --prepare
  python backend/training/entrenar_ct_monai_200_pulmon_amp.py --train --epochs 3

Los modos --prepare y --train solo utilizan train y validation.
No lee imágenes del conjunto test ni calcula métricas de test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path("outputs/monai_dataset/lote_ct_3d_200.parquet")
DATA_DIR = Path("data/monai")
CACHE_BASE = Path("outputs/monai_ct_200_cache")
OUT = Path("outputs/monai_ct_200_pulmon_amp")
QUOTAS = {"train": (140, 70), "validation": (30, 15), "test": (30, 15)}
REQUIRED = {"PatientID", "split", "etiqueta_cancer_12m", "ruta_imagen", "SeriesDescription"}
SEED = 2026
WARNING = (
    "SOLO INVESTIGACION: 200 CT seleccionadas 50/50; etiquetas de diagnostico "
    "a 12 meses, NO de malignidad visible. Sin validacion clinica. "
    "No usar como diagnostico ni integrar como predictor de cancer. "
    "Conjunto test reservado."
)


def load_manifest(path: Path, data_dir: Path):
    if not path.is_file():
        raise SystemExit(f"No se encontro el manifiesto: {path}")
    df = pd.read_parquet(path)
    absent = REQUIRED.difference(df.columns)
    if absent:
        raise SystemExit(f"Faltan columnas en el manifiesto: {sorted(absent)}")
    if len(df) != 200:
        raise SystemExit(f"Se esperaban 200 filas; hay {len(df)}")
    if df["PatientID"].isna().any() or df["ruta_imagen"].isna().any():
        raise SystemExit("Existen identificadores o rutas vacios")
    df["PatientID"] = df["PatientID"].astype(str).str.strip()
    if df["PatientID"].eq("").any() or df["PatientID"].duplicated().any():
        raise SystemExit("Pacientes vacios o repetidos entre conjuntos")
    if not df["split"].isin(QUOTAS).all() or not df["etiqueta_cancer_12m"].isin([0, 1]).all():
        raise SystemExit("Split o etiquetas inesperadas")
    if not df["SeriesDescription"].astype(str).str.split(",").str[0].eq("0").all():
        raise SystemExit("Al menos una serie no es del cribado T0")

    root = data_dir.resolve()
    groups = {"train": [], "validation": []}
    for split, (expected, positive) in QUOTAS.items():
        part = df.loc[df["split"].eq(split)]
        if len(part) != expected or int(part["etiqueta_cancer_12m"].sum()) != positive:
            raise SystemExit(f"Distribucion inesperada en {split}")
        for row in part.itertuples(index=False):
            relative = Path(str(row.ruta_imagen).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit("Ruta insegura en el manifiesto")
            file = (root / relative).resolve()
            if not file.is_relative_to(root) or not file.is_file() or file.stat().st_size <= 0:
                raise SystemExit(f"CT no disponible o ruta invalida en {split}")
            if split != "test":  # Nunca abre las CT de prueba.
                groups[split].append((file, int(row.etiqueta_cancer_12m)))
        print(f"{split}: {expected} CT; positivas={positive}, negativas={expected-positive}", flush=True)
    return groups


def cache_dir(base: Path, size: tuple[int, int, int], margin: float) -> Path:
    return base / f"pulmon_xyz_{size[0]}_{size[1]}_{size[2]}_margen_{int(margin)}mm_v1"


def cache_name(image: Path) -> str:
    stat = image.stat()
    fingerprint = f"{image.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:24] + ".npy"


def preprocess(image: Path, inferer, size: tuple[int, int, int], margin_mm: float):
    import SimpleITK as sitk
    import numpy as np

    ct = sitk.ReadImage(str(image))
    if ct.GetDimension() != 3 or min(ct.GetSize()) <= 2 or min(ct.GetSpacing()) <= 0:
        raise ValueError("Geometria CT invalida")
    mask = np.asarray(inferer.apply(ct))  # Z,Y,X; mascara segmentada por lungmask
    expected_shape = (ct.GetSize()[2], ct.GetSize()[1], ct.GetSize()[0])
    if mask.shape != expected_shape:
        raise ValueError("lungmask devolvio mascara de forma distinta")
    pts = np.argwhere(mask > 0)
    if len(pts) < 1000:
        raise ValueError("Mascara pulmonar vacia o sospechosamente pequena")
    low_zyx = pts.min(axis=0)
    high_zyx = pts.max(axis=0)
    low = low_zyx[::-1].astype(np.int64)  # XYZ
    high = high_zyx[::-1].astype(np.int64)
    orig_size = np.asarray(ct.GetSize(), dtype=np.int64)
    spacing = np.asarray(ct.GetSpacing(), dtype=np.float64)
    margin_vox = np.ceil(float(margin_mm) / spacing).astype(np.int64)
    low = np.maximum(0, low - margin_vox)
    high = np.minimum(orig_size - 1, high + margin_vox)
    width = high - low + 1
    if np.any(width < 20):
        raise ValueError("Recorte pulmonar demasiado pequeno")

    # Conserva la geometría física: misma dirección y límites de la CT.
    physical_origin = ct.TransformIndexToPhysicalPoint(tuple(int(i) for i in low))
    new_spacing = tuple(float((width[i] - 1) * spacing[i] / (size[i] - 1)) for i in range(3))
    resampled = sitk.Resample(
        ct, list(size), sitk.Transform(), sitk.sitkLinear,
        physical_origin, new_spacing, ct.GetDirection(),
        -1000.0, sitk.sitkFloat32,
    )
    array = sitk.GetArrayFromImage(resampled)  # Z Y X
    if not np.isfinite(array).all():
        raise ValueError("CT con intensidades no finitas")
    np.clip(array, -1000.0, 400.0, out=array)
    array = (array + 1000.0) / 1400.0
    return array.astype(np.float16), tuple(round(s, 3) for s in new_spacing)


def prepare(groups, folder: Path, size, margin: float, limit: int):
    # Importamos la segmentación sólo en esta fase; entrenar desde caché no la requiere.
    from lungmask import LMInferer

    folder.mkdir(parents=True, exist_ok=True)
    inferer = LMInferer(modelname="R231", force_cpu=True, tqdm_disable=True)
    records = groups["train"] + groups["validation"]
    created = reused = 0
    for index, (image, _) in enumerate(records, 1):
        destination = folder / cache_name(image)
        if destination.is_file():
            reused += 1
            continue
        if limit and created >= limit:
            break
        print(f"Preparando recorte pulmonar nuevo {created+1} | posicion {index}/{len(records)}...", flush=True)
        try:
            array, mm = preprocess(image, inferer, size, margin)
            temporary = destination.with_suffix(".tmp.npy")
            np.save(temporary, array, allow_pickle=False)
            os.replace(temporary, destination)
            created += 1
            print(f"  OK: recorte {array.shape} ZYX; espaciado final XYZ {mm} mm", flush=True)
        except Exception as exc:
            raise SystemExit(f"Se detuvo el preprocesado en la posicion {index}: {type(exc).__name__}: {exc}") from exc
    ready = sum((folder / cache_name(image)).is_file() for image, _ in records)
    print(f"CACHE PULMONAR: {ready}/{len(records)} preparados; nuevos={created}, reutilizados={reused}", flush=True)
    if ready != len(records):
        print("Faltan recortes. Ejecuta otra vez --prepare sin --limit. No hay que repetir los preparados.", flush=True)


class CachedCT:
    def __init__(self, records, folder: Path):
        self.records = [(folder / cache_name(image), label) for image, label in records]
        absent = [item for item, _ in self.records if not item.is_file()]
        if absent:
            raise SystemExit(f"Hay {len(absent)} CT sin preprocesar; ejecuta --prepare antes de --train")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import torch
        file, label = self.records[idx]
        arr = np.load(file, allow_pickle=False)
        if arr.ndim != 3 or not np.isfinite(arr).all():
            raise ValueError("CT cacheada invalida")
        # Copia a RAM una sola CT: evita conservar 170 tomografías en memoria.
        x = torch.from_numpy(arr.astype(np.float32, copy=True)).unsqueeze(0)
        return x, torch.tensor(label, dtype=torch.long)


def auc_binary(labels: list[int], scores: list[float]):
    positive = np.array(labels, dtype=np.int64) == 1
    values = np.array(scores, dtype=np.float64)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0 or not np.isfinite(values).all():
        return None
    # Mann–Whitney U con empates a mitad de crédito.
    good = values[positive][:, None]
    bad = values[~positive][None, :]
    return float(((good > bad).sum() + 0.5 * (good == bad).sum()) / (n_pos * n_neg))


def evaluate(model, loader, loss_fn, device, amp):
    import torch
    model.eval()
    total_loss, labels, scores = 0.0, [], []
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            total_loss += float(loss.detach().item()) * y.numel()
            scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
            labels.extend(y.cpu().tolist())
    if not np.isfinite(total_loss):
        raise RuntimeError("Pérdida de validación no finita")
    predictions = [int(s >= 0.5) for s in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    return {"val_loss": total_loss/len(labels), "val_aciertos": tp+tn,
            "val_total": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "roc_auc_exploratorio": auc_binary(labels, scores),
            "predicciones_clase_1": tp+fp}


def train(groups, folder: Path, output: Path, manifest: Path, size, margin, epochs: int, use_amp: bool):
    import torch
    from monai.networks.nets import resnet10
    from torch import nn
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise SystemExit("Este experimento necesita CUDA; no se iniciará en CPU accidentalmente")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"El directorio de salida no está vacío: {output}. No se sobrescriben resultados")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    train_set = CachedCT(groups["train"], folder)
    val_set = CachedCT(groups["validation"], folder)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda:0")
    model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    weights = output / "modelo_resnet10_ct200_pulmon.pt"
    report = output / "entrenamiento.json"
    print(f"TRAIN: {len(train_set)} CT, VALIDATION: {len(val_set)} CT, TEST: NO UTILIZADO", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} | AMP: {use_amp} | XYZ: {size}", flush=True)
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Pérdida de entrenamiento no finita. Se cancela.")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += float(loss.detach().item())
            correct += int((logits.argmax(1) == y).sum().item())
        result = evaluate(model, val_loader, loss_fn, device, use_amp)
        result.update({"epoch": epoch, "train_loss": train_loss/len(train_set),
                       "train_aciertos": correct, "train_total": len(train_set)})
        history.append(result)
        print(f"Epoch {epoch}/{epochs}: train_loss={result['train_loss']:.4f} "
              f"train={correct}/{len(train_set)}; val_loss={result['val_loss']:.4f} "
              f"val={result['val_aciertos']}/{len(val_set)}; "
              f"val_ROC_AUC={result['roc_auc_exploratorio']:.3f} "
              f"TP={result['tp']} FP={result['fp']} TN={result['tn']} FN={result['fn']}", flush=True)
        if result["val_loss"] < best_loss:
            best_loss = result["val_loss"]
            best_epoch = epoch
            tmp = output / "modelo_resnet10_ct200_pulmon.tmp"
            torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "architecture": "monai.resnet10_3d", "size_xyz": list(size),
                        "margin_mm": margin, "lung_crop": "lungmask_R231_bbox_plus_margin",
                        "normalization": "clip_HU_-1000_400_then_0_1", "amp": use_amp,
                        "best_epoch": best_epoch, "warning": WARNING}, tmp)
            os.replace(tmp, weights)
        payload = {"tipo": "experimento_ct200_pulmon_no_clinico", "objetivo": "cancer_diagnosticado_12m",
                   "train_ct": len(train_set), "validation_ct": len(val_set), "test_ct_utilizadas": 0,
                   "muestreo_balanceado_artificial": True, "size_xyz": list(size),
                   "margin_mm": margin, "amp": use_amp, "best_epoch": best_epoch,
                   "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   "history": history, "warning": WARNING, "finished": epoch == epochs}
        tmpj = output / "entrenamiento.tmp.json"
        tmpj.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmpj, report)
    print(f"MEJOR EPOCA: {best_epoch}; pesos: {weights}; reporte: {report}", flush=True)
    print("SIN METRICAS CLINICAS; TEST RESERVADO. No integrar como predictor a la web.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_BASE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--size", type=int, nargs=3, default=(128, 128, 112), metavar=("X", "Y", "Z"))
    parser.add_argument("--margin-mm", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=0, help="Sólo con --prepare: cantidad de recortes nuevos (0=todos)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if any(d < 32 for d in args.size) or not 0 < args.margin_mm <= 50 or not 1 <= args.epochs <= 30:
        raise SystemExit("Configuración fuera de rango")
    if args.limit < 0 or (args.limit and not args.prepare):
        raise SystemExit("--limit solo se permite con --prepare")
    print(WARNING, flush=True)
    print("Revisando manifiesto y existencia local (no lee CT de test)...", flush=True)
    groups = load_manifest(args.manifest, args.data_dir)
    if args.check_only:
        print("REVISION APROBADA: 200/200 CT registradas, splits y etiquetas correctos.")
        print("Sin escribir archivos ni examinar el test.")
        return
    folder = cache_dir(args.cache, tuple(args.size), args.margin_mm)
    if args.prepare:
        prepare(groups, folder, tuple(args.size), args.margin_mm, args.limit)
    else:
        train(groups, folder, args.output, args.manifest, tuple(args.size), args.margin_mm, args.epochs, not args.no_amp)


if __name__ == "__main__":
    main()
