"""
LungAI — entrena un clasificador de nodulos (maligno o benigno) con los recortes.

Usa la tabla de etiquetar_nodulos.py y los recortes de procesar_lotes.py. Entrena una
ResNet 3D de MONAI con los pacientes de train, elige la mejor epoca con validation y
guarda los pesos y las metricas. Compara siempre contra dos reglas simples: usar solo el
tamano de la caja y usar solo el puntaje del detector.

Uso:
    python backend/training/entrenar_clasificador.py

Importante:
- La salida es una probabilidad aprendida de etiquetas aproximadas; NO es un diagnostico.
- Si el modelo no supera a la regla del tamano, no esta aportando nada: hay que decirlo.
- test solo se evalua con --eval-test, una vez, cuando ya no se vaya a cambiar nada.
- Hay muchos mas benignos que malignos: se usan todos los malignos y una parte de los
  benignos (primero los que reporto el radiologo), y cada lote se arma balanceado.
- outputs/ esta en .gitignore. No subir pesos ni predicciones a GitHub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_roc_auc
from monai.networks.nets import resnet10
from monai.transforms import Compose, RandFlip, RandGaussianNoise, RandRotate90
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

HU_MIN, HU_MAX = -1000, 400

# Volteos y giros de 90 grados dentro del corte: un nodulo no tiene orientacion preferida.
AUGMENT = Compose([
    RandFlip(prob=0.5, spatial_axis=0),
    RandFlip(prob=0.5, spatial_axis=1),
    RandFlip(prob=0.5, spatial_axis=2),
    RandRotate90(prob=0.5, spatial_axes=(1, 2)),
    RandGaussianNoise(prob=0.3, std=0.02),
])


def select_samples(table: pd.DataFrame, benign_per_malignant: int, max_days: float | None, seed: int) -> pd.DataFrame:
    """Elige los recortes con etiqueta: todos los malignos y una parte de los benignos."""
    table = table[table["recorte"].notna() & table["etiqueta"].isin(["maligno", "benigno"])]
    table = table[table["split"].isin(["train", "validation", "test"])]
    if max_days is not None:
        late = table["etiqueta"].eq("maligno") & table["dias_al_diagnostico"].gt(max_days)
        table = table[~late]

    chosen = []
    for _, group in table.groupby("split"):
        malignant = group[group["etiqueta"].eq("maligno")]
        benign = group[group["etiqueta"].eq("benigno")].sample(frac=1, random_state=seed)
        # Los que reporto el radiologo son los negativos mas parecidos a un nodulo maligno.
        benign = benign.sort_values("reportado", ascending=False, kind="stable")
        chosen += [malignant, benign.head(benign_per_malignant * max(len(malignant), 1))]
    samples = pd.concat(chosen, ignore_index=True)
    samples["maligno"] = samples["etiqueta"].eq("maligno").astype(int)

    splits_per_patient = samples.groupby("PatientID")["split"].nunique()
    if splits_per_patient.gt(1).any():
        raise ValueError("Hay pacientes en mas de un conjunto: revisar la division.")
    return samples


def load_crops(samples: pd.DataFrame, crops_dir: Path) -> np.ndarray:
    """Lee de los .npz el recorte de cada fila, en el mismo orden."""
    crops = np.empty((len(samples), 40, 64, 64), dtype=np.int16)
    for (patient, series), group in samples.groupby(["PatientID", "SeriesInstanceUID"]):
        with np.load(crops_dir / patient / f"{series}.npz") as archive:
            crops[group.index] = archive["recortes"][group["recorte"].astype(int)]
    return crops


class CropDataset(Dataset):
    def __init__(self, crops: np.ndarray, labels: np.ndarray, augment: bool):
        self.crops, self.labels, self.augment = crops, labels, augment

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int):
        crop = torch.from_numpy(self.crops[index].astype(np.float32))
        crop = ((crop - HU_MIN) / (HU_MAX - HU_MIN)).clamp(0, 1).unsqueeze(0)  # canal, z, y, x
        if self.augment:
            crop = torch.as_tensor(AUGMENT(crop))
        return crop, torch.tensor(self.labels[index], dtype=torch.float32)


def auc(scores, labels) -> float | None:
    """Area bajo la curva ROC; None si el conjunto no tiene las dos clases."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    return round(float(compute_roc_auc(torch.as_tensor(np.asarray(scores, dtype=np.float32)),
                                       torch.as_tensor(labels))), 4)


def predict(model, crops: np.ndarray, labels: np.ndarray, device, batch_size: int) -> np.ndarray:
    """Regresa la salida sin sigmoide: asi el AUC no pierde orden cuando la probabilidad se satura."""
    model.eval()
    loader = DataLoader(CropDataset(crops, labels, augment=False), batch_size=batch_size)
    with torch.no_grad():
        return torch.cat([model(x.to(device)).squeeze(1).float().cpu() for x, _ in loader]).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("outputs/etiquetas/candidatos_etiquetados.parquet"))
    parser.add_argument("--crops-dir", type=Path, default=Path("outputs/recortes"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clasificador"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--benign-per-malignant", type=int, default=10)
    parser.add_argument("--max-days", type=float, help="Ignorar malignos diagnosticados despues de estos dias.")
    parser.add_argument("--eval-test", action="store_true", help="Evaluar en test al terminar (solo una vez).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    AUGMENT.set_random_state(args.seed)

    samples = select_samples(pd.read_parquet(args.labels), args.benign_per_malignant, args.max_days, args.seed)
    counts = pd.crosstab(samples["split"], samples["etiqueta"]).reindex(
        index=["train", "validation", "test"], columns=["maligno", "benigno"], fill_value=0
    )
    print("Recortes por conjunto:")
    print(counts.to_string())
    if counts.loc["train"].min() == 0:
        raise SystemExit("train necesita recortes malignos y benignos. Procesa mas series antes de entrenar.")

    crops = load_crops(samples, args.crops_dir)
    labels = samples["maligno"].to_numpy()
    index = {split: np.flatnonzero(samples["split"].eq(split).to_numpy()) for split in counts.index}
    train, validation = index["train"], index["validation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Con pocos datos BatchNorm se comporta distinto al evaluar; InstanceNorm no depende del lote.
    model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=1, norm=("instance", {"affine": True})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = torch.nn.BCEWithLogitsLoss()

    # Cada lote se arma con la misma cantidad esperada de malignos y benignos.
    class_weight = 1.0 / np.bincount(labels[train])
    sampler = WeightedRandomSampler(class_weight[labels[train]], num_samples=len(train), replacement=True)
    loader = DataLoader(CropDataset(crops[train], labels[train], augment=True),
                        batch_size=args.batch_size, sampler=sampler, drop_last=len(train) > args.batch_size)
    print(f"Dispositivo: {device} | recortes de train: {len(train)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "modelo.pt"
    history, best = [], {"epoca": 0, "auc_validation": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_function(model(x.to(device)).squeeze(1), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()

        score = auc(predict(model, crops[validation], labels[validation], device, args.batch_size),
                    labels[validation]) if len(validation) else None
        history.append({"epoca": epoch, "perdida_train": round(total / len(loader), 4), "auc_validation": score})
        print(f"Epoca {epoch}/{args.epochs} | perdida {history[-1]['perdida_train']} | AUC validation {score}", flush=True)

        # Sin validation con las dos clases no se puede elegir: se queda la ultima epoca.
        if score is None or best["auc_validation"] is None or score > best["auc_validation"]:
            best = {"epoca": epoch, "auc_validation": score}
            torch.save(model.state_dict(), weights_path)

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    metrics = {
        "recortes": counts.to_dict("index"),
        "mejor_epoca": best["epoca"],
        "historial": history,
        "opciones": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    evaluated = ["validation", "test"] if args.eval_test else ["validation"]
    predictions = []
    for split in evaluated:
        rows = index[split]
        if not len(rows):
            continue
        output = predict(model, crops[rows], labels[rows], device, args.batch_size)
        subset = samples.iloc[rows]
        metrics[split] = {
            "auc_modelo": auc(output, labels[rows]),
            "auc_solo_tamano": auc(subset["tamano_mm"], labels[rows]),
            "auc_solo_puntaje_detector": auc(subset["puntaje"], labels[rows]),
        }
        predictions.append(subset[["PatientID", "SeriesInstanceUID", "recorte", "split", "etiqueta",
                                   "tamano_mm", "puntaje"]].assign(probabilidad_maligno=(1 / (1 + np.exp(-output))).round(4)))

    (args.output_dir / "metricas.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if predictions:
        pd.concat(predictions).to_csv(args.output_dir / "predicciones.csv", index=False)

    print(f"\nMejor epoca: {best['epoca']}")
    for split in evaluated:
        if split in metrics:
            print(f"{split}: " + " | ".join(f"{name} {value}" for name, value in metrics[split].items()))
    if not args.eval_test:
        print("test no se evaluo (usa --eval-test cuando el modelo ya este decidido).")
    print(f"Archivos guardados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
