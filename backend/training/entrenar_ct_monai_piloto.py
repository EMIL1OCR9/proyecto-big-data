"""LungAI: prueba de extremo a extremo de clasificación de CT con MONAI.

NO es un modelo clínico: piloto deliberadamente seleccionado con 8 CT de
entrenamiento (4 positivas/4 negativas) y 4 de validación (2/2).

Ejecutar desde la raíz de proyecto-big-data, en Ubuntu/WSL y lungai-venv:
    python backend/training/entrenar_ct_monai_piloto.py --epochs 3

Dependencias: torch, monai, numpy, SimpleITK, pandas, pyarrow.
No usa ni modifica la red de detección de nódulos ni el modelo basado en scr_res0.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.networks.nets import resnet10
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

COLUMNAS = {"PatientID", "split", "etiqueta_cancer_12m", "ruta_imagen"}
SPLITS = ("train", "validation")
AVISO = (
    "PILOTO TECNICO, no diagnostico: 8 CT para entrenamiento, 4 CT para validacion, "
    "ninguna prueba independiente. Los puntajes NO son probabilidades clinicas; "
    "no se debe usar en pacientes ni integrar como detector de cancer."
)


def cargar_piloto(parquet: Path, data_dir: Path) -> dict[str, list[tuple[Path, int]]]:
    if not parquet.is_file():
        raise SystemExit(f"No se encontro el piloto: {parquet}")
    tabla = pd.read_parquet(parquet)
    faltan = COLUMNAS.difference(tabla.columns)
    if faltan:
        raise SystemExit(f"Faltan columnas en el piloto: {sorted(faltan)}")
    if tabla.empty or tabla["PatientID"].isna().any() or tabla["PatientID"].duplicated().any():
        raise SystemExit("Piloto vacio, con ID ausente o con pacientes repetidos.")
    if not tabla["split"].isin(SPLITS).all():
        raise SystemExit("El piloto contiene splits inesperados. No usar TEST aqui.")
    if not tabla["etiqueta_cancer_12m"].isin((0, 1)).all():
        raise SystemExit("El piloto contiene etiquetas invalidas.")
    if len(tabla) != 12:
        raise SystemExit(f"El piloto debe contener 12 CT; se encontraron {len(tabla)}.")

    base = data_dir.resolve()
    grupos: dict[str, list[tuple[Path, int]]] = {nombre: [] for nombre in SPLITS}
    for r in tabla.itertuples(index=False):
        relativa = Path(str(r.ruta_imagen).replace("\\", "/"))
        if relativa.is_absolute() or ".." in relativa.parts:
            raise SystemExit("Ruta relativa de imagen no permitida.")
        imagen = (base / relativa).resolve()
        if not imagen.is_relative_to(base) or not imagen.is_file() or imagen.stat().st_size == 0:
            raise SystemExit("Falta una CT del piloto o su ruta es invalida.")
        grupos[str(r.split)].append((imagen, int(r.etiqueta_cancer_12m)))

    for nombre, total, positivos in (("train", 8, 4), ("validation", 4, 2)):
        etiquetas = [y for _, y in grupos[nombre]]
        if len(etiquetas) != total or sum(etiquetas) != positivos:
            raise SystemExit(f"Distribucion inesperada en {nombre}; se cancela el entrenamiento.")
        print(f"{nombre}: {len(etiquetas)} CT, {sum(etiquetas)} positivas, {len(etiquetas)-sum(etiquetas)} negativas", flush=True)
    return grupos


def preparar_ct(ruta: Path, size_xyz: tuple[int, int, int]) -> np.ndarray:
    """Remuestrea toda la CT a una forma pequeña manteniendo su campo físico.

    Este remuestreo anisotrópico es SOLO para comprobar el pipeline, no es un
    preprocesamiento clínicamente validado.
    """
    imagen = sitk.ReadImage(str(ruta))
    if imagen.GetDimension() != 3:
        raise ValueError("Se encontro una imagen que no es 3D.")
    old_size = imagen.GetSize()
    old_spacing = imagen.GetSpacing()
    if min(old_size) < 2:
        raise ValueError("Imagen con dimension degenerada.")
    new_spacing = tuple(
        float((old_size[i] - 1) * old_spacing[i] / (size_xyz[i] - 1))
        for i in range(3)
    )
    remuestreada = sitk.Resample(
        imagen,
        list(size_xyz),
        sitk.Transform(),
        sitk.sitkLinear,
        imagen.GetOrigin(),
        new_spacing,
        imagen.GetDirection(),
        -1000.0,
        sitk.sitkFloat32,
    )
    # SimpleITK retorna [Z, Y, X]; MONAI/torch usa [C, D, H, W].
    x = sitk.GetArrayFromImage(remuestreada)
    if not np.isfinite(x).all():
        raise ValueError("Imagen con valores no finitos.")
    # HU (aire:-1000, tejidos hasta 400), escala [0, 1].
    np.clip(x, -1000, 400, out=x)
    x += 1000.0
    x /= 1400.0
    return x[np.newaxis, ...].astype(np.float32, copy=False)


def construir_dataset(grupo: list[tuple[Path, int]], size_xyz: tuple[int, int, int], nombre: str) -> TensorDataset:
    imagenes: list[torch.Tensor] = []
    etiquetas: list[int] = []
    for numero, (ruta, etiqueta) in enumerate(grupo, 1):
        # No imprime IDs ni nombres de archivos por privacidad.
        print(f"Preparando CT {nombre} {numero}/{len(grupo)}...", flush=True)
        imagenes.append(torch.from_numpy(preparar_ct(ruta, size_xyz)))
        etiquetas.append(etiqueta)
    return TensorDataset(torch.stack(imagenes), torch.tensor(etiquetas, dtype=torch.long))


def evaluar(modelo: nn.Module, cargador: DataLoader, loss_fn: nn.Module, dispositivo: torch.device) -> tuple[float, int, int]:
    modelo.eval()
    perdida, correctas, n = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in cargador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logits = modelo(x)
            perdida += float(loss_fn(logits, y).item()) * len(y)
            correctas += int((logits.argmax(dim=1) == y).sum().item())
            n += len(y)
    return perdida / n, correctas, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("outputs/monai_dataset/piloto_ct_3d.parquet"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/monai"))
    parser.add_argument("--output", type=Path, default=Path("outputs/monai_ct_piloto"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--size", type=int, nargs=3, metavar=("X", "Y", "Z"), default=(96, 96, 64))
    parser.add_argument("--cpu", action="store_true", help="Usar CPU si CUDA no tiene memoria.")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or min(args.size) < 32:
        raise SystemExit("Epochs debe estar entre 1 y 30, dimensiones al menos 32.")

    salida_modelo = args.output / "modelo_resnet10_piloto.pt"
    salida_metricas = args.output / "entrenamiento.json"
    if salida_modelo.exists() or salida_metricas.exists():
        raise SystemExit("Ya existen resultados de este piloto. No se sobrescribieron.")

    print(AVISO, flush=True)
    grupos = cargar_piloto(args.pilot, args.data_dir)
    semilla = 42
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.benchmark = False
    dispositivo = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"Dispositivo: {dispositivo} | tamano reducido XYZ: {tuple(args.size)}", flush=True)

    train = construir_dataset(grupos["train"], tuple(args.size), "train")
    val = construir_dataset(grupos["validation"], tuple(args.size), "validation")
    generador = torch.Generator().manual_seed(semilla)
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0, generator=generador)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)

    modelo = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    historial = []
    mejor_val = float("inf")
    mejor_pesos = None
    mejor_epoch = 0

    for epoch in range(1, args.epochs + 1):
        modelo.train()
        suma, aciertos = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad(set_to_none=True)
            logits = modelo(x)
            perdida = loss_fn(logits, y)
            perdida.backward()
            optimizador.step()
            suma += float(perdida.item())
            aciertos += int((logits.argmax(dim=1) == y).sum().item())
        val_loss, val_ok, val_n = evaluar(modelo, val_loader, loss_fn, dispositivo)
        registro = {
            "epoch": epoch, "train_loss": suma / len(train),
            "train_aciertos": aciertos, "train_total": len(train),
            "val_loss": val_loss, "val_aciertos": val_ok, "val_total": val_n,
        }
        historial.append(registro)
        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={registro['train_loss']:.4f} "
            f"train={aciertos}/{len(train)}; val_loss={val_loss:.4f} "
            f"val={val_ok}/{val_n}", flush=True,
        )
        if val_loss < mejor_val:
            mejor_val = val_loss
            mejor_epoch = epoch
            mejor_pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}

    assert mejor_pesos is not None
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": mejor_pesos,
        "architecture": "monai.resnet10_3d",
        "size_xyz": tuple(args.size),
        "best_epoch": mejor_epoch,
        "warning": AVISO,
    }
    torch.save(checkpoint, salida_modelo)
    reporte = {
        "objetivo": "etiqueta_cancer_12m",
        "tipo": "piloto_tecnico_no_validado",
        "usa_pixeles_ct": True,
        "train_ct": len(train), "validation_ct": len(val),
        "test_ct": 0,
        "best_epoch_por_val_loss": mejor_epoch,
        "size_xyz": list(args.size),
        "historial": historial,
        "advertencia": AVISO,
    }
    salida_metricas.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo piloto guardado: {salida_modelo}", flush=True)
    print(f"Reporte del piloto: {salida_metricas}", flush=True)
    print("SOLO demostracion tecnica: NO usar como clasificador de cancer en la web.", flush=True)


if __name__ == "__main__":
    main()
