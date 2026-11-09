"""LungAI | Prueba tecnica de una red 3D MONAI con UNA CT real.

NO entrena, NO calibra, NO diagnostica cancer ni guarda pesos.

Desde la raiz del proyecto:
    uv run python backend/data_pipeline/probar_modelo_monai.py

Requiere MONAI y PyTorch instalados. Usa una unica NIfTI de training
que ya exista en data/monai/images y figure en el datalist generado.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from monai.networks.nets import resnet10
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Resized,
    ScaleIntensityRanged,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist", type=Path, default=Path("outputs/monai_dataset/datalist.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/monai"))
    parser.add_argument("--size", type=int, default=48, help="Lado de la prueba 3D; recomendado 48 en CPU.")
    args = parser.parse_args()

    if args.size < 32 or args.size > 128 or args.size % 16:
        parser.error("--size debe ser multiplo de 16 y estar entre 32 y 128.")
    if not args.datalist.is_file():
        parser.error(f"No existe el datalist: {args.datalist}")

    data = json.loads(args.datalist.read_text(encoding="utf-8"))
    train = data.get("training", [])
    disponible = None
    for item in train:
        ruta_rel = item.get("image")
        if isinstance(ruta_rel, str):
            ruta = args.image_root / ruta_rel
            if ruta.is_file():
                disponible = (ruta, item)
                break
    if disponible is None:
        parser.error("No hay una CT de training descargada que coincida con el datalist.")

    ruta, item = disponible
    etiqueta = int(item["label"])
    if etiqueta not in (0, 1):
        parser.error("La etiqueta de esta muestra debe ser 0 o 1.")

    torch.manual_seed(42)
    # CPU-only: limitar hilos para que la prueba no monopolice el equipo.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print("=== LUNGAI | PRUEBA DE RED 3D CON MONAI ===", flush=True)
    print("NIfTI de training encontrado: SI", flush=True)
    print("Etiqueta de seguimiento a 12 meses:", etiqueta, flush=True)
    print("Dispositivo: CPU (prueba sin GPU)", flush=True)

    # Se usa Resize SOLO para la prueba de conectividad: cambia la geometria
    # y NO es un preprocesamiento clinico ni una receta final de entrenamiento.
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(args.size,) * 3,
                mode="trilinear", align_corners=False),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400,
                             b_min=0, b_max=1, clip=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    img = transform({"image": str(ruta)})["image"]
    if img.shape != (1, args.size, args.size, args.size):
        raise RuntimeError(f"Forma inesperada despues de MONAI: {tuple(img.shape)}")
    if not bool(torch.isfinite(img).all()):
        raise RuntimeError("La imagen transformada contiene valores no finitos.")

    print("Tensor MONAI (C, X, Y, Z):", tuple(img.shape), flush=True)
    batch = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0).to("cpu")  # (B, C, X, Y, Z)
    model = resnet10(spatial_dims=3, n_input_channels=1,
                     num_classes=2, pretrained=False, conv1_t_stride=2)
    model.eval()
    with torch.inference_mode():
        output = model(batch)
    if output.shape != (1, 2) or not bool(torch.isfinite(output).all()):
        raise RuntimeError(f"Salida de red invalida: {tuple(output.shape)}")

    print("Entrada a red (B,C,X,Y,Z):", tuple(batch.shape))
    print("Salida de red (B,clases):", tuple(output.shape))
    print("Pesos entrenados: NO")
    print("Prueba de ejecucion 3D: OK")
    print("ADVERTENCIA: la salida NO es una prediccion de cancer. "
          "Es una red SIN ENTRENAR, usada solo para verificar el flujo.")


if __name__ == "__main__":
    main()
