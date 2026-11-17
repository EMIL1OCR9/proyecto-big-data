"""LungAI: entrenamiento técnico 3D sobre CT locales válidas del lote ampliado.

- Usa exclusivamente imágenes locales de los splits train y validation.
- Reserva test: NO lee sus píxeles ni utiliza sus etiquetas en métricas.
- Excluye las CT ausentes o rechazadas por control de calidad, SIN inventar cortes.
- Reproduce el preprocesado y arquitectura del piloto técnico previo.
- NO es un modelo clínico, ni una estimación de riesgo calibrada.

Desde la raíz del proyecto, con ~/lungai-venv:
  python backend/training/entrenar_ct_monai_ampliado.py --check-only
  python backend/training/entrenar_ct_monai_ampliado.py --epochs 3
"""
from __future__ import annotations

import argparse
import hashlib
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

MANIFIESTO = Path("outputs/monai_dataset/lote_ct_3d_ampliado.parquet")
DIRECTORIO_CT = Path("data/monai")
SALIDA = Path("outputs/monai_ct_ampliado")
CUOTAS = {"train": (40, 20), "validation": (12, 6), "test": (12, 6)}
NECESARIAS = {"PatientID", "split", "etiqueta_cancer_12m", "ruta_imagen"}
AVISO = (
    "DEMOSTRACION TECNICA NO VALIDADA. Seleccion artificialmente balanceada de NLST; "
    "las etiquetas corresponden a diagnostico en 12 meses, no a una anotacion "
    "radiologica de malignidad en la CT. NO usar en pacientes, para diagnostico, "
    "ni como predictor de cancer en la web. Test reservado."
)


def revisar_lote(ruta: Path, directorio: Path):
    if not ruta.is_file():
        raise SystemExit(f"No se encontro el manifiesto: {ruta}")
    tabla = pd.read_parquet(ruta)
    faltan = NECESARIAS - set(tabla.columns)
    if faltan:
        raise SystemExit(f"Faltan columnas: {sorted(faltan)}")
    if len(tabla) != 64:
        raise SystemExit(f"El manifiesto debe contener 64 filas; tiene {len(tabla)}")
    if tabla["PatientID"].isna().any():
        raise SystemExit("Hay identificadores ausentes")
    tabla["PatientID"] = tabla["PatientID"].astype(str).str.strip()
    if tabla["PatientID"].eq("").any() or tabla["PatientID"].duplicated().any():
        raise SystemExit("Hay identificadores vacios o pacientes repetidos")
    if not tabla["split"].isin(CUOTAS).all():
        raise SystemExit("Se encontraron splits desconocidos")
    if not tabla["etiqueta_cancer_12m"].isin([0, 1]).all():
        raise SystemExit("Se encontraron etiquetas distintas de 0 y 1")
    for nombre, (cantidad, positivos) in CUOTAS.items():
        segmento = tabla.loc[tabla["split"].eq(nombre)]
        if len(segmento) != cantidad or int(segmento["etiqueta_cancer_12m"].sum()) != positivos:
            raise SystemExit(f"Distribucion del manifiesto inesperada en {nombre}")

    raiz = directorio.resolve()
    grupos = {nombre: [] for nombre in CUOTAS}
    ausentes = {nombre: {0: 0, 1: 0} for nombre in CUOTAS}
    for _, fila in tabla.iterrows():
        split = str(fila["split"])
        etiqueta = int(fila["etiqueta_cancer_12m"])
        relativa = Path(str(fila["ruta_imagen"]).replace("\\", "/"))
        if relativa.is_absolute() or ".." in relativa.parts:
            raise SystemExit("Ruta insegura en el manifiesto")
        ct = (raiz / relativa).resolve()
        if not ct.is_relative_to(raiz):
            raise SystemExit("Una ruta sale del directorio de CT")
        if not ct.is_file() or ct.stat().st_size <= 0:
            ausentes[split][etiqueta] += 1
            continue
        # Solo se llevan las rutas del test en el conteo, jamás se lee su imagen.
        if split != "test":
            grupos[split].append((ct, etiqueta))

    print("LOTE AMPLIADO: IMAGENES LOCALES VALIDAS POR EXISTENCIA", flush=True)
    for nombre, (cantidad, positivos) in CUOTAS.items():
        presentes = cantidad - sum(ausentes[nombre].values())
        pres_positivos = positivos - ausentes[nombre][1]
        pres_negativos = (cantidad - positivos) - ausentes[nombre][0]
        print(f"{nombre}: {presentes}/{cantidad} CT; positivas={pres_positivos}, "
              f"negativas={pres_negativos}; ausentes={sum(ausentes[nombre].values())}", flush=True)
    if len(grupos["train"]) < 16 or sum(y for _, y in grupos["train"]) < 4 or sum(1-y for _, y in grupos["train"]) < 4:
        raise SystemExit("Insuficientes CT locales de entrenamiento de ambas clases")
    if len(grupos["validation"]) < 8 or sum(y for _, y in grupos["validation"]) < 3 or sum(1-y for _, y in grupos["validation"]) < 3:
        raise SystemExit("Insuficientes CT locales de validacion de ambas clases")
    print("TEST RESERVADO: no se cargaran sus imagenes ni se calcularan metricas de test.", flush=True)
    return grupos


def preparar_ct(ruta: Path, tamaño_xyz: tuple[int, int, int]) -> np.ndarray:
    """Remuestreo del volumen completo: solo un compromiso para prueba técnica."""
    imagen = sitk.ReadImage(str(ruta))
    if imagen.GetDimension() != 3:
        raise ValueError("Se encontro una imagen que no es tridimensional")
    original = imagen.GetSize()
    espaciado = imagen.GetSpacing()
    if min(original) < 2 or min(espaciado) <= 0:
        raise ValueError("Geometria CT no valida")
    nuevo_espaciado = tuple(
        float((original[i] - 1) * espaciado[i] / (tamaño_xyz[i] - 1))
        for i in range(3)
    )
    reducida = sitk.Resample(
        imagen, list(tamaño_xyz), sitk.Transform(), sitk.sitkLinear,
        imagen.GetOrigin(), nuevo_espaciado, imagen.GetDirection(),
        -1000.0, sitk.sitkFloat32,
    )
    x = sitk.GetArrayFromImage(reducida)  # [Z, Y, X]
    if not np.isfinite(x).all():
        raise ValueError("Intensidades CT no finitas")
    np.clip(x, -1000.0, 400.0, out=x)
    x += 1000.0
    x /= 1400.0
    return x[np.newaxis, ...].astype(np.float32, copy=False)


def crear_dataset(registros, tamaño_xyz, nombre):
    volumenes = []
    etiquetas = []
    for i, (ruta, etiqueta) in enumerate(registros, 1):
        print(f"Preparando CT {nombre} {i}/{len(registros)}...", flush=True)
        volumenes.append(torch.from_numpy(preparar_ct(ruta, tamaño_xyz)))
        etiquetas.append(etiqueta)
    return TensorDataset(torch.stack(volumenes), torch.tensor(etiquetas, dtype=torch.long))


def evaluar(modelo, loader, criterio, dispositivo):
    modelo.eval()
    suma, aciertos, total = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logits = modelo(x)
            suma += float(criterio(logits, y).item()) * len(y)
            aciertos += int((logits.argmax(1) == y).sum().item())
            total += len(y)
    return suma / total, aciertos, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFIESTO)
    p.add_argument("--data-dir", type=Path, default=DIRECTORIO_CT)
    p.add_argument("--output", type=Path, default=SALIDA)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--size", type=int, nargs=3, metavar=("X", "Y", "Z"), default=(96, 96, 64))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Solo revisa manifiesto y CT presentes; no entrena")
    a = p.parse_args()
    if not 1 <= a.epochs <= 30 or any(v < 32 for v in a.size):
        raise SystemExit("--epochs entre 1 y 30; dimensiones de --size al menos 32")

    print(AVISO, flush=True)
    grupos = revisar_lote(a.manifest, a.data_dir)
    if a.check_only:
        print("Comprobacion terminada. No se entreno ni modifico ningun archivo.")
        return

    pesos = a.output / "modelo_resnet10_ampliado.pt"
    reporte = a.output / "entrenamiento.json"
    if pesos.exists() or reporte.exists():
        raise SystemExit("Ya existen resultados en el destino; se canceló para no sobrescribirlos")

    semilla = 42
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.benchmark = False
    dispositivo = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"Dispositivo: {dispositivo}; tamaño XYZ: {tuple(a.size)}", flush=True)

    train = crear_dataset(grupos["train"], tuple(a.size), "train")
    val = crear_dataset(grupos["validation"], tuple(a.size), "validation")
    generador = torch.Generator().manual_seed(semilla)
    train_dl = DataLoader(train, batch_size=1, shuffle=True, num_workers=0, generator=generador)
    val_dl = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)

    red = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2).to(dispositivo)
    opt = torch.optim.Adam(red.parameters(), lr=1e-4)
    criterio = nn.CrossEntropyLoss()
    mejor = float("inf")
    epoca_mejor = 0
    mejores_pesos = None
    historia = []
    for epoca in range(1, a.epochs + 1):
        red.train()
        suma, aciertos = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dispositivo), y.to(dispositivo)
            opt.zero_grad(set_to_none=True)
            logits = red(x)
            perdida = criterio(logits, y)
            perdida.backward()
            opt.step()
            suma += float(perdida.item())
            aciertos += int((logits.argmax(1) == y).sum().item())
        val_loss, val_ok, val_n = evaluar(red, val_dl, criterio, dispositivo)
        r = {
            "epoch": epoca,
            "train_loss": suma / len(train),
            "train_aciertos": aciertos,
            "train_total": len(train),
            "val_loss": val_loss,
            "val_aciertos": val_ok,
            "val_total": val_n,
        }
        historia.append(r)
        print(f"Epoch {epoca}/{a.epochs}: train_loss={r['train_loss']:.4f} "
              f"train={aciertos}/{len(train)}; val_loss={val_loss:.4f} "
              f"val={val_ok}/{val_n}", flush=True)
        if val_loss < mejor:
            mejor = val_loss
            epoca_mejor = epoca
            mejores_pesos = {k: v.detach().cpu().clone() for k, v in red.state_dict().items()}

    if mejores_pesos is None:
        raise RuntimeError("El entrenamiento no genero pesos")
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": mejores_pesos,
        "architecture": "monai.resnet10_3d",
        "size_xyz": tuple(a.size),
        "best_epoch": epoca_mejor,
        "warning": AVISO,
    }, pesos)
    metricas = {
        "tipo": "experimento_ct_ampliado_no_validado",
        "usa_pixeles_ct": True,
        "objetivo": "diagnostico_cancer_12m_en_cohorte_NLST",
        "balanceado_artificialmente": True,
        "train_ct": len(train),
        "validation_ct": len(val),
        "test_ct_utilizadas": 0,
        "mejor_epoca_por_val_loss": epoca_mejor,
        "size_xyz": list(a.size),
        "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "historial": historia,
        "advertencia": AVISO,
    }
    reporte.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo ampliado guardado: {pesos}", flush=True)
    print(f"Reporte: {reporte}", flush=True)
    print("Modelo de investigacion, no utilizar en pacientes ni agregar como diagnostico en la web.", flush=True)


if __name__ == "__main__":
    main()
