"""
LungAI — procesa la muestra por lotes sin guardar las CT.

Por cada lote descarga las series de IDC, las convierte a NIfTI, detecta nodulos, recorta
un cubo alrededor de cada candidato y borra las CT. Solo quedan los JSON, los recortes y
un registro de avance, asi que la muestra completa (260 GB) pasa por el programa sin
ocupar ese espacio. Volver a correrlo sigue donde se quedo y reintenta lo que fallo.

Uso:
    python backend/inference/procesar_lotes.py --limit 8

Importante:
- --limit es obligatorio. Cada lote ocupa disco temporal (unos 2 GB con --workers 1).
- Para correr a escala necesita GPU: en CPU el detector tarda varios minutos por CT.
- Con --keep-images las CT convertidas no se borran (por ejemplo, para el visor 3D).
  Las que ya existian antes de la corrida nunca se borran.
- Los recortes son para entrenar un clasificador; por si solos NO son un diagnostico.
- outputs/ y data/ estan en .gitignore. No subir resultados a GitHub.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from convert_to_nifti import AIR_HU, SERIES_PER_WORKER, SPACING, convert_batch, limit_threads

CROP = (64, 64, 40)  # voxeles en x, y, z: unos 45 x 45 x 50 mm


def crop_candidate(volume: np.ndarray, center: list[int]) -> np.ndarray:
    """Recorta un cubo centrado en el candidato; lo que queda fuera de la CT es aire."""
    crop = np.full(CROP[::-1], AIR_HU, dtype=np.int16)
    source, target = [], []
    for middle, size, limit in zip(center[::-1], CROP[::-1], volume.shape):
        first = middle - size // 2
        start, stop = max(first, 0), min(first + size, limit)
        source.append(slice(start, stop))
        target.append(slice(start - first, stop - first))
    crop[tuple(target)] = volume[tuple(source)]
    return crop


def dicom_slice(z: int, info: dict) -> int:
    """Pasa un corte del NIfTI al numero de corte DICOM, que es el que anota el radiologo."""
    original = z * SPACING[2] / info["espaciado_z_original_mm"]
    step = (info["instancia_ultimo_corte"] - info["instancia_primer_corte"]) / (info["cortes_originales"] - 1)
    return round(info["instancia_primer_corte"] + original * step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/monai_dataset/cohorte_cancer_12m.parquet"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/monai"), help="Carpeta temporal de las CT.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/modelos"))
    parser.add_argument("--split", choices=["train", "validation", "test"], help="Por defecto, todos.")
    parser.add_argument("--limit", type=int, required=True, help="Maximo de series a procesar.")
    parser.add_argument("--workers", type=int, default=1, help="Series que se convierten a la vez.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Puntaje minimo para contar como nodulo.")
    parser.add_argument("--crop-threshold", type=float, default=0.1, help="Puntaje minimo para guardar el recorte.")
    parser.add_argument("--keep-images", action="store_true", help="No borrar las CT convertidas.")
    args = parser.parse_args()

    cohort = pd.read_parquet(args.input)
    sample = cohort[cohort["en_muestra"]]
    if args.split:
        sample = sample[sample["split"].eq(args.split)]

    progress_path = args.output_dir / "lotes_progreso.csv"
    progress = pd.read_csv(progress_path, dtype={"PatientID": "string"}) if progress_path.exists() else pd.DataFrame()
    done = set(progress["SeriesInstanceUID"]) if len(progress) else set()
    pending = sample[~sample["SeriesInstanceUID"].isin(done)]
    selected = pending.head(args.limit)
    print(
        f"Muestra: {len(sample)} | pendientes: {len(pending)} | ahora: {len(selected)} "
        f"(~{selected['series_size_MB'].sum() / 1024:.1f} GB de descarga)"
    )
    if selected.empty:
        return

    # Se importan aqui para que los procesos de conversion no carguen los modelos.
    from idc_index import IDCClient
    from lungmask import LMInferer

    from detectar_nodulos import detect_series, load_detector, save_report

    args.data_dir.mkdir(parents=True, exist_ok=True)
    client = IDCClient()
    detector, device = load_detector(args.model_dir)
    lung_model = LMInferer(modelname="R231", tqdm_disable=True)
    lobe_model = LMInferer(modelname="LTRCLobes", tqdm_disable=True)
    print("Dispositivo:", device)

    batch_size = args.workers * SERIES_PER_WORKER
    failures, processed = [], 0
    start = time.time()

    with ProcessPoolExecutor(args.workers, initializer=limit_threads, initargs=(args.workers,)) as pool:
        for first in range(0, len(selected), batch_size):
            batch = selected.iloc[first:first + batch_size]
            existed = [(args.data_dir / p).exists() for p in batch["ruta_imagen"]]
            rows = []
            for row, info, keep in zip(batch.itertuples(), convert_batch(client, batch, args.data_dir, pool), existed):
                ids = {"PatientID": row.PatientID, "SeriesInstanceUID": row.SeriesInstanceUID}
                image_path = args.data_dir / row.ruta_imagen
                try:
                    if info["error"]:
                        raise ValueError(info["error"])
                    began = time.time()
                    candidates = detect_series(image_path, detector, device, lung_model, lobe_model)

                    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
                    crops = []
                    for candidate in candidates:
                        candidate["corte_dicom"] = dicom_slice(candidate["centro_voxel"][2], info)
                        if candidate["puntaje"] >= args.crop_threshold and candidate["en_pulmon"]:
                            candidate["recorte"] = len(crops)  # posicion dentro del archivo de recortes
                            crops.append(crop_candidate(volume, candidate["centro_voxel"]))
                    if crops:
                        crop_path = args.output_dir / "recortes" / row.PatientID / f"{row.SeriesInstanceUID}.npz"
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(crop_path, recortes=np.stack(crops))

                    report = save_report(row, candidates, args.threshold,
                                         args.output_dir / "nodulos" / row.PatientID / f"{row.SeriesInstanceUID}.json")
                    rows.append({
                        **ids, "split": row.split, "series_size_MB": row.series_size_MB,
                        **{k: v for k, v in info.items() if k != "error"},
                        "candidatos": len(candidates), "nodulos": len(report["nodulos"]),
                        "recortes": len(crops), "segundos_deteccion": round(time.time() - began),
                    })
                except Exception as error:  # Una serie mala no debe detener las demas.
                    failures.append({**ids, "error": str(error)})
                finally:
                    if not (keep or args.keep_images):
                        image_path.unlink(missing_ok=True)

            # El avance se guarda en cada lote, por si la corrida se interrumpe.
            progress = pd.concat([progress, pd.DataFrame(rows)], ignore_index=True)
            progress.to_csv(progress_path, index=False)
            processed += len(batch)
            print(f"[{processed}/{len(selected)}] fallidas: {len(failures)} | {time.time() - start:.0f} s", flush=True)

    # Solo guarda los fallos de esta corrida: al reintentar, el archivo se actualiza.
    log = args.output_dir / "lotes_fallidas.csv"
    pd.DataFrame(failures, columns=["PatientID", "SeriesInstanceUID", "error"]).to_csv(log, index=False)
    if failures:
        print(f"{len(failures)} series fallaron; detalle en {log}. Vuelve a correr el script para reintentarlas.")

    total = cohort[cohort["en_muestra"]]
    finished = total[total["SeriesInstanceUID"].isin(set(progress["SeriesInstanceUID"]))]
    print(
        f"Procesadas: {len(finished)}/{len(total)} series | {finished['series_size_MB'].sum() / 1024:.1f} de "
        f"{total['series_size_MB'].sum() / 1024:.1f} GB (DICOM) | nodulos: {int(progress['nodulos'].sum())} | "
        f"recortes: {int(progress['recortes'].sum())}"
    )


if __name__ == "__main__":
    main()
