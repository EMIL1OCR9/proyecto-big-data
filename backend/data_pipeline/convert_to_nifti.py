"""
LungAI — descarga series de la muestra y las guarda en NIfTI para MONAI.

Usa la tabla de prepare_monai_dataset.py. Baja las series de IDC por lotes, las
convierte en paralelo y borra sus DICOM. Volver a correrlo salta lo ya convertido y
reintenta lo que fallo.

Uso:
    python backend/data_pipeline/convert_to_nifti.py --split test --limit 10
    python backend/data_pipeline/convert_to_nifti.py --limit 2841 --workers 8

Importante:
- --limit es obligatorio para no llenar el disco por accidente.
- Antes de remuestrear se revisa la geometria de la serie y el relleno fuera del campo
  de vision se cambia por aire. Ambas partes vienen de procesar_lote_t0.py (Iris).
  Ademas del PixelPaddingValue, todo lo que esta debajo de -1024 HU se toma como relleno.
- Las imagenes se guardan en HU (int16) y se remuestrean una sola vez aqui, con
  SimpleITK, al espaciado del detector. La ventana de HU se aplica despues, en MONAI.
- Cada worker usa ~2 GB de RAM y cada lote ocupa disco temporal. Usa --workers 1 en
  una PC.
- data/ esta en .gitignore. No subir imagenes a GitHub.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

SPACING = (0.703125, 0.703125, 1.25)  # mm, el del detector de MONAI
SERIES_PER_WORKER = 4  # tamano de cada lote de descarga
AIR_HU = -1000  # valor que reemplaza al relleno
SUSPICIOUS_HU = -1024  # nada real queda debajo: tambien se toma como relleno


def limit_threads(workers: int) -> None:
    """Evita que cada proceso use todos los nucleos cuando hay varios workers."""
    if workers > 1:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def normalize_padding(value: int, bits: int, representation: int) -> int:
    """Lee el valor de relleno respetando BitsStored y PixelRepresentation.

    Algunos DICOM guardan 63536 en el tag para un pixel con signo de -2000.
    """
    modulo = 1 << bits
    value = int(value) % modulo
    if representation == 1 and value >= modulo >> 1:
        value -= modulo
    return value


def read_headers(files: list[str]) -> tuple[list[tuple[float, float] | None], float]:
    """Revisa la geometria y regresa el rango de relleno en HU de cada corte."""
    positions, padding = [], []
    reference = None

    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        spacing = np.asarray(ds.PixelSpacing, dtype=float)
        shape = (int(ds.Rows), int(ds.Columns))

        if reference is None:
            normal = np.cross(orientation[:3], orientation[3:])
            if not 0.99 <= np.linalg.norm(normal) <= 1.01:
                raise ValueError("ImageOrientationPatient no es ortonormal")
            reference = (orientation, spacing, shape, normal)
        elif (
            not np.allclose(orientation, reference[0], atol=1e-3, rtol=0)
            or not np.allclose(spacing, reference[1], atol=1e-3, rtol=0)
            or shape != reference[2]
        ):
            raise ValueError("la serie mezcla orientacion, espaciado o tamanos")
        positions.append(float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), reference[3])))

        pad = ds.get("PixelPaddingValue")
        if pad is None:
            padding.append(None)
            continue
        # El relleno se declara en valores crudos: se pasa a HU con el rescale del corte.
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        bits, representation = int(ds.BitsStored), int(ds.PixelRepresentation)
        limit = ds.get("PixelPaddingRangeLimit", pad)
        raw = [normalize_padding(v, bits, representation) for v in (pad, limit)]
        padding.append(tuple(sorted(v * slope + intercept for v in raw)))

    steps = np.diff(positions)
    if not (np.all(steps > 0.01) or np.all(steps < -0.01)):
        raise ValueError("cortes duplicados o desordenados")
    z_spacing = float(np.median(np.abs(steps)))
    if np.any(np.abs(np.abs(steps) - z_spacing) > max(0.25 * z_spacing, 0.2)):
        raise ValueError("espaciado Z irregular")
    return padding, z_spacing


def convert_series(job: dict) -> dict:
    """Convierte una serie ya descargada. Regresa el error o su resumen de calidad."""
    try:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(job["source"], job["uid"])
        if len(files) != job["slices"]:
            raise ValueError(f"{len(files)} cortes, se esperaban {job['slices']}")
        padding, z_spacing = read_headers(files)

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        image = reader.Execute()

        # El corte k del arreglo es files[k]: cada uno usa su propio rango de relleno.
        hu = sitk.GetArrayFromImage(image)
        declared = np.zeros(hu.shape, dtype=bool)
        for k, interval in enumerate(padding):
            if interval is not None:
                declared[k] = (hu[k] >= interval[0]) & (hu[k] <= interval[1])
        # Hay series cuyo tag no coincide con el relleno real de la imagen.
        mask = declared | (hu < SUSPICIOUS_HU)
        hu[mask] = AIR_HU

        clean = sitk.GetImageFromArray(hu.astype(np.int16))
        clean.CopyInformation(image)
        size = [round(n * s / t) for n, s, t in zip(clean.GetSize(), clean.GetSpacing(), SPACING)]
        clean = sitk.Resample(clean, size, sitk.Transform(), sitk.sitkLinear,
                              clean.GetOrigin(), SPACING, clean.GetDirection(), AIR_HU)

        destination = Path(job["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Solo aparece con su nombre final si termino de escribirse.
        partial = destination.with_name(destination.name.replace(".nii.gz", ".parcial.nii.gz"))
        sitk.WriteImage(clean, str(partial), useCompression=True)
        partial.replace(destination)

        tagged = sum(interval is not None for interval in padding)
        return {
            "error": None,
            "relleno": "tag" if tagged == len(padding) else "heuristica" if tagged == 0 else "mixto",
            "porcentaje_relleno": round(100 * float(mask.mean()), 3),
            "porcentaje_fuera_del_tag": round(100 * float((mask & ~declared).mean()), 3),
            "espaciado_z_original_mm": round(z_spacing, 3),
        }
    except Exception as error:  # Una serie mala no debe detener las demas.
        return {"error": str(error)}


def convert_batch(client, batch: pd.DataFrame, data_dir: Path, pool) -> list[dict]:
    """Descarga un lote junto y lo convierte con los workers."""
    temporary = Path(tempfile.mkdtemp(dir=data_dir))
    try:
        client.download_from_selection(
            downloadDir=str(temporary),
            seriesInstanceUID=batch["SeriesInstanceUID"].tolist(),
            dirTemplate="%SeriesInstanceUID",
            show_progress_bar=False,
            quiet=True,
        )
        jobs = [
            {
                "source": str(temporary / row.SeriesInstanceUID),
                "uid": row.SeriesInstanceUID,
                "slices": row.instanceCount,
                "destination": str(data_dir / row.ruta_imagen),
            }
            for row in batch.itertuples()
        ]
        return list(pool.map(convert_series, jobs))
    except Exception as error:
        return [{"error": f"lote fallido: {error}"}] * len(batch)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/monai_dataset/cohorte_cancer_12m.parquet"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/monai"))
    parser.add_argument("--split", choices=["train", "validation", "test"], help="Por defecto, todos.")
    parser.add_argument("--limit", type=int, required=True, help="Maximo de series a convertir.")
    parser.add_argument("--workers", type=int, default=1, help="Series que se procesan a la vez.")
    args = parser.parse_args()

    from idc_index import IDCClient

    args.data_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(args.input)
    sample = cohort[cohort["en_muestra"]]
    if args.split:
        sample = sample[sample["split"].eq(args.split)]

    pending = sample[~sample["ruta_imagen"].map(lambda p: (args.data_dir / p).exists())]
    selected = pending.head(args.limit)
    print(
        f"Muestra: {len(sample)} | pendientes: {len(pending)} | ahora: {len(selected)} "
        f"(~{selected['series_size_MB'].sum() / 1024:.1f} GB de descarga)"
    )

    client = IDCClient()
    batch_size = args.workers * SERIES_PER_WORKER
    failures, converted, done = [], [], 0
    start = time.time()

    with ProcessPoolExecutor(args.workers, initializer=limit_threads, initargs=(args.workers,)) as pool:
        for first in range(0, len(selected), batch_size):
            batch = selected.iloc[first:first + batch_size]
            for row, result in zip(batch.itertuples(), convert_batch(client, batch, args.data_dir, pool)):
                ids = {"PatientID": row.PatientID, "SeriesInstanceUID": row.SeriesInstanceUID}
                if result["error"]:
                    failures.append({**ids, "error": result["error"]})
                else:
                    converted.append({**ids, **{k: v for k, v in result.items() if k != "error"}})
            done += len(batch)
            print(f"[{done}/{len(selected)}] fallidas: {len(failures)} | {time.time() - start:.0f} s", flush=True)

    # El resumen de calidad se acumula entre corridas; los fallos solo son de esta corrida.
    summary = args.data_dir / "conversion_resumen.csv"
    if converted:
        pd.DataFrame(converted).to_csv(summary, mode="a", header=not summary.exists(), index=False)
    log = args.data_dir / "conversion_fallidas.csv"
    pd.DataFrame(failures, columns=["PatientID", "SeriesInstanceUID", "error"]).to_csv(log, index=False)
    if failures:
        print(f"{len(failures)} series fallaron; detalle en {log}. Vuelve a correr el script para reintentarlas.")


if __name__ == "__main__":
    main()
