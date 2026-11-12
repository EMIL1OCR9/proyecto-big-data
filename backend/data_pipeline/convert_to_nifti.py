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
  de vision se cambia por aire. Ambas partes vienen del preprocesamiento de Iris
  (procesar_lote_t0.py, en la rama Iris).
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
NIFTI_RATIO = 0.85  # el NIfTI ocupa ~81 % del DICOM (piloto de 54 series), con margen
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


def same_pixels(first: str, second: str) -> bool:
    """Compara la imagen de dos cortes DICOM."""
    return np.array_equal(
        sitk.GetArrayFromImage(sitk.ReadImage(first)),
        sitk.GetArrayFromImage(sitk.ReadImage(second)),
    )


def read_headers(files: list[str]) -> tuple[list[str], list[tuple[float, float] | None], float, int, tuple[int, int]]:
    """Revisa la geometria, ordena los cortes y quita los repetidos.

    Regresa los archivos en orden, el rango de relleno en HU de cada uno, el espaciado Z,
    cuantos cortes repetidos se quitaron y el InstanceNumber del primer y del ultimo corte.
    """
    slices = []
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
        position = float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), reference[3]))

        padding = None
        pad = ds.get("PixelPaddingValue")
        if pad is not None:
            # El relleno se declara en valores crudos: se pasa a HU con el rescale del corte.
            slope = float(ds.get("RescaleSlope", 1))
            intercept = float(ds.get("RescaleIntercept", 0))
            bits, representation = int(ds.BitsStored), int(ds.PixelRepresentation)
            limit = ds.get("PixelPaddingRangeLimit", pad)
            raw = [normalize_padding(v, bits, representation) for v in (pad, limit)]
            padding = tuple(sorted(v * slope + intercept for v in raw))
        slices.append((position, path, padding, int(ds.InstanceNumber)))

    # GDCM no ordena las series con cortes repetidos, asi que se ordena aqui.
    slices.sort(key=lambda s: s[0])
    unique = [slices[0]]
    for current in slices[1:]:
        if current[0] - unique[-1][0] > 0.01:
            unique.append(current)
        elif not same_pixels(current[1], unique[-1][1]):
            raise ValueError("dos cortes en la misma posicion con imagen distinta")

    
    steps = np.diff([s[0] for s in unique])
    z_spacing = float(np.median(steps))

    tolerancia = max(0.25 * z_spacing, 0.2)

    irregulares = np.flatnonzero(
        np.abs(steps - z_spacing) > tolerancia
    )

    if len(irregulares) > 0:
        ejemplos = [
            round(float(steps[i]), 3)
            for i in irregulares[:10]
        ]

        raise ValueError(
            f"espaciado Z irregular | "
            f"mediana={z_spacing:.3f} mm | "
            f"minimo={steps.min():.3f} mm | "
            f"maximo={steps.max():.3f} mm | "
            f"intervalos_irregulares={len(irregulares)} "
            f"de {len(steps)} | "
            f"primeros_intervalos={ejemplos}"
        )

    instances = (unique[0][3], unique[-1][3])
    return [s[1] for s in unique], [s[2] for s in unique], z_spacing, len(slices) - len(unique), instances


def convert_series(job: dict) -> dict:
    """Convierte una serie ya descargada. Regresa el error o su resumen de calidad."""
    try:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(job["source"], job["uid"])
        if len(files) != job["slices"]:
            raise ValueError(f"{len(files)} cortes, se esperaban {job['slices']}")
        files, padding, z_spacing, duplicates, instances = read_headers(files)

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
            "cortes_repetidos_quitados": duplicates,
            # Con esto se recupera el numero de corte DICOM, que es el que anota el radiologo.
            "cortes_originales": len(files),
            "instancia_primer_corte": instances[0],
            "instancia_ultimo_corte": instances[1],
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


def report_progress(sample: pd.DataFrame, data_dir: Path) -> pd.Series:
    """Muestra cuanto de la muestra ya esta convertido y regresa que filas lo estan."""
    done = sample["ruta_imagen"].map(lambda p: (data_dir / p).exists())
    on_disk = sum((data_dir / p).stat().st_size for p in sample.loc[done, "ruta_imagen"]) / 1024**3
    print(
        f"Convertidas: {int(done.sum())}/{len(sample)} series | descargado: "
        f"{sample.loc[done, 'series_size_MB'].sum() / 1024:.1f} de {sample['series_size_MB'].sum() / 1024:.1f} GB "
        f"(DICOM) | en disco: {on_disk:.1f} GB"
    )
    return done


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

    pending = sample[~report_progress(sample, args.data_dir)]
    selected = pending.head(args.limit)
    print(
        f"Muestra: {len(sample)} | pendientes: {len(pending)} | ahora: {len(selected)} "
        f"(~{selected['series_size_MB'].sum() / 1024:.1f} GB de descarga)"
    )

    batch_size = args.workers * SERIES_PER_WORKER
    # Espacio para los NIfTI de esta corrida, mas un lote de DICOM temporales.
    needed = (
        selected["series_size_MB"].sum() * NIFTI_RATIO
        + selected["series_size_MB"].nlargest(batch_size).sum()
    ) / 1024
    free = shutil.disk_usage(args.data_dir).free / 1024**3
    if needed > free:
        raise SystemExit(f"Hacen falta ~{needed:.1f} GB y solo hay {free:.1f} GB libres. Baja --limit o libera espacio.")

    client = IDCClient()
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
        rows = pd.DataFrame(converted)
        if summary.exists():  # Se reescribe para que las columnas nuevas no queden desalineadas.
            rows = pd.concat([pd.read_csv(summary, dtype={"PatientID": "string"}), rows], ignore_index=True)
        rows.to_csv(summary, index=False)
    log = args.data_dir / "conversion_fallidas.csv"
    pd.DataFrame(failures, columns=["PatientID", "SeriesInstanceUID", "error"]).to_csv(log, index=False)
    if failures:
        print(f"{len(failures)} series fallaron; detalle en {log}. Vuelve a correr el script para reintentarlas.")
    report_progress(sample, args.data_dir)


if __name__ == "__main__":
    main()
