"""Descarga una muestra de tomografias completas del NLST desde la API publica NBIA de TCIA.

Uso: uv run python nlst/descargar_nlst.py   (ajusta LIMITE_GB abajo; volver a correrlo reanuda)
"""
import csv
import hashlib
import io
import random
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd
import pydicom
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

LIMITE_GB = 1                # tamano en disco de los DICOM (1 GB = 1e9 bytes); prueba 10, 50, 250...
PROPORCION_CANCER = 0.5      # fraccion de pacientes con cancer confirmado en la muestra (None = aleatorio)
SEMILLA = 42
KERNELS_SUAVES = ("B30f", "STANDARD", "FC10", "C")  # preferidos por estudio (Siemens, GE, Toshiba, Philips)

# NLST tiene su propio servidor NBIA (no el general services.cancerimagingarchive.net)
API = "https://nlst.cancerimagingarchive.net/nbia-api/services/v1"
CLINICO_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/package-nlst-780.2021-05-28.zip"
DATOS = Path(__file__).resolve().parents[1] / "data"
SALIDA = DATOS / "NLST_DICOM"
CLINICO = DATOS / "NLST_clinico"
PLAN = SALIDA / "plan.csv"
REPORTE = SALIDA / "series_descargadas.csv"
TIMEOUT = (15, 120)          # (conexion, lectura entre bloques) en segundos
INTENTOS = 5
BLOQUE = 1024 * 1024

sesion = requests.Session()
sesion.mount("https://", HTTPAdapter(max_retries=Retry(
    total=INTENTOS, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))))


def api(endpoint, **params):
    r = sesion.get(f"{API}/{endpoint}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json() if r.text else []  # NBIA responde cuerpo vacio cuando no hay resultados


def etiquetas_clinicas():
    """Tabla de participantes del paquete clinico publico (CC BY 4.0, sin registro)."""
    csv_prsn = CLINICO / "nlst_780_prsn_idc_20210527.csv"
    if not csv_prsn.exists():
        CLINICO.mkdir(exist_ok=True)
        r = sesion.get(CLINICO_URL, timeout=TIMEOUT)
        r.raise_for_status()
        interno = zipfile.ZipFile(io.BytesIO(r.content))
        entrega = zipfile.ZipFile(io.BytesIO(interno.read("nlst_780/nlst780.idc.delivery.052821.zip")))
        for nombre in entrega.namelist():
            datos = entrega.read(nombre)
            if nombre.endswith(".csv.zip"):
                zipfile.ZipFile(io.BytesIO(datos)).extractall(CLINICO)
            else:
                (CLINICO / nombre).write_bytes(datos)  # diccionarios .rtf
    return pd.read_csv(csv_prsn, usecols=["pid", "can_scr", "cancyr", "candx_days"],
                       dtype="Int64").set_index("pid")


def orden_pacientes(clin):
    """Orden determinista de pacientes; intercala casos con cancer si PROPORCION_CANCER no es None."""
    rng = random.Random(SEMILLA)
    pids = sorted(int(p["PatientId"]) for p in api("getPatient", Collection="NLST"))
    if PROPORCION_CANCER is None:
        rng.shuffle(pids)
        return pids
    cancer = [p for p in pids if clin.at[p, "can_scr"] > 0]
    sanos = [p for p in pids if clin.at[p, "can_scr"] == 0]
    rng.shuffle(cancer)
    rng.shuffle(sanos)
    orden, n_cancer = [], 0
    while cancer or sanos:
        toca_cancer = cancer and (not sanos or n_cancer < PROPORCION_CANCER * (len(orden) + 1))
        n_cancer += bool(toca_cancer)
        orden.append((cancer if toca_cancer else sanos).pop())
    return orden


def series_del_paciente(pid):
    """Una serie axial original (OPA) por estudio: kernel suave si hay, si no la de mas cortes."""
    por_estudio = {}
    for s in api("getSeries", Collection="NLST", PatientID=pid):
        campos = (s.get("SeriesDescription") or "").split(",")  # p. ej. "1,OPA,SE,VZOOM,B30f,..."
        if s.get("Modality") != "CT" or len(campos) < 5 or campos[1] != "OPA":
            continue
        s["study_yr"] = f"T{campos[0]}"
        s["kernel"] = campos[4]
        por_estudio.setdefault(s["StudyInstanceUID"], []).append(s)
    return [max(v, key=lambda s: (s["kernel"] in KERNELS_SUAVES, s["ImageCount"], -s["SeriesNumber"]))
            for v in por_estudio.values()]


def planear(clin):
    """Agrega pacientes completos al plan hasta llenar LIMITE_GB. El plan se conserva entre corridas."""
    limite = LIMITE_GB * 1e9
    plan = pd.read_csv(PLAN, dtype=str) if PLAN.exists() else pd.DataFrame()
    total = plan["FileSize"].astype(int).sum() if len(plan) else 0
    en_plan = set(plan["PatientID"]) if len(plan) else set()
    nuevas, rechazos = [], 0
    with tqdm(desc="Planeando", unit=" pac") as barra:
        for pid in orden_pacientes(clin):
            if total >= limite or rechazos >= 30:  # 30 seguidos sin caber: ya no queda espacio util
                break
            if str(pid) in en_plan:
                continue
            series = series_del_paciente(pid)
            tam = sum(s["FileSize"] for s in series)
            barra.update()
            if not series or total + tam > limite:
                rechazos += bool(series)
                continue
            rechazos = 0
            total += tam
            et = clin.loc[pid]
            nuevas += [{"PatientID": str(pid), "StudyInstanceUID": s["StudyInstanceUID"],
                        "SeriesInstanceUID": s["SeriesInstanceUID"], "study_yr": s["study_yr"],
                        "kernel": s["kernel"], "ImageCount": s["ImageCount"], "FileSize": s["FileSize"],
                        "SeriesDescription": s["SeriesDescription"], "can_scr": et.can_scr,
                        "cancyr": et.cancyr, "candx_days": et.candx_days} for s in series]
            barra.set_postfix(GB=f"{total / 1e9:.2f}")
    if nuevas:
        plan = pd.concat([plan, pd.DataFrame(nuevas)], ignore_index=True)
        SALIDA.mkdir(exist_ok=True)
        plan.to_csv(PLAN, index=False)
    return plan


def descargar_serie(fila, destino):
    """Baja el ZIP por streaming, lo valida (MD5 + DICOM) y lo mueve a destino solo si todo esta bien."""
    uid = fila["SeriesInstanceUID"]
    parcial = destino.with_name(destino.name + ".zip.part")
    temporal = destino.with_name(destino.name + ".tmp")
    for intento in range(1, INTENTOS + 1):
        try:
            # El ZIP se genera al vuelo: sin Content-Length y sin soporte de Range, asi que
            # cada intento empieza la serie desde cero.
            with sesion.get(f"{API}/getImageWithMD5Hash", params={"SeriesInstanceUID": uid},
                            stream=True, timeout=TIMEOUT) as r, open(parcial, "wb") as f, \
                    tqdm(desc=f"  {uid[-12:]}", unit="B", unit_scale=True, leave=False) as barra:
                r.raise_for_status()
                for bloque in r.iter_content(BLOQUE):
                    f.write(bloque)
                    barra.update(len(bloque))
            n = validar_y_extraer(parcial, temporal, fila)
            temporal.rename(destino)
            return n
        except Exception as e:  # red, ZIP corrupto, MD5 o DICOM invalido: se reintenta la serie
            tqdm.write(f"  intento {intento}/{INTENTOS} fallo en {uid}: {e}")
            shutil.rmtree(temporal, ignore_errors=True)
            if intento == INTENTOS:
                raise
            time.sleep(2 ** intento)
        finally:
            parcial.unlink(missing_ok=True)


def validar_y_extraer(zip_path, carpeta, fila):
    shutil.rmtree(carpeta, ignore_errors=True)
    carpeta.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        md5 = dict(csv.reader(io.TextIOWrapper(z.open("md5hashes.csv"))))
        dcms = [n for n in z.namelist() if n.endswith(".dcm")]
        if len(dcms) != int(fila["ImageCount"]):
            raise ValueError(f"{len(dcms)} archivos, se esperaban {fila['ImageCount']}")
        for nombre in dcms:
            h = hashlib.md5()
            with z.open(nombre) as src, open(carpeta / nombre, "wb") as dst:
                while bloque := src.read(BLOQUE):
                    h.update(bloque)
                    dst.write(bloque)
            if h.hexdigest() != md5.get(nombre):
                raise ValueError(f"MD5 no coincide en {nombre}")
            ds = pydicom.dcmread(carpeta / nombre)  # falla si no es DICOM valido
            if "PixelData" not in ds or ds.SeriesInstanceUID != fila["SeriesInstanceUID"]:
                raise ValueError(f"{nombre} no trae imagen de la serie esperada")
    return len(dcms)


def main():
    clin = etiquetas_clinicas()
    plan = planear(clin)
    print(f"Plan: {plan.PatientID.nunique()} pacientes, {len(plan)} series, "
          f"{plan.FileSize.astype(int).sum() / 1e9:.2f} GB (limite {LIMITE_GB} GB)")
    filas, fallidas = [], []
    with tqdm(total=plan.FileSize.astype(int).sum(), unit="B", unit_scale=True, desc="En disco") as total:
        for fila in plan.to_dict("records"):
            destino = SALIDA / fila["PatientID"] / fila["StudyInstanceUID"] / fila["SeriesInstanceUID"]
            if not destino.exists():  # solo existe si la serie quedo completa y validada
                destino.parent.mkdir(parents=True, exist_ok=True)
                try:
                    descargar_serie(fila, destino)
                except Exception as e:
                    fallidas.append(fila["SeriesInstanceUID"])
                    tqdm.write(f"Se omite {fila['SeriesInstanceUID']}: {e}")
                    continue
            archivos = list(destino.glob("*.dcm"))
            filas.append({**fila, "ImageCount": len(archivos),
                          "bytes_descargados": sum(a.stat().st_size for a in archivos)})
            total.update(int(fila["FileSize"]))
    pd.DataFrame(filas).to_csv(REPORTE, index=False)
    print(f"Listo: {len(filas)} series en {SALIDA}, reporte en {REPORTE.name}")
    if fallidas:
        print(f"{len(fallidas)} series fallaron; vuelve a correr el script para reintentarlas.")


if __name__ == "__main__":
    main()
