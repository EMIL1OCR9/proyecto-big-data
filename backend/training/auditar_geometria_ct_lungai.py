"""LungAI: auditoría de geometría CT de TRAIN/VALIDATION (sin test).

Solo consulta las cabeceras NIfTI, sin leer píxeles ni modificar archivos.
Calcula la resolución espacial efectiva al reducir un volumen completo a
96 x 96 x 64, tal como hace el entrenador 3D ampliado.

Ejecutar desde la raíz del repositorio con ~/lungai-venv:
    python backend/training/auditar_geometria_ct_lungai.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

MANIFIESTO = Path('outputs/monai_dataset/lote_ct_3d_ampliado.parquet')
BASE = Path('data/monai').resolve()
DESTINO = np.array([96.0, 96.0, 64.0], dtype=np.float64)


def mediana(vector, decimales=2):
    return f'{np.median(vector):.{decimales}f}' if len(vector) else 'n/d'


def main():
    columnas = ['split', 'etiqueta_cancer_12m', 'ruta_imagen', 'SeriesDescription']
    datos = pd.read_parquet(MANIFIESTO, columns=columnas)
    datos = datos.loc[datos['split'].isin(['train', 'validation'])].copy()
    if not datos['etiqueta_cancer_12m'].isin([0, 1]).all():
        raise SystemExit('Se encontraron etiquetas inesperadas.')

    resultados = []
    ausentes = 0
    ilegibles = 0
    geometria_invalida = 0
    t0_opa_incorrectas = 0

    for fila in datos.itertuples(index=False):
        descrip = str(fila.SeriesDescription).split(',')
        if len(descrip) < 2 or descrip[0] != '0' or descrip[1] != 'OPA':
            t0_opa_incorrectas += 1

        relativa = Path(str(fila.ruta_imagen).replace('\\', '/'))
        if relativa.is_absolute() or '..' in relativa.parts:
            raise SystemExit('Ruta fuera del directorio autorizado.')
        archivo = (BASE / relativa).resolve()
        if not archivo.is_relative_to(BASE):
            raise SystemExit('Ruta fuera del directorio autorizado.')
        if not archivo.is_file() or archivo.stat().st_size <= 0:
            ausentes += 1
            continue

        lector = sitk.ImageFileReader()
        lector.SetFileName(str(archivo))
        try:
            lector.ReadImageInformation()  # solo cabecera; no carga volumen CT
            tam = np.asarray(lector.GetSize(), dtype=np.float64)
            esp = np.asarray(lector.GetSpacing(), dtype=np.float64)
        except RuntimeError:
            ilegibles += 1
            continue
        if (len(tam) != 3 or len(esp) != 3 or
            not np.isfinite(tam).all() or not np.isfinite(esp).all() or
            np.any(tam <= 1) or np.any(esp <= 0)):
            geometria_invalida += 1
            continue

        fov = (tam - 1) * esp
        nuevo = fov / (DESTINO - 1)
        resultados.append({
            'split': fila.split,
            'etiqueta': int(fila.etiqueta_cancer_12m),
            'fov_x': fov[0], 'fov_y': fov[1], 'fov_z': fov[2],
            'nuevo_x': nuevo[0], 'nuevo_y': nuevo[1], 'nuevo_z': nuevo[2],
            'original_z': esp[2],
            'grueso': int(np.any(nuevo > 3.0)),
        })

    datos_geo = pd.DataFrame(resultados)
    print('\nAUDITORIA GEOMETRICA CT — SOLO TRAIN Y VALIDATION')
    print('Objetivo de remuestreo: 96 x 96 x 64 voxeles (volumen completo)')
    print(f'CT locales auditadas: {len(datos_geo)}')
    print(f'No disponibles: {ausentes}; cabeceras ilegibles: {ilegibles}; geometria invalida: {geometria_invalida}')
    print(f'Series sin identificacion T0 OPA: {t0_opa_incorrectas}')
    print('Test reservado: no se analizaron sus imagenes ni sus metricas.')

    if datos_geo.empty:
        print('Sin CT auditables. No se modifico ningun archivo.')
        return

    print('\nRESUMEN POR GRUPO (etiqueta 1 = cancer diagnosticado en 12 meses)')
    for split in ('train', 'validation'):
        for etiqueta in (0, 1):
            g = datos_geo[(datos_geo['split'] == split) & (datos_geo['etiqueta'] == etiqueta)]
            if g.empty:
                print(f'{split} clase {etiqueta}: 0 CT')
                continue
            print(
                f'{split} clase {etiqueta}: {len(g)} CT; '
                f'FOV XYZ mediano (mm) '
                f'{mediana(g.fov_x)}/{mediana(g.fov_y)}/{mediana(g.fov_z)}; '
                f'voxel tras reduccion XYZ (mm) '
                f'{mediana(g.nuevo_x)}/{mediana(g.nuevo_y)}/{mediana(g.nuevo_z)}; '
                f'espaciado Z original mediano {mediana(g.original_z)} mm'
            )

    print('\nCONCLUSION DEL CHEQUEO')
    print(f'CT con al menos un eje > 3 mm/voxel tras reduccion: {int(datos_geo.grueso.sum())}/{len(datos_geo)}')
    print('Los detalles pequenos pueden perderse al reducir el volumen completo.')
    print('Solo se consultaron cabeceras. No se leyo el test ni se modificaron archivos.')


if __name__ == '__main__':
    main()
