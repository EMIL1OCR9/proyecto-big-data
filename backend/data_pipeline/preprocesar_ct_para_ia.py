"""LungAI | Paso 10: preprocesar una CT en HU para experimentos de Deep Learning.

Entrada: volumen_hu.npz generado por preprocesar_primera_ct.py (paso 9).
No descarga DICOM, no usa etiquetas de cáncer, no entrena ni diagnostica.

Ejecutar DESDE LA RAÍZ del proyecto:
    uv add scipy
    uv run python backend/data_pipeline/preprocesar_ct_para_ia.py

Opciones:
    uv run python backend/data_pipeline/preprocesar_ct_para_ia.py --espaciado-objetivo 2 1 1

Nota: el NPZ del paso 9 no guarda PixelPaddingValue/PixelPaddingRangeLimit.
Valores HU menores de -1024 se TRATAN HEURÍSTICAMENTE como valores
sospechosos de relleno y se reemplazan con -1000 HU, sin afirmarlo como
una identificación DICOM definitiva. Para una implementación clínica habría
que procesar los atributos DICOM de relleno antes de la conversión a HU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


ENTRADA_PREDETERMINADA = Path("outputs/preprocesamiento/primer_ct/volumen_hu.npz")
SALIDA_PREDETERMINADA = Path("outputs/preprocesamiento/primer_ct_dl")
HU_AIRE = -1000.0
UMBRAL_HU_SOSPECHOSO = -1024.0


def leer_npz(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo: {ruta}. Ejecuta antes el paso 9.")
    with np.load(ruta, allow_pickle=False) as datos:
        requeridas = {"volumen_hu", "espaciado_mm_zyx"}
        faltantes = requeridas.difference(datos.files)
        if faltantes:
            raise ValueError(f"El NPZ no contiene {sorted(faltantes)}; claves disponibles: {datos.files}.")
        hu = np.asarray(datos["volumen_hu"], dtype=np.float32)
        espaciado = np.asarray(datos["espaciado_mm_zyx"], dtype=np.float64)

    if hu.ndim != 3 or min(hu.shape) < 2:
        raise ValueError(f"Se esperaba un volumen 3D de varios cortes, se recibió {hu.shape}.")
    if espaciado.shape != (3,) or not np.all(np.isfinite(espaciado)) or np.any(espaciado <= 0):
        raise ValueError(f"Espaciado Z-Y-X inválido: {espaciado}.")
    if not np.isfinite(hu).all():
        raise ValueError("Hay NaN/inf en el volumen HU. No se corregirán silenciosamente.")
    return hu, espaciado


def revisar_resumen_original(entrada: Path) -> None:
    resumen = entrada.parent / "resumen_preprocesamiento.json"
    if resumen.is_file():
        datos = json.loads(resumen.read_text(encoding="utf-8"))
        if datos.get("espaciado_irregular_advertencia") is True:
            raise ValueError(
                "El paso 9 detectó espaciado irregular. No interpolar esta serie "
                "como si tuviera separación uniforme; revisar primero los DICOM originales."
            )
    else:
        print("AVISO: no se encontró el resumen del paso 9. Comprueba la geometría de la serie.")


def nueva_forma_y_espaciado(
    forma: tuple[int, int, int],
    espaciado: np.ndarray,
    objetivo: np.ndarray,
) -> tuple[tuple[int, int, int], np.ndarray]:
    # Mantiene aproximadamente la distancia entre los CENTROS de los voxeles extremos.
    # La separación lograda puede diferir ligeramente del objetivo debido al redondeo.
    dimensiones = np.array(forma, dtype=np.int64)
    nueva_forma = np.rint((dimensiones - 1) * espaciado / objetivo).astype(np.int64) + 1
    nueva_forma = np.maximum(nueva_forma, 2)
    if int(np.prod(nueva_forma, dtype=np.int64)) > 100_000_000:
        raise ValueError(
            "El volumen de salida superaría 100 millones de voxeles; "
            "escoge un espaciado objetivo más grueso para esta prueba."
        )
    logrado = (dimensiones - 1) * espaciado / (nueva_forma - 1)
    return tuple(int(x) for x in nueva_forma), logrado


def imagen_comparativa(
    original: np.ndarray,
    normalizado: np.ndarray,
    mascara: np.ndarray,
    destino: Path,
    minimo_hu: float,
    maximo_hu: float,
) -> None:
    a = original[original.shape[0] // 2]
    b = normalizado[normalizado.shape[0] // 2]
    c = mascara[mascara.shape[0] // 2]

    fig, ejes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    ejes[0].imshow(a, cmap="gray", vmin=minimo_hu, vmax=maximo_hu)
    ejes[0].set_title("Original HU (ventana DL)")
    ejes[1].imshow(b, cmap="gray", vmin=0, vmax=1)
    ejes[1].set_title("Remuestreado y normalizado [0,1]")
    ejes[2].imshow(c, cmap="gray", vmin=0, vmax=1)
    ejes[2].set_title("HU < -1024 (posible relleno)")
    for eje in ejes:
        eje.axis("off")
    fig.suptitle("LungAI | Paso 10 - preparación de imágenes, SIN diagnóstico")
    fig.savefig(destino, dpi=125)
    plt.close(fig)


def ejecutar(
    entrada: Path,
    salida: Path,
    objetivo: np.ndarray,
    hu_min: float,
    hu_max: float,
) -> dict:
    if not (np.all(np.isfinite(objetivo)) and np.all(objetivo > 0)):
        raise ValueError("--espaciado-objetivo debe contener tres números positivos.")
    if not (np.isfinite(hu_min) and np.isfinite(hu_max) and hu_min < hu_max):
        raise ValueError("Los límites HU deben ser finitos y HU mínimo < HU máximo.")
    revisar_resumen_original(entrada)
    hu, espaciado = leer_npz(entrada)
    forma_original = hu.shape
    hu_original_min, hu_original_max = float(hu.min()), float(hu.max())

    # El NPZ del paso 9 ya contiene HU, NO aplicar RescaleSlope/Intercept de nuevo.
    sospechoso = hu < UMBRAL_HU_SOSPECHOSO
    cantidad_sospechosa = int(np.count_nonzero(sospechoso))
    porcentaje_sospechoso = 100.0 * cantidad_sospechosa / hu.size

    # En este prototipo, sustituimos los valores no plausibles de baja densidad
    # por aire, evitando que -3024 produzca un borde espurio al interpolar.
    # No equivale a identificar PixelPaddingValue desde los DICOM originales.
    if cantidad_sospechosa:
        hu[sospechoso] = HU_AIRE

    # Interpolación lineal (orden 1) para imágenes; vecino cercano (orden 0)
    # para la máscara de control. No reorienta ni segmenta pulmones.
    np.clip(hu, hu_min, hu_max, out=hu)
    hu -= hu_min
    hu /= (hu_max - hu_min)

    nueva_forma, espaciado_logrado = nueva_forma_y_espaciado(forma_original, espaciado, objetivo)
    factores = np.asarray(nueva_forma, dtype=np.float64) / np.asarray(forma_original, dtype=np.float64)
    if nueva_forma == forma_original:
        volumen_dl = hu
        mascara_dl = sospechoso.astype(np.uint8)
    else:
        volumen_dl = zoom(hu, zoom=factores, order=1, mode="nearest", prefilter=False, grid_mode=False)
        mascara_dl = zoom(sospechoso.astype(np.uint8), zoom=factores, order=0, mode="nearest", prefilter=False, grid_mode=False)

    if volumen_dl.shape != nueva_forma or mascara_dl.shape != nueva_forma:
        raise ValueError("La interpolación devolvió dimensiones inesperadas.")
    if not np.isfinite(volumen_dl).all():
        raise ValueError("La normalización introdujo valores no finitos.")
    if float(volumen_dl.min()) < -1e-5 or float(volumen_dl.max()) > 1.00001:
        raise ValueError("La normalización no quedó dentro de [0,1].")
    if float(volumen_dl.max() - volumen_dl.min()) < 1e-5:
        raise ValueError("La tomografía quedó casi constante: revisar ventanas HU e imagen de origen.")

    salida.mkdir(parents=True, exist_ok=True)
    archivo_npz = salida / "volumen_normalizado.npz"
    # Guardamos ÚNICAMENTE la entrada de imagen y su geometría: nunca etiquetas
    # clínicas ni variables de seguimiento como canales de entrada al modelo.
    np.savez_compressed(
        archivo_npz,
        volumen_normalizado=volumen_dl.astype(np.float32),
        espaciado_mm_zyx=np.asarray(espaciado_logrado, dtype=np.float32),
        mascara_hu_menor_menos1024=mascara_dl.astype(np.uint8),
    )
    with np.load(entrada, allow_pickle=False) as datos_originales:
        original_hu = np.asarray(datos_originales["volumen_hu"], dtype=np.float32)
    imagen_comparativa(
        original=original_hu,
        normalizado=volumen_dl,
        mascara=mascara_dl,
        destino=salida / "comparacion_dl.png",
        minimo_hu=hu_min,
        maximo_hu=hu_max,
    )

    informe = {
        "estado": "procesado; NO entrenado y NO inferencia clínica",
        "forma_original_zyx": list(forma_original),
        "forma_salida_zyx": list(nueva_forma),
        "espaciado_original_mm_zyx": [round(float(v), 5) for v in espaciado],
        "espaciado_objetivo_mm_zyx": [round(float(v), 5) for v in objetivo],
        "espaciado_logrado_mm_zyx": [round(float(v), 5) for v in espaciado_logrado],
        "hu_min_original": hu_original_min,
        "hu_max_original": hu_original_max,
        "voxeles_hu_menor_menos1024": cantidad_sospechosa,
        "porcentaje_hu_menor_menos1024": round(porcentaje_sospechoso, 3),
        "criterio_valores_sospechosos": (
            "HU < -1024 se aproxima como aire (-1000) antes de interpolar. "
            "Esto no identifica PixelPaddingValue DICOM de forma definitiva."
        ),
        "ventana_hu_para_normalizacion": [hu_min, hu_max],
        "rango_normalizado": [round(float(volumen_dl.min()), 5), round(float(volumen_dl.max()), 5)],
        "interpolacion_imagen": "lineal, scipy.ndimage.zoom, grid_mode=False",
        "interpolacion_mascara": "vecino más cercano",
        "orden_ejes": "Z, Y, X",
        "tipo_dato_volumen": "float32",
        "nota": (
            "Todavía falta definir recortes/padding y orientación física completos según el modelo elegido. "
            "No diagnostica ni localiza lesiones; nunca usar una etiqueta clínica como canal de entrada."
        ),
    }
    (salida / "resumen_dl.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return informe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=ENTRADA_PREDETERMINADA)
    parser.add_argument("--salida", type=Path, default=SALIDA_PREDETERMINADA)
    parser.add_argument("--espaciado-objetivo", type=float, nargs=3, metavar=("Z", "Y", "X"), default=(2.0, 1.0, 1.0))
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=400.0)
    args = parser.parse_args()

    print("\n=== LUNGAI | PASO 10: PREPROCESAMIENTO PARA IA ===", flush=True)
    resultado = ejecutar(
        entrada=args.entrada,
        salida=args.salida,
        objetivo=np.asarray(args.espaciado_objetivo, dtype=np.float64),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
    )
    print("Forma original Z-Y-X:", tuple(resultado["forma_original_zyx"]))
    print("Forma procesada Z-Y-X:", tuple(resultado["forma_salida_zyx"]))
    print("Espaciado logrado (mm Z-Y-X):", tuple(resultado["espaciado_logrado_mm_zyx"]))
    print("HU originales: ", resultado["hu_min_original"], "a", resultado["hu_max_original"])
    print("HU < -1024 sospechosos:", resultado["voxeles_hu_menor_menos1024"], "(", resultado["porcentaje_hu_menor_menos1024"], "%)")
    print("Rango normalizado:", tuple(resultado["rango_normalizado"]))
    print("Archivos guardados en:", args.salida.resolve())
    print(" - volumen_normalizado.npz")
    print(" - resumen_dl.json")
    print(" - comparacion_dl.png")
    print("Esto NO entrena un modelo ni diagnostica cáncer.")


if __name__ == "__main__":
    main()
