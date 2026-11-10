"""
LungAI — genera una imagen PNG por serie con los nodulos detectados.

Lee los JSON de detectar_nodulos.py y su NIfTI. Por cada nodulo dibuja el corte
completo con la caja en rojo y, debajo, un acercamiento de 6 x 6 cm. La imagen queda
junto al JSON, con el mismo nombre. Volver a correrlo salta las que ya existen.

Uso:
    python backend/inference/ver_nodulos.py

Importante:
- Muestra lo que marco el detector; NO es un diagnostico.
- Las series sin nodulos no generan imagen.
- outputs/ esta en .gitignore. No subir imagenes con identificadores a GitHub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Guarda la imagen sin abrir una ventana.
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.patches import Rectangle

ZOOM_MM = 30  # el acercamiento muestra 3 cm hacia cada lado


def draw_report(report: dict, image_path: Path, output: Path, maximum: int) -> None:
    """Dibuja los nodulos de una serie, del puntaje mas alto al mas bajo."""
    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayViewFromImage(image)  # orden Z, Y, X
    sx, sy, _ = image.GetSpacing()
    nodules = report["nodulos"][:maximum]

    figure, axes = plt.subplots(2, len(nodules), figsize=(4 * len(nodules), 8.4), squeeze=False)
    for column, nodule in enumerate(nodules):
        x, y, z = nodule["centro_voxel"]
        width, height = nodule["tamano_mm"][0] / sx, nodule["tamano_mm"][1] / sy
        for axis in axes[:, column]:
            axis.imshow(volume[z], cmap="gray", vmin=-1000, vmax=400)
            axis.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                                     fill=False, edgecolor="red", linewidth=1.5))
            axis.axis("off")
        axes[1, column].set_xlim(x - ZOOM_MM / sx, x + ZOOM_MM / sx)
        axes[1, column].set_ylim(y + ZOOM_MM / sy, y - ZOOM_MM / sy)
        axes[0, column].set_title(
            f"Puntaje {nodule['puntaje']:.2f} | corte {z}\n"
            f"{nodule['lobulo'] or 'sin lobulo'} | caja de {max(nodule['tamano_mm'][:2]):.0f} mm"
        )

    total = len(report["nodulos"])
    shown = f"{len(nodules)} de {total} nodulos" if total > len(nodules) else f"{total} nodulos"
    figure.suptitle(f"Paciente {report['patient_id']} | {shown} | deteccion automatica, sin diagnostico")
    figure.tight_layout()
    figure.savefig(output, dpi=80)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodules-dir", type=Path, default=Path("outputs/nodulos"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/monai"))
    parser.add_argument("--max", type=int, default=5, help="Maximo de nodulos por imagen.")
    args = parser.parse_args()

    created = without_nodules = 0
    for path in sorted(args.nodules_dir.glob("*/*.json")):
        output = path.with_suffix(".png")
        if output.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report["nodulos"]:
            without_nodules += 1
            continue
        draw_report(report, args.data_dir / report["imagen"], output, args.max)
        created += 1
        print(f"{report['patient_id']}: {len(report['nodulos'])} nodulos -> {output}")

    print(f"Imagenes nuevas: {created} | series sin nodulos: {without_nodules}")


if __name__ == "__main__":
    main()
