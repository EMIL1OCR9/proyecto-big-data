"""Muestra un corte DICOM. Uso: uv run python nlst/ver_dicom.py <archivo.dcm | carpeta_de_serie>"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pydicom


def mostrar_dicom(ruta, ventana=(-600, 1500)):
    """Abre un .dcm (o el corte central de una carpeta de serie) y lo grafica en HU con ventana pulmonar."""
    ruta = Path(ruta)
    if ruta.is_dir():
        cortes = sorted(ruta.glob("*.dcm"),
                        key=lambda f: float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]))
        ruta = cortes[len(cortes) // 2]
    ds = pydicom.dcmread(ruta)
    hu = ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    centro, ancho = ventana
    plt.imshow(hu, cmap="gray", vmin=centro - ancho / 2, vmax=centro + ancho / 2)
    plt.title(f"Paciente {ds.PatientID} | {ds.get('ClinicalTrialTimePointID', '')} | "
              f"corte {ds.InstanceNumber} | {ds.SliceThickness} mm")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    mostrar_dicom(sys.argv[1])
