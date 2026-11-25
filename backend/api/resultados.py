"""Resumen del proyecto para la pagina de resultados. No corre modelos: lee lo que ya existe.

- outputs/monai_dataset/cohorte_cancer_12m.parquet y outputs/nodulos/ (los JSON de Colab).
- outputs/etiquetas/ (nodulos_reportados.parquet y candidatos_etiquetados.parquet).
- data/modelos/clasificador/metricas.json.
- Opcionales, copiados de Drive: outputs/lotes/lotes_progreso*.csv y lotes_fallidas*.csv (tiempo por
  serie y causas de exclusion) y outputs/clasificador/predicciones.csv (aciertos y errores en test).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SPLITS = ["train", "validation", "test"]
THRESHOLD = 0.5  # puntaje minimo del detector para contar como nodulo
CAUSES = [
    ("espaciado Z irregular", "Espaciado irregular entre cortes"),
    ("dos cortes en la misma posicion", "Cortes repetidos con imagen distinta"),
    ("mezcla orientacion", "Orientación o tamaño mezclados"),
]
RADS = [(6, "2"), (8, "3"), (15, "4A"), (float("inf"), "4B")]  # Lung-RADS por diametro medio, en mm


def lung_rads(size: list[float]) -> str:
    """Categoria Lung-RADS de un nodulo solido en la primera CT, solo por su diametro medio axial."""
    mean = (size[0] + size[1]) / 2
    return next(category for limit, category in RADS if mean < limit)


def read_csvs(folder: Path, pattern: str) -> pd.DataFrame:
    parts = [pd.read_csv(p, dtype={"PatientID": "string"}) for p in sorted(folder.glob(pattern))]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


class ResultsService:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.outputs = self.root / "outputs"
        self._summary: dict[str, Any] | None = None

    def summary(self, refresh: bool = False) -> dict[str, Any]:
        """Se calcula una vez (lee los 2,790 JSON); con refresh se vuelve a leer todo."""
        if self._summary is None or refresh:
            self._summary = {
                "procesamiento": self.processing(),
                "detector": self.detector(),
                "etiquetas": self.labels(),
                "clasificador": self.classifier(),
            }
        return self._summary

    def processing(self) -> dict[str, Any]:
        cohort = pd.read_parquet(self.outputs / "monai_dataset" / "cohorte_cancer_12m.parquet",
                                 columns=["SeriesInstanceUID", "split", "en_muestra", "series_size_MB"])
        sample = cohort[cohort["en_muestra"]]
        processed, nodules, rads = set(), 0, {category: 0 for _, category in RADS}
        for path in (self.outputs / "nodulos").glob("*/*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            processed.add(report["series_uid"])
            nodules += len(report["nodulos"])
            for nodule in report["nodulos"]:
                rads[lung_rads(nodule["tamano_mm"])] += 1
        done = sample["SeriesInstanceUID"].isin(processed)

        progress = read_csvs(self.outputs / "lotes", "lotes_progreso*.csv")
        if len(progress):
            progress = progress.drop_duplicates("SeriesInstanceUID", keep="last")
        failures = read_csvs(self.outputs / "lotes", "lotes_fallidas*.csv")
        causes = None
        if len(failures):
            failures = failures.drop_duplicates("SeriesInstanceUID", keep="last")
            failures = failures[~failures["SeriesInstanceUID"].isin(processed)]
            names = failures["error"].astype(str).map(
                lambda error: next((name for key, name in CAUSES if key in error), "Otra causa"))
            causes = [{"causa": name, "series": int(count)} for name, count in names.value_counts().items()]
            missing = int((~done).sum()) - len(failures)
            if missing > 0:  # excluidas cuyo archivo de fallas no se copio
                causes.append({"causa": "Causa no registrada (falta su lotes_fallidas)", "series": missing})

        return {
            "series_total": len(sample),
            "series_procesadas": int(done.sum()),
            "gb_total": round(float(sample["series_size_MB"].sum()) / 1024, 1),
            "gb_procesados": round(float(sample.loc[done, "series_size_MB"].sum()) / 1024, 1),
            "por_conjunto": [{"conjunto": split, "procesadas": int((done & sample["split"].eq(split)).sum()),
                              "total": int(sample["split"].eq(split).sum())} for split in SPLITS],
            "nodulos": nodules,
            "lung_rads": [{"categoria": category, "nodulos": count} for category, count in rads.items()],
            "segundos_deteccion": round(float(progress["segundos_deteccion"].mean()))
            if "segundos_deteccion" in progress else None,
            "causas_exclusion": causes,
        }

    def detector(self) -> dict[str, Any] | None:
        path = self.outputs / "etiquetas" / "nodulos_reportados.parquet"
        if not path.is_file():
            return None
        found = pd.read_parquet(path, columns=["detectado"])["detectado"].value_counts()
        return {"reportados": int(found.sum()), "encontrados": int(found.get("si", 0)),
                "bajo_umbral": int(found.get("bajo umbral", 0)), "no_encontrados": int(found.get("no", 0))}

    def labels(self) -> list[dict[str, Any]] | None:
        path = self.outputs / "etiquetas" / "candidatos_etiquetados.parquet"
        if not path.is_file():
            return None
        table = pd.read_parquet(path, columns=["recorte", "etiqueta", "split"])
        table = table[table["recorte"].notna() & table["etiqueta"].isin(["maligno", "benigno"])]
        counts = pd.crosstab(table["split"], table["etiqueta"]).reindex(index=SPLITS, columns=["maligno", "benigno"],
                                                                         fill_value=0)
        return [{"conjunto": split, "malignos": int(row["maligno"]), "benignos": int(row["benigno"])}
                for split, row in counts.iterrows()]

    def classifier(self) -> dict[str, Any] | None:
        path = self.root / "data" / "modelos" / "clasificador" / "metricas.json"
        if not path.is_file():
            return None
        metrics = json.loads(path.read_text(encoding="utf-8"))
        return {key: metrics.get(key) for key in ("mejor_epoca", "historial", "validation", "test", "recortes")}

    def cases(self, per_group: int = 5) -> dict[str, Any] | None:
        """Aciertos y errores del clasificador en test, solo con nodulos que el visor muestra."""
        path = self.outputs / "clasificador" / "predicciones.csv"
        if not path.is_file():
            return None
        predictions = pd.read_csv(path, dtype={"PatientID": "string"})
        test = predictions[predictions["split"].eq("test") & predictions["puntaje"].ge(THRESHOLD)]
        malignant = test[test["etiqueta"].eq("maligno")]
        benign = test[test["etiqueta"].eq("benigno")]

        def rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
            return [{"patient_id": r.PatientID, "series_uid": r.SeriesInstanceUID, "recorte": int(r.recorte),
                     "etiqueta": r.etiqueta, "probabilidad_maligno": round(float(r.probabilidad_maligno), 3),
                     "diametro_mm": round(float(r.tamano_mm), 1), "puntaje": round(float(r.puntaje), 3)}
                    for r in frame.head(per_group).itertuples()]

        positive = test["probabilidad_maligno"].ge(0.5)
        return {
            "umbral": 0.5,
            "malignos": len(malignant), "benignos": len(benign),
            "malignos_detectados": int((positive & test["etiqueta"].eq("maligno")).sum()),
            "benignos_marcados": int((positive & test["etiqueta"].eq("benigno")).sum()),
            "aciertos": rows(malignant.sort_values("probabilidad_maligno", ascending=False)),
            "falsos_negativos": rows(malignant.sort_values("probabilidad_maligno")),
            "falsos_positivos": rows(benign.sort_values("probabilidad_maligno", ascending=False)),
        }
