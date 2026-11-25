import { formatSize, lungRads } from './medidas'
import './Analisis.css'

export type Nodule = {
  numero: number
  puntaje: number
  diametro_mm: number
  /** Ancho, alto y profundidad de la caja del detector, en mm. */
  tamano_mm: number[]
  lobulo: string
  centro_ras_mm: number[]
  probabilidad_maligno: number | null
  /** Posición del recorte en el archivo de Colab; une el nódulo con sus etiquetas y predicciones. */
  recorte?: number | null
  /** Solo en la cohorte: lo que dice el NLST de este nódulo. */
  etiqueta_nlst?: string | null
  reportado_radiologo?: boolean | null
}
type TestMetrics = { auc_modelo: number; auc_solo_tamano: number; auc_solo_puntaje_detector: number }
export type ModelInfo = { umbral: number; clasificador: boolean; metricas_test: TestMetrics | null }

const LABELS: Record<string, string> = {
  maligno: 'Cáncer en ese lóbulo', benigno: 'Benigno', 'sin etiqueta': 'Sin etiqueta',
}

type TableProps = {
  nodules: Nodule[]
  discarded: number
  model: ModelInfo
  selected: Nodule | null
  onSelect: (nodule: Nodule) => void
  /** Columnas del radiólogo y del NLST: solo en la cohorte, cuando se copiaron las etiquetas. */
  withTruth?: boolean
}

export function NoduleTable({ nodules, discarded, model, selected, onSelect, withTruth = false }: TableProps) {
  return <div className="table-card an-table">
    <div className="table-caption">
      <strong>Nódulos encontrados</strong>
      <span>{nodules.length} con puntaje del detector ≥ {model.umbral} · {discarded} candidatos descartados</span>
    </div>
    {nodules.length ? <div className="table-scroll"><table>
      <thead><tr>
        <th>#</th><th>Lóbulo</th><th>Diámetro mayor</th><th title="Transversal (izquierda-derecha) × anteroposterior (frente-espalda) × craneocaudal (cabeza-pies)">Medidas (T × AP × CC)</th><th title="Categoría de la guía Lung-RADS solo por el diámetro medio; ver la nota abajo">Lung-RADS (tamaño)</th><th>Puntaje del detector</th><th>Probabilidad del modelo</th>
        {withTruth && <><th>Lo anotó el radiólogo</th><th>Etiqueta del NLST</th></>}
        <th></th>
      </tr></thead>
      <tbody>{nodules.map(nodule => <tr key={nodule.numero} className={selected?.numero === nodule.numero ? 'an-selected' : ''}>
        <td>{nodule.numero}</td>
        <td>{nodule.lobulo || 'Sin lóbulo'}</td>
        <td>{nodule.diametro_mm.toFixed(1)} mm</td>
        <td>{formatSize(nodule.tamano_mm)}</td>
        <td><span className={`an-rads an-rads-${lungRads(nodule.tamano_mm).category}`} title={`Diámetro medio ${lungRads(nodule.tamano_mm).meanMm} mm`}>{lungRads(nodule.tamano_mm).category}</span></td>
        <td>{nodule.puntaje.toFixed(2)}</td>
        <td>{nodule.probabilidad_maligno == null ? '—' : <span className="an-probability">
          <span className="an-bar"><span style={{ width: `${Math.round(nodule.probabilidad_maligno * 100)}%` }} /></span>
          {Math.round(nodule.probabilidad_maligno * 100)} %
        </span>}</td>
        {withTruth && <>
          <td>{nodule.reportado_radiologo == null ? '—' : nodule.reportado_radiologo ? 'Sí' : 'No'}</td>
          <td>{nodule.etiqueta_nlst ? <span className={`an-truth an-truth-${nodule.etiqueta_nlst.replace(' ', '-')}`}>
            {LABELS[nodule.etiqueta_nlst] ?? nodule.etiqueta_nlst}</span> : '—'}</td>
        </>}
        <td><button type="button" className="text-btn" onClick={() => onSelect(nodule)}>Ver en la CT →</button></td>
      </tr>)}</tbody>
    </table></div> : <p className="an-empty">El detector no encontró nódulos con puntaje de {model.umbral} o más.</p>}
  </div>
}

type NotesProps = { model: ModelInfo; warning: string; split?: string; withTruth?: boolean }

const SPLIT_NOTES: Record<string, string> = {
  train: 'Este paciente se usó para entrenar el modelo, así que su probabilidad no sirve como prueba: el modelo ya lo vio.',
  validation: 'Este paciente se usó para elegir la mejor época del modelo; el modelo no entrenó con él.',
  test: 'Este paciente es de test: el modelo nunca lo vio, así que su probabilidad es una prueba justa.',
}

export function ModelNotes({ model, warning, split, withTruth }: NotesProps) {
  const metrics = model.metricas_test
  return <div className="an-notes">
    {split && SPLIT_NOTES[split] && <p><b>Conjunto: {split}.</b> {SPLIT_NOTES[split]}</p>}
    {withTruth && <p><b>Etiqueta del NLST.</b> "Cáncer en ese lóbulo" quiere decir que el radiólogo anotó este nódulo y el paciente tuvo cáncer confirmado en ese lóbulo; no confirma que haya salido de este nódulo exacto. "Benigno": el paciente nunca tuvo cáncer y se le siguió al menos 2 años.</p>}
    <p><b>Medidas.</b> Son las de la caja que dibuja el detector alrededor del nódulo: transversal (izquierda-derecha) × anteroposterior (frente-espalda) × craneocaudal (cabeza-pies). Son aproximadas (±1–2 mm). "Ver en la CT" las dibuja en amarillo sobre cada corte; para medir a mano, usa el botón Medir del visor y arrastra sobre la imagen.</p>
    <p><b>Lung-RADS por tamaño.</b> Es la categoría que da la guía clínica Lung-RADS a un nódulo sólido en la primera tomografía según su diámetro medio: menos de 6 mm → 2 (probablemente benigno), 6 a 8 → 3, 8 a 15 → 4A y 15 o más → 4B (sospechoso). Aquí se calcula solo con la medida del detector; la guía también considera el tipo de nódulo, su forma y si creció. En la prueba final, el tamaño solo predijo el cáncer igual o mejor que el modelo.</p>
    {!model.clasificador && <p><b>Sin clasificador.</b> No se encontró <code>data/modelos/clasificador/modelo.pt</code>: solo se muestra la detección.</p>}
    {model.clasificador && <p><b>Cómo leer la probabilidad.</b> El modelo se entrenó viendo 1 nódulo maligno por cada 10 benignos, así que el número sirve para comparar nódulos entre sí, no es el riesgo real de cáncer.</p>}
    {metrics && <p><b>Qué tan bien funcionó.</b> En la prueba final el modelo obtuvo un AUC de {metrics.auc_modelo}; medir solo el tamaño, {metrics.auc_solo_tamano}; el puntaje del detector, {metrics.auc_solo_puntaje_detector}. {metrics.auc_modelo < metrics.auc_solo_tamano ? 'El modelo no superó a medir el tamaño.' : 'El modelo quedó a la par o por encima de medir el tamaño.'}</p>}
    <p><b>Importante.</b> {warning}</p>
  </div>
}
