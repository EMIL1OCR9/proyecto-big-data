import { useEffect, useState } from 'react'
import { readJson } from './api'
import './Resultados.css'

type Split = { conjunto: string; procesadas: number; total: number }
type Summary = {
  procesamiento: {
    series_total: number; series_procesadas: number; gb_total: number; gb_procesados: number
    por_conjunto: Split[]; nodulos: number; lung_rads: { categoria: string; nodulos: number }[]
    segundos_deteccion: number | null; causas_exclusion: { causa: string; series: number }[] | null
  }
  detector: { reportados: number; encontrados: number; bajo_umbral: number; no_encontrados: number } | null
  etiquetas: { conjunto: string; malignos: number; benignos: number }[] | null
  clasificador: {
    mejor_epoca: number
    historial: { epoca: number; auc_validation: number | null }[]
    validation: Aucs; test: Aucs | null
  } | null
}
type Aucs = { auc_modelo: number; auc_solo_tamano: number; auc_solo_puntaje_detector: number }
type Case = { patient_id: string; series_uid: string; recorte: number; etiqueta: string; probabilidad_maligno: number; diametro_mm: number }
type Cases = {
  umbral: number; malignos: number; benignos: number; malignos_detectados: number; benignos_marcados: number
  aciertos: Case[]; falsos_negativos: Case[]; falsos_positivos: Case[]
}
export type OpenCase = (patientId: string, seriesUid: string, recorte: number) => void

const percent = (part: number, total: number) => `${(100 * part / total).toFixed(1).replace('.0', '')} %`
const share = (part: number, total: number) => `${100 * part / total}%` // para CSS: sin espacio
const number = (value: number) => value.toLocaleString('es-MX')

export default function Resultados({ apiBase, onOpenCase }: { apiBase: string; onOpenCase: OpenCase }) {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [cases, setCases] = useState<Cases | null>(null)
  const [casesError, setCasesError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${apiBase}/resultados`, { signal: controller.signal })
      .then(response => readJson<Summary>(response)).then(setSummary)
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message) })
    fetch(`${apiBase}/resultados/casos`, { signal: controller.signal })
      .then(response => readJson<Cases>(response)).then(setCases)
      .catch(reason => { if (!controller.signal.aborted) setCasesError(reason.message) })
    return () => controller.abort()
  }, [apiBase])

  if (error) return <section className="rs-page"><p className="alert">No se pudieron leer los resultados: {error}</p></section>
  if (!summary) return <section className="rs-page"><p className="rs-loading">Leyendo los resultados de las 2,790 tomografías…</p></section>
  const p = summary.procesamiento

  return <section className="rs-page">
    <div className="page-heading"><div>
      <span className="eyebrow dark">RESULTADOS DEL PROYECTO</span>
      <h1>LungAI en números</h1>
      <p>Lo que salió de procesar la muestra del NLST en Colab: el procesamiento, el detector, las etiquetas y el clasificador.</p>
    </div></div>

    <div className="stats-grid rs-stats">
      <div className="stat-card"><span>Tomografías procesadas</span><strong>{number(p.series_procesadas)}</strong><small>de {number(p.series_total)} ({percent(p.series_procesadas, p.series_total)})</small></div>
      <div className="stat-card"><span>Datos procesados</span><strong>{p.gb_procesados} GB</strong><small>de {p.gb_total} GB de DICOM, sin guardarlos</small></div>
      <div className="stat-card"><span>Nódulos detectados</span><strong>{number(p.nodulos)}</strong><small>con puntaje del detector de 0.5 o más</small></div>
      <div className="stat-card"><span>Detección por tomografía</span><strong>{p.segundos_deteccion ? `${p.segundos_deteccion} s` : '—'}</strong><small>{p.segundos_deteccion ? 'en promedio, con GPU en Colab' : 'copia los lotes_progreso*.csv a outputs/lotes'}</small></div>
    </div>

    <div className="rs-grid">
      <article className="rs-card">
        <h2>Procesamiento por conjunto</h2>
        <p className="rs-lead">Se excluyó el {percent(p.series_total - p.series_procesadas, p.series_total)} de las tomografías por defectos de origen.</p>
        {p.por_conjunto.map(split => <div className="rs-bar-row" key={split.conjunto}>
          <span>{split.conjunto}</span>
          <div className="rs-bar"><div style={{ width: `${100 * split.procesadas / split.total}%` }} /></div>
          <b>{number(split.procesadas)} / {number(split.total)}</b>
        </div>)}
        {p.causas_exclusion && <ul className="rs-list">{p.causas_exclusion.map(cause =>
          <li key={cause.causa}><b>{cause.series}</b> {cause.causa.toLowerCase()}</li>)}</ul>}
      </article>

      {summary.detector && <article className="rs-card">
        <h2>Detector contra el radiólogo</h2>
        <p className="rs-lead">De {number(summary.detector.reportados)} nódulos que anotó el radiólogo, el detector encontró {percent(summary.detector.encontrados, summary.detector.reportados)}, y {percent(summary.detector.encontrados + summary.detector.bajo_umbral, summary.detector.reportados)} contando los de puntaje bajo.</p>
        <div className="rs-stack">
          <div className="rs-found" style={{ width: share(summary.detector.encontrados, summary.detector.reportados) }} />
          <div className="rs-low" style={{ width: share(summary.detector.bajo_umbral, summary.detector.reportados) }} />
          <div className="rs-miss" style={{ width: share(summary.detector.no_encontrados, summary.detector.reportados) }} />
        </div>
        <ul className="rs-legend">
          <li><i className="rs-found" />Encontrados: {number(summary.detector.encontrados)}</li>
          <li><i className="rs-low" />Con puntaje bajo: {number(summary.detector.bajo_umbral)}</li>
          <li><i className="rs-miss" />No encontrados: {number(summary.detector.no_encontrados)}</li>
        </ul>
      </article>}

      <article className="rs-card">
        <h2>Tamaño de los nódulos (Lung-RADS)</h2>
        <p className="rs-lead">La mayoría son menores de 6 mm, que la guía considera probablemente benignos.</p>
        <Bars rows={p.lung_rads.map(row => ({ label: `Categoría ${row.categoria}`, value: row.nodulos, tone: `rads-${row.categoria}` }))} />
        <p className="rs-note">2: menos de 6 mm · 3: de 6 a 8 · 4A: de 8 a 15 · 4B: 15 mm o más. Solo por tamaño.</p>
      </article>

      {summary.etiquetas && <article className="rs-card">
        <h2>Etiquetas para entrenar</h2>
        <p className="rs-lead">Hay más de 50 benignos por cada maligno: por eso el modelo se entrenó con 10 benignos por maligno.</p>
        <table className="rs-table"><thead><tr><th>Conjunto</th><th>Malignos</th><th>Benignos</th></tr></thead>
          <tbody>{summary.etiquetas.map(row => <tr key={row.conjunto}><td>{row.conjunto}</td><td>{number(row.malignos)}</td><td>{number(row.benignos)}</td></tr>)}</tbody>
        </table>
      </article>}
    </div>

    {summary.clasificador && <div className="rs-grid">
      <article className="rs-card">
        <h2>Clasificador contra reglas simples</h2>
        <p className="rs-lead">{summary.clasificador.test && summary.clasificador.test.auc_modelo < summary.clasificador.test.auc_solo_tamano
          ? 'En test, medir solo el tamaño separó malignos de benignos un poco mejor que el modelo.'
          : 'En test, el modelo quedó a la par o por encima de medir solo el tamaño.'}</p>
        <AucChart validation={summary.clasificador.validation} test={summary.clasificador.test} />
        <p className="rs-note">AUC: 0.5 es tirar una moneda y 1 es perfecto. Con 32 malignos en test, diferencias de 0.02 a 0.03 pueden ser azar.</p>
      </article>
      <article className="rs-card">
        <h2>Entrenamiento</h2>
        <p className="rs-lead">AUC en validation por época; se guardó la época {summary.clasificador.mejor_epoca}, la mejor.</p>
        <EpochChart history={summary.clasificador.historial} best={summary.clasificador.mejor_epoca} />
        <p className="rs-note">El AUC sube y baja varios puntos de una época a otra: con pocos malignos, el resultado es inestable.</p>
      </article>
    </div>}

    <article className="rs-card rs-cases">
      <h2>Aciertos y errores del modelo en test</h2>
      {cases ? <>
        <p className="rs-lead">Con un umbral de {cases.umbral * 100} %, el modelo marcó {cases.malignos_detectados} de {cases.malignos} nódulos malignos y {cases.benignos_marcados} de {cases.benignos} benignos. Abre cualquier caso para verlo en la tomografía.</p>
        <p className="rs-note">Solo cuenta los nódulos de test con los que se midió el modelo (todos los malignos y 10 benignos por cada uno) que el detector marcó con puntaje de 0.5 o más. Fíjate en el tamaño: las falsas alarmas suelen ser nódulos grandes y los cánceres que no vio, chicos.</p>
        <div className="rs-case-grid">
          <CaseList title="Cáncer que el modelo sí detectó" note="Malignos con la probabilidad más alta" cases={cases.aciertos} onOpen={onOpenCase} />
          <CaseList title="Cáncer que el modelo no vio" note="Malignos con la probabilidad más baja" cases={cases.falsos_negativos} onOpen={onOpenCase} />
          <CaseList title="Falsas alarmas" note="Benignos con la probabilidad más alta" cases={cases.falsos_positivos} onOpen={onOpenCase} />
        </div>
      </> : <p className="rs-lead">{casesError ?? 'Cargando…'}</p>}
    </article>

    <p className="footnote">Proyecto académico. Las etiquetas son aproximadas (cáncer en el mismo lóbulo) y nada de esto es un diagnóstico.</p>
  </section>
}

function Bars({ rows }: { rows: { label: string; value: number; tone: string }[] }) {
  const max = Math.max(...rows.map(row => row.value))
  return <div className="rs-bars">{rows.map(row => <div className="rs-bar-row" key={row.label}>
    <span>{row.label}</span>
    <div className="rs-bar"><div className={`rs-${row.tone}`} style={{ width: `${100 * row.value / max}%` }} /></div>
    <b>{number(row.value)}</b>
  </div>)}</div>
}

function AucChart({ validation, test }: { validation: Aucs; test: Aucs | null }) {
  const rows: [string, keyof Aucs][] = [['Modelo (red 3D)', 'auc_modelo'], ['Solo tamaño', 'auc_solo_tamano'], ['Solo puntaje del detector', 'auc_solo_puntaje_detector']]
  // El eje empieza en 0.5 (azar), para que se vean las diferencias que importan.
  const width = (auc: number) => `${Math.max(0, (auc - 0.5) / 0.5) * 100}%`
  return <div className="rs-bars">
    {rows.map(([label, key]) => <div className="rs-auc" key={key}>
      <span>{label}</span>
      <div className="rs-bar-row"><small>validation</small><div className="rs-bar"><div className="rs-validation" style={{ width: width(validation[key]) }} /></div><b>{validation[key].toFixed(3)}</b></div>
      {test && <div className="rs-bar-row"><small>test</small><div className="rs-bar"><div className="rs-test" style={{ width: width(test[key]) }} /></div><b>{test[key].toFixed(3)}</b></div>}
    </div>)}
    <p className="rs-note">Las barras empiezan en 0.5, que es tirar una moneda.</p>
  </div>
}

function EpochChart({ history, best }: { history: { epoca: number; auc_validation: number | null }[]; best: number }) {
  const points = history.filter(row => row.auc_validation != null) as { epoca: number; auc_validation: number }[]
  if (!points.length) return null
  const [w, h, pad] = [460, 180, 30]
  const low = Math.floor(Math.min(...points.map(row => row.auc_validation)) * 20) / 20
  const x = (epoch: number) => pad + (epoch - 1) / Math.max(points.length - 1, 1) * (w - 2 * pad)
  const y = (auc: number) => h - pad - (auc - low) / (1 - low) * (h - 2 * pad)
  const bestRow = points.find(row => row.epoca === best)
  return <svg className="rs-epochs" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="AUC de validation por época">
    {[low, (low + 1) / 2, 1].map(tick => <g key={tick}>
      <line x1={pad} x2={w - pad} y1={y(tick)} y2={y(tick)} className="rs-grid-line" />
      <text x={pad - 6} y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text>
    </g>)}
    <polyline points={points.map(row => `${x(row.epoca)},${y(row.auc_validation)}`).join(' ')} className="rs-line" />
    {bestRow && <g>
      <circle cx={x(bestRow.epoca)} cy={y(bestRow.auc_validation)} r={5} className="rs-best" />
      <text x={x(bestRow.epoca)} y={y(bestRow.auc_validation) - 10} textAnchor="middle" className="rs-best-label">época {best}: {bestRow.auc_validation.toFixed(3)}</text>
    </g>}
    <text x={pad} y={h - 8}>época 1</text>
    <text x={w - pad} y={h - 8} textAnchor="end">época {points.at(-1)!.epoca}</text>
  </svg>
}

function CaseList({ title, note, cases, onOpen }: { title: string; note: string; cases: Case[]; onOpen: OpenCase }) {
  return <div className="rs-case-list">
    <h3>{title}</h3><small>{note}</small>
    <ul>{cases.map(item => <li key={`${item.series_uid}-${item.recorte}`}>
      <span>Paciente <b>{item.patient_id}</b> · {item.diametro_mm} mm</span>
      <span className="rs-prob">{Math.round(item.probabilidad_maligno * 100)} %</span>
      <button type="button" className="text-btn" onClick={() => onOpen(item.patient_id, item.series_uid, item.recorte)}>Abrir →</button>
    </li>)}</ul>
  </div>
}
