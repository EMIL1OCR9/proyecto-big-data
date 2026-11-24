import { useEffect, useState, type FormEvent } from 'react'
import NiftiCanvas, { ViewButtons, type View } from './NiftiCanvas'
import './Analisis.css'

type Status = 'en_cola' | 'procesando' | 'listo' | 'error'
type Study = { id: string; archivo: string; estado: Status; paso: string; error: string | null; creado: number }
type Nodule = {
  numero: number
  puntaje: number
  diametro_mm: number
  lobulo: string
  centro_ras_mm: number[]
  probabilidad_maligno: number | null
}
type TestMetrics = { auc_modelo: number; auc_solo_tamano: number; auc_solo_puntaje_detector: number }
type Result = {
  nodulos: Nodule[]
  candidatos_descartados: number
  segundos: number
  advertencia: string
  fuente: { tipo: string; cortes: number; series_en_el_zip?: number }
  modelo: { umbral: number; clasificador: boolean; metricas_test: TestMetrics | null }
}
type StudyDetail = Study & { resultado: Result | null }

const STATUS_LABELS: Record<Status, string> = {
  en_cola: 'En cola', procesando: 'Analizando', listo: 'Listo', error: 'Error',
}
const running = (status: Status) => status === 'en_cola' || status === 'procesando'

async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail ?? `La API respondió HTTP ${response.status}`)
  return data as T
}

export default function Analisis({ apiBase }: { apiBase: string }) {
  const [studies, setStudies] = useState<Study[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<StudyDetail | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<View>('multiplanar')
  const [target, setTarget] = useState<Nodule | null>(null)
  const [refresh, setRefresh] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${apiBase}/estudios`, { signal: controller.signal })
      .then(response => readJson<{ estudios: Study[] }>(response))
      .then(data => setStudies(data.estudios))
      .catch(reason => { if (!controller.signal.aborted) setError(`No se pudo consultar la API: ${reason.message}`) })
    return () => controller.abort()
  }, [apiBase, refresh])

  // Mientras el estudio se analiza, se pregunta su estado cada 3 segundos.
  useEffect(() => {
    if (!selected) return
    let timer: ReturnType<typeof setTimeout> | undefined
    const controller = new AbortController()
    async function poll() {
      try {
        const data = await readJson<StudyDetail>(await fetch(`${apiBase}/estudios/${selected}`, { signal: controller.signal }))
        setDetail(data)
        if (running(data.estado)) timer = setTimeout(poll, 3000)
        else setRefresh(value => value + 1)
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Error desconocido')
      }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [apiBase, selected])

  function open(id: string) {
    setSelected(id)
    setDetail(null)
    setTarget(null)
    setError(null)
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!file) { setError('Elige un archivo primero.'); return }
    setUploading(true)
    setError(null)
    const body = new FormData()
    body.append('archivo', file)
    try {
      const study = await readJson<Study>(await fetch(`${apiBase}/estudios`, { method: 'POST', body }))
      open(study.id)
      setRefresh(value => value + 1)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'No se pudo subir el archivo.')
    } finally {
      setUploading(false)
    }
  }

  const result = detail?.estado === 'listo' ? detail.resultado : null

  return <section className="ct-page">
    <div className="page-heading ct-heading"><div>
      <span className="eyebrow dark">ANÁLISIS DE TOMOGRAFÍAS</span>
      <h1>Analizar estudio</h1>
      <p>Sube una tomografía de tórax: el programa busca nódulos, los mide, los ubica por lóbulo y estima qué tan sospechosos son.</p>
    </div></div>

    <form className="ct-search" onSubmit={event => void submit(event)}>
      <label htmlFor="an-file">Tomografía: un .zip con los DICOM de una serie, o un archivo .nii / .nii.gz</label>
      <div className="ct-search-controls">
        <input id="an-file" type="file" accept=".zip,.nii,.gz" onChange={event => setFile(event.target.files?.[0] ?? null)} />
        <button type="submit" className="outline-btn" disabled={uploading || !file}>{uploading ? 'Subiendo…' : 'Subir y analizar'}</button>
      </div>
      <small>El análisis corre en esta computadora; la tomografía no se envía a servicios externos. Sin tarjeta gráfica tarda varios minutos.</small>
    </form>

    {error && <p role="alert" className="alert">{error}</p>}

    {studies.length > 0 && <div className="ct-series-list">
      <strong>Estudios subidos</strong>
      <div className="ct-series-buttons">
        {studies.map(study => <button type="button" key={study.id} onClick={() => open(study.id)}
          className={selected === study.id ? 'ct-series-choice selected' : 'ct-series-choice'}>
          <span>{study.archivo}</span>
          <small>{STATUS_LABELS[study.estado]} · {new Date(study.creado * 1000).toLocaleString('es-MX')}</small>
        </button>)}
      </div>
    </div>}

    {detail && running(detail.estado) && <div className="an-progress" role="status">
      <span className="an-spinner" aria-hidden="true" />
      <div><strong>{STATUS_LABELS[detail.estado]}: {detail.archivo}</strong><small>{detail.paso}…</small></div>
    </div>}

    {detail?.estado === 'error' && <p role="alert" className="alert">No se pudo analizar {detail.archivo}: {detail.error}</p>}

    {selected && result && <>
      <div className="ct-viewer-card">
        <div className="ct-toolbar">
          <div className="ct-toolbar-title">
            <strong>{detail?.archivo}</strong>
            <small>{result.fuente.tipo} · {result.fuente.cortes} cortes · analizado en {Math.round(result.segundos / 60)} min · nódulos en rojo</small>
          </div>
          <ViewButtons view={view} onChange={setView} />
        </div>
        <NiftiCanvas key={selected} view={view} target={target?.centro_ras_mm}
          imageUrl={`${apiBase}/estudios/${selected}/imagen`} overlayUrl={`${apiBase}/estudios/${selected}/marcas`} />
      </div>

      <div className="table-card an-table">
        <div className="table-caption">
          <strong>Nódulos encontrados</strong>
          <span>{result.nodulos.length} con puntaje del detector ≥ {result.modelo.umbral} · {result.candidatos_descartados} candidatos descartados</span>
        </div>
        {result.nodulos.length ? <div className="table-scroll"><table>
          <thead><tr><th>#</th><th>Lóbulo</th><th>Diámetro mayor</th><th>Puntaje del detector</th><th>Probabilidad del modelo</th><th></th></tr></thead>
          <tbody>{result.nodulos.map(nodule => <tr key={nodule.numero} className={target?.numero === nodule.numero ? 'an-selected' : ''}>
            <td>{nodule.numero}</td>
            <td>{nodule.lobulo || 'Sin lóbulo'}</td>
            <td>{nodule.diametro_mm.toFixed(1)} mm</td>
            <td>{nodule.puntaje.toFixed(2)}</td>
            <td>{nodule.probabilidad_maligno == null ? '—' : <span className="an-probability">
              <span className="an-bar"><span style={{ width: `${Math.round(nodule.probabilidad_maligno * 100)}%` }} /></span>
              {Math.round(nodule.probabilidad_maligno * 100)} %
            </span>}</td>
            <td><button type="button" className="text-btn" onClick={() => setTarget(nodule)}>Ver en la CT →</button></td>
          </tr>)}</tbody>
        </table></div> : <p className="an-empty">El detector no encontró nódulos con puntaje de {result.modelo.umbral} o más.</p>}
      </div>

      <div className="an-notes">
        {!result.modelo.clasificador && <p><b>Sin clasificador.</b> No se encontró <code>data/modelos/clasificador/modelo.pt</code>: solo se muestra la detección.</p>}
        {result.modelo.clasificador && <p><b>Cómo leer la probabilidad.</b> El modelo se entrenó viendo 1 nódulo maligno por cada 10 benignos, así que el número sirve para comparar nódulos entre sí, no es el riesgo real de cáncer.</p>}
        {result.modelo.metricas_test && <p><b>Qué tan bien funcionó.</b> En la prueba final el modelo obtuvo un AUC de {result.modelo.metricas_test.auc_modelo}; medir solo el tamaño, {result.modelo.metricas_test.auc_solo_tamano}; el puntaje del detector, {result.modelo.metricas_test.auc_solo_puntaje_detector}. {result.modelo.metricas_test.auc_modelo < result.modelo.metricas_test.auc_solo_tamano ? 'El modelo no superó a medir el tamaño.' : 'El modelo quedó a la par o por encima de medir el tamaño.'}</p>}
        <p><b>Importante.</b> {result.advertencia}</p>
      </div>
    </>}

    {!selected && studies.length === 0 && <div className="ct-empty">
      <span aria-hidden="true">⇪</span>
      <strong>Todavía no hay estudios</strong>
      <p>Sube una tomografía para empezar. Los estudios quedan guardados en esta computadora, en <code>data/estudios</code>.</p>
    </div>}
    <p className="footnote">Prototipo de investigación sin validación clínica. No usar para tomar decisiones médicas.</p>
  </section>
}
