import { useEffect, useMemo, useState, type FormEvent } from 'react'
import NiftiCanvas, { ViewButtons, type View } from './NiftiCanvas'
import { readJson } from './api'
import { toFocus } from './medidas'
import { ModelNotes, NoduleTable, type ModelInfo, type Nodule } from './Nodulos'
import './Analisis.css'

type Status = 'en_cola' | 'procesando' | 'listo' | 'error'
type Study = { id: string; archivo: string; estado: Status; paso: string; error: string | null; creado: number }
type Result = {
  nodulos: Nodule[]
  candidatos_descartados: number
  segundos: number
  advertencia: string
  fuente: { tipo: string; cortes: number; series_en_el_zip?: number }
  modelo: ModelInfo
}
type StudyDetail = Study & { resultado: Result | null }

const STATUS_LABELS: Record<Status, string> = {
  en_cola: 'En cola', procesando: 'Analizando', listo: 'Listo', error: 'Error',
}
const running = (status: Status) => status === 'en_cola' || status === 'procesando'

export default function Analisis({ apiBase }: { apiBase: string }) {
  const [studies, setStudies] = useState<Study[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<StudyDetail | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<View>('multiplanar')
  const [target, setTarget] = useState<Nodule | null>(null)
  const focusOn = useMemo(() => (target ? toFocus(target) : null), [target])
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
        <NiftiCanvas key={selected} view={view} focusOn={focusOn}
          imageUrl={`${apiBase}/estudios/${selected}/imagen`} overlayUrl={`${apiBase}/estudios/${selected}/marcas`} />
      </div>

      <NoduleTable nodules={result.nodulos} discarded={result.candidatos_descartados} model={result.modelo}
        selected={target} onSelect={setTarget} />
      <ModelNotes model={result.modelo} warning={result.advertencia} />
    </>}

    {!selected && studies.length === 0 && <div className="ct-empty">
      <span aria-hidden="true">⇪</span>
      <strong>Todavía no hay estudios</strong>
      <p>Sube una tomografía para empezar. Los estudios quedan guardados en esta computadora, en <code>data/estudios</code>.</p>
    </div>}
    <p className="footnote">Prototipo de investigación sin validación clínica. No usar para tomar decisiones médicas.</p>
  </section>
}
