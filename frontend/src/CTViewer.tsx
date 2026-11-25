import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import NiftiCanvas, { ViewButtons, type View } from './NiftiCanvas'
import { readJson } from './api'
import { toFocus } from './medidas'
import { ModelNotes, NoduleTable, type ModelInfo, type Nodule } from './Nodulos'
import './CTViewer.css'

type InitialSelection = { patientId: string; seriesUid: string; recorte?: number }
type Props = { apiBase: string; initialSelection: InitialSelection | null }

type Series = {
  series_uid: string
  split: string
  imagen_disponible: boolean
  analisis_disponible: boolean
  nodulos_detectados: number | null
}
type PatientDetail = { patient_id: string; series: Series[] }

const openable = (series: Series) => series.imagen_disponible || series.analisis_disponible

type Preparation = {
  estado: 'pendiente' | 'en_cola' | 'procesando' | 'listo' | 'error'
  paso: string
  error: string | null
  resultado: {
    split: string
    etiqueta_cancer_12m: number | null
    nodulos: Nodule[]
    candidatos_descartados: number
    modelo: ModelInfo
    etiquetas_disponibles: boolean
    advertencia: string
  } | null
}

type CohortProps = {
  apiBase: string
  uid: string
  view: View
  onView: (view: View) => void
  /** Al abrir un caso desde Resultados: se prepara sola y se acerca a ese nódulo. */
  recorte?: number
}

/** Serie ya analizada en Colab: si falta, la CT se baja de IDC y se marcan sus nódulos del JSON. */
function CohortSeries({ apiBase, uid, view, onView, recorte }: CohortProps) {
  const [state, setState] = useState<Preparation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [target, setTarget] = useState<Nodule | null>(null)
  const focusOn = useMemo(() => (target ? toFocus(target) : null), [target])
  const [requests, setRequests] = useState(0)
  const pending = useRef(recorte)
  const base = `${apiBase}/series/${encodeURIComponent(uid)}`

  // Mientras se baja y se marca, se pregunta el estado cada 2 segundos.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined
    const controller = new AbortController()
    async function poll() {
      try {
        const data = await readJson<Preparation>(await fetch(`${base}/preparacion`, { signal: controller.signal }))
        if (data.estado === 'pendiente' && pending.current != null) {
          await readJson(await fetch(`${base}/preparar`, { method: 'POST', signal: controller.signal }))
          timer = setTimeout(poll, 1000)
          return
        }
        setState(data)
        if (data.estado === 'listo' && pending.current != null) {
          const nodule = data.resultado?.nodulos.find(item => item.recorte === pending.current)
          pending.current = undefined
          if (nodule) setTarget(nodule)
        }
        if (data.estado === 'en_cola' || data.estado === 'procesando') timer = setTimeout(poll, 2000)
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Error desconocido')
      }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [base, requests])

  async function prepare(redo = false) {
    setError(null)
    try {
      await readJson(await fetch(`${base}/preparar${redo ? '?rehacer=true' : ''}`, { method: 'POST' }))
      setRequests(value => value + 1)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'No se pudo preparar la serie.')
    }
  }

  const result = state?.estado === 'listo' ? state.resultado : null
  return <>
    {error && <p role="alert" className="alert">{error}</p>}
    {state?.estado === 'pendiente' && <div className="ct-empty">
      <span aria-hidden="true">⇩</span>
      <strong>Sus nódulos ya se detectaron en Colab</strong>
      <p>Para verlos sobre la tomografía hay que bajarla del archivo público IDC: unos 100–200 MB y alrededor de un minuto. No se vuelve a correr el detector.</p>
      <button type="button" className="outline-btn" onClick={() => void prepare()}>Bajar de IDC y abrir</button>
    </div>}
    {(state?.estado === 'en_cola' || state?.estado === 'procesando') && <div className="an-progress" role="status">
      <span className="an-spinner" aria-hidden="true" />
      <div><strong>Preparando la serie</strong><small>{state.paso}…</small></div>
    </div>}
    {state?.estado === 'error' && <div className="alert" role="alert">No se pudo preparar la serie: {state.error} <button type="button" onClick={() => void prepare()}>Reintentar</button></div>}
    {result && <>
      <div className="an-patient">
        <span>Conjunto: <b>{result.split}</b></span>
        {result.etiqueta_cancer_12m != null && <span>{result.etiqueta_cancer_12m ? 'Tuvo cáncer de pulmón en los 12 meses siguientes' : 'No tuvo cáncer de pulmón en los 12 meses siguientes'}</span>}
      </div>
      <div className="ct-viewer-card">
        <div className="ct-toolbar">
          <div className="ct-toolbar-title"><strong>Tomografía de la cohorte</strong><small>Nódulos del análisis de Colab, marcados en rojo</small></div>
          <ViewButtons view={view} onChange={onView} />
        </div>
        <NiftiCanvas view={view} focusOn={focusOn} imageUrl={`${base}/imagen`} overlayUrl={`${base}/marcas`} />
      </div>
      <NoduleTable nodules={result.nodulos} discarded={result.candidatos_descartados} model={result.modelo}
        selected={target} onSelect={setTarget} withTruth={result.etiquetas_disponibles} />
      <ModelNotes model={result.modelo} warning={result.advertencia} split={result.split} withTruth={result.etiquetas_disponibles} />
      {!result.etiquetas_disponibles && <p className="ct-warning">Para ver si cada nódulo fue cáncer según el NLST, copia <code>etiquetas/candidatos_etiquetados.parquet</code> de Drive a <code>outputs/etiquetas/</code> y <button type="button" className="text-btn" onClick={() => void prepare(true)}>vuelve a preparar la serie</button> (la CT ya bajada se reutiliza).</p>}
    </>}
  </>
}

export default function CTViewer({ apiBase, initialSelection }: Props) {
  const [patientInput, setPatientInput] = useState(initialSelection?.patientId ?? '')
  const [patient, setPatient] = useState<PatientDetail | null>(null)
  const [selectedUid, setSelectedUid] = useState<string | null>(null)
  const [view, setView] = useState<View>('multiplanar')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestNumber = useRef(0)

  const lookup = useCallback(async (rawId: string, preferredUid?: string) => {
    const id = rawId.trim()
    if (!id) { setError('Escribe un ID de paciente de la cohorte.'); return }
    // No exponemos búsquedas generales ni archivos fuera de la cohorte.
    const sequence = ++requestNumber.current
    setBusy(true)
    setError(null)
    setPatient(null)
    setSelectedUid(null)
    try {
      const response = await fetch(`${apiBase}/pacientes/${encodeURIComponent(id)}`)
      if (!response.ok) {
        if (response.status === 404) throw new Error('Este paciente no está en la muestra.')
        throw new Error(`La API respondió HTTP ${response.status}.`)
      }
      const data: PatientDetail = await response.json()
      if (!Array.isArray(data.series)) throw new Error('Formato inesperado del backend.')
      if (sequence !== requestNumber.current) return
      setPatient(data)
      const usable = data.series.filter(openable)
      const preferred = usable.find(s => s.series_uid === preferredUid) ?? (usable.length === 1 ? usable[0] : undefined)
      if (preferred) setSelectedUid(preferred.series_uid)
      else if (!usable.length) {
        setError('Este paciente todavía no tiene análisis ni tomografía en este equipo. Copia la carpeta nodulos de Drive a outputs/nodulos.')
      }
    } catch (reason) {
      if (sequence !== requestNumber.current) return
      setError(reason instanceof Error ? reason.message : 'No fue posible consultar el paciente.')
    } finally {
      if (sequence === requestNumber.current) setBusy(false)
    }
  }, [apiBase])

  useEffect(() => {
    if (initialSelection) void lookup(initialSelection.patientId, initialSelection.seriesUid)
    return () => { requestNumber.current++ }
  }, [initialSelection, lookup])

  const selected = patient?.series.find(series => series.series_uid === selectedUid)

  return <section className="ct-page">
    <div className="page-heading ct-heading"><div>
      <span className="eyebrow dark">VISUALIZACIÓN DE IMÁGENES</span>
      <h1>Visor CT</h1>
      <p>Cortes axiales, coronales y sagitales, además de una vista volumétrica experimental.</p>
    </div></div>

    <form className="ct-search" onSubmit={event => { event.preventDefault(); void lookup(patientInput) }}>
      <label htmlFor="ct-patient">ID de paciente de la cohorte</label>
      <div className="ct-search-controls">
        <input id="ct-patient" value={patientInput} onChange={event => setPatientInput(event.target.value)} placeholder="Por ejemplo, un paciente de la tabla de Pacientes" autoComplete="off" />
        <button type="submit" className="outline-btn" disabled={busy}>{busy ? 'Buscando…' : 'Buscar CT'}</button>
      </div>
      <small>Se consulta FastAPI en tu computadora. Las tomografías que falten se bajan del archivo público IDC; no se envían datos a ningún lado.</small>
    </form>

    {error && <p role="alert" className="alert">{error}</p>}
    {patient && <div className="ct-series-list">
      <strong>Series del paciente {patient.patient_id}</strong>
      <div className="ct-series-buttons">
        {patient.series.map(series => <button
          type="button" key={series.series_uid}
          className={selectedUid === series.series_uid ? 'ct-series-choice selected' : 'ct-series-choice'}
          disabled={!openable(series)}
          onClick={() => { setSelectedUid(series.series_uid); setView('multiplanar') }}
          title={openable(series) ? 'Abrir esta CT' : 'No hay análisis ni CT en este equipo'}>
          <span>{series.split || 'Sin split'} · {series.analisis_disponible ? `${series.nodulos_detectados} nódulos (Colab)` : 'Sin análisis'}</span>
          <small>{series.imagen_disponible ? 'CT en este equipo' : series.analisis_disponible ? 'La CT se baja de IDC al abrirla' : 'CT no disponible'}</small>
        </button>)}
      </div>
    </div>}

    {selectedUid && selected?.analisis_disponible ? <CohortSeries key={selectedUid} apiBase={apiBase} uid={selectedUid} view={view} onView={setView}
      recorte={selectedUid === initialSelection?.seriesUid ? initialSelection.recorte : undefined} />
      : selectedUid ? <div className="ct-viewer-card">
        <div className="ct-toolbar">
          <div className="ct-toolbar-title"><strong>Tomografía NIfTI</strong><small>Escala de grises · ventana aproximada de pulmón (-1000 a 400 HU)</small></div>
          <ViewButtons view={view} onChange={setView} />
        </div>
        <NiftiCanvas key={selectedUid} imageUrl={`${apiBase}/series/${encodeURIComponent(selectedUid)}/imagen`} view={view} />
        <p className="ct-warning">Esta serie no tiene resultado del detector, así que solo se muestra la imagen.</p>
      </div> : <div className="ct-empty">
        <span aria-hidden="true">◫</span>
        <strong>Selecciona una tomografía de la cohorte</strong>
        <p>Escribe el ID de un paciente procesado en Colab, o ábrelo desde su ficha en la tabla de Pacientes. Si su tomografía no está en este equipo, se baja de IDC al abrirla.</p>
      </div>}
    <p className="footnote">Prototipo de investigación sin validación clínica. Los archivos NIfTI se consultan desde FastAPI en localhost.</p>
  </section>
}
