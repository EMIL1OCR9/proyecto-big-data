import { useCallback, useEffect, useRef, useState } from 'react'
import { Niivue, SLICE_TYPE } from '@niivue/niivue'
import './CTViewer.css'

type InitialSelection = { patientId: string; seriesUid: string }
type Props = { apiBase: string; initialSelection: InitialSelection | null }

type Series = {
  series_uid: string
  split: string
  imagen_disponible: boolean
  analisis_disponible: boolean
  nodulos_detectados: number | null
}
type PatientDetail = { patient_id: string; series: Series[] }
type View = 'multiplanar' | 'axial' | 'coronal' | 'sagittal' | 'render'

const VIEW_TYPES: Record<View, SLICE_TYPE> = {
  multiplanar: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  coronal: SLICE_TYPE.CORONAL,
  sagittal: SLICE_TYPE.SAGITTAL,
  render: SLICE_TYPE.RENDER,
}

/** Un único canvas NiiVue por CT. No descarga DICOM ni ejecuta inferencia. */
function NiftiCanvas({ apiBase, uid, view }: { apiBase: string; uid: string; view: View }) {
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const nv = useRef<Niivue | null>(null)
  const viewRef = useRef(view)
  const [phase, setPhase] = useState<'cargando' | 'lista' | 'error'>('cargando')
  const [message, setMessage] = useState('Cargando tomografía NIfTI…')

  useEffect(() => { viewRef.current = view; nv.current?.setSliceType(VIEW_TYPES[view]) }, [view])

  useEffect(() => {
    const element = canvas.current
    if (!element) return
    let cancelled = false
    const viewer = new Niivue({
      backColor: [0.035, 0.070, 0.115, 1],
      crosshairColor: [0.20, 0.88, 0.76, 1],
      crosshairWidth: 1,
      isResizeCanvas: true,
    })
    nv.current = viewer

    async function load() {
      try {
        await viewer.attachToCanvas(element!)
        if (cancelled) return
        // La API entrega NIfTI comprimido. El nombre ayuda a reconocer .nii.gz.
        await viewer.loadVolumes([{
          url: `${apiBase}/series/${encodeURIComponent(uid)}/imagen`,
          name: 'tomografia.nii.gz',
          colormap: 'gray',
          cal_min: -1000,
          cal_max: 400,
        }])
        if (cancelled) return
        viewer.setSliceType(VIEW_TYPES[viewRef.current])
        setPhase('lista')
        setMessage('Tomografía cargada. Usa la rueda del ratón para recorrer los cortes.')
      } catch (reason) {
        if (cancelled) return
        setPhase('error')
        const detail = reason instanceof Error ? reason.message : 'Error desconocido'
        setMessage(`No se pudo abrir la tomografía: ${detail}`)
      }
    }
    void load()

    return () => {
      cancelled = true
      if (nv.current === viewer) nv.current = null
      // Liberar recursos WebGL al cambiar de serie o abandonar la pantalla.
      try { viewer.cleanup() } catch { /* La limpieza puede coincidir con un montaje interrumpido. */ }
    }
  }, [apiBase, uid])

  return <div className="ct-canvas-shell">
    <canvas className="ct-canvas" ref={canvas} aria-label="Visor interactivo de tomografía" />
    {phase !== 'lista' && <div className={`ct-canvas-message ct-${phase}`} role="status">{message}</div>}
    {phase === 'lista' && <p className="ct-canvas-caption">{message} Imagen de investigación, no validada para uso clínico.</p>}
  </div>
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
      const available = data.series.find(s => s.series_uid === preferredUid && s.imagen_disponible)
      if (available) setSelectedUid(available.series_uid)
      else if (!data.series.some(s => s.imagen_disponible)) {
        setError('El paciente existe, pero su NIfTI todavía no está descargado en este equipo.')
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

  return <section className="ct-page">
    <div className="page-heading ct-heading"><div>
      <span className="eyebrow dark">VISUALIZACIÓN DE IMÁGENES</span>
      <h1>Visor CT</h1>
      <p>Cortes axiales, coronales y sagitales, además de una vista volumétrica experimental.</p>
    </div></div>

    <form className="ct-search" onSubmit={event => { event.preventDefault(); void lookup(patientInput) }}>
      <label htmlFor="ct-patient">ID de paciente de la cohorte</label>
      <div className="ct-search-controls">
        <input id="ct-patient" value={patientInput} onChange={event => setPatientInput(event.target.value)} placeholder="Escribe un ID con NIfTI disponible" autoComplete="off" />
        <button type="submit" className="outline-btn" disabled={busy}>{busy ? 'Buscando…' : 'Buscar CT'}</button>
      </div>
      <small>Se consulta FastAPI en tu computadora. No se envían tomografías a servicios externos.</small>
    </form>

    {error && <p role="alert" className="alert">{error}</p>}
    {patient && <div className="ct-series-list">
      <strong>Series del paciente {patient.patient_id}</strong>
      <div className="ct-series-buttons">
        {patient.series.map(series => <button
          type="button" key={series.series_uid}
          className={selectedUid === series.series_uid ? 'ct-series-choice selected' : 'ct-series-choice'}
          disabled={!series.imagen_disponible}
          onClick={() => { setSelectedUid(series.series_uid); setView('multiplanar') }}
          title={series.imagen_disponible ? 'Abrir esta CT' : 'No hay NIfTI disponible'}>
          <span>{series.split || 'Sin split'} · {series.imagen_disponible ? 'CT disponible' : 'CT no descargada'}</span>
          <small>{series.analisis_disponible ? 'Existe resultado de detector' : 'Detector pendiente'}</small>
        </button>)}
      </div>
    </div>}

    {selectedUid ? <div className="ct-viewer-card">
      <div className="ct-toolbar">
        <div className="ct-toolbar-title"><strong>Tomografía NIfTI</strong><small>Escala de grises · ventana aproximada de pulmón (-1000 a 400 HU)</small></div>
        <div className="ct-view-buttons" aria-label="Cambiar vista">
          {([
            ['multiplanar', '4 planos'], ['axial', 'Axial'], ['coronal', 'Coronal'],
            ['sagittal', 'Sagital'], ['render', '3D'],
          ] as [View, string][]).map(([type, label]) => <button key={type} type="button"
            className={view === type ? 'selected' : ''} aria-pressed={view === type} onClick={() => setView(type)}>{label}</button>)}
        </div>
      </div>
      <NiftiCanvas key={selectedUid} apiBase={apiBase} uid={selectedUid} view={view} />
      <p className="ct-warning">El visor solo muestra imágenes. No identifica nódulos nuevos y todavía no dibuja marcadores: primero validaremos la orientación y las coordenadas físicas del JSON del detector.</p>
    </div> : <div className="ct-empty">
      <span aria-hidden="true">◫</span>
      <strong>Selecciona una tomografía disponible</strong>
      <p>Introduce el ID de un paciente que ya tenga una CT NIfTI en tu equipo, o abre el visor desde su ficha en la tabla de Pacientes.</p>
    </div>}
    <p className="footnote">Prototipo de investigación sin validación clínica. Los archivos NIfTI se consultan desde FastAPI en localhost.</p>
  </section>
}
