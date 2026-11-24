import { useCallback, useEffect, useRef, useState } from 'react'
import NiftiCanvas, { ViewButtons, type View } from './NiftiCanvas'
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
        <ViewButtons view={view} onChange={setView} />
      </div>
      <NiftiCanvas key={selectedUid} imageUrl={`${apiBase}/series/${encodeURIComponent(selectedUid)}/imagen`} view={view} />
      <p className="ct-warning">Este visor solo muestra las tomografías de la cohorte. Para buscar y marcar nódulos en un estudio, súbelo en la sección Analizar estudio.</p>
    </div> : <div className="ct-empty">
      <span aria-hidden="true">◫</span>
      <strong>Selecciona una tomografía disponible</strong>
      <p>Introduce el ID de un paciente que ya tenga una CT NIfTI en tu equipo, o abre el visor desde su ficha en la tabla de Pacientes.</p>
    </div>}
    <p className="footnote">Prototipo de investigación sin validación clínica. Los archivos NIfTI se consultan desde FastAPI en localhost.</p>
  </section>
}
