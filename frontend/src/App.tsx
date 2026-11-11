import { lazy, Suspense, useEffect, useState } from 'react'
import './App.css'

const CTViewer = lazy(() => import('./CTViewer'))

type Page = 'inicio' | 'pacientes' | 'visor'
type Split = '' | 'train' | 'validation' | 'test'

type Patient = {
  patient_id: string
  series_total: number
  series_analizadas: number
  nodulos_detectados: number | null
  nodulo_mayor_mm: number | null
  estado: 'pendiente' | 'disponible' | 'parcial'
}

type PatientList = {
  total: number
  limit: number
  offset: number
  pacientes: Patient[]
}

type PatientSeries = {
  series_uid: string
  split: string
  imagen_disponible: boolean
  analisis_disponible: boolean
  nodulos_detectados: number | null
}

type PatientDetail = Patient & { series: PatientSeries[] }

const PAGE_SIZE = 12
// Solo se utiliza en desarrollo local. El backend ya permite el origen 5173.
const API = 'http://127.0.0.1:8081'

function statusLabel(status: Patient['estado']) {
  if (status === 'disponible') return 'Analizado'
  if (status === 'parcial') return 'Parcial'
  return 'Pendiente'
}

function App() {
  const [page, setPage] = useState<Page>('inicio')
  const [split, setSplit] = useState<Split>('')
  const [offset, setOffset] = useState(0)
  const [search, setSearch] = useState('')
  const [list, setList] = useState<PatientList | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refresh, setRefresh] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<PatientDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [viewerSelection, setViewerSelection] = useState<{ patientId: string; seriesUid: string } | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    async function loadPatients() {
      setLoading(true)
      setError(null)
      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(offset),
        solo_con_resultados: 'false',
      })
      if (split) params.set('split', split)
      try {
        const response = await fetch(`${API}/pacientes?${params}`, { signal: controller.signal })
        if (!response.ok) throw new Error(`La API respondió HTTP ${response.status}`)
        const json = (await response.json()) as PatientList
        if (!Array.isArray(json.pacientes) || typeof json.total !== 'number') {
          throw new Error('La API devolvió un formato inesperado')
        }
        setList(json)
      } catch (reason) {
        if (controller.signal.aborted) return
        setError(reason instanceof Error ? reason.message : 'No se pudo consultar la API')
        setList(null)
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    void loadPatients()
    return () => controller.abort()
  }, [offset, split, refresh])

  useEffect(() => {
    if (!selectedId) {
      setDetail(null)
      return
    }
    const controller = new AbortController()
    setDetail(null)
    setDetailError(null)
    setDetailLoading(true)
    async function loadDetail() {
      try {
        const response = await fetch(`${API}/pacientes/${encodeURIComponent(selectedId!)}`, {
          signal: controller.signal,
        })
        if (!response.ok) throw new Error(`La API respondió HTTP ${response.status}`)
        setDetail((await response.json()) as PatientDetail)
      } catch (reason) {
        if (!controller.signal.aborted) {
          setDetailError(reason instanceof Error ? reason.message : 'Error desconocido')
        }
      } finally {
        if (!controller.signal.aborted) setDetailLoading(false)
      }
    }
    void loadDetail()
    return () => controller.abort()
  }, [selectedId])

  const patients = (list?.pacientes ?? []).filter(patient => patient.patient_id.includes(search.trim()))
  const analyzedOnPage = (list?.pacientes ?? []).filter(patient => patient.series_analizadas > 0).length

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">✳</span><div><strong>LungAI</strong><small>Imágenes pulmonares · investigación</small></div></div>
        <div className="menu-label">PLATAFORMA</div>
        <nav aria-label="Secciones principales">
          <button className={page === 'inicio' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('inicio')}>⌂ <span>Inicio</span></button>
          <button className={page === 'pacientes' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('pacientes')}>▤ <span>Pacientes</span></button>
          <button className={page === 'visor' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('visor')}>◫ <span>Visor CT</span></button>
        </nav>
        <div className="sidebar-note"><span className="status-dot"/> Prototipo académico<br/><small>Sin validación clínica</small></div>
      </aside>

      <main className="main-content">
        <header className="topbar"><span>PROYECTO LUNGAI / {page.toUpperCase()}</span><span className="local-tag">● Entorno local</span></header>

        {page === 'inicio' && <>
          <section className="hero">
            <span className="eyebrow">VISUALIZACIÓN DE TOMOGRAFÍAS</span>
            <h1>Exploración inteligente<br/>de imágenes pulmonares.</h1>
            <p>Una plataforma académica para consultar estudios de tomografía y visualizar posibles nódulos identificados por modelos experimentales.</p>
            <button className="primary-btn" onClick={() => setPage('pacientes')}>Explorar pacientes <span>→</span></button>
            <div className="orb orb-a"/><div className="orb orb-b"/>
          </section>
          <section className="section-title"><div><span className="eyebrow dark">PANORAMA</span><h2>Datos del proyecto</h2></div></section>
          <div className="stats-grid">
            <div className="stat-card"><span>Pacientes en muestra</span><strong>{loading ? '—' : list?.total.toLocaleString('es-MX') ?? '—'}</strong><small>{split ? `Filtro: ${split}` : 'Consulta desde Parquet'}</small></div>
            <div className="stat-card"><span>Pacientes visibles</span><strong>{loading ? '—' : list?.pacientes.length ?? '—'}</strong><small>En la página actual</small></div>
            <div className="stat-card"><span>Análisis disponibles</span><strong>{loading ? '—' : analyzedOnPage}</strong><small>Solo en la página actual</small></div>
          </div>
          {error && <div className="alert">No se pudo conectar con FastAPI: {error}. Comprueba que esté ejecutándose en el puerto 8081. <button onClick={() => setRefresh(v => v + 1)}>Reintentar</button></div>}
          <div className="disclaimer">Esta plataforma es un prototipo académico de investigación. Los hallazgos de un modelo no están validados para tomar decisiones clínicas.</div>
        </>}

        {page === 'pacientes' && <>
          <section className="page-heading"><div><span className="eyebrow dark">EXPLORADOR DE DATOS</span><h1>Pacientes</h1><p>Registros de la cohorte de investigación, consultados desde tu API FastAPI.</p></div><button className="outline-btn" onClick={() => setRefresh(v => v + 1)}>↻ Actualizar</button></section>
          <div className="filters">
            <label>División de datos
              <select value={split} onChange={event => {setSplit(event.target.value as Split);setOffset(0);setSearch('')}}>
                <option value="">Todos</option><option value="train">Entrenamiento</option><option value="validation">Validación</option><option value="test">Prueba</option>
              </select>
            </label>
            <label>Filtrar ID en esta página
              <input value={search} onChange={event => setSearch(event.target.value)} placeholder="Buscar en los registros visibles" />
            </label>
          </div>
          {error && <div className="alert">No pudimos leer los pacientes: {error}. <button onClick={() => setRefresh(v => v + 1)}>Reintentar</button></div>}
          <div className="table-card"><div className="table-caption"><strong>Listado de pacientes</strong><span>{list ? `${list.total.toLocaleString('es-MX')} registros` : 'Sin conexión'}</span></div>
            <div className="table-scroll"><table><thead><tr><th>Paciente</th><th>Series</th><th>Análisis</th><th>Nódulos reportados</th><th>Mayor tamaño</th><th></th></tr></thead>
              <tbody>{loading ? <tr><td colSpan={6} className="empty-cell">Cargando pacientes…</td></tr> : patients.length ? patients.map(patient => <tr key={patient.patient_id}>
                <td className="mono">{patient.patient_id}</td><td>{patient.series_total}</td><td><span className={`state state-${patient.estado}`}>{statusLabel(patient.estado)}</span></td>
                <td>{patient.nodulos_detectados ?? '—'}</td><td>{patient.nodulo_mayor_mm == null ? '—' : `${patient.nodulo_mayor_mm} mm`}</td>
                <td><button className="text-btn" onClick={() => setSelectedId(patient.patient_id)}>Ver detalle →</button></td>
              </tr>) : <tr><td colSpan={6} className="empty-cell">No hay pacientes para esta consulta.</td></tr>}</tbody>
            </table></div>
            <div className="pagination"><span>{list ? `${offset + (list.pacientes.length ? 1 : 0)}–${offset + list.pacientes.length} de ${list.total}` : '—'}</span><div><button disabled={loading || offset === 0} onClick={() => {setOffset(v => Math.max(0, v - PAGE_SIZE));setSearch('')}}>Anterior</button><button disabled={loading || !list || offset + PAGE_SIZE >= list.total} onClick={() => {setOffset(v => v + PAGE_SIZE);setSearch('')}}>Siguiente</button></div></div>
          </div>
          <p className="footnote">“Pendiente” significa que todavía no existe un JSON del detector, no que se hayan encontrado cero nódulos.</p>
        </>}

        {page === 'visor' && <Suspense fallback={<section className="placeholder"><p>Preparando el visor NiiVue…</p></section>}>
          <CTViewer key={viewerSelection?.seriesUid ?? 'manual'} apiBase={API} initialSelection={viewerSelection} />
        </Suspense>}
      </main>

      {selectedId && <div className="modal-shade" onClick={() => setSelectedId(null)}><section className="detail-panel" role="dialog" aria-modal="true" aria-label="Detalle de paciente" onClick={event => event.stopPropagation()}>
        <div className="detail-header"><div><span className="eyebrow dark">FICHA DE INVESTIGACIÓN</span><h2>Paciente {selectedId}</h2></div><button className="close-btn" aria-label="Cerrar detalle" onClick={() => setSelectedId(null)}>✕</button></div>
        {detailLoading && <p>Cargando información…</p>}{detailError && <p className="alert">{detailError}</p>}
        {detail && <><p className="muted">Series registradas: {detail.series_total} · Estado: {statusLabel(detail.estado)}</p>
          {detail.series.map(series => <div className="series-card" key={series.series_uid}><strong>Serie de tomografía</strong><small className="uid">{series.series_uid}</small><div className="series-row"><span>Conjunto</span><b>{series.split}</b></div><div className="series-row"><span>Archivo NIfTI</span><b>{series.imagen_disponible ? 'Disponible' : 'No disponible'}</b></div><div className="series-row"><span>Análisis de nódulos</span><b>{series.analisis_disponible ? 'Disponible' : 'Pendiente'}</b></div><div className="series-row"><span>Nódulos reportados</span><b>{series.nodulos_detectados ?? '—'}</b></div>{series.imagen_disponible && <button type="button" className="outline-btn" style={{ marginTop: 16, width: '100%' }} onClick={() => { setViewerSelection({ patientId: detail.patient_id, seriesUid: series.series_uid }); setSelectedId(null); setPage('visor') }}>Abrir tomografía en visor →</button>}</div>)}
          <p className="footnote">Resultados experimentales sin validación clínica.</p></>}
      </section></div>}
    </div>
  )
}

export default App
