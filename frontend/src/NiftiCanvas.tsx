import { useEffect, useRef, useState } from 'react'
import { DRAG_MODE, MULTIPLANAR_TYPE, Niivue, SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue'
import './CTViewer.css'

export type View = 'multiplanar' | 'axial' | 'coronal' | 'sagittal' | 'render'

const VIEW_TYPES: Record<View, SLICE_TYPE> = {
  multiplanar: SLICE_TYPE.MULTIPLANAR,
  axial: SLICE_TYPE.AXIAL,
  coronal: SLICE_TYPE.CORONAL,
  sagittal: SLICE_TYPE.SAGITTAL,
  render: SLICE_TYPE.RENDER,
}

const VIEW_LABELS: [View, string][] = [
  ['multiplanar', '4 planos'], ['axial', 'Axial'], ['coronal', 'Coronal'], ['sagittal', 'Sagital'], ['render', '3D'],
]

const MAX_ZOOM = 8
const MARK_OPACITY = 0.8

// Ventanas de contraste estándar en HU (nivel ± ancho / 2).
type Window = 'pulmon' | 'mediastino' | 'hueso'
const WINDOWS: Record<Window, { label: string; range: [number, number] }> = {
  pulmon: { label: 'Pulmón', range: [-1350, 150] }, // nivel -600, ancho 1500
  mediastino: { label: 'Mediastino', range: [-135, 215] }, // nivel 40, ancho 350: grasa, vasos, calcio
  hueso: { label: 'Hueso', range: [-500, 1500] }, // nivel 500, ancho 2000
}
const NODULE_ZOOM = 3 // al ir a un nódulo: unos 12 cm de campo, el nódulo se ve con su entorno

export function ViewButtons({ view, onChange }: { view: View; onChange: (view: View) => void }) {
  return <div className="ct-view-buttons" aria-label="Cambiar vista">
    {VIEW_LABELS.map(([type, label]) => <button key={type} type="button"
      className={view === type ? 'selected' : ''} aria-pressed={view === type} onClick={() => onChange(type)}>{label}</button>)}
  </div>
}

/** Un nódulo al que acercarse: centro en mm (RAS) y, si se conoce, su tamaño (x, y, z) en mm. */
export type Focus = { center: number[]; size?: number[]; label?: string }

type Props = {
  imageUrl: string
  /** Máscara con los nódulos, misma geometría que la CT; se dibuja en rojo encima. */
  overlayUrl?: string
  view: View
  /** Nódulo al que se acerca la vista; con `size`, se dibuja su medida en cada corte. */
  focusOn?: Focus | null
}

// Clic: mover la cruz · clic derecho o central y arrastrar: mover la imagen · Mayús: contraste.
const BROWSE = {
  leftButton: { primary: DRAG_MODE.crosshair, withShift: DRAG_MODE.contrast },
  rightButton: DRAG_MODE.pan,
  centerButton: DRAG_MODE.pan,
}
// Con la regla, arrastrar con el clic traza una medida en mm.
const MEASURE = { ...BROWSE, leftButton: { primary: DRAG_MODE.measurement, withShift: DRAG_MODE.contrast } }

/** Un único canvas NiiVue por CT; móntalo con `key` para cambiar de estudio. No ejecuta inferencia. */
export default function NiftiCanvas({ imageUrl, overlayUrl, view, focusOn }: Props) {
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const nv = useRef<Niivue | null>(null)
  const viewRef = useRef(view)
  const focusRef = useRef(focusOn)
  const [phase, setPhase] = useState<'cargando' | 'lista' | 'error'>('cargando')
  const [message, setMessage] = useState('Cargando tomografía NIfTI…')
  const [zoom, setZoom] = useState(1)
  const [showMarks, setShowMarks] = useState(true)
  const [measuring, setMeasuring] = useState(false)
  const [windowName, setWindowName] = useState<Window>('pulmon')

  useEffect(() => { viewRef.current = view; nv.current?.setSliceType(VIEW_TYPES[view]) }, [view])

  useEffect(() => {
    focusRef.current = focusOn
    const viewer = nv.current
    if (phase !== 'lista' || !viewer || !focusOn) return
    const frame = requestAnimationFrame(() => setZoom(focus(viewer, focusOn, NODULE_ZOOM)))
    return () => cancelAnimationFrame(frame)
  }, [phase, focusOn])

  useEffect(() => {
    const element = canvas.current
    if (!element) return
    let cancelled = false
    const viewer = new Niivue({
      backColor: [0.035, 0.070, 0.115, 1],
      crosshairColor: [0.20, 0.88, 0.76, 0.45], // semitransparente: pasa justo encima del nódulo
      crosshairWidth: 1,
      isResizeCanvas: true,
      // Cuadrícula fija de 2 x 2 con la vista 3D: el acomodo automático pone los cortes en
      // una fila que no cabe en pantallas anchas.
      multiplanarLayout: MULTIPLANAR_TYPE.GRID,
      multiplanarShowRender: SHOW_RENDER.ALWAYS,
      mouseEventConfig: BROWSE,
      // Medidas en amarillo, para no confundirlas con los anillos rojos de los nódulos.
      measureLineColor: [1, 0.84, 0.25, 1],
      measureTextColor: [1, 0.84, 0.25, 1],
    })
    nv.current = viewer

    // Ctrl + rueda acerca o aleja; la rueda sola sigue cambiando de corte.
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey) return
      event.preventDefault()
      event.stopImmediatePropagation()
      setZoom(zoomBy(viewer, event.deltaY < 0 ? 1.25 : 0.8))
    }
    element.addEventListener('wheel', onWheel, { capture: true, passive: false })

    async function load() {
      try {
        await viewer.attachToCanvas(element!)
        if (cancelled) return
        // La API entrega NIfTI comprimido. El nombre ayuda a reconocer .nii.gz.
        const [low, high] = WINDOWS.pulmon.range
        const volumes = [{ url: imageUrl, name: 'tomografia.nii.gz', colormap: 'gray', cal_min: low, cal_max: high }]
        if (overlayUrl) volumes.push({ url: overlayUrl, name: 'nodulos.nii.gz', colormap: 'red', cal_min: 0.5, cal_max: 1 })
        await viewer.loadVolumes(volumes)
        if (cancelled) return
        if (overlayUrl) viewer.setOpacity(1, MARK_OPACITY)
        viewer.setSliceType(VIEW_TYPES[viewRef.current])
        if (focusRef.current) setZoom(focus(viewer, focusRef.current, NODULE_ZOOM))
        setPhase('lista')
        setMessage('Rueda: cambiar de corte · Ctrl + rueda o botones: acercar · clic derecho y arrastrar: mover la imagen.')
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
      element.removeEventListener('wheel', onWheel, { capture: true })
      if (nv.current === viewer) nv.current = null
      // Liberar recursos WebGL al cambiar de serie o abandonar la pantalla.
      try { viewer.cleanup() } catch { /* La limpieza puede coincidir con un montaje interrumpido. */ }
    }
  }, [imageUrl, overlayUrl])

  function zoomWith(factor: number) {
    if (nv.current) setZoom(factor === 1 ? fit(nv.current) : zoomBy(nv.current, factor))
  }

  function toggleMarks() {
    nv.current?.setOpacity(1, showMarks ? 0 : MARK_OPACITY)
    setShowMarks(!showMarks)
  }

  function toggleRuler() {
    nv.current?.setMouseEventConfig(measuring ? BROWSE : MEASURE)
    setMeasuring(!measuring)
  }

  function clearMeasures() {
    nv.current?.clearMeasurements()
    nv.current?.drawScene()
  }

  function applyWindow(name: Window) {
    const viewer = nv.current
    if (!viewer?.volumes.length) return
    ;[viewer.volumes[0].cal_min, viewer.volumes[0].cal_max] = WINDOWS[name].range
    viewer.updateGLVolume()
    setWindowName(name)
  }

  function capture() {
    // Guarda como PNG lo que se ve en el visor, con marcas y medidas.
    void nv.current?.saveScene(`lungai-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`)
  }

  return <div className="ct-canvas-shell">
    <canvas className={measuring ? 'ct-canvas ct-measuring' : 'ct-canvas'} ref={canvas} aria-label="Visor interactivo de tomografía" onContextMenu={event => event.preventDefault()} />
    {phase !== 'lista' && <div className={`ct-canvas-message ct-${phase}`} role="status">{message}</div>}
    {phase === 'lista' && <>
      <div className="ct-window" aria-label="Ventana de contraste">
        {(Object.keys(WINDOWS) as Window[]).map(name => <button key={name} type="button"
          className={windowName === name ? 'on' : ''} aria-pressed={windowName === name} onClick={() => applyWindow(name)}>{WINDOWS[name].label}</button>)}
        <button type="button" onClick={capture} title="Guardar como imagen lo que se ve en el visor">Guardar imagen</button>
      </div>
      {focusOn?.label && <div className="ct-focus" role="status">{focusOn.label}</div>}
      <div className="ct-zoom" aria-label="Zoom y medidas">
        <button type="button" onClick={() => zoomWith(0.8)} disabled={zoom <= 1} aria-label="Alejar">−</button>
        <span>{zoom.toFixed(1)}×</span>
        <button type="button" onClick={() => zoomWith(1.25)} disabled={zoom >= MAX_ZOOM} aria-label="Acercar">+</button>
        <button type="button" onClick={() => zoomWith(1)}>Ajustar</button>
        <button type="button" onClick={toggleRuler} aria-pressed={measuring} className={measuring ? 'on' : ''}>{measuring ? 'Midiendo…' : 'Medir'}</button>
        <button type="button" onClick={clearMeasures}>Borrar medidas</button>
        {overlayUrl && <button type="button" onClick={toggleMarks} aria-pressed={showMarks}>{showMarks ? 'Ocultar marcas' : 'Ver marcas'}</button>}
      </div>
      <p className="ct-canvas-caption">{measuring
        ? 'Regla: arrastra con el clic sobre un corte para medir en mm; la medida se queda en ese corte. Vuelve a pulsar Medir para terminar.'
        : message} Imagen de investigación, no validada para uso clínico.</p>
    </>}
  </div>
}

/**
 * NiiVue muestra en 2D la ventana (extremos − pan) / zoom, en mm. Para dejar el punto P en el
 * centro con zoom Z: pan = C − Z · P, donde C es el centro del volumen.
 */
function center(viewer: Niivue, mm: number[], zoom: number): number {
  const middle = viewer.frac2mm([0.5, 0.5, 0.5])
  viewer.setPan2Dxyzmm([0, 1, 2].map(i => middle[i] - zoom * mm[i]).concat(zoom) as [number, number, number, number])
  return zoom
}

/** Acerca o aleja alrededor de la cruz. */
function zoomBy(viewer: Niivue, factor: number): number {
  const zoom = Math.min(MAX_ZOOM, Math.max(1, viewer.scene.pan2Dxyzmm[3] * factor))
  if (zoom === 1) return fit(viewer)
  const mm = viewer.frac2mm(viewer.scene.crosshairPos)
  return center(viewer, [mm[0], mm[1], mm[2]], zoom)
}

/** Vuelve a la vista completa. */
function fit(viewer: Niivue): number {
  viewer.setPan2Dxyzmm([0, 0, 0, 1])
  return 1
}

/** Pone la cruz en el nódulo, lo deja centrado con ese zoom y dibuja su medida en cada corte. */
function focus(viewer: Niivue, target: Focus, zoom: number): number {
  const [x, y, z] = target.center
  viewer.scene.crosshairPos = viewer.mm2frac([x, y, z])
  viewer.createOnLocationChange()
  viewer.clearMeasurements()
  if (target.size) {
    const [sx, sy, sz] = target.size
    // Una medida por corte, para que los textos no se encimen: la transversal (izquierda-derecha)
    // en el axial, la craneocaudal (cabeza-pies) en el coronal y la anteroposterior en el sagital.
    addMeasure(viewer, SLICE_TYPE.AXIAL, [x - sx / 2, y, z], [x + sx / 2, y, z])
    addMeasure(viewer, SLICE_TYPE.CORONAL, [x, y, z - sz / 2], [x, y, z + sz / 2])
    addMeasure(viewer, SLICE_TYPE.SAGITTAL, [x, y - sy / 2, z], [x, y + sy / 2, z])
  }
  return center(viewer, target.center, zoom)
}

/** Agrega una medida como las de la regla, en el corte donde está la cruz. */
function addMeasure(viewer: Niivue, sliceType: SLICE_TYPE, start: number[], end: number[]) {
  viewer.document.completedMeasurements.push({
    startMM: Float32Array.from(start),
    endMM: Float32Array.from(end),
    distance: Math.hypot(...start.map((value, i) => end[i] - value)),
    sliceIndex: 0,
    sliceType,
    slicePosition: viewer.getCurrentSlicePosition(sliceType),
  })
}
