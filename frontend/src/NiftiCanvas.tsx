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
const NODULE_ZOOM = 3 // al ir a un nódulo: unos 12 cm de campo, el nódulo se ve con su entorno

export function ViewButtons({ view, onChange }: { view: View; onChange: (view: View) => void }) {
  return <div className="ct-view-buttons" aria-label="Cambiar vista">
    {VIEW_LABELS.map(([type, label]) => <button key={type} type="button"
      className={view === type ? 'selected' : ''} aria-pressed={view === type} onClick={() => onChange(type)}>{label}</button>)}
  </div>
}

type Props = {
  imageUrl: string
  /** Máscara con los nódulos, misma geometría que la CT; se dibuja en rojo encima. */
  overlayUrl?: string
  view: View
  /** Punto en mm (RAS) al que se acerca la vista, por ejemplo el centro de un nódulo. */
  target?: number[] | null
}

/** Un único canvas NiiVue por CT; móntalo con `key` para cambiar de estudio. No ejecuta inferencia. */
export default function NiftiCanvas({ imageUrl, overlayUrl, view, target }: Props) {
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const nv = useRef<Niivue | null>(null)
  const viewRef = useRef(view)
  const targetRef = useRef(target)
  const [phase, setPhase] = useState<'cargando' | 'lista' | 'error'>('cargando')
  const [message, setMessage] = useState('Cargando tomografía NIfTI…')
  const [zoom, setZoom] = useState(1)
  const [showMarks, setShowMarks] = useState(true)

  useEffect(() => { viewRef.current = view; nv.current?.setSliceType(VIEW_TYPES[view]) }, [view])

  useEffect(() => {
    targetRef.current = target
    const viewer = nv.current
    if (phase !== 'lista' || !viewer || !target) return
    const frame = requestAnimationFrame(() => setZoom(focus(viewer, target, NODULE_ZOOM)))
    return () => cancelAnimationFrame(frame)
  }, [phase, target])

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
      // Clic: mover la cruz · clic derecho o central y arrastrar: mover la imagen · Mayús: contraste.
      mouseEventConfig: {
        leftButton: { primary: DRAG_MODE.crosshair, withShift: DRAG_MODE.contrast },
        rightButton: DRAG_MODE.pan,
        centerButton: DRAG_MODE.pan,
      },
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
        const volumes = [{ url: imageUrl, name: 'tomografia.nii.gz', colormap: 'gray', cal_min: -1000, cal_max: 400 }]
        if (overlayUrl) volumes.push({ url: overlayUrl, name: 'nodulos.nii.gz', colormap: 'red', cal_min: 0.5, cal_max: 1 })
        await viewer.loadVolumes(volumes)
        if (cancelled) return
        if (overlayUrl) viewer.setOpacity(1, MARK_OPACITY)
        viewer.setSliceType(VIEW_TYPES[viewRef.current])
        if (targetRef.current) setZoom(focus(viewer, targetRef.current, NODULE_ZOOM))
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

  return <div className="ct-canvas-shell">
    <canvas className="ct-canvas" ref={canvas} aria-label="Visor interactivo de tomografía" onContextMenu={event => event.preventDefault()} />
    {phase !== 'lista' && <div className={`ct-canvas-message ct-${phase}`} role="status">{message}</div>}
    {phase === 'lista' && <>
      <div className="ct-zoom" aria-label="Zoom">
        <button type="button" onClick={() => zoomWith(0.8)} disabled={zoom <= 1} aria-label="Alejar">−</button>
        <span>{zoom.toFixed(1)}×</span>
        <button type="button" onClick={() => zoomWith(1.25)} disabled={zoom >= MAX_ZOOM} aria-label="Acercar">+</button>
        <button type="button" onClick={() => zoomWith(1)}>Ajustar</button>
        {overlayUrl && <button type="button" onClick={toggleMarks} aria-pressed={showMarks}>{showMarks ? 'Ocultar marcas' : 'Ver marcas'}</button>}
      </div>
      <p className="ct-canvas-caption">{message} Imagen de investigación, no validada para uso clínico.</p>
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

/** Pone la cruz en el punto y lo deja centrado con ese zoom. */
function focus(viewer: Niivue, mm: number[], zoom: number): number {
  viewer.scene.crosshairPos = viewer.mm2frac([mm[0], mm[1], mm[2]])
  viewer.createOnLocationChange()
  return center(viewer, mm, zoom)
}
