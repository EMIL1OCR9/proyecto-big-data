import { useEffect, useRef, useState } from 'react'
import { Niivue, SLICE_TYPE } from '@niivue/niivue'
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
  /** Punto en mm (RAS) al que se mueve la cruz, por ejemplo el centro de un nódulo. */
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

  useEffect(() => { viewRef.current = view; nv.current?.setSliceType(VIEW_TYPES[view]) }, [view])

  useEffect(() => {
    targetRef.current = target
    if (phase === 'lista' && nv.current && target) moveTo(nv.current, target)
  }, [phase, target])

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
        const volumes = [{ url: imageUrl, name: 'tomografia.nii.gz', colormap: 'gray', cal_min: -1000, cal_max: 400 }]
        if (overlayUrl) volumes.push({ url: overlayUrl, name: 'nodulos.nii.gz', colormap: 'red', cal_min: 0.5, cal_max: 1 })
        await viewer.loadVolumes(volumes)
        if (cancelled) return
        if (overlayUrl) viewer.setOpacity(1, 0.45)
        viewer.setSliceType(VIEW_TYPES[viewRef.current])
        if (targetRef.current) moveTo(viewer, targetRef.current)
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
  }, [imageUrl, overlayUrl])

  return <div className="ct-canvas-shell">
    <canvas className="ct-canvas" ref={canvas} aria-label="Visor interactivo de tomografía" />
    {phase !== 'lista' && <div className={`ct-canvas-message ct-${phase}`} role="status">{message}</div>}
    {phase === 'lista' && <p className="ct-canvas-caption">{message} Imagen de investigación, no validada para uso clínico.</p>}
  </div>
}

function moveTo(viewer: Niivue, mm: number[]) {
  viewer.scene.crosshairPos = viewer.mm2frac([mm[0], mm[1], mm[2]])
  viewer.createOnLocationChange()
  viewer.drawScene()
}
