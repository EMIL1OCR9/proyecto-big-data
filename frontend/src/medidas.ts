import type { Focus } from './NiftiCanvas'

type Measured = { numero: number; lobulo: string; centro_ras_mm: number[]; tamano_mm: number[] }

/** Transversal × anteroposterior × craneocaudal, en mm, como lo midió la caja del detector. */
export function formatSize(size: number[]): string {
  return `${size.map(value => value.toFixed(1)).join(' × ')} mm`
}

/**
 * Categoría Lung-RADS (v2022) de un nódulo sólido en la primera tomografía, solo por tamaño.
 * Usa el diámetro medio en el corte axial: el promedio de la medida transversal y la anteroposterior.
 * Es una aproximación: la guía también considera si el nódulo es sólido, su forma y si creció.
 */
export function lungRads(size: number[]): { category: string; meanMm: number } {
  const meanMm = Math.round(((size[0] + size[1]) / 2) * 10) / 10
  const category = meanMm < 6 ? '2' : meanMm < 8 ? '3' : meanMm < 15 ? '4A' : '4B'
  return { category, meanMm }
}

/** El nódulo como lo necesita el visor: centro, tamaño y una etiqueta con su medida. */
export function toFocus(nodule: Measured): Focus {
  return {
    center: nodule.centro_ras_mm,
    size: nodule.tamano_mm,
    label: `Nódulo ${nodule.numero} · ${formatSize(nodule.tamano_mm)}${nodule.lobulo ? ` · ${nodule.lobulo}` : ''}`,
  }
}
