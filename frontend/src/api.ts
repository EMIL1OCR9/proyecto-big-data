/** Lee la respuesta de la API; si falló, lanza el mensaje que mandó FastAPI. */
export async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail ?? `La API respondió HTTP ${response.status}`)
  return data as T
}
