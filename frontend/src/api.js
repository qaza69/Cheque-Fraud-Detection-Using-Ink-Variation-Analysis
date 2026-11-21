// Thin client for the cheque-fraud API (FastAPI on Render).
export const API = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')

class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, options = {}, timeoutMs = 120000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(API + path, { ...options, signal: controller.signal })
    if (!res.ok) {
      let detail = `Request failed (${res.status})`
      try {
        const body = await res.json()
        if (body && body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      } catch { /* body was not JSON */ }
      throw new ApiError(detail, res.status)
    }
    return await res.json()
  } catch (err) {
    if (err.name === 'AbortError') throw new ApiError('The server took too long to answer. Please try again.', 0)
    if (err instanceof ApiError) throw err
    throw new ApiError('Cannot reach the server. Check your connection and try again.', 0)
  } finally {
    clearTimeout(timer)
  }
}

/**
 * Free hosting puts the API to sleep when idle; the first call can take about a minute.
 * Polls /health until it answers; `onWaiting` fires once the wait becomes noticeable.
 */
export async function waitForApi({ onWaiting, maxMs = 120000, signal } = {}) {
  const started = Date.now()
  let notified = false
  while (Date.now() - started < maxMs) {
    if (signal?.aborted) throw new ApiError('cancelled', 0)
    try {
      return await request('/health', {}, 8000)
    } catch {
      if (!notified && Date.now() - started > 2500) {
        notified = true
        onWaiting?.()
      }
      await new Promise((r) => setTimeout(r, 2500))
    }
  }
  throw new ApiError('The server did not wake up in time.', 0)
}

export const getModels = () => request('/api/models')
export const getSamples = () => request('/api/samples')
export const sampleUrl = (name) => `${API}/api/samples/${encodeURIComponent(name)}`

export function detect({ file, sample, model, threshold, fields }) {
  const form = new FormData()
  if (file) form.append('file', file)
  else if (sample) form.append('sample', sample)
  form.append('model', model)
  if (threshold != null) form.append('threshold', String(threshold))
  fields.forEach((f) => form.append('fields', f))
  return request('/api/detect', { method: 'POST', body: form }, 150000)
}
