import { useCallback, useEffect, useMemo, useState } from 'react'
import { API, detect, getModels, getSamples, sampleUrl, waitForApi } from './api'
import { ModelTable, Result, StatusPill, Tabs, Uploader } from './components'

export default function App() {
  const [status, setStatus] = useState('checking')          // checking | waking | online | offline
  const [info, setInfo] = useState({ models: [], fields: [], layouts: [] })
  const [samples, setSamples] = useState([])

  const [file, setFile] = useState(null)
  const [sample, setSample] = useState('')                  // demo cheque name chosen by the user
  const [model, setModel] = useState('')
  const [threshold, setThreshold] = useState(null)          // null = use the model's calibrated value
  const [fields, setFields] = useState([])

  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [page, setPage] = useState('detect')

  /* ---- connect to the API (it may be asleep) ---- */
  const connect = useCallback(async () => {
    setStatus('checking'); setError('')
    try {
      await waitForApi({ onWaiting: () => setStatus('waking') })
      const [m, s] = await Promise.all([getModels(), getSamples().catch(() => [])])
      setInfo(m); setSamples(s)
      setModel((cur) => cur || m.models.find((x) => x.recommended)?.id || m.models[0]?.id || '')
      setFields((cur) => (cur.length ? cur : m.fields.filter((f) => f.default).map((f) => f.id)))
      setStatus('online')
    } catch {
      setStatus('offline')
    }
  }, [])
  useEffect(() => { connect() }, [connect])

  const activeModel = useMemo(() => info.models.find((m) => m.id === model), [info, model])
  const calibrated = activeModel?.metrics?.threshold ?? 0.5
  const shownThreshold = threshold ?? calibrated

  /* ---- what is currently selected as the cheque image ---- */
  const [localPreview, setLocalPreview] = useState('')
  useEffect(() => {
    if (!file) { setLocalPreview(''); return }
    const url = URL.createObjectURL(file)
    setLocalPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  const preview = file ? localPreview : sample ? sampleUrl(sample) : ''

  const chooseFile = (f) => { setFile(f); setSample(''); setResult(null); setError('') }
  const chooseSample = (name) => { setSample(name); setFile(null); setResult(null); setError('') }
  const toggleField = (id) => setFields((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]))

  const run = async () => {
    setBusy(true); setError(''); setResult(null)
    try {
      setResult(await detect({ file, sample: file ? '' : sample, model, threshold, fields }))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }
  const ready = status === 'online' && !busy && (file || sample) && fields.length > 0 && model

  return (
    <div className="shell">
      <header className="banner">
        <div>
          <h1>Cheque Fraud Detection</h1>
          <p>Ink-variation analysis of the handwritten fields: finds cheques altered with a second pen, even when both inks look the same colour.</p>
        </div>
        <StatusPill status={status} />
      </header>

      <Tabs value={page} onChange={setPage} label="Sections" tabs={[
        { id: 'detect', label: 'Detect' }, { id: 'models', label: 'Model comparison' }, { id: 'how', label: 'How it works' }]} />

      {status === 'offline' && (
        <div className="alert" role="alert">
          The API at <code>{API}</code> is not reachable.{' '}
          <button className="btn small" onClick={connect}>Try again</button>
        </div>
      )}

      {page === 'detect' && (
        <main className="grid">
          <section className="panel">
            <h2>1. Choose a cheque</h2>
            <Uploader file={file} previewUrl={preview} onFile={chooseFile} disabled={busy} />
            {samples.length > 0 && (
              <label className="field">
                <span>or try a demo cheque</span>
                <select value={sample} onChange={(e) => chooseSample(e.target.value)} disabled={busy}>
                  <option value="">Select a demo cheque...</option>
                  {samples.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
                </select>
              </label>
            )}

            <h2>2. Settings</h2>
            <label className="field">
              <span>Detection model</span>
              <select value={model} onChange={(e) => { setModel(e.target.value); setThreshold(null) }} disabled={busy || !info.models.length}>
                {info.models.map((m) => <option key={m.id} value={m.id}>{m.label}{m.recommended ? ' (recommended)' : ''}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Decision threshold: <b>{shownThreshold.toFixed(2)}</b>
                {threshold != null && <button className="link" onClick={() => setThreshold(null)}>reset to {calibrated.toFixed(2)}</button>}</span>
              <input type="range" min="0.05" max="0.95" step="0.01" value={shownThreshold} disabled={busy}
                onChange={(e) => setThreshold(parseFloat(e.target.value))} />
              <small>Lower = stricter (flags more cheques); the default was chosen by cross-validation.</small>
            </label>
            <fieldset className="field">
              <legend>Handwritten fields to compare</legend>
              {info.fields.map((f) => (
                <label key={f.id} className="check">
                  <input type="checkbox" checked={fields.includes(f.id)} onChange={() => toggleField(f.id)} disabled={busy} />
                  {f.label}
                </label>
              ))}
              <small>Date and signature are often written in advance by the account holder, so they are off by default.</small>
            </fieldset>

            <button className="btn primary" disabled={!ready} onClick={run}>
              {busy ? 'Analysing...' : 'Detect fraud'}
            </button>
            {busy && <p className="hint">Isolating the handwriting and comparing inks. This takes a few seconds.</p>}
            {error && <p className="alert" role="alert">{error}</p>}
            <p className="hint privacy">Privacy: the cheque is analysed in memory and is not stored.</p>
          </section>

          <section className="stage">
            {result ? <Result r={result} />
              : (
                <div className="placeholder">
                  {preview ? <img src={preview} alt="Selected cheque" /> : <p>Upload a cheque scan or pick a demo cheque to begin.</p>}
                  {preview && <p className="hint">Press <b>Detect fraud</b> to analyse this cheque.</p>}
                </div>
              )}
          </section>
        </main>
      )}

      {page === 'models' && (
        <section className="panel solo">
          <h2>Cross-validated comparison on the IDRBT dataset</h2>
          <ModelTable models={info.models} />
          {info.layouts.length > 0 && <p className="hint">Known cheque layouts: {info.layouts.join(', ')}. Other layouts are analysed in a weaker generic mode.</p>}
        </section>
      )}

      {page === 'how' && (
        <section className="panel solo prose">
          <h2>Why ink variation?</h2>
          <p>A fraudster alters a cheque after it was signed, usually with a pen that looks the same. Different pens use
            different dyes, so their inks absorb red, green and blue light in different proportions: a difference a
            scanner can measure even when the eye cannot.</p>
          <h2>Pipeline</h2>
          <ol>
            <li><b>Layout matching and alignment:</b> the scan is matched to a blank template of its bank's cheque and aligned to it.</li>
            <li><b>Handwriting isolation:</b> a pixel counts as ink only if it is darker than anything printed there on the blank
              cheque, so the logo, labels, lines, security pattern and coloured background are never analysed.</li>
            <li><b>Ink measurement:</b> handwriting is cut into small units (about one letter or digit). The optical density
              <code> OD = -log(I / I_paper)</code> removes the paper colour; 25 features describe the dye ratio, density, stroke width and texture.</li>
            <li><b>Pen comparison:</b> a model gives, for every pair of units, the probability of different pens.</li>
            <li><b>Verdict:</b> the cheque is flagged when a coherent group of units disagrees with the rest.</li>
          </ol>
          <h2>Limits</h2>
          <p>Trained on 112 IDRBT cheques from four banks. Every dataset cheque is tampered, so the genuine class was derived
            (second pen removed) or simulated; results are an estimate, not a measurement on real single-pen cheques. The
            output is a flag for human review, never an automatic rejection.</p>
        </section>
      )}

      <footer>Cheque Fraud Detection using Ink Variation Analysis - IDRBT Cheque Image Dataset</footer>
    </div>
  )
}
