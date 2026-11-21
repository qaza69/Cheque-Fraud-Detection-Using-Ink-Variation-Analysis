import { useEffect, useRef, useState } from 'react'

export const pct = (x, d = 1) => (x == null ? '-' : `${(x * 100).toFixed(d)}%`)

/* ------------------------------------------------------------------ small pieces */
export function StatusPill({ status }) {
  const text = {
    checking: 'Connecting to the server...',
    waking: 'Waking the server up (free hosting sleeps when idle, up to a minute)...',
    online: 'Server online',
    offline: 'Server unreachable',
  }[status]
  return (
    <span className={`pill pill-${status}`} role="status" aria-live="polite">
      <i aria-hidden="true" /> {text}
    </span>
  )
}

export function Tile({ value, label }) {
  return (
    <div className="tile">
      <div className="tile-value">{value}</div>
      <div className="tile-label">{label}</div>
    </div>
  )
}

export function Tabs({ tabs, value, onChange, label }) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} className={value === t.id ? 'tab active' : 'tab'}
          onClick={() => onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ cheque input */
export function Uploader({ file, previewUrl, onFile, disabled }) {
  const [drag, setDrag] = useState(false)
  const inputRef = useRef(null)
  const take = (f) => {
    if (!f) return
    if (!/^image\//.test(f.type) && !/\.(tif|tiff|bmp)$/i.test(f.name)) return alert('Please choose an image file (JPG, PNG, BMP or TIFF).')
    onFile(f)
  }
  return (
    <div
      className={`drop ${drag ? 'drag' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files?.[0]) }}
    >
      <input ref={inputRef} type="file" accept="image/*,.tif,.tiff,.bmp" hidden onChange={(e) => take(e.target.files?.[0])} />
      <button type="button" className="btn ghost" disabled={disabled} onClick={() => inputRef.current?.click()}>
        {file ? 'Choose another file' : 'Browse a cheque scan'}
      </button>
      <p className="hint">or drag and drop here - JPG, PNG, BMP, TIFF up to 12 MB</p>
      {file && <p className="file-name">{file.name} ({(file.size / 1024).toFixed(0)} KB)</p>}
      {previewUrl && !file && <p className="file-name">Demo cheque selected</p>}
    </div>
  )
}

/* ------------------------------------------------------------------ pair matrix */
function colour(v) {
  // green (same pen) -> yellow -> red (different pen)
  const g = v < 0.5 ? 200 : Math.round(200 * (1 - (v - 0.5) * 2))
  const r = v < 0.5 ? Math.round(60 + v * 2 * 195) : 255
  return `rgb(${r},${Math.max(g, 30)},70)`
}

export function MatrixCanvas({ matrix }) {
  const ref = useRef(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !matrix?.length) return
    const n = matrix.length
    const cell = Math.max(4, Math.floor(420 / n))
    canvas.width = canvas.height = n * cell
    const ctx = canvas.getContext('2d')
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        ctx.fillStyle = i === j ? '#1b2638' : colour(matrix[i][j])
        ctx.fillRect(j * cell, i * cell, cell, cell)
      }
  }, [matrix])
  if (!matrix?.length) return <p className="hint">Not enough ink units to compare.</p>
  return (
    <div className="matrix">
      <canvas ref={ref} aria-label="Pairwise ink comparison matrix" />
      <div className="legend" aria-hidden="true">
        <span>same pen</span><div className="bar" /><span>different pen</span>
      </div>
      <p className="hint">Every cell is the probability that two handwritten units were written with different pens
        (units sorted from most typical to most distinct). A red block in a corner means a group of strokes that
        disagrees with the rest of the cheque.</p>
    </div>
  )
}

/* ------------------------------------------------------------------ result */
export function Result({ r }) {
  const [tab, setTab] = useState('ink')
  const fraud = r.verdict === 'FRAUD'
  const unknown = r.verdict !== 'FRAUD' && r.verdict !== 'GENUINE'
  const title = fraud ? 'FRAUD - more than one pen detected'
    : unknown ? 'NOT ENOUGH HANDWRITING' : 'GENUINE - a single pen'
  const cls = fraud ? 'fraud' : unknown ? 'unknown' : 'genuine'
  const summary = fraud
    ? `${r.units_suspicious} handwritten ink units disagree with the rest of the cheque's ink - they were written with a different pen.`
    : unknown ? 'Too little handwriting was found in the selected fields to compare inks.'
      : 'All handwritten strokes are consistent with a single pen.'
  const tabs = [
    { id: 'ink', label: 'Suspicious ink' }, { id: 'hand', label: 'Handwriting analysed' },
    { id: 'heat', label: 'Heat-map' }, { id: 'pairs', label: 'Pen comparison' }, { id: 'fields', label: 'By field' },
  ]
  return (
    <section className="result" aria-live="polite">
      <div className={`verdict ${cls}`}>
        <h2>{title}</h2>
        <p>{summary}</p>
      </div>
      {r.notes?.map((n) => <p key={n} className="note">{n}</p>)}
      <div className="tiles">
        <Tile value={pct(r.confidence, 0)} label="Confidence" />
        <Tile value={r.score.toFixed(2)} label={`Cheque score (threshold ${r.threshold.toFixed(2)})`} />
        <Tile value={r.units_total} label="Ink units compared" />
        <Tile value={r.units_suspicious} label="Suspicious ink units" />
        <Tile value={r.layout} label="Cheque layout" />
      </div>
      <Tabs tabs={tabs} value={tab} onChange={setTab} label="Result views" />
      <div className="tabpanel" role="tabpanel">
        {tab === 'ink' && <Figure src={r.images.highlighted}
          caption={fraud ? 'Ink written with a different pen is filled red and boxed.' : 'No ink was attributed to a second pen.'} />}
        {tab === 'hand' && <Figure src={r.images.ink}
          caption="Only these pixels were analysed: logo, printed text, lines, security pattern and background were removed using the bank's blank template." />}
        {tab === 'heat' && <Figure src={r.images.heatmap} caption="How strongly each stroke's ink disagrees with the rest of the cheque." />}
        {tab === 'pairs' && <MatrixCanvas matrix={r.pair_matrix_sorted} />}
        {tab === 'fields' && <FieldTable summary={r.field_summary} />}
      </div>
      <p className="hint">Model: {r.model_label}. This is a screening aid: a flagged cheque should be examined by a person.</p>
    </section>
  )
}

function Figure({ src, caption }) {
  return (
    <figure>
      <img src={src} alt={caption} />
      <figcaption>{caption}</figcaption>
    </figure>
  )
}

function FieldTable({ summary }) {
  const rows = Object.entries(summary || {})
  if (!rows.length) return <p className="hint">No field results.</p>
  return (
    <table className="table">
      <thead><tr><th>Field</th><th>Ink units</th><th>Suspicious</th><th>Max distinctness</th></tr></thead>
      <tbody>{rows.map(([name, v]) => (
        <tr key={name}><td>{name}</td><td>{v.units}</td><td>{v.suspicious}</td><td>{v.max_distinct.toFixed(2)}</td></tr>
      ))}</tbody>
    </table>
  )
}

/* ------------------------------------------------------------------ model comparison */
export function ModelTable({ models }) {
  if (!models?.length) return <p className="hint">No model information yet.</p>
  return (
    <>
      <table className="table wide">
        <thead><tr><th>Model</th><th>Cheque AUC</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Pair AUC</th></tr></thead>
        <tbody>{models.map((m) => (
          <tr key={m.id}>
            <td>{m.label}{m.recommended ? ' *' : ''}</td>
            <td>{m.metrics.cheque_auc?.toFixed(3) ?? '-'}</td><td>{pct(m.metrics.accuracy)}</td>
            <td>{pct(m.metrics.precision)}</td><td>{pct(m.metrics.recall)}</td><td>{m.metrics.pair_auc?.toFixed(3) ?? '-'}</td>
          </tr>
        ))}</tbody>
      </table>
      <p className="hint">5-fold cross-validation on the IDRBT dataset (112 cheques), folds grouped by cheque. Every held-out
        cheque is scored as scanned (tampered) and with the second pen's additions removed (genuine). * = recommended.</p>
    </>
  )
}
