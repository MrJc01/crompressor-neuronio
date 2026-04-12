import { useState, useEffect, useRef } from 'react'
import { Send } from 'lucide-react'
import './index.css'

const API = 'http://localhost:9999'

const BRAIN_COLORS = {
  'brain_codigo.crom': '#4e8cff',
  'brain_matematica.crom': '#c850f9',
  'brain_literario.crom': '#ffa34e',
  'base': '#6b6f82',
}

const BRAIN_LABELS = {
  'brain_codigo.crom': 'Código',
  'brain_matematica.crom': 'Matemática',
  'brain_literario.crom': 'Literário',
  'base': 'Base (Frozen)',
}

const BRAIN_ICONS = {
  'brain_codigo.crom': '⚡',
  'brain_matematica.crom': '📐',
  'brain_literario.crom': '🎨',
  'base': '🧊',
}

// ──────────────────────────────────────────
// Pipeline Graph (Canvas) with Zoom+Pan
// Shows the actual CROM architecture flow
// ──────────────────────────────────────────
function PipelineGraph({ activeBrain, routingInfo, touchedBlocks }) {
  const canvasRef = useRef(null)
  const nodesRef = useRef(null)
  const edgesRef = useRef(null)
  const animRef = useRef(null)
  const camRef = useRef({ x: 0, y: 0, zoom: 1 })
  const dragRef = useRef({ dragging: false, lx: 0, ly: 0 })
  const timeRef = useRef(0)

  useEffect(() => {
    if (nodesRef.current) return
    const nodes = []
    const edges = []

    // Pipeline nodes positioned in a flow layout
    nodes.push({ id: 'prompt', label: 'PROMPT\nInput', x: -300, y: 0, r: 24, color: '#4e8cff', type: 'io', fixed: true })
    nodes.push({ id: 'hnsw', label: 'HNSW\nRouter', x: -100, y: 0, r: 30, color: '#fff', type: 'hub', fixed: true })

    // Brain nodes (the actual .crom neurons) — fanned out from router
    nodes.push({ id: 'brain_codigo.crom', label: 'brain_codigo\n.crom', x: 100, y: -120, r: 24, color: BRAIN_COLORS['brain_codigo.crom'], type: 'brain', fixed: true, active: false, weight: 0 })
    nodes.push({ id: 'brain_matematica.crom', label: 'brain_matematica\n.crom', x: 100, y: 0, r: 24, color: BRAIN_COLORS['brain_matematica.crom'], type: 'brain', fixed: true, active: false, weight: 0 })
    nodes.push({ id: 'brain_literario.crom', label: 'brain_literario\n.crom', x: 100, y: 120, r: 24, color: BRAIN_COLORS['brain_literario.crom'], type: 'brain', fixed: true, active: false, weight: 0 })

    // XOR Delta compositor
    nodes.push({ id: 'delta', label: 'XOR\nDelta ⊕', x: 280, y: 0, r: 22, color: '#00ffa3', type: 'process', fixed: true })

    // FUSE / SSD block
    nodes.push({ id: 'fuse', label: 'FUSE\nKernel VFS', x: 280, y: 130, r: 20, color: '#3a3d5e', type: 'fuse', fixed: true })

    // Output
    nodes.push({ id: 'output', label: 'OUTPUT\nResponse', x: 440, y: 0, r: 24, color: '#00ffa3', type: 'io', fixed: true })

    // SSD chunk dots orbiting FUSE
    for (let i = 0; i < 30; i++) {
      const a = (i / 30) * Math.PI * 2
      const d = 50 + Math.random() * 30
      nodes.push({
        id: `chunk_${i}`, label: '', x: 280 + Math.cos(a) * d, y: 130 + Math.sin(a) * d,
        r: 2 + Math.random() * 2, color: '#1a1c28', type: 'chunk', blockIndex: i, heat: 0, fixed: true
      })
    }

    // Edges (pipeline flow)
    edges.push({ from: 'prompt', to: 'hnsw', type: 'flow', weight: 1 })
    edges.push({ from: 'hnsw', to: 'brain_codigo.crom', type: 'route', weight: 0 })
    edges.push({ from: 'hnsw', to: 'brain_matematica.crom', type: 'route', weight: 0 })
    edges.push({ from: 'hnsw', to: 'brain_literario.crom', type: 'route', weight: 0 })
    edges.push({ from: 'brain_codigo.crom', to: 'delta', type: 'route', weight: 0 })
    edges.push({ from: 'brain_matematica.crom', to: 'delta', type: 'route', weight: 0 })
    edges.push({ from: 'brain_literario.crom', to: 'delta', type: 'route', weight: 0 })
    edges.push({ from: 'fuse', to: 'delta', type: 'data', weight: 0.3 })
    edges.push({ from: 'delta', to: 'output', type: 'flow', weight: 1 })

    nodesRef.current = nodes
    edgesRef.current = edges
  }, [])

  // Update routing weights — ALL brains get their weight (multi-brain composition)
  useEffect(() => {
    if (!edgesRef.current || !nodesRef.current || !routingInfo) return

    // Update brain node weights
    nodesRef.current.forEach(n => {
      if (n.type === 'brain') {
        const ri = routingInfo.find(r => r.brain === n.id)
        n.weight = ri ? ri.weight : 0
        n.active = n.weight > 0.05 // Any brain with >5% weight is "active"
      }
    })

    // Update edge weights from router to brains AND brains to delta
    edgesRef.current.forEach(e => {
      if (e.type === 'route') {
        const brainId = e.from.startsWith('brain_') ? e.from : e.to
        const ri = routingInfo.find(r => r.brain === brainId)
        e.weight = ri ? ri.weight : 0
      }
    })
  }, [routingInfo])

  // Chunk heatmap
  useEffect(() => {
    if (!nodesRef.current || !touchedBlocks) return
    nodesRef.current.forEach(n => {
      if (n.type === 'chunk') {
        const hits = touchedBlocks[String(n.blockIndex)] || 0
        n.heat = Math.min(hits / 3, 1)
        n.color = hits > 8 ? '#00ffa3' : hits > 3 ? '#c850f9' : hits > 0 ? '#4e8cff' : '#1a1c28'
      }
    })
  }, [touchedBlocks])

  // Mouse: zoom + pan
  useEffect(() => {
    const c = canvasRef.current; if (!c) return
    const onW = e => { e.preventDefault(); camRef.current.zoom = Math.max(0.4, Math.min(2.5, camRef.current.zoom * (e.deltaY > 0 ? 0.93 : 1.07))) }
    const onD = e => { dragRef.current = { dragging: true, lx: e.clientX, ly: e.clientY } }
    const onM = e => { if (!dragRef.current.dragging) return; camRef.current.x += (e.clientX - dragRef.current.lx) / camRef.current.zoom; camRef.current.y += (e.clientY - dragRef.current.ly) / camRef.current.zoom; dragRef.current.lx = e.clientX; dragRef.current.ly = e.clientY }
    const onU = () => { dragRef.current.dragging = false }
    c.addEventListener('wheel', onW, { passive: false }); c.addEventListener('mousedown', onD); window.addEventListener('mousemove', onM); window.addEventListener('mouseup', onU)
    return () => { c.removeEventListener('wheel', onW); c.removeEventListener('mousedown', onD); window.removeEventListener('mousemove', onM); window.removeEventListener('mouseup', onU) }
  }, [])

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d')
    let w, h
    const resize = () => { const r = canvas.parentElement.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; w = r.width; h = r.height; canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.width = w + 'px'; canvas.style.height = h + 'px'; ctx.setTransform(dpr, 0, 0, dpr, 0, 0) }
    resize(); window.addEventListener('resize', resize)
    const find = id => nodesRef.current?.find(n => n.id === id)

    const tick = () => {
      if (!nodesRef.current) { animRef.current = requestAnimationFrame(tick); return }
      timeRef.current += 0.016
      const nodes = nodesRef.current, edges = edgesRef.current, cam = camRef.current, t = timeRef.current

      ctx.clearRect(0, 0, w, h)
      ctx.save()
      ctx.translate(w / 2 + cam.x, h / 2 + cam.y)
      ctx.scale(cam.zoom, cam.zoom)

      // Draw edges
      for (const e of edges) {
        const a = find(e.from), b = find(e.to); if (!a || !b) continue
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y)

        if (e.type === 'route' && e.weight > 0.05) {
          const brainId = e.from.startsWith('brain_') ? e.from : e.to
          const col = BRAIN_COLORS[brainId] || '#fff'
          ctx.strokeStyle = col
          ctx.lineWidth = 1 + e.weight * 4
          ctx.globalAlpha = 0.3 + e.weight * 0.7
          ctx.shadowColor = col
          ctx.shadowBlur = e.weight * 20
        } else if (e.type === 'flow') {
          ctx.strokeStyle = 'rgba(0,255,163,0.25)'
          ctx.lineWidth = 2
          ctx.globalAlpha = 0.6
          ctx.shadowBlur = 0
        } else if (e.type === 'data') {
          ctx.strokeStyle = 'rgba(58,61,94,0.4)'
          ctx.lineWidth = 1.5
          ctx.globalAlpha = 0.5
          ctx.shadowBlur = 0
          ctx.setLineDash([4, 4])
        } else {
          ctx.strokeStyle = 'rgba(255,255,255,0.04)'
          ctx.lineWidth = 0.5
          ctx.globalAlpha = 0.3
          ctx.shadowBlur = 0
        }
        ctx.stroke()
        ctx.setLineDash([])
        ctx.globalAlpha = 1; ctx.shadowBlur = 0

        // Animated particle along active route edges
        if (e.type === 'route' && e.weight > 0.1) {
          const progress = (t * 0.5 * (1 + e.weight)) % 1
          const px = a.x + (b.x - a.x) * progress
          const py = a.y + (b.y - a.y) * progress
          const brainId = e.from.startsWith('brain_') ? e.from : e.to
          ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2)
          ctx.fillStyle = BRAIN_COLORS[brainId] || '#fff'
          ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 10
          ctx.fill(); ctx.shadowBlur = 0
        }
      }

      // Draw nodes
      for (const n of nodes) {
        if (n.type === 'chunk') {
          ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
          if (n.heat > 0) { ctx.fillStyle = n.color; ctx.shadowColor = n.color; ctx.shadowBlur = n.heat * 12 }
          else { ctx.fillStyle = 'rgba(30,32,48,0.4)'; ctx.shadowBlur = 0 }
          ctx.fill(); ctx.shadowBlur = 0
          continue
        }

        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)

        if (n.type === 'hub') {
          const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r)
          g.addColorStop(0, '#fff'); g.addColorStop(1, '#8888aa')
          ctx.fillStyle = g; ctx.shadowColor = '#fff'; ctx.shadowBlur = 15
        } else if (n.type === 'brain') {
          // Multi-brain: opacity based on weight, not binary
          const alpha = n.active ? Math.max(0.4, n.weight) : 0.12
          ctx.fillStyle = n.color + Math.round(alpha * 255).toString(16).padStart(2, '0')
          if (n.active && n.weight > 0.3) { ctx.shadowColor = n.color; ctx.shadowBlur = 20 + n.weight * 20 }
        } else if (n.type === 'process') {
          ctx.fillStyle = n.color + '88'
          ctx.shadowColor = n.color; ctx.shadowBlur = 10
        } else if (n.type === 'fuse') {
          const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r)
          g.addColorStop(0, '#3a3d5e'); g.addColorStop(1, '#1e2030')
          ctx.fillStyle = g
        } else if (n.type === 'io') {
          ctx.fillStyle = n.color + '60'
          ctx.strokeStyle = n.color; ctx.lineWidth = 1.5
        }

        ctx.fill(); ctx.shadowBlur = 0; ctx.shadowColor = 'transparent'
        if (n.type === 'io') ctx.stroke()

        // Weight percentage on brain nodes
        if (n.type === 'brain' && n.weight > 0.01) {
          ctx.fillStyle = n.active ? '#fff' : 'rgba(255,255,255,0.4)'
          ctx.font = "bold 10px 'JetBrains Mono'"
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
          ctx.fillText(Math.round(n.weight * 100) + '%', n.x, n.y + n.r + 14)
        }

        // Labels
        if (n.label) {
          ctx.fillStyle = n.type === 'hub' ? '#111' : (n.active || n.type === 'process' || n.type === 'io' || n.type === 'fuse' ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.4)')
          ctx.font = `${n.type === 'hub' ? 'bold 10px' : n.type === 'brain' ? '600 8px' : '500 8px'} 'Outfit', sans-serif`
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
          n.label.split('\n').forEach((l, i, a) => ctx.fillText(l, n.x, n.y + (i - (a.length - 1) / 2) * 12))
        }
      }

      ctx.restore()
      animRef.current = requestAnimationFrame(tick)
    }
    animRef.current = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(animRef.current); window.removeEventListener('resize', resize) }
  }, [])

  return <canvas ref={canvasRef} />
}

// ──────────────────────────────────────────
// Metrics Panel
// ──────────────────────────────────────────
function MetricsPanel({ metrics, model }) {
  if (!metrics && !model) return null
  return (
    <div className="metrics-panel">
      {model && (
        <div className="metrics-section">
          <h4>🧠 Modelo</h4>
          <div className="metric-row"><span>Nome</span><strong>{model.name}</strong></div>
          <div className="metric-row"><span>Quant.</span><strong>{model.quantization}</strong></div>
          <div className="metric-row"><span>Formato</span><strong>{model.format}</strong></div>
          <div className="metric-row"><span>Engine</span><strong>{model.engine}</strong></div>
          <div className="metric-row"><span>Vertente</span><strong>{model.vertente}</strong></div>
        </div>
      )}
      {metrics && (
        <div className="metrics-section">
          <h4>📊 Última Inferência</h4>
          <div className="metric-row"><span>Tempo</span><strong>{(metrics.inference_time_ms / 1000).toFixed(1)}s</strong></div>
          <div className="metric-row"><span>Tok/s</span><strong>{metrics.tokens_per_second}</strong></div>
          <div className="metric-row"><span>HNSW</span><strong>{metrics.hnsw_decision_us}μs</strong></div>
          <div className="metric-row"><span>Chunks</span><strong>{metrics.chunks_read}</strong></div>
          <div className="metric-row"><span>Tokens</span><strong>{metrics.completion_tokens}</strong></div>
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────
// Main App
// ──────────────────────────────────────────
function App() {
  const [messages, setMessages] = useState([
    { role: 'system', text: 'Crompressor-Neurônio online. O cérebro base está congelado no SSD e é lido via FUSE. Envie uma mensagem para ativar o roteamento HNSW multi-brain.' },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [fuseOnline, setFuseOnline] = useState(false)
  const [activeBrain, setActiveBrain] = useState('base')
  const [routingInfo, setRoutingInfo] = useState(null)
  const [touchedBlocks, setTouchedBlocks] = useState({})
  const [modelInfo, setModelInfo] = useState(null)
  const [lastMetrics, setLastMetrics] = useState(null)
  const endRef = useRef(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  useEffect(() => {
    const check = async () => { try { const r = await fetch(`${API}/stats`); setFuseOnline(r.ok) } catch { setFuseOnline(false) } }
    check(); const iv = setInterval(check, 5000); return () => clearInterval(iv)
  }, [])

  const sendMessage = async (e) => {
    e.preventDefault()
    const text = input.trim(); if (!text || loading) return
    setMessages(prev => [...prev, { role: 'user', text }])
    setInput(''); setLoading(true)

    try {
      const res = await fetch(`${API}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt: text }) })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      setActiveBrain(data.brain || 'base')
      setRoutingInfo(data.routing || [])
      setTouchedBlocks(data.touchedBlocks || {})
      if (data.model) setModelInfo(data.model)
      if (data.metrics) setLastMetrics(data.metrics)

      // Build multi-brain routing summary
      const activeRoutes = (data.routing || []).filter(r => r.weight > 0.01)
      const routeSummary = activeRoutes.map(r => `${BRAIN_LABELS[r.brain] || r.brain}: ${(r.weight * 100).toFixed(0)}%`).join(' · ')

      setMessages(prev => [...prev, {
        role: 'brain', text: data.response || '(sem resposta)',
        brain: data.brain,
        routing: data.routing,
        metrics: data.metrics,
        model: data.model,
        routeSummary,
        error: data.error,
      }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'system', text: `Erro: ${err.message}` }])
    } finally { setLoading(false) }
  }

  return (
    <div className="app-shell">
      <div className="topbar">
        <h1>🧠 Crompressor-Neurônio</h1>
        <div className="topbar-status">
          <div className={`status-dot ${fuseOnline ? '' : 'offline'}`} />
          {fuseOnline ? 'FUSE Online · :9999' : 'FUSE Offline'}
        </div>
      </div>

      <div className="graph-panel">
        <PipelineGraph activeBrain={activeBrain} routingInfo={routingInfo} touchedBlocks={touchedBlocks} />

        <div className="info-overlay">
          <h3>Pipeline CROM Multi-Brain</h3>
          <div style={{fontSize:'0.72rem', color:'#8a8ea0', lineHeight:1.5}}>
            <strong style={{color:'#00ffa3'}}>Prompt</strong> → <strong style={{color:'#fff'}}>HNSW Router</strong> (cosine similarity) →
            <strong style={{color:'#4e8cff'}}> brain.crom</strong> (pesos ponderados) →
            <strong style={{color:'#00ffa3'}}> XOR Delta ⊕</strong> → <strong style={{color:'#00ffa3'}}>Output</strong>
          </div>
          <div style={{marginTop: 6, fontSize: '0.65rem', opacity: 0.5}}>Scroll = zoom · Arrastar = mover</div>
        </div>

        {routingInfo && routingInfo.length > 0 && (
          <div className="routing-flash">
            <h4>Composição Multi-Brain</h4>
            {routingInfo.map((r, i) => (
              <div key={i} className="route-bar">
                <span className="route-bar-label" style={{ color: BRAIN_COLORS[r.brain] || '#888' }}>
                  {BRAIN_LABELS[r.brain] || r.brain}
                </span>
                <div style={{ flex: 1, background: 'rgba(255,255,255,0.05)', borderRadius: 3, height: 5, overflow: 'hidden' }}>
                  <div className="route-bar-fill" style={{ width: `${(r.weight * 100).toFixed(0)}%`, background: BRAIN_COLORS[r.brain] || '#888' }} />
                </div>
                <span className="route-bar-pct">{(r.weight * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}

        <div className="graph-legend">
          <div className="legend-item"><div className="legend-dot" style={{ background: '#4e8cff' }} /> Código</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: '#c850f9' }} /> Matemática</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: '#ffa34e' }} /> Literário</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: '#00ffa3' }} /> Pipeline</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: '#3a3d5e' }} /> FUSE/SSD</div>
        </div>
      </div>

      <div className="chat-panel">
        <div className="chat-header">💬 Neural Chat</div>

        <MetricsPanel metrics={lastMetrics} model={modelInfo} />

        <div className="chat-messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.role === 'brain' && (
                <div className="msg-route">
                  <span className="brain-tag" style={{ borderColor: BRAIN_COLORS[m.brain] || '#555', color: BRAIN_COLORS[m.brain] || '#555' }}>
                    {BRAIN_ICONS[m.brain] || '🧊'} {BRAIN_LABELS[m.brain] || m.brain}
                  </span>
                  {m.metrics && (
                    <span className="msg-timing">{(m.metrics.inference_time_ms / 1000).toFixed(1)}s · {m.metrics.tokens_per_second} tok/s</span>
                  )}
                </div>
              )}

              {m.text}

              {m.role === 'brain' && m.routing && (
                <div className="routing-annotation">
                  <div><strong>Composição:</strong> {m.routeSummary}</div>
                  {m.metrics && (
                    <div><strong>HNSW:</strong> {m.metrics.hnsw_decision_us}μs · <strong>Chunks FUSE:</strong> {m.metrics.chunks_read} blocos lidos via XOR Δ</div>
                  )}
                  <div style={{marginTop:4, opacity: 0.7}}>
                    O roteador calculou a similaridade de cosseno entre o embedding do seu prompt e o centroide
                    de cada brain.crom registrado, compondo os pesos acima via softmax.
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="msg brain">
              <div style={{fontSize:'0.72rem',color:'#6b6f82',marginBottom:4}}>⏳ Inferindo via llama-server...</div>
              <div className="typing-dots"><span/><span/><span/></div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <form className="chat-input-bar" onSubmit={sendMessage}>
          <input type="text" placeholder={fuseOnline ? 'Pergunte algo ao neurônio...' : 'FUSE offline'} value={input} onChange={e => setInput(e.target.value)} disabled={loading || !fuseOnline} />
          <button type="submit" disabled={loading || !input.trim() || !fuseOnline}><Send size={16} /></button>
        </form>
      </div>
    </div>
  )
}

export default App
