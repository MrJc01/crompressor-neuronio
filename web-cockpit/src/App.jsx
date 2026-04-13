import { useState, useEffect, useRef } from 'react'
import { Send, Plus, Cpu, Database } from 'lucide-react'
import './index.css'

const API = 'http://localhost:9999'

// Deterministic color generation based on brain ID
function getBrainColor(id) {
  if (id === 'base') return '#6b6f82';
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = id.charCodeAt(i) + ((hash << 5) - hash);
  const h = Math.abs(hash) % 360;
  return `hsl(${h}, 85%, 65%)`;
}

function formatLabel(id) {
  if (id === 'base') return 'Base (Frozen)'
  return id.replace('brain_', '').replace('.crom', '').toUpperCase()
}

// ──────────────────────────────────────────
// Pipeline Graph (Canvas) with Dynamic Scale (Multi-Brain)
// ──────────────────────────────────────────
function PipelineGraph({ activeBrain, routingInfo, touchedBlocks, registeredBrains }) {
  const canvasRef = useRef(null)
  const nodesRef = useRef([])
  const edgesRef = useRef([])
  const animRef = useRef(null)
  const camRef = useRef({ x: 0, y: 0, zoom: 1 })
  const dragRef = useRef({ dragging: false, lx: 0, ly: 0 })
  const timeRef = useRef(0)

  useEffect(() => {
    const nodes = []
    const edges = []

    // Pipeline Fixed Nodes
    nodes.push({ id: 'prompt', label: 'PROMPT\nInput', x: -300, y: 0, r: 24, color: '#4e8cff', type: 'io', fixed: true })
    nodes.push({ id: 'hnsw', label: 'HNSW\nRouter', x: -100, y: 0, r: 30, color: '#fff', type: 'hub', fixed: true })
    nodes.push({ id: 'delta', label: 'XOR\nDelta ⊕', x: 280, y: 0, r: 22, color: '#00ffa3', type: 'process', fixed: true })
    nodes.push({ id: 'fuse', label: 'FUSE\nKernel VFS', x: 280, y: 130, r: 20, color: '#3a3d5e', type: 'fuse', fixed: true })
    nodes.push({ id: 'output', label: 'OUTPUT\nResponse', x: 440, y: 0, r: 24, color: '#00ffa3', type: 'io', fixed: true })

    // SSD chunks orbiting FUSE
    for (let i = 0; i < 30; i++) {
        const a = (i / 30) * Math.PI * 2
        const d = 40 + Math.random() * 20
        nodes.push({
          id: `chunk_${i}`, label: '', x: 280 + Math.cos(a) * d, y: 130 + Math.sin(a) * d,
          r: 2 + Math.random() * 2, color: '#1a1c28', type: 'chunk', blockIndex: i, heat: 0, fixed: true
        })
    }

    edges.push({ from: 'prompt', to: 'hnsw', type: 'flow', weight: 1 })
    edges.push({ from: 'fuse', to: 'delta', type: 'data', weight: 0.3 })
    edges.push({ from: 'delta', to: 'output', type: 'flow', weight: 1 })

    // Build dynamic brains
    const brainList = registeredBrains || []
    
    brainList.forEach((b, index) => {
      // spread brains vertically between router and delta
      const total = brainList.length;
      let by = 0;
      if (total > 1) {
        const spreadY = Math.min(300, total * 60)
        by = -spreadY/2 + (spreadY / (total - 1)) * index
      }
      
      const bColor = getBrainColor(b.id)
      nodes.push({
        id: b.id, label: formatLabel(b.id) + '\n.crom', 
        x: 90, y: by, r: 22, color: bColor, type: 'brain', fixed: true, active: false, weight: 0
      })
      edges.push({ from: 'hnsw', to: b.id, type: 'route', weight: 0 })
      edges.push({ from: b.id, to: 'delta', type: 'route', weight: 0 })
    })

    nodesRef.current = nodes
    edgesRef.current = edges
  }, [registeredBrains])

  useEffect(() => {
    if (!edgesRef.current || !nodesRef.current || !routingInfo) return
    nodesRef.current.forEach(n => {
      if (n.type === 'brain') {
        const ri = routingInfo.find(r => r.brain === n.id)
        n.weight = ri ? ri.weight : 0
        n.active = n.weight > 0.05
      }
    })
    edgesRef.current.forEach(e => {
      if (e.type === 'route') {
        const brainId = e.from.startsWith('brain_') ? e.from : e.to
        const ri = routingInfo.find(r => r.brain === brainId)
        e.weight = ri ? ri.weight : 0
      }
    })
  }, [routingInfo, registeredBrains])

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
      animRef.current = requestAnimationFrame(tick)
      if (!nodesRef.current) return
      timeRef.current += 0.016
      const nodes = nodesRef.current, edges = edgesRef.current, cam = camRef.current, t = timeRef.current

      ctx.clearRect(0, 0, w, h)
      ctx.save()
      ctx.translate(w / 2 + cam.x, h / 2 + cam.y)
      ctx.scale(cam.zoom, cam.zoom)

      for (const e of edges) {
        const a = find(e.from), b = find(e.to); if (!a || !b) continue
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y)

        if (e.type === 'route' && e.weight > 0.05) {
          const brainId = e.from.startsWith('brain_') ? e.from : e.to
          const col = getBrainColor(brainId)
          ctx.strokeStyle = col
          ctx.lineWidth = 1 + e.weight * 4
          ctx.globalAlpha = 0.3 + e.weight * 0.7
          ctx.shadowColor = col
          ctx.shadowBlur = e.weight * 20
        } else if (e.type === 'flow') {
          ctx.strokeStyle = 'rgba(0,255,163,0.25)'; ctx.lineWidth = 2; ctx.globalAlpha = 0.6; ctx.shadowBlur = 0
        } else if (e.type === 'data') {
          ctx.strokeStyle = 'rgba(58,61,94,0.4)'; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.5; ctx.setLineDash([4, 4])
        } else {
          ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 0.5; ctx.globalAlpha = 0.3; ctx.shadowBlur = 0
        }
        ctx.stroke()
        ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.shadowBlur = 0

        // Animated particle
        if (e.type === 'route' && e.weight > 0.1) {
          const progress = (t * 0.5 * (1 + e.weight)) % 1
          const px = a.x + (b.x - a.x) * progress, py = a.y + (b.y - a.y) * progress
          const brainId = e.from.startsWith('brain_') ? e.from : e.to
          ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2)
          ctx.fillStyle = getBrainColor(brainId); ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 10; ctx.fill()
        }
      }

      for (const n of nodes) {
        if (n.type === 'chunk') {
          ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
          if (n.heat > 0) { ctx.fillStyle = n.color; ctx.shadowColor = n.color; ctx.shadowBlur = n.heat * 12 }
          else { ctx.fillStyle = 'rgba(30,32,48,0.4)'; ctx.shadowBlur = 0 }
          ctx.fill(); ctx.shadowBlur = 0; continue
        }

        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
        if (n.type === 'hub') {
          const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r)
          g.addColorStop(0, '#fff'); g.addColorStop(1, '#8888aa')
          ctx.fillStyle = g; ctx.shadowColor = '#fff'; ctx.shadowBlur = 15
        } else if (n.type === 'brain') {
          const alpha = n.active ? Math.max(0.4, n.weight) : 0.12
          ctx.fillStyle = n.color + Math.round(alpha * 255).toString(16).padStart(2, '0')
          if (n.active && n.weight > 0.3) { ctx.shadowColor = n.color; ctx.shadowBlur = 20 + n.weight * 20 }
        } else if (n.type === 'process') {
          ctx.fillStyle = n.color + '88'; ctx.shadowColor = n.color; ctx.shadowBlur = 10
        } else if (n.type === 'fuse') {
          const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r)
          g.addColorStop(0, '#3a3d5e'); g.addColorStop(1, '#1e2030'); ctx.fillStyle = g
        } else if (n.type === 'io') {
          ctx.fillStyle = n.color + '60'; ctx.strokeStyle = n.color; ctx.lineWidth = 1.5
        }
        ctx.fill(); ctx.shadowBlur = 0; ctx.shadowColor = 'transparent'; if (n.type === 'io') ctx.stroke()

        if (n.type === 'brain' && n.weight > 0.01) {
          ctx.fillStyle = n.active ? '#fff' : 'rgba(255,255,255,0.4)'
          ctx.font = "bold 9px 'JetBrains Mono'"; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
          ctx.fillText(Math.round(n.weight * 100) + '%', n.x, n.y + n.r + 14)
        }
        if (n.label) {
          ctx.fillStyle = n.type === 'hub' ? '#111' : (n.active || n.type === 'process' || n.type === 'io' || n.type === 'fuse' ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.4)')
          ctx.font = `${n.type === 'hub' ? 'bold 10px' : n.type === 'brain' ? '600 7px' : '500 8px'} 'Outfit', sans-serif`
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
          n.label.split('\n').forEach((l, i, a) => ctx.fillText(l, n.x, n.y + (i - (a.length - 1) / 2) * 11))
        }
      }
      ctx.restore()
    }
    animRef.current = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(animRef.current); window.removeEventListener('resize', resize) }
  }, [])

  return <canvas ref={canvasRef} />
}

// ──────────────────────────────────────────
// Main App
// ──────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([
    { role: 'system', text: 'Neural Cockpit Online. Dimensão Vetorial (896-Dim) do HNSW Inicializada via Embedding LLM.' },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [fuseOnline, setFuseOnline] = useState(false)
  
  const [registeredBrains, setRegisteredBrains] = useState([])
  const [activeBrain, setActiveBrain] = useState('base')
  const [routingInfo, setRoutingInfo] = useState(null)
  const [touchedBlocks, setTouchedBlocks] = useState({})
  
  const [modelInfo, setModelInfo] = useState(null)
  const [lastMetrics, setLastMetrics] = useState(null)
  
  const [ingestModal, setIngestModal] = useState(false)
  const [ingestDomain, setIngestDomain] = useState('')
  const [ingestText, setIngestText] = useState('')
  const [ingesting, setIngesting] = useState(false)

  const endRef = useRef(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  const fetchBrains = async () => {
    try {
      const res = await fetch(`${API}/brains`)
      if (res.ok) setRegisteredBrains(await res.json())
      setFuseOnline(true)
    } catch {
      setFuseOnline(false)
    }
  }

  useEffect(() => {
    fetchBrains()
    const iv = setInterval(fetchBrains, 5000)
    return () => clearInterval(iv)
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

      const activeRoutes = (data.routing || []).filter(r => r.weight > 0.01)
      const routeSummary = activeRoutes.map(r => `${formatLabel(r.brain)}: ${(r.weight * 100).toFixed(0)}%`).join(' · ')

      setMessages(prev => [...prev, {
        role: 'brain', text: data.response || '(sem resposta)',
        brain: data.brain,
        routing: data.routing, metrics: data.metrics, model: data.model, routeSummary, error: data.error,
      }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'system', text: `Erro: ${err.message}` }])
    } finally { setLoading(false) }
  }

  const doIngest = async (e) => {
    e.preventDefault()
    if(!ingestDomain || !ingestText) return
    setIngesting(true)
    try {
      const res = await fetch(`${API}/ingest`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain: ingestDomain.toLowerCase(), text: ingestText })
      })
      const r = await res.json()
      if (res.ok) {
        setIngestModal(false)
        setIngestDomain('')
        setIngestText('')
        await fetchBrains()
        setMessages(prev => [...prev, { role: 'system', text: `Novo Cérebro injetado com sucesso no ecossistema: ${r.brainID} (Dimensão: ${r.dim})` }])
      } else {
        alert("Erro: " + r)
      }
    } catch (err){
      alert(err.message)
    } finally {
      setIngesting(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="topbar">
        <h1>🧠 Crompressor-Neurônio</h1>
        <button className="ingest-btn" onClick={() => setIngestModal(true)}>
          <Plus size={14}/> Ingerir Novo Conhecimento
        </button>
        <div className="topbar-status">
          <div className={`status-dot ${fuseOnline ? '' : 'offline'}`} />
          {fuseOnline ? 'API & Embeddings Online' : 'Sistema Offline'}
        </div>
      </div>

      <div className="graph-panel">
        <PipelineGraph activeBrain={activeBrain} routingInfo={routingInfo} touchedBlocks={touchedBlocks} registeredBrains={registeredBrains} />

        <div className="info-overlay">
          <h3>Ecossistema Dinâmico</h3>
          <div style={{fontSize:'0.72rem', color:'#8a8ea0', lineHeight:1.5}}>
            Você está operando {registeredBrains.length} cérebros em runtime. <br/>
            O gráfico gera órbitas reativas com roteamento de Dimensão Múltipla. 
          </div>
          <div style={{marginTop: 6, fontSize: '0.65rem', opacity: 0.5}}>Scroll = zoom · Arrastar = mover</div>
        </div>

        {routingInfo && routingInfo.length > 0 && (
          <div className="routing-flash">
            <h4>Composição Vetorial {lastMetrics?.hnsw_decision_us}μs</h4>
            {routingInfo.map((r, i) => r.weight > 0.01 && (
              <div key={i} className="route-bar">
                <span className="route-bar-label" style={{ color: getBrainColor(r.brain) }}>
                  {formatLabel(r.brain)}
                </span>
                <div style={{ flex: 1, background: 'rgba(255,255,255,0.05)', borderRadius: 3, height: 5, overflow: 'hidden' }}>
                  <div className="route-bar-fill" style={{ width: `${(r.weight * 100).toFixed(0)}%`, background: getBrainColor(r.brain) }} />
                </div>
                <span className="route-bar-pct">{(r.weight * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="chat-panel">
        <div className="chat-header">💬 Neural Chat (Terminal Real)</div>
        <div className="chat-messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.role === 'brain' && (
                <div className="msg-route">
                  <span className="brain-tag" style={{ borderColor: getBrainColor(m.brain), color: getBrainColor(m.brain) }}>
                    🧊 {formatLabel(m.brain)}
                  </span>
                  {m.metrics && ( <span className="msg-timing">{(m.metrics.inference_time_ms / 1000).toFixed(1)}s · {m.metrics.tokens_per_second} tok/s</span> )}
                </div>
              )}
              {m.text}
              {m.role === 'brain' && m.routing && (
                <div className="routing-annotation">
                  <div><strong>Composição:</strong> {m.routeSummary}</div>
                  <div style={{marginTop:4, opacity: 0.7}}>
                    A IA gerou um Embedding [896-Dim] da sua pergunta e fez dot-product contra os {registeredBrains.length} volumes registrados para determinar qual Delta injetar no FUSE Kernel.
                  </div>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="msg brain">
              <div style={{fontSize:'0.72rem',color:'#6b6f82',marginBottom:4}}>⏳ Inferência Real (Llama.cpp)...</div>
              <div className="typing-dots"><span/><span/><span/></div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <form className="chat-input-bar" onSubmit={sendMessage}>
          <input type="text" placeholder={fuseOnline ? 'Interaja usando linguagem...' : 'Servidor Inacessível'} value={input} onChange={e => setInput(e.target.value)} disabled={loading || !fuseOnline} />
          <button type="submit" disabled={loading || !input.trim() || !fuseOnline}><Send size={16} /></button>
        </form>
      </div>

      {ingestModal && (
        <div className="modal-backdrop">
          <div className="modal">
            <h3>Ingerir Novo Conhecimento (.crom)</h3>
            <p>O sistema irá compactar este conhecimento analisando a entropia e gerando Embeddings para criar um Delta Dinâmico que se espalhará no HNSW Router.</p>
            <input type="text" placeholder="Domínio Principal (ex: Direito, Fisica...)" value={ingestDomain} onChange={e=>setIngestDomain(e.target.value.replace(/[^a-zA-Z]/g, ''))} />
            <textarea placeholder="Insira o texto de conhecimento..." rows={6} value={ingestText} onChange={e=>setIngestText(e.target.value)}></textarea>
            <div className="modal-actions">
              <button className="btn-cancel" onClick={() => setIngestModal(false)} disabled={ingesting}>Cancelar</button>
              <button className="btn-ingest" onClick={doIngest} disabled={ingesting || !ingestDomain || !ingestText}>
                {ingesting ? 'Extraindo Camadas 896-Dim...' : 'Ingerir Cérebro Neural'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
