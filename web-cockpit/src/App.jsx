import { useState, useEffect, useRef } from 'react'
import { Send } from 'lucide-react'
import './index.css'

const API = 'http://localhost:9999'
const PERSONA_COLORS = {
  persona_code: '#4e8cff',
  persona_math: '#c850f9',
  persona_creative: '#ffa34e',
  base: '#555',
}
const PERSONA_LABELS = {
  persona_code: 'CODE',
  persona_math: 'MATH',
  persona_creative: 'CREATIVE',
}

// ──────────────────────────────────────────
// Force-Directed Network Graph with Zoom+Pan
// ──────────────────────────────────────────
function NetworkGraph({ activePersona, routingInfo, touchedBlocks }) {
  const canvasRef = useRef(null)
  const nodesRef = useRef(null)
  const edgesRef = useRef(null)
  const animFrameRef = useRef(null)
  const camRef = useRef({ x: 0, y: 0, zoom: 1 })
  const dragRef = useRef({ dragging: false, lastX: 0, lastY: 0 })

  useEffect(() => {
    if (nodesRef.current) return

    const nodes = []
    const edges = []

    nodes.push({ id: 'hnsw', label: 'HNSW\nRouter', x: 0, y: 0, vx: 0, vy: 0, r: 32, color: '#fff', type: 'hub', fixed: true })

    const brainDist = 200
    const bd = [
      { id: 'persona_code',     label: 'CODE',     angle: -Math.PI/2 },
      { id: 'persona_math',     label: 'MATH',     angle: -Math.PI/2 + Math.PI*2/3 },
      { id: 'persona_creative', label: 'CREATIVE', angle: -Math.PI/2 + Math.PI*4/3 },
    ]

    bd.forEach(b => {
      nodes.push({
        id: b.id, label: b.label,
        x: Math.cos(b.angle) * brainDist, y: Math.sin(b.angle) * brainDist,
        vx: 0, vy: 0, r: 26,
        color: PERSONA_COLORS[b.id], type: 'brain', fixed: false, active: false
      })
      edges.push({ from: 'hnsw', to: b.id, weight: 0 })
    })

    nodes.push({ id: 'frozen', label: 'FROZEN\n.gguf', x: 0, y: 160, vx: 0, vy: 0, r: 22, color: '#2a2d3e', type: 'frozen', fixed: false })
    edges.push({ from: 'hnsw', to: 'frozen', weight: 0.3 })

    for (let i = 0; i < 50; i++) {
      const angle = (i / 50) * Math.PI * 2
      const orbit = 80 + Math.random() * 60
      nodes.push({
        id: `chunk_${i}`, label: '',
        x: Math.cos(angle) * orbit + (Math.random() - 0.5) * 30,
        y: 160 + Math.sin(angle) * orbit + (Math.random() - 0.5) * 30,
        vx: 0, vy: 0, r: 2.5 + Math.random() * 2.5,
        color: '#1a1c28', type: 'chunk', blockIndex: i, fixed: false, heat: 0
      })
      edges.push({ from: 'frozen', to: `chunk_${i}`, weight: 0.05 })
    }

    edges.push({ from: 'persona_code', to: 'persona_math', weight: 0.02 })
    edges.push({ from: 'persona_math', to: 'persona_creative', weight: 0.02 })
    edges.push({ from: 'persona_creative', to: 'persona_code', weight: 0.02 })

    nodesRef.current = nodes
    edgesRef.current = edges
  }, [])

  useEffect(() => {
    if (!edgesRef.current || !routingInfo) return
    edgesRef.current.forEach(e => {
      if (e.from === 'hnsw' && e.to.startsWith('persona_')) {
        const ri = routingInfo.find(r => r.brain === e.to)
        e.weight = ri ? ri.weight : 0
      }
    })
  }, [routingInfo])

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

  useEffect(() => {
    if (!nodesRef.current) return
    nodesRef.current.forEach(n => { if (n.type === 'brain') n.active = (n.id === `persona_${activePersona}`) })
  }, [activePersona])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const onWheel = (e) => { e.preventDefault(); camRef.current.zoom = Math.max(0.3, Math.min(3, camRef.current.zoom * (e.deltaY > 0 ? 0.92 : 1.08))) }
    const onDown = (e) => { dragRef.current = { dragging: true, lastX: e.clientX, lastY: e.clientY } }
    const onMove = (e) => { if (!dragRef.current.dragging) return; camRef.current.x += (e.clientX - dragRef.current.lastX) / camRef.current.zoom; camRef.current.y += (e.clientY - dragRef.current.lastY) / camRef.current.zoom; dragRef.current.lastX = e.clientX; dragRef.current.lastY = e.clientY }
    const onUp = () => { dragRef.current.dragging = false }

    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('mousedown', onDown)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { canvas.removeEventListener('wheel', onWheel); canvas.removeEventListener('mousedown', onDown); window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let w, h
    const resize = () => { const r = canvas.parentElement.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; w = r.width; h = r.height; canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.width = w + 'px'; canvas.style.height = h + 'px'; ctx.setTransform(dpr, 0, 0, dpr, 0, 0) }
    resize()
    window.addEventListener('resize', resize)
    const findNode = id => nodesRef.current?.find(n => n.id === id)

    const tick = () => {
      if (!nodesRef.current) { animFrameRef.current = requestAnimationFrame(tick); return }
      const nodes = nodesRef.current, edges = edgesRef.current, cam = camRef.current

      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]; if (a.fixed) continue
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]; let dx = a.x - b.x, dy = a.y - b.y, dist = Math.sqrt(dx*dx+dy*dy)||1, minD = a.r+b.r+10
          if (dist < minD * 3) { let f = (minD/dist)*0.25; if (!a.fixed){a.vx+=(dx/dist)*f;a.vy+=(dy/dist)*f}; if (!b.fixed){b.vx-=(dx/dist)*f;b.vy-=(dy/dist)*f} }
        }
      }
      for (const e of edges) { const a=findNode(e.from),b=findNode(e.to); if(!a||!b)continue; let dx=b.x-a.x,dy=b.y-a.y,dist=Math.sqrt(dx*dx+dy*dy)||1; let t=(a.type==='hub'||b.type==='hub')?200:(a.type==='frozen'||b.type==='frozen')?60:140; let f=(dist-t)*0.001; if(!a.fixed){a.vx+=(dx/dist)*f;a.vy+=(dy/dist)*f}; if(!b.fixed){b.vx-=(dx/dist)*f;b.vy-=(dy/dist)*f} }
      for (const n of nodes) { if(n.fixed)continue; n.vx-=n.x*0.0003; n.vy-=(n.y-(n.type==='chunk'?160:0))*0.0003; n.vx*=0.93; n.vy*=0.93; n.x+=n.vx; n.y+=n.vy }

      ctx.clearRect(0, 0, w, h)
      ctx.save()
      ctx.translate(w/2+cam.x, h/2+cam.y)
      ctx.scale(cam.zoom, cam.zoom)

      for (const e of edges) {
        const a=findNode(e.from),b=findNode(e.to); if(!a||!b)continue
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y)
        if (e.weight>0.2 && e.from==='hnsw') { const c=PERSONA_COLORS[e.to]||'#fff'; ctx.strokeStyle=c; ctx.lineWidth=1.5+e.weight*3; ctx.globalAlpha=0.4+e.weight*0.6; ctx.shadowColor=c; ctx.shadowBlur=12+e.weight*15 }
        else { ctx.strokeStyle='rgba(255,255,255,0.03)'; ctx.lineWidth=0.5; ctx.globalAlpha=0.5; ctx.shadowBlur=0 }
        ctx.stroke(); ctx.globalAlpha=1; ctx.shadowBlur=0
      }

      for (const n of nodes) {
        ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,Math.PI*2)
        if (n.type==='hub') { const g=ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,n.r); g.addColorStop(0,'#fff'); g.addColorStop(1,'#8888aa'); ctx.fillStyle=g; ctx.shadowColor='#fff'; ctx.shadowBlur=12 }
        else if (n.type==='brain') { ctx.fillStyle=n.active?n.color:n.color+'30'; if(n.active){ctx.shadowColor=n.color;ctx.shadowBlur=30} }
        else if (n.type==='chunk') { if(n.heat>0){ctx.fillStyle=n.color;ctx.shadowColor=n.color;ctx.shadowBlur=n.heat*15}else{ctx.fillStyle='rgba(30,32,48,0.5)'} }
        else if (n.type==='frozen') { const g=ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,n.r); g.addColorStop(0,'#3a3d5e'); g.addColorStop(1,'#1e2030'); ctx.fillStyle=g }
        ctx.fill(); ctx.shadowBlur=0; ctx.shadowColor='transparent'
        if (n.label && n.type!=='chunk') { ctx.fillStyle=n.type==='hub'?'#111':(n.active?'#fff':'rgba(255,255,255,0.5)'); ctx.font=n.type==='hub'?"bold 11px 'Outfit'":'600 9px Outfit'; ctx.textAlign='center'; ctx.textBaseline='middle'; n.label.split('\n').forEach((l,i,a) => ctx.fillText(l,n.x,n.y+(i-(a.length-1)/2)*13)) }
      }

      ctx.restore()
      animFrameRef.current = requestAnimationFrame(tick)
    }
    animFrameRef.current = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(animFrameRef.current); window.removeEventListener('resize', resize) }
  }, [])

  return <canvas ref={canvasRef} />
}

// ──────────────────────────────────────────
// Main App
// ──────────────────────────────────────────
function App() {
  const [messages, setMessages] = useState([
    { role: 'system', text: 'Neural Cockpit Online · FUSE Kernel Driver ativo. Envie uma mensagem para conversar com a IA.' },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [fuseOnline, setFuseOnline] = useState(false)
  const [activePersona, setActivePersona] = useState('base')
  const [routingInfo, setRoutingInfo] = useState(null)
  const [touchedBlocks, setTouchedBlocks] = useState({})
  const [modelInfo, setModelInfo] = useState(null)
  const endRef = useRef(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  useEffect(() => {
    const check = async () => { try { const r = await fetch(`${API}/stats`); setFuseOnline(r.ok) } catch { setFuseOnline(false) } }
    check()
    const iv = setInterval(check, 4000)
    return () => clearInterval(iv)
  }, [])

  const sendMessage = async (e) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages(prev => [...prev, { role: 'user', text }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      setActivePersona(data.persona || 'base')
      setRoutingInfo(data.routing || [])
      setTouchedBlocks(data.touchedBlocks || {})
      if (data.model) setModelInfo(data.model)

      const routingLines = (data.routing || []).map(r => {
        const pct = (r.weight * 100).toFixed(0)
        return `${PERSONA_LABELS[r.brain] || r.brain}: ${pct}%`
      }).join(' · ')

      setMessages(prev => [...prev, {
        role: 'brain',
        text: data.response || '(sem resposta)',
        persona: data.persona,
        annotation: {
          persona: data.persona,
          routingSummary: routingLines,
          blockCount: Object.keys(data.touchedBlocks || {}).length,
          model: data.model,
        },
        error: data.error,
      }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'system', text: `Erro: ${err.message}. Daemon FUSE rodando?` }])
    } finally {
      setLoading(false)
    }
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
        <NetworkGraph activePersona={activePersona} routingInfo={routingInfo} touchedBlocks={touchedBlocks} />

        <div className="info-overlay">
          <h3>Multi-Brain HNSW</h3>
          {modelInfo ? (
            <>
              <div style={{marginBottom: 6}}>
                <strong>{modelInfo.name}</strong> · {modelInfo.quantization} · {modelInfo.size}
              </div>
              Engine: {modelInfo.engine}<br/>
              Formato: {modelInfo.format}
            </>
          ) : (
            <>Aguardando primeira mensagem para carregar dados do modelo...</>
          )}
          <div style={{marginTop: 8, fontSize: '0.72rem', opacity: 0.6}}>
            Scroll = zoom · Arrastar = mover
          </div>
        </div>

        {routingInfo && routingInfo.length > 0 && (
          <div className="routing-flash">
            <h4>Roteamento HNSW</h4>
            {routingInfo.map((r, i) => (
              <div key={i} className="route-bar">
                <span className="route-bar-label" style={{ color: PERSONA_COLORS[r.brain] || '#888' }}>
                  {PERSONA_LABELS[r.brain] || r.brain}
                </span>
                <div style={{ flex: 1, background: 'rgba(255,255,255,0.05)', borderRadius: 3, height: 5, overflow: 'hidden' }}>
                  <div className="route-bar-fill" style={{ width: `${(r.weight * 100).toFixed(0)}%`, background: PERSONA_COLORS[r.brain] || '#888' }} />
                </div>
                <span className="route-bar-pct">{(r.weight * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}

        <div className="graph-legend">
          <div className="legend-item"><div className="legend-dot" style={{ background: '#4e8cff' }} /> Code</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: '#c850f9' }} /> Math</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: '#ffa34e' }} /> Creative</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: '#fff' }} /> Router</div>
          <div className="legend-item"><div className="legend-dot" style={{ background: '#3a3d5e' }} /> Frozen</div>
        </div>
      </div>

      <div className="chat-panel">
        <div className="chat-header">💬 Neural Chat</div>

        <div className="chat-messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.role === 'brain' && m.persona && (
                <div className="msg-route">
                  <span className={`persona-tag ${m.persona}`}>
                    {m.persona === 'code' ? '⚡ CODE' : m.persona === 'math' ? '📐 MATH' : m.persona === 'creative' ? '🎨 CREATIVE' : '🔘 BASE'}
                  </span>
                  {m.annotation?.model && (
                    <span style={{fontSize:'0.6rem',color:'#6b6f82',marginLeft:4}}>{m.annotation.model.name} · {m.annotation.model.quantization}</span>
                  )}
                </div>
              )}

              {m.text}

              {m.role === 'brain' && m.annotation && (
                <div className="routing-annotation">
                  <strong>Roteamento:</strong> {m.annotation.routingSummary}<br/>
                  <strong>SSD Δ:</strong> {m.annotation.blockCount} blocos mutados via XOR<br/>
                  <strong>Por quê?</strong> O HNSW analisou o embedding do seu prompt e determinou que a persona <strong>{m.annotation.persona}</strong> tem a maior similaridade de cosseno com o contexto semântico da pergunta.
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="msg brain">
              <div style={{fontSize:'0.72rem',color:'#6b6f82',marginBottom:4}}>
                ⏳ Inferindo via llama-server (modelo na RAM)...
              </div>
              <div className="typing-dots"><span/><span/><span/></div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <form className="chat-input-bar" onSubmit={sendMessage}>
          <input
            type="text"
            placeholder={fuseOnline ? 'Pergunte algo à IA...' : 'FUSE offline'}
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={loading || !fuseOnline}
          />
          <button type="submit" disabled={loading || !input.trim() || !fuseOnline}>
            <Send size={16} />
          </button>
        </form>
      </div>
    </div>
  )
}

export default App
