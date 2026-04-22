# crompressor-video — Extração Sensorial do World Model

## Papel no Ecossistema 5D

O **crompressor-video** é o **córtex visual**: a unidade que extrai codificações dinâmicas sensoriais do ambiente para alimentar o World Model.

## Além da Compressão de Frames

Não se trata apenas de comprimir vídeo — trata-se de extrair **o momento físico** de cada frame:

| Extração | O Que Captura |
|:---------|:-------------|
| Velocidade vetorial | Para onde e quão rápido objetos se movem |
| Oclusões latentes | O que está escondido atrás de objetos visíveis |
| Cinemática | Inércia, aceleração, trajetórias previstas |
| Semântica visual | O que é cada objeto e como se comporta |

## Como Alimenta o World Model

```
Câmera captura frame
    ↓
crompressor-video extrai:
    ├── Objetos detectados + bounding boxes
    ├── Vetores de movimento (optical flow)
    ├── Profundidade estimada (depth map)
    └── Semântica (carro, pessoa, árvore)
    ↓
World Model recebe modelo geométrico matricial
    ↓
ToT gera branches de predição:
    ├── "O carro vai virar à direita" (80%)
    ├── "O carro vai frear" (15%)
    └── "O carro vai acelerar" (5%)
```

## Observador Virtual via Vídeo

Quando um objeto some atrás de um prédio, o sistema **continua "vendo-o"** via simulação:
- Baseado em inércia e dados prévios
- O crompressor-video mantém o "fantasma" do objeto
- Quando o objeto reaparece → `diff` confirma ou corrige

## Conexão com FPS Cognitivo

Humanos processam vídeo a ~60Hz com latência de 80-100ms. O crompressor-video pode operar a taxas muito superiores, extraindo informação que humanos nunca perceberiam — vibrações, micro-expressões, padrões de movimento invisíveis ao olho.

## Questões Abertas

- [ ] O crompressor-video pode gerar observadores virtuais a partir de um stream?
- [ ] Como integrar audio (crompressor-sinapse?) para reconstrução 3D sonora?
- [ ] O encoding neural do video pode usar Codebook Learning do tensor-vivo?
