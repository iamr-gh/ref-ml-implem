## Basic GPT-style transformer language model.
##
## This intentionally mirrors the MNIST implementation style: flat seq[float32]
## buffers, explicit loops, small helper procs, and preallocated workspace.
##
## The full transformer stack is used in the forward pass. For simplicity and
## readability, the training update here applies SGD to the language-model head
## and the token/position embeddings through the head gradient. Attention/FFN
## parameters participate in forward inference and can be extended with full
## backprop later.

import std/[math, random]

{.push checks: off.}

const
  DefaultDModel* = 64
  DefaultHeads* = 4
  DefaultBlocks* = 2
  DefaultFFDim* = 256

type
  Metrics* = object
    loss*: float32
    accuracy*: float32

  Linear* = object
    inDim*, outDim*: int
    w*: seq[float32]  # row-major: w[i*outDim + j]
    b*: seq[float32]

  Block* = object
    q*, k*, v*, o*: Linear
    ff1*, ff2*: Linear

  Transformer* = object
    vocabSize*, seqLen*, dModel*, nHeads*, dHead*, nBlocks*, ffDim*: int
    tokenEmb*: seq[float32] # vocabSize * dModel
    posEmb*: seq[float32]   # seqLen * dModel
    blocks*: seq[Block]
    head*: Linear           # dModel -> vocabSize

  Workspace* = object
    x*: seq[float32]        # seqLen * dModel
    x2*: seq[float32]       # seqLen * dModel
    q*, k*, v*: seq[float32]
    att*: seq[float32]      # nHeads * seqLen * seqLen
    ff*: seq[float32]       # seqLen * ffDim
    logits*: seq[float32]   # seqLen * vocabSize
    probs*: seq[float32]    # vocabSize scratch
    gradH*: seq[float32]    # dModel scratch

proc relu(x: float32): float32 {.inline.} =
  if x > 0'f32: x else: 0'f32

proc initLinear*(inDim, outDim: int; reluScale = true): Linear =
  result.inDim = inDim
  result.outDim = outDim
  result.w = newSeq[float32](inDim * outDim)
  result.b = newSeq[float32](outDim)
  let scale = if reluScale: sqrt(2.0'f32 / float32(inDim)) else: sqrt(1.0'f32 / float32(inDim))
  for i in 0 ..< result.w.len:
    result.w[i] = rand(2.0'f32 * scale) - scale

proc initTransformer*(vocabSize, seqLen: int; dModel = DefaultDModel; nHeads = DefaultHeads; nBlocks = DefaultBlocks; ffDim = DefaultFFDim): Transformer =
  doAssert dModel mod nHeads == 0
  result.vocabSize = vocabSize
  result.seqLen = seqLen
  result.dModel = dModel
  result.nHeads = nHeads
  result.dHead = dModel div nHeads
  result.nBlocks = nBlocks
  result.ffDim = ffDim

  result.tokenEmb = newSeq[float32](vocabSize * dModel)
  result.posEmb = newSeq[float32](seqLen * dModel)
  let embScale = 0.02'f32
  for i in 0 ..< result.tokenEmb.len:
    result.tokenEmb[i] = rand(2.0'f32 * embScale) - embScale
  for i in 0 ..< result.posEmb.len:
    result.posEmb[i] = rand(2.0'f32 * embScale) - embScale

  result.blocks = newSeq[Block](nBlocks)
  for i in 0 ..< nBlocks:
    result.blocks[i].q = initLinear(dModel, dModel, false)
    result.blocks[i].k = initLinear(dModel, dModel, false)
    result.blocks[i].v = initLinear(dModel, dModel, false)
    result.blocks[i].o = initLinear(dModel, dModel, false)
    result.blocks[i].ff1 = initLinear(dModel, ffDim, true)
    result.blocks[i].ff2 = initLinear(ffDim, dModel, false)

  result.head = initLinear(dModel, vocabSize, false)

proc initWorkspace*(m: Transformer): Workspace =
  result.x = newSeq[float32](m.seqLen * m.dModel)
  result.x2 = newSeq[float32](m.seqLen * m.dModel)
  result.q = newSeq[float32](m.seqLen * m.dModel)
  result.k = newSeq[float32](m.seqLen * m.dModel)
  result.v = newSeq[float32](m.seqLen * m.dModel)
  result.att = newSeq[float32](m.nHeads * m.seqLen * m.seqLen)
  result.ff = newSeq[float32](m.seqLen * m.ffDim)
  result.logits = newSeq[float32](m.seqLen * m.vocabSize)
  result.probs = newSeq[float32](m.vocabSize)
  result.gradH = newSeq[float32](m.dModel)

proc linearForward(inp: openArray[float32]; outp: var seq[float32]; layer: Linear; rows: int) =
  for r in 0 ..< rows:
    let inBase = r * layer.inDim
    let outBase = r * layer.outDim
    for j in 0 ..< layer.outDim:
      var s = layer.b[j]
      for i in 0 ..< layer.inDim:
        s += inp[inBase + i] * layer.w[i * layer.outDim + j]
      outp[outBase + j] = s

proc softmaxInto(logits: openArray[float32]; probs: var seq[float32]; n: int) =
  var maxVal = logits[0]
  for i in 1 ..< n:
    if logits[i] > maxVal: maxVal = logits[i]
  var sum: float32 = 0
  for i in 0 ..< n:
    let e = exp(logits[i] - maxVal)
    probs[i] = e
    sum += e
  for i in 0 ..< n:
    probs[i] /= sum

proc forward*(m: Transformer; tokens: openArray[int32]; ws: var Workspace) =
  let T = min(tokens.len, m.seqLen)
  let C = m.dModel

  # Token + positional embedding.
  for t in 0 ..< T:
    let tok = int(tokens[t])
    for c in 0 ..< C:
      ws.x[t*C + c] = m.tokenEmb[tok*C + c] + m.posEmb[t*C + c]

  for bi in 0 ..< m.nBlocks:
    let blk = m.blocks[bi]
    linearForward(ws.x, ws.q, blk.q, T)
    linearForward(ws.x, ws.k, blk.k, T)
    linearForward(ws.x, ws.v, blk.v, T)

    # Causal multi-head self-attention.
    for h in 0 ..< m.nHeads:
      let headOff = h * m.dHead
      for tq in 0 ..< T:
        var maxScore = -1.0e30'f32
        for tk in 0 ..< T:
          let attIdx = (h * m.seqLen + tq) * m.seqLen + tk
          if tk > tq:
            ws.att[attIdx] = 0
          else:
            var score: float32 = 0
            for d in 0 ..< m.dHead:
              score += ws.q[tq*C + headOff + d] * ws.k[tk*C + headOff + d]
            score /= sqrt(float32(m.dHead))
            ws.att[attIdx] = score
            if score > maxScore: maxScore = score
        var sum: float32 = 0
        for tk in 0 .. tq:
          let attIdx = (h * m.seqLen + tq) * m.seqLen + tk
          let e = exp(ws.att[attIdx] - maxScore)
          ws.att[attIdx] = e
          sum += e
        for tk in 0 .. tq:
          let attIdx = (h * m.seqLen + tq) * m.seqLen + tk
          ws.att[attIdx] /= sum

      for tq in 0 ..< T:
        for d in 0 ..< m.dHead:
          var s: float32 = 0
          for tk in 0 .. tq:
            let attIdx = (h * m.seqLen + tq) * m.seqLen + tk
            s += ws.att[attIdx] * ws.v[tk*C + headOff + d]
          ws.x2[tq*C + headOff + d] = s

    # Output projection + residual.
    linearForward(ws.x2, ws.q, blk.o, T)
    for i in 0 ..< T*C:
      ws.x[i] += ws.q[i]

    # FFN dense -> ReLU -> dense + residual.
    linearForward(ws.x, ws.ff, blk.ff1, T)
    for i in 0 ..< T*m.ffDim:
      ws.ff[i] = relu(ws.ff[i])
    linearForward(ws.ff, ws.x2, blk.ff2, T)
    for i in 0 ..< T*C:
      ws.x[i] += ws.x2[i]

  linearForward(ws.x, ws.logits, m.head, T)

proc trainOne*(m: var Transformer; tokens: openArray[int32]; lr: float32; ws: var Workspace): Metrics =
  m.forward(tokens, ws)
  let T = min(tokens.len, m.seqLen) - 1
  let C = m.dModel
  let V = m.vocabSize
  var correct = 0
  var totalLoss: float32 = 0

  for t in 0 ..< T:
    let y = int(tokens[t + 1])
    let logBase = t * V
    softmaxInto(ws.logits.toOpenArray(logBase, logBase + V - 1), ws.probs, V)
    totalLoss += -ln(max(ws.probs[y], 1e-7'f32))

    var pred = 0
    var best = ws.probs[0]
    for j in 1 ..< V:
      if ws.probs[j] > best:
        best = ws.probs[j]
        pred = j
    if pred == y: inc correct

    for c in 0 ..< C: ws.gradH[c] = 0

    # SGD update for LM head. Also accumulate gradient wrt hidden state.
    for j in 0 ..< V:
      var g = ws.probs[j]
      if j == y: g -= 1.0'f32
      for c in 0 ..< C:
        let idx = c * V + j
        ws.gradH[c] += m.head.w[idx] * g
        m.head.w[idx] -= lr * ws.x[t*C + c] * g
      m.head.b[j] -= lr * g

    # Lightweight embedding update through head gradient.
    let tok = int(tokens[t])
    for c in 0 ..< C:
      m.tokenEmb[tok*C + c] -= lr * ws.gradH[c]
      m.posEmb[t*C + c] -= lr * ws.gradH[c]

  result.loss = totalLoss / float32(T)
  result.accuracy = float32(correct) / float32(T)

proc evaluate*(m: Transformer; tokens: openArray[int32]; ws: var Workspace): Metrics =
  m.forward(tokens, ws)
  let T = min(tokens.len, m.seqLen) - 1
  let V = m.vocabSize
  var correct = 0
  var totalLoss: float32 = 0
  for t in 0 ..< T:
    let y = int(tokens[t + 1])
    let logBase = t * V
    softmaxInto(ws.logits.toOpenArray(logBase, logBase + V - 1), ws.probs, V)
    totalLoss += -ln(max(ws.probs[y], 1e-7'f32))
    var pred = 0
    var best = ws.probs[0]
    for j in 1 ..< V:
      if ws.probs[j] > best:
        best = ws.probs[j]
        pred = j
    if pred == y: inc correct
  result.loss = totalLoss / float32(T)
  result.accuracy = float32(correct) / float32(T)

{.pop.}
