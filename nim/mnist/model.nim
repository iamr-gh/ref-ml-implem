## Models and training utilities.
##
## Performance-critical inner loops use raw pointers + {.checks:off.}
## to minimize overhead. The algorithmic structure mirrors model.nim exactly.

import std/[math, random]
import data

{.push checks: off.}

type
  Metrics* = object
    loss*: float32
    accuracy*: float32

  Layer* = object
    inDim*, outDim*: int
    w*: seq[float32]   # row-major: w[i * outDim + j]
    b*: seq[float32]

  Mlp* = object
    layers*: seq[Layer]

  MlpWorkspace* = object
    activations*: seq[seq[float32]]
    deltas*: seq[seq[float32]]

proc initLayer*(inDim, outDim: int): Layer =
  result.inDim = inDim
  result.outDim = outDim
  result.w = newSeq[float32](inDim * outDim)
  result.b = newSeq[float32](outDim)

  let scale = sqrt(2.0'f32 / float32(inDim))
  for i in 0 ..< result.w.len:
    result.w[i] = rand(2.0'f32 * scale) - scale

proc initMlp*(inputDim, outputDim: int; hidden: openArray[int]): Mlp =
  var prev = inputDim
  for h in hidden:
    doAssert h > 0, "hidden layer sizes must be positive"
    result.layers.add(initLayer(prev, h))
    prev = h
  result.layers.add(initLayer(prev, outputDim))

proc initWorkspace*(m: Mlp): MlpWorkspace =
  result.activations = newSeq[seq[float32]](m.layers.len + 1)
  result.deltas = newSeq[seq[float32]](m.layers.len)

  result.activations[0] = newSeq[float32](m.layers[0].inDim)
  for li, layer in m.layers:
    result.activations[li + 1] = newSeq[float32](layer.outDim)
    result.deltas[li] = newSeq[float32](layer.outDim)

proc relu(x: float32): float32 {.inline.} =
  if x > 0'f32: x else: 0'f32

proc softmaxInPlace(x: ptr UncheckedArray[float32]; len: int) =
  var maxVal = x[0]
  for i in 1 ..< len:
    if x[i] > maxVal: maxVal = x[i]

  var sum: float32 = 0
  for i in 0 ..< len:
    x[i] = exp(x[i] - maxVal)
    sum += x[i]
  for i in 0 ..< len:
    x[i] /= sum

proc crossEntropyDyn(probs: openArray[float32]; y: int): float32 =
  const eps = 1e-7'f32
  -ln(max(probs[y], eps))

proc argmaxDyn(x: openArray[float32]): int =
  var best = 0
  var bestVal = x[0]
  for i in 1 ..< x.len:
    if x[i] > bestVal:
      best = i
      bestVal = x[i]
  best

template ua(s: seq[float32]): ptr UncheckedArray[float32] =
  cast[ptr UncheckedArray[float32]](s[0].unsafeAddr)

proc forward*(m: Mlp; x: openArray[float32]; ws: var MlpWorkspace) =
  let nLayers = m.layers.len

  # Copy input
  let srcAddr = x[0].unsafeAddr
  let dstAddr = addr ws.activations[0][0]
  copyMem(dstAddr, srcAddr, x.len * sizeof(float32))

  for li in 0 ..< nLayers:
    let inD = m.layers[li].inDim
    let outD = m.layers[li].outDim
    let actIn = ua(ws.activations[li])
    let actOut = ua(ws.activations[li + 1])
    let w = ua(m.layers[li].w)
    let b = ua(m.layers[li].b)

    for j in 0 ..< outD:
      var s = b[j]
      for i in 0 ..< inD:
        s += actIn[i] * w[i * outD + j]
      actOut[j] = s

    if li < nLayers - 1:
      for j in 0 ..< outD:
        actOut[j] = relu(actOut[j])
    else:
      softmaxInPlace(actOut, outD)

proc predict*(m: Mlp; x: openArray[float32]; ws: var MlpWorkspace): int =
  m.forward(x, ws)
  argmaxDyn(ws.activations[^1])

proc trainOne*(m: var Mlp; x: openArray[float32]; y: int; lr: float32; ws: var MlpWorkspace): float32 =
  m.forward(x, ws)
  let probs = ws.activations[^1]
  result = crossEntropyDyn(probs, y)

  let lastLi = m.layers.len - 1
  let outD = m.layers[lastLi].outDim
  let delta = ua(ws.deltas[lastLi])
  let probsArr = ua(ws.activations[^1])
  for j in 0 ..< outD:
    delta[j] = probsArr[j]
  delta[y] -= 1.0'f32

  for li in countdown(m.layers.len - 1, 0):
    let inD = m.layers[li].inDim
    let outD = m.layers[li].outDim
    let w = ua(m.layers[li].w)
    let b = ua(m.layers[li].b)
    let actIn = ua(ws.activations[li])
    let dlt = ua(ws.deltas[li])

    if li > 0:
      # Compute prev delta with current weights before updating.
      let prevDelta = ua(ws.deltas[li - 1])
      let prevAct = ua(ws.activations[li])
      for i in 0 ..< inD:
        var s: float32 = 0
        for j in 0 ..< outD:
          s += w[i * outD + j] * dlt[j]
        prevDelta[i] = if prevAct[i] > 0'f32: s else: 0'f32

    # SGD update for this layer.
    for i in 0 ..< inD:
      let xi = actIn[i]
      let base = i * outD
      for j in 0 ..< outD:
        w[base + j] -= lr * xi * dlt[j]
    for j in 0 ..< outD:
      b[j] -= lr * dlt[j]

proc trainEpoch*[I: static int](m: var Mlp; ds: Dataset[I]; lr: float32; ws: var MlpWorkspace): float32 =
  var total: float32 = 0
  for i in 0 ..< ds.len:
    total += m.trainOne(ds.sample(i), ds.label(i), lr, ws)
  total / float32(ds.len)

proc evaluate*[I: static int](m: Mlp; ds: Dataset[I]; ws: var MlpWorkspace): Metrics =
  var correct = 0
  var totalLoss: float32 = 0
  for i in 0 ..< ds.len:
    m.forward(ds.sample(i), ws)
    let probs = ws.activations[^1]
    let y = ds.label(i)
    totalLoss += crossEntropyDyn(probs, y)
    if argmaxDyn(probs) == y: inc correct
  result.loss = totalLoss / float32(ds.len)
  result.accuracy = correct.float32 / ds.len.float32

{.pop.}