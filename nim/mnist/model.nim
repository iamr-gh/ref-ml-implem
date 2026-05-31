## Models and training utilities.

import std/[math, random]
import data

type
  Metrics* = object
    loss*: float32
    accuracy*: float32

  Layer* = object
    inDim*, outDim*: int
    w*: seq[float32]   # row-major: w[input * outDim + output]
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

  # He-ish init works well with ReLU hidden layers.
  let scale = sqrt(2.0'f32 / float32(inDim))
  for i in 0 ..< result.w.len:
    result.w[i] = rand(2.0'f32 * scale) - scale

proc initMlp*(inputDim, outputDim: int; hidden: openArray[int]): Mlp =
  ## Create input -> hidden... -> output MLP.
  ## Hidden layers use ReLU; final layer uses softmax.
  var prev = inputDim
  for h in hidden:
    doAssert h > 0, "hidden layer sizes must be positive"
    result.layers.add(initLayer(prev, h))
    prev = h
  result.layers.add(initLayer(prev, outputDim))

proc initWorkspace*(m: Mlp): MlpWorkspace =
  ## Reusable buffers. activations[0] is the input copy, activations[^1] probs.
  result.activations = newSeq[seq[float32]](m.layers.len + 1)
  result.deltas = newSeq[seq[float32]](m.layers.len)

  result.activations[0] = newSeq[float32](m.layers[0].inDim)
  for li, layer in m.layers:
    result.activations[li + 1] = newSeq[float32](layer.outDim)
    result.deltas[li] = newSeq[float32](layer.outDim)

proc relu(x: float32): float32 {.inline.} =
  if x > 0: x else: 0

proc softmaxDyn(x: var seq[float32]) =
  var maxVal = x[0]
  for v in x:
    if v > maxVal: maxVal = v

  var sum: float32 = 0
  for i in 0 ..< x.len:
    x[i] = exp(x[i] - maxVal)
    sum += x[i]
  for i in 0 ..< x.len:
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

proc forward*(m: Mlp; x: openArray[float32]; ws: var MlpWorkspace): lent seq[float32] =
  ## Fill workspace activations and return output probabilities.
  for i in 0 ..< x.len:
    ws.activations[0][i] = x[i]

  for li, layer in m.layers:
    let prev = ws.activations[li]
    var outv = ws.activations[li + 1]

    for j in 0 ..< layer.outDim:
      var s = layer.b[j]
      for i in 0 ..< layer.inDim:
        s += prev[i] * layer.w[i * layer.outDim + j]
      outv[j] = s

    if li < m.layers.len - 1:
      for j in 0 ..< outv.len: outv[j] = relu(outv[j])
    else:
      softmaxDyn(outv)

  ws.activations[^1]

proc predictProbs*(m: Mlp; x: openArray[float32]; ws: var MlpWorkspace): lent seq[float32] =
  m.forward(x, ws)

proc predict*(m: Mlp; x: openArray[float32]; ws: var MlpWorkspace): int =
  argmaxDyn(m.predictProbs(x, ws))

proc trainOne*(m: var Mlp; x: openArray[float32]; y: int; lr: float32; ws: var MlpWorkspace): float32 =
  let probs = m.forward(x, ws)
  result = crossEntropyDyn(probs, y)

  # Output delta = softmax probs - oneHot(y)
  let lastLi = m.layers.len - 1
  for j in 0 ..< probs.len:
    ws.deltas[lastLi][j] = probs[j]
  ws.deltas[lastLi][y] -= 1.0'f32

  for li in countdown(m.layers.len - 1, 0):
    let prev = ws.activations[li]
    let delta = ws.deltas[li]

    if li > 0:
      var prevDelta = ws.deltas[li - 1]
      # Compute with current weights before updating them.
      for i in 0 ..< m.layers[li].inDim:
        var s: float32 = 0
        for j in 0 ..< m.layers[li].outDim:
          s += m.layers[li].w[i * m.layers[li].outDim + j] * delta[j]
        prevDelta[i] = if prev[i] > 0: s else: 0

    # SGD update for this layer.
    for i in 0 ..< m.layers[li].inDim:
      let xi = prev[i]
      for j in 0 ..< m.layers[li].outDim:
        m.layers[li].w[i * m.layers[li].outDim + j] -= lr * xi * delta[j]
    for j in 0 ..< m.layers[li].outDim:
      m.layers[li].b[j] -= lr * delta[j]

proc trainEpoch*[I: static int](m: var Mlp; ds: Dataset[I]; lr: float32; ws: var MlpWorkspace): float32 =
  var total: float32 = 0
  for i in 0 ..< ds.len:
    total += m.trainOne(ds.sample(i), ds.label(i), lr, ws)
  total / float32(ds.len)

proc evaluate*[I: static int](m: Mlp; ds: Dataset[I]; ws: var MlpWorkspace): Metrics =
  var correct = 0
  var totalLoss: float32 = 0
  for i in 0 ..< ds.len:
    let probs = m.predictProbs(ds.sample(i), ws)
    let y = ds.label(i)
    totalLoss += crossEntropyDyn(probs, y)
    if argmaxDyn(probs) == y: inc correct
  result.loss = totalLoss / float32(ds.len)
  result.accuracy = correct.float32 / ds.len.float32