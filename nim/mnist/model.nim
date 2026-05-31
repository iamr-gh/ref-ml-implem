## Models and training utilities.

import std/[math, random]
import tensor, data

type
  Metrics* = object
    loss*: float32
    accuracy*: float32

  SoftmaxClassifier*[I, O: static int] = object
    w*: Mat[I, O]
    b*: Vec[O]

  Layer* = object
    inDim*, outDim*: int
    w*: seq[float32]   # row-major: w[input * outDim + output]
    b*: seq[float32]

  Mlp* = object
    layers*: seq[Layer]

# ── Static softmax classifier, kept as a minimal baseline ────────────────

proc initSoftmaxClassifier*[I, O: static int](): SoftmaxClassifier[I, O] =
  result.w = randMat[I, O](xavierScale(I))

proc logits*[I, O: static int](m: SoftmaxClassifier[I, O]; x: Vec[I]): Vec[O] =
  result = matvec(m.w, x)
  for j in 0 ..< O:
    result[j] += m.b[j]

proc predictProbs*[I, O: static int](m: SoftmaxClassifier[I, O]; x: Vec[I]): Vec[O] =
  result = m.logits(x)
  softmax(result)

proc trainOne*[I, O: static int](m: var SoftmaxClassifier[I, O]; x: Vec[I]; y: int; lr: float32): float32 =
  var probs = m.predictProbs(x)
  result = crossEntropy(probs, y)
  probs[y] -= 1.0'f32

  for i in 0 ..< I:
    for j in 0 ..< O:
      m.w[i][j] -= lr * x[i] * probs[j]
  for j in 0 ..< O:
    m.b[j] -= lr * probs[j]

proc trainEpoch*[I, O: static int](m: var SoftmaxClassifier[I, O]; ds: Dataset[I]; lr: float32): float32 =
  var total: float32 = 0
  for i in 0 ..< ds.len:
    total += m.trainOne(ds.xs[i], ds.ys[i], lr)
  total / float32(ds.len)

proc evaluate*[I, O: static int](m: SoftmaxClassifier[I, O]; ds: Dataset[I]): Metrics =
  var correct = 0
  var totalLoss: float32 = 0
  for i in 0 ..< ds.len:
    let probs = m.predictProbs(ds.xs[i])
    totalLoss += crossEntropy(probs, ds.ys[i])
    if argmax(probs) == ds.ys[i]: inc correct
  result.loss = totalLoss / float32(ds.len)
  result.accuracy = correct.float32 / ds.len.float32

# ── Runtime-configurable MLP with ReLU hidden layers ─────────────────────

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
  var dims = newSeq[int]()
  dims.add(inputDim)
  for h in hidden:
    doAssert h > 0, "hidden layer sizes must be positive"
    dims.add(h)
  dims.add(outputDim)

  for i in 0 ..< dims.len - 1:
    result.layers.add(initLayer(dims[i], dims[i + 1]))

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

proc crossEntropyDyn(probs: seq[float32]; y: int): float32 =
  const eps = 1e-7'f32
  -ln(max(probs[y], eps))

proc argmaxDyn(x: seq[float32]): int =
  var best = 0
  var bestVal = x[0]
  for i in 1 ..< x.len:
    if x[i] > bestVal:
      best = i
      bestVal = x[i]
  best

proc forward*(m: Mlp; x: openArray[float32]): seq[seq[float32]] =
  ## Return activations, including input at index 0 and output probabilities at ^1.
  result = newSeq[seq[float32]](m.layers.len + 1)
  result[0] = newSeq[float32](x.len)
  for i in 0 ..< x.len: result[0][i] = x[i]

  for li, layer in m.layers:
    let prev = result[li]
    var outv = newSeq[float32](layer.outDim)

    for j in 0 ..< layer.outDim:
      var s = layer.b[j]
      for i in 0 ..< layer.inDim:
        s += prev[i] * layer.w[i * layer.outDim + j]
      outv[j] = s

    if li < m.layers.len - 1:
      for j in 0 ..< outv.len: outv[j] = relu(outv[j])
    else:
      softmaxDyn(outv)

    result[li + 1] = outv

proc predictProbs*(m: Mlp; x: openArray[float32]): seq[float32] =
  m.forward(x)[^1]

proc predict*(m: Mlp; x: openArray[float32]): int =
  argmaxDyn(m.predictProbs(x))

proc trainOne*(m: var Mlp; x: openArray[float32]; y: int; lr: float32): float32 =
  let activations = m.forward(x)
  result = crossEntropyDyn(activations[^1], y)

  var delta = activations[^1]
  delta[y] -= 1.0'f32       # d logits for softmax + CE

  for li in countdown(m.layers.len - 1, 0):
    let prev = activations[li]
    var prevDelta: seq[float32]

    if li > 0:
      prevDelta = newSeq[float32](m.layers[li].inDim)
      # Compute with current weights before updating them.
      for i in 0 ..< m.layers[li].inDim:
        var s: float32 = 0
        for j in 0 ..< m.layers[li].outDim:
          s += m.layers[li].w[i * m.layers[li].outDim + j] * delta[j]
        # ReLU derivative for previous hidden activation.
        prevDelta[i] = if prev[i] > 0: s else: 0

    # SGD update for this layer.
    for i in 0 ..< m.layers[li].inDim:
      for j in 0 ..< m.layers[li].outDim:
        m.layers[li].w[i * m.layers[li].outDim + j] -= lr * prev[i] * delta[j]
    for j in 0 ..< m.layers[li].outDim:
      m.layers[li].b[j] -= lr * delta[j]

    if li > 0:
      delta = prevDelta

proc trainEpoch*[I: static int](m: var Mlp; ds: Dataset[I]; lr: float32): float32 =
  var total: float32 = 0
  for i in 0 ..< ds.len:
    total += m.trainOne(ds.xs[i], ds.ys[i], lr)
  total / float32(ds.len)

proc evaluate*[I: static int](m: Mlp; ds: Dataset[I]): Metrics =
  var correct = 0
  var totalLoss: float32 = 0
  for i in 0 ..< ds.len:
    let probs = m.predictProbs(ds.xs[i])
    totalLoss += crossEntropyDyn(probs, ds.ys[i])
    if argmaxDyn(probs) == ds.ys[i]: inc correct
  result.loss = totalLoss / float32(ds.len)
  result.accuracy = correct.float32 / ds.len.float32