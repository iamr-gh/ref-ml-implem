## Minimal softmax classifier and training utilities.
## This structure is intentionally easy to generalize to Linear layers later.

import tensor, data

type
  Metrics* = object
    loss*: float32
    accuracy*: float32

  SoftmaxClassifier*[I, O: static int] = object
    w*: Mat[I, O]
    b*: Vec[O]

proc initSoftmaxClassifier*[I, O: static int](): SoftmaxClassifier[I, O] =
  result.w = randMat[I, O](xavierScale(I))
  # b defaults to zero

proc logits*[I, O: static int](m: SoftmaxClassifier[I, O]; x: Vec[I]): Vec[O] =
  result = matvec(m.w, x)
  for j in 0 ..< O:
    result[j] += m.b[j]

proc predictProbs*[I, O: static int](m: SoftmaxClassifier[I, O]; x: Vec[I]): Vec[O] =
  result = m.logits(x)
  softmax(result)

proc predict*[I, O: static int](m: SoftmaxClassifier[I, O]; x: Vec[I]): int =
  argmax(m.predictProbs(x))

proc trainOne*[I, O: static int](m: var SoftmaxClassifier[I, O]; x: Vec[I]; y: int; lr: float32): float32 =
  ## One SGD step. Returns sample cross-entropy loss.
  var probs = m.predictProbs(x)
  result = crossEntropy(probs, y)

  # dlogits = softmax(logits) - oneHot(y)
  probs[y] -= 1.0'f32

  # gradients and update, fused to avoid storing grad matrices
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
    if argmax(probs) == ds.ys[i]:
      inc correct

  result.loss = totalLoss / float32(ds.len)
  result.accuracy = correct.float32 / ds.len.float32