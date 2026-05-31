## Small fixed-size tensor helpers for toy neural networks.

import std/[math, random]

type
  Vec*[N: static int] = array[N, float32]
  Mat*[R, C: static int] = array[R, array[C, float32]]

proc zeros*[N: static int](_: typedesc[Vec[N]]): Vec[N] = discard
proc zeros*[R, C: static int](_: typedesc[Mat[R, C]]): Mat[R, C] = discard

proc randMat*[R, C: static int](scale: float32): Mat[R, C] =
  for i in 0 ..< R:
    for j in 0 ..< C:
      result[i][j] = rand(2.0'f32 * scale) - scale

proc matvec*[R, C: static int](w: Mat[R, C]; x: Vec[R]): Vec[C] =
  for j in 0 ..< C:
    var s: float32 = 0
    for i in 0 ..< R:
      s += x[i] * w[i][j]
    result[j] = s

proc add*[N: static int](a, b: Vec[N]): Vec[N] =
  for i in 0 ..< N:
    result[i] = a[i] + b[i]

proc sub*[N: static int](a, b: Vec[N]): Vec[N] =
  for i in 0 ..< N:
    result[i] = a[i] - b[i]

proc softmax*[N: static int](x: var Vec[N]) =
  var maxVal = x[0]
  for i in 1 ..< N:
    if x[i] > maxVal: maxVal = x[i]

  var sum: float32 = 0
  for i in 0 ..< N:
    x[i] = exp(x[i] - maxVal)
    sum += x[i]

  for i in 0 ..< N:
    x[i] /= sum

proc oneHot*[N: static int](label: int): Vec[N] =
  result[label] = 1.0'f32

proc argmax*[N: static int](x: Vec[N]): int =
  var best = 0
  var bestVal = x[0]
  for i in 1 ..< N:
    if x[i] > bestVal:
      best = i
      bestVal = x[i]
  best

proc crossEntropy*[N: static int](probs: Vec[N]; label: int): float32 =
  const eps = 1e-7'f32
  -ln(max(probs[label], eps))

proc xavierScale*(inputDim: int): float32 =
  1.0'f32 / sqrt(float32(inputDim))