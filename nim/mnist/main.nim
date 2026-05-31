## A from-scratch neural network for MNIST digit classification.
## Direct port of the Odin mnist/main.odin implementation.
## Single linear layer -> softmax, with SGD training.

import std/[math, strformat]

const
  ImgSize* = 30          # image is 30x30 (matching Odin code)
  InputDim* = ImgSize * ImgSize   # 900
  OutputDim* = 10        # digits 0-9

type
  Image* = object
    data*: array[ImgSize, array[ImgSize, float32]]

  Matrix*[M: static int; N: static int] = array[M, array[N, float32]]

# ── Templates for type-flexible generic ops ──────────────────────────────
# Nim's generics work differently from Odin's dollar-sign polymorphism.
# We use static ints for dimensions and float32 throughout for simplicity.

proc transpose*[M, N: static int](
  x: Matrix[M, N];
  xT: var Matrix[N, M]
) =
  for i in 0 ..< M:
    for j in 0 ..< N:
      xT[j][i] = x[i][j]

proc softmax*[A: static int](vec: var Matrix[1, A]) =
  ## In-place softmax along the single row.
  ## Uses the max-subtraction trick for numerical stability.
  var maxVal = vec[0][0]
  for j in 0 ..< A:
    if vec[0][j] > maxVal:
      maxVal = vec[0][j]

  var sum: float32 = 0
  for j in 0 ..< A:
    vec[0][j] = exp(vec[0][j] - maxVal)
    sum += vec[0][j]
  for j in 0 ..< A:
    vec[0][j] /= sum

proc matmul*[A, B, C: static int](
  a: Matrix[A, B];
  b: Matrix[B, C];
  o: var Matrix[A, C]
) =
  for i in 0 ..< A:
    for j in 0 ..< C:
      var s: float32 = 0
      for k in 0 ..< B:
        s += a[i][k] * b[k][j]
      o[i][j] = s

proc flatten*(img: Image; flat: var Matrix[1, InputDim]) =
  ## Flatten a 30x30 image into a 1x900 row vector.
  for i in 0 ..< ImgSize:
    for j in 0 ..< ImgSize:
      flat[0][i * ImgSize + j] = img.data[i][j]

proc forward*(
  weights: Matrix[InputDim, OutputDim];
  input: Matrix[1, InputDim];
  output: var Matrix[1, OutputDim]
) =
  matmul(input, weights, output)
  softmax(output)

proc predict*(img: Image; weights: Matrix[InputDim, OutputDim]): int =
  ## Return the predicted digit (0-9) for an image.
  var input: Matrix[1, InputDim]
  flatten(img, input)

  var output: Matrix[1, OutputDim]
  forward(weights, input, output)

  var maxV = output[0][0]
  var maxI = 0
  for j in 0 ..< OutputDim:
    if output[0][j] > maxV:
      maxV = output[0][j]
      maxI = j
  return maxI

proc backward*(
  grad: var Matrix[InputDim, OutputDim];
  input: Matrix[1, InputDim];
  dldy: Matrix[1, OutputDim]
) =
  ## Gradient of the loss w.r.t. weights:  input^T @ (y_hat - y_one_hot)
  ## We skip explicit transpose and index directly.
  for i in 0 ..< InputDim:
    for j in 0 ..< OutputDim:
      grad[i][j] = input[0][i] * dldy[0][j]

proc train*(
  xs: seq[Image];
  ys: seq[int];
  weights: var Matrix[InputDim, OutputDim];
  lr: float32;
  batch: int
) =
  ## SGD training loop (one sample at a time, batch param unused for now).
  var
    grad: Matrix[InputDim, OutputDim]
    input: Matrix[1, InputDim]
    yHat: Matrix[1, OutputDim]

  for idx in 0 ..< xs.len:
    let img = xs[idx]
    let y = ys[idx]

    var yOneHot: Matrix[1, OutputDim]
    yOneHot[0][y] = 1.0'f32

    # Clear gradients
    zeroMem(addr grad, sizeof(grad))

    flatten(img, input)
    forward(weights, input, yHat)

    # dldy = y_hat - y_one_hot
    var dldy: Matrix[1, OutputDim]
    for j in 0 ..< OutputDim:
      dldy[0][j] = yHat[0][j] - yOneHot[0][j]

    backward(grad, input, dldy)

    # Update step
    for i in 0 ..< InputDim:
      for j in 0 ..< OutputDim:
        weights[i][j] -= grad[i][j] * lr

# ── Main ─────────────────────────────────────────────────────────────────

when isMainModule:
  var img: Image
  for i in 0 ..< ImgSize:
    for j in 0 ..< ImgSize:
      img.data[i][j] = 1.0'f32

  var weights: Matrix[InputDim, OutputDim]
  weights[10][8] = 1.0'f32

  # Assuming pretrained with 1 for bias
  let pred = predict(img, weights)
  echo fmt"prediction: {pred}"

  var xs = @[img]
  var ys = @[0]

  train(xs, ys, weights, 0.1'f32, 0)

  echo "training step complete"