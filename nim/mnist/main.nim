## A from-scratch neural network for MNIST digit classification.
## Direct port of the Odin mnist/main.odin implementation.
## Single linear layer -> softmax, with SGD training.
##
## Binary data format (.bin):
##   [uint32 num_images] [uint32 rows] [uint32 cols]
##   [float32[num_images * rows * cols] pixels]   (normalized 0-1)
##   [uint8[num_images] labels]

import std/[math, strformat, streams, random]

const
  RawImgSize* = 28       # original MNIST image size
  ImgSize* = 30           # padded to 30x30 (1-pixel border of 1s for bias)
  InputDim* = ImgSize * ImgSize   # 900
  OutputDim* = 10                # digits 0-9

type
  Image* = object
    data*: array[ImgSize, array[ImgSize, float32]]

  Matrix*[M: static int; N: static int] = array[M, array[N, float32]]

# ── Core ops ────────────────────────────────────────────────────────────

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

proc pad28to30*(raw: array[RawImgSize, array[RawImgSize, float32]]): Image =
  ## Pad a 28x28 image into 30x30 with a 1-pixel border of 1.0 (bias trick).
  for i in 0 ..< ImgSize:
    for j in 0 ..< ImgSize:
      if i == 0 or i == ImgSize - 1 or j == 0 or j == ImgSize - 1:
        result.data[i][j] = 1.0'f32
      else:
        result.data[i][j] = raw[i - 1][j - 1]

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
  for i in 0 ..< InputDim:
    for j in 0 ..< OutputDim:
      grad[i][j] = input[0][i] * dldy[0][j]

proc crossEntropyLoss*(yHat: Matrix[1, OutputDim]; y: int): float32 =
  ## Cross-entropy loss for a single sample.
  const eps: float32 = 1e-7'f32
  return -ln(max(yHat[0][y], eps))

# ── Training ────────────────────────────────────────────────────────────

proc train*(
  xs: seq[Image];
  ys: seq[int];
  weights: var Matrix[InputDim, OutputDim];
  lr: float32;
  batch: int
): float32 =
  ## SGD training loop (one sample at a time, batch param unused for now).
  ## Returns average cross-entropy loss over the epoch.
  var
    grad: Matrix[InputDim, OutputDim]
    input: Matrix[1, InputDim]
    yHat: Matrix[1, OutputDim]

  var totalLoss: float32 = 0

  for idx in 0 ..< xs.len:
    let img = xs[idx]
    let y = ys[idx]

    var yOneHot: Matrix[1, OutputDim]
    yOneHot[0][y] = 1.0'f32

    # Clear gradients
    zeroMem(addr grad, sizeof(grad))

    flatten(img, input)
    forward(weights, input, yHat)

    totalLoss += crossEntropyLoss(yHat, y)

    # dldy = y_hat - y_one_hot
    var dldy: Matrix[1, OutputDim]
    for j in 0 ..< OutputDim:
      dldy[0][j] = yHat[0][j] - yOneHot[0][j]

    backward(grad, input, dldy)

    # Update step
    for i in 0 ..< InputDim:
      for j in 0 ..< OutputDim:
        weights[i][j] -= grad[i][j] * lr

    if (idx + 1) mod 10000 == 0:
      echo fmt"  trained {idx + 1}/{xs.len} samples"

  return totalLoss / float32(xs.len)

# ── Data loading ────────────────────────────────────────────────────────

proc loadDataset*(path: string): (seq[Image], seq[int]) =
  ## Load the binary dataset format produced by download_mnist.py.
  ## Format: [uint32 num] [uint32 rows] [uint32 cols]
  ##         [float32[num*rows*cols] pixels] [uint8[num] labels]
  var fs = newFileStream(path, fmRead)
  if fs == nil:
    echo fmt"ERROR: could not open {path}"
    quit(1)

  let numImages = fs.readUint32().int
  let rows = fs.readUint32().int
  let cols = fs.readUint32().int
  doAssert rows == RawImgSize and cols == RawImgSize,
    fmt"Expected 28x28 images, got {rows}x{cols}"

  var images = newSeq[Image](numImages)
  var labels = newSeq[int](numImages)

  for i in 0 ..< numImages:
    var raw: array[RawImgSize, array[RawImgSize, float32]]
    for r in 0 ..< rows:
      for c in 0 ..< cols:
        raw[r][c] = fs.readFloat32()

    images[i] = pad28to30(raw)

  for i in 0 ..< numImages:
    labels[i] = fs.readUint8().int

  fs.close()
  return (images, labels)

# ── Shuffling ───────────────────────────────────────────────────────────

proc shuffleData*(xs: var seq[Image]; ys: var seq[int]) =
  ## Fisher-Yates shuffle of paired images and labels.
  doAssert xs.len == ys.len
  for i in countdown(xs.len - 1, 1):
    let j = rand(i)
    swap(xs[i], xs[j])
    swap(ys[i], ys[j])

# ── Evaluation ──────────────────────────────────────────────────────────

proc evaluate*(
  xs: seq[Image];
  ys: seq[int];
  weights: Matrix[InputDim, OutputDim]
): (int, float32) =
  ## Returns (correct_count, avg_loss) on the given dataset.
  var input: Matrix[1, InputDim]
  var output: Matrix[1, OutputDim]

  var correct = 0
  var totalLoss: float32 = 0

  for idx in 0 ..< xs.len:
    flatten(xs[idx], input)
    forward(weights, input, output)

    var maxV = output[0][0]
    var maxI = 0
    for j in 0 ..< OutputDim:
      if output[0][j] > maxV:
        maxV = output[0][j]
        maxI = j

    if maxI == ys[idx]:
      inc correct

    totalLoss += crossEntropyLoss(output, ys[idx])

  return (correct, totalLoss / float32(xs.len))

# ── Main ─────────────────────────────────────────────────────────────────

when isMainModule:
  randomize()

  echo "Loading training data..."
  let (trainImages, trainLabels) = loadDataset("data/train.bin")
  echo fmt"  {trainImages.len} training images loaded"

  echo "Loading test data..."
  let (testImages, testLabels) = loadDataset("data/test.bin")
  echo fmt"  {testImages.len} test images loaded"

  var weights: Matrix[InputDim, OutputDim]

  # Small Xavier-like init: uniform in [-1/sqrt(InputDim), 1/sqrt(InputDim)]
  let scale = 1.0'f32 / sqrt(float32(InputDim))
  for i in 0 ..< InputDim:
    for j in 0 ..< OutputDim:
      weights[i][j] = rand(2.0'f32 * scale) - scale

  # Evaluate before training (random baseline)
  echo "\nPre-training evaluation on test set..."
  var (preCorrect, preLoss) = evaluate(testImages, testLabels, weights)
  echo fmt"  accuracy: {preCorrect}/{testImages.len} ({100.0 * preCorrect.float / testImages.len.float:.2f}%)"
  echo fmt"  avg loss: {preLoss:.4f}"

  # Train 1 epoch
  echo "\nTraining 1 epoch..."
  var trainXs = trainImages
  var trainYs = trainLabels
  shuffleData(trainXs, trainYs)

  let avgLoss = train(trainXs, trainYs, weights, lr = 0.01'f32, batch = 0)
  echo fmt"  avg train loss: {avgLoss:.4f}"

  # Evaluate on test set
  echo "\nEvaluating on test set..."
  let (correct, testLoss) = evaluate(testImages, testLabels, weights)
  echo fmt"  accuracy: {correct}/{testImages.len} ({100.0 * correct.float / testImages.len.float:.2f}%)"
  echo fmt"  avg test loss: {testLoss:.4f}"