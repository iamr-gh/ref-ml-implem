## MNIST binary dataset loading.

import std/[strformat, random]

const
  MnistRows* = 28
  MnistCols* = 28
  MnistInputDim* = MnistRows * MnistCols
  MnistClasses* = 10

type
  Dataset*[N: static int] = object
    xs*: seq[float32]  # flat: sample i starts at i*N
    ys*: seq[int]
    order*: seq[int]   # shuffle indices instead of moving pixels
    n*: int

proc len*[N: static int](ds: Dataset[N]): int = ds.n

proc sample*[N: static int](ds: Dataset[N]; logicalIdx: int): openArray[float32] =
  let actual = if ds.order.len == 0: logicalIdx else: ds.order[logicalIdx]
  let offset = actual * N
  ds.xs.toOpenArray(offset, offset + N - 1)

proc label*[N: static int](ds: Dataset[N]; logicalIdx: int): int =
  let actual = if ds.order.len == 0: logicalIdx else: ds.order[logicalIdx]
  ds.ys[actual]

proc readU32Le(buf: string; offset: var int): uint32 {.inline.} =
  result = uint32(buf[offset].ord) or
    (uint32(buf[offset + 1].ord) shl 8) or
    (uint32(buf[offset + 2].ord) shl 16) or
    (uint32(buf[offset + 3].ord) shl 24)
  offset += 4

proc readF32Le(buf: string; offset: var int): float32 {.inline.} =
  let bits = readU32Le(buf, offset)
  cast[float32](bits)

proc loadMnist*(path: string): Dataset[MnistInputDim] =
  ## Load .bin format produced by download_mnist.py:
  ## [uint32 num] [uint32 rows] [uint32 cols]
  ## [float32[num*rows*cols] pixels] [uint8[num] labels]
  let buf = readFile(path)
  var offset = 0

  let numImages = readU32Le(buf, offset).int
  let rows = readU32Le(buf, offset).int
  let cols = readU32Le(buf, offset).int
  doAssert rows == MnistRows and cols == MnistCols,
    fmt"expected 28x28 images, got {rows}x{cols}"

  result.n = numImages
  result.xs = newSeq[float32](numImages * MnistInputDim)
  result.ys = newSeq[int](numImages)
  result.order = newSeq[int](numImages)

  for i in 0 ..< numImages * MnistInputDim:
    result.xs[i] = readF32Le(buf, offset)

  for i in 0 ..< numImages:
    result.ys[i] = buf[offset].ord
    result.order[i] = i
    inc offset

proc shuffle*[N: static int](ds: var Dataset[N]) =
  ## Fisher-Yates shuffle over indices only.
  for i in countdown(ds.n - 1, 1):
    let j = rand(i)
    swap(ds.order[i], ds.order[j])