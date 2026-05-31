## MNIST binary dataset loading.

import std/[strformat, streams, random]
import tensor

const
  MnistRows* = 28
  MnistCols* = 28
  MnistInputDim* = MnistRows * MnistCols
  MnistClasses* = 10

type
  Dataset*[N: static int] = object
    xs*: seq[Vec[N]]
    ys*: seq[int]

proc len*[N: static int](ds: Dataset[N]): int = ds.xs.len

proc loadMnist*(path: string): Dataset[MnistInputDim] =
  ## Load .bin format produced by download_mnist.py:
  ## [uint32 num] [uint32 rows] [uint32 cols]
  ## [float32[num*rows*cols] pixels] [uint8[num] labels]
  var fs = newFileStream(path, fmRead)
  if fs == nil:
    raise newException(IOError, fmt"could not open {path}")

  let numImages = fs.readUint32().int
  let rows = fs.readUint32().int
  let cols = fs.readUint32().int
  doAssert rows == MnistRows and cols == MnistCols,
    fmt"expected 28x28 images, got {rows}x{cols}"

  result.xs = newSeq[Vec[MnistInputDim]](numImages)
  result.ys = newSeq[int](numImages)

  for i in 0 ..< numImages:
    for p in 0 ..< MnistInputDim:
      result.xs[i][p] = fs.readFloat32()

  for i in 0 ..< numImages:
    result.ys[i] = fs.readUint8().int

  fs.close()

proc shuffle*[N: static int](ds: var Dataset[N]) =
  ## Fisher-Yates shuffle of paired samples and labels.
  doAssert ds.xs.len == ds.ys.len
  for i in countdown(ds.xs.len - 1, 1):
    let j = rand(i)
    swap(ds.xs[i], ds.xs[j])
    swap(ds.ys[i], ds.ys[j])