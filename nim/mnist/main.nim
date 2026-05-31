## Minimal MNIST MLP experiment.
## Usage:
##   nim c -d:release -r main.nim              # default: hidden=[128]
##   ./main 256 128                            # hidden=[256, 128]
##   ./main none                               # no hidden layers, softmax classifier

import std/[strformat, random, os, strutils]
import data, model, plot

const
  Epochs = 10
  Lr = 0.01'f32

proc parseHidden(): seq[int] =
  ## Hidden layer sizes are passed as CLI args.
  ## Examples: ./main 128 | ./main 256 128 | ./main none
  let args = commandLineParams()
  if args.len == 0:
    return @[128]
  if args.len == 1 and args[0].toLowerAscii in ["none", "0", "linear"]:
    return @[]

  for arg in args:
    let h = parseInt(arg)
    doAssert h > 0, "hidden layer sizes must be positive integers"
    result.add(h)

when isMainModule:
  randomize()

  let hidden = parseHidden()
  echo fmt"Model: input={MnistInputDim}, hidden={hidden}, output={MnistClasses}"

  echo "Loading data..."
  var train = loadMnist("data/train.bin")
  let test = loadMnist("data/test.bin")
  echo fmt"  train: {train.len} samples"
  echo fmt"  test:  {test.len} samples"

  var net = initMlp(MnistInputDim, MnistClasses, hidden)
  var history: History

  let baseline = net.evaluate(test)
  history.add(0, baseline.loss, baseline)
  echo ""
  echo fmt"Epoch 0: test acc={100.0 * baseline.accuracy:5.2f}% loss={baseline.loss:.4f}"

  for epoch in 1 .. Epochs:
    shuffle(train)
    let trainLoss = net.trainEpoch(train, Lr)
    let metrics = net.evaluate(test)
    history.add(epoch, trainLoss, metrics)

    echo fmt"Epoch {epoch:2d}: train loss={trainLoss:.4f}  test acc={100.0 * metrics.accuracy:5.2f}% loss={metrics.loss:.4f}"

  printTable(history)
  plotHistory(history)