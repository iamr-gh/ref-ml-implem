## Minimal MNIST MLP experiment.
## Usage:
##   nim c -d:release -r main.nim              # default: hidden=[128]
##   ./main 256 128                            # hidden=[256, 128]
##   ./main none                               # no hidden layers, softmax classifier

import std/[strformat, random, os, strutils, times]
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

proc timed[T](label: string; body: proc(): T): T =
  let t0 = cpuTime()
  result = body()
  echo fmt"TIMING {label:<20} {cpuTime() - t0:.6f}s"

proc timed(label: string; body: proc()) =
  let t0 = cpuTime()
  body()
  echo fmt"TIMING {label:<20} {cpuTime() - t0:.6f}s"

when isMainModule:
  let totalT0 = cpuTime()
  randomize()

  let hidden = parseHidden()
  echo fmt"Model: input={MnistInputDim}, hidden={hidden}, output={MnistClasses}"

  echo "Loading data..."
  var train = timed("load_train", proc(): Dataset[MnistInputDim] = loadMnist("data/train.bin"))
  let test = timed("load_test", proc(): Dataset[MnistInputDim] = loadMnist("data/test.bin"))
  echo fmt"  train: {train.len} samples"
  echo fmt"  test:  {test.len} samples"

  var net = timed("init_model", proc(): Mlp = initMlp(MnistInputDim, MnistClasses, hidden))
  var ws = timed("init_workspace", proc(): MlpWorkspace = initWorkspace(net))
  var history: History

  let baseline = timed("baseline_eval", proc(): Metrics = net.evaluate(test, ws))
  history.add(0, baseline.loss, baseline)
  echo ""
  echo fmt"Epoch 0: test acc={100.0 * baseline.accuracy:5.2f}% loss={baseline.loss:.4f}"

  for epoch in 1 .. Epochs:
    timed(fmt"epoch_{epoch}_shuffle", proc() = shuffle(train))
    let trainLoss = timed(fmt"epoch_{epoch}_train", proc(): float32 = net.trainEpoch(train, Lr, ws))
    let metrics = timed(fmt"epoch_{epoch}_eval", proc(): Metrics = net.evaluate(test, ws))
    history.add(epoch, trainLoss, metrics)

    echo fmt"Epoch {epoch:2d}: train loss={trainLoss:.4f}  test acc={100.0 * metrics.accuracy:5.2f}% loss={metrics.loss:.4f}"

  timed("report", proc() =
    printTable(history)
    plotHistory(history)
  )

  echo fmt"TIMING total                {cpuTime() - totalT0:.6f}s"