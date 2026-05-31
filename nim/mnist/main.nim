## Minimal MNIST softmax-classifier experiment.

import std/[strformat, random]
import data, model, plot

const
  Epochs = 10
  Lr = 0.01'f32

when isMainModule:
  randomize()

  echo "Loading data..."
  var train = loadMnist("data/train.bin")
  let test = loadMnist("data/test.bin")
  echo fmt"  train: {train.len} samples"
  echo fmt"  test:  {test.len} samples"

  var net = initSoftmaxClassifier[MnistInputDim, MnistClasses]()
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