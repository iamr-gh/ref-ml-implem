## Terminal-based plotting utilities for training metrics.
## No external dependencies - just ASCII/Unicode box drawing.

import std/[strformat, sequtils]
import model

const
  plotW* = 50
  plotH* = 12

type
  History* = object
    epochs*: seq[int]
    trainLoss*: seq[float32]
    test*: seq[Metrics]

proc add*(h: var History; epoch: int; trainLoss: float32; test: Metrics) =
  h.epochs.add(epoch)
  h.trainLoss.add(trainLoss)
  h.test.add(test)

proc plotLineChart*(title: string; epochs: seq[int]; values: seq[float]) =
  let n = epochs.len
  if n < 2: return

  var vMin = values[0]
  var vMax = values[0]
  for v in values:
    if v < vMin: vMin = v
    if v > vMax: vMax = v

  let margin = max(0.001, vMax - vMin) * 0.05
  vMin = max(0.0, vMin - margin)
  vMax += margin

  var grid = newSeqWith(plotH, newString(plotW))
  for r in 0 ..< plotH:
    for c in 0 ..< plotW:
      grid[r][c] = ' '

  for i in 0 ..< n:
    let x = int(float(plotW - 1) * float(epochs[i] - epochs[0]) / float(max(1, epochs[^1] - epochs[0]))).clamp(0, plotW - 1)
    let y = int(float(plotH - 1) * (1.0 - (values[i] - vMin) / (vMax - vMin))).clamp(0, plotH - 1)
    grid[y][x] = '*'

  var dashes = ""
  for _ in 0 ..< max(0, 54 - title.len): dashes.add("─")

  echo ""
  echo fmt"┌─ {title} {dashes}┐"
  for r in 0 ..< plotH:
    let yVal = vMin + (vMax - vMin) * float(plotH - 1 - r) / float(plotH - 1)
    var line = fmt"│{yVal:6.2f} │"
    for c in 0 ..< plotW: line.add(grid[r][c])
    line.add("│")
    echo line

  var bottom = "│       └"
  for _ in 0 ..< plotW: bottom.add("─")
  bottom.add("│")
  echo bottom

  var label = fmt"│       epochs {epochs[0]}-{epochs[^1]}"
  while label.len < plotW + 10: label.add(" ")
  label.add("│")
  echo label
  echo "└───────────────────────────────────────────────────────┘"

proc printTable*(h: History) =
  echo ""
  echo "Epoch │ Train Loss │ Test Loss │ Test Acc"
  echo "──────┼────────────┼───────────┼─────────"
  for i in 0 ..< h.epochs.len:
    echo fmt"  {h.epochs[i]:2d}  │  {h.trainLoss[i]:.4f}  │  {h.test[i].loss:.4f}  │ {100.0 * h.test[i].accuracy:5.2f}%"

proc plotHistory*(h: History) =
  plotLineChart("Accuracy %", h.epochs, h.test.mapIt(100.0 * float(it.accuracy)))
  plotLineChart("Avg Loss", h.epochs, h.test.mapIt(float(it.loss)))