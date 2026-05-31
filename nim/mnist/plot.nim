## Terminal-based plotting utilities for training metrics.
## No external dependencies - just ASCII/Unicode box drawing.

import std/[strformat, sequtils]

const
  plotW* = 50   # plot area width in chars
  plotH* = 12   # plot area height in lines

proc plotLineChart*(
  title: string;
  epochs: seq[int];
  values: seq[float];
  unit: string
) =
  ## Draw a small ASCII line chart of values over epochs.
  let n = epochs.len
  if n < 2:
    return

  # Find range
  var
    vMin = values[0]
    vMax = values[0]
  for i in 0 ..< n:
    if values[i] < vMin: vMin = values[i]
    if values[i] > vMax: vMax = values[i]

  # Add 5% margin
  let vRange = max(0.001, vMax - vMin)
  vMin = max(0.0, vMin - vRange * 0.05)
  vMax = vMax + vRange * 0.05

  # Build 2D grid
  var grid = newSeqWith(plotH, newString(plotW))
  for r in 0 ..< plotH:
    for c in 0 ..< plotW:
      grid[r][c] = ' '

  for i in 0 ..< n:
    let xc = int(float(plotW - 1) * float(epochs[i] - epochs[0]) / float(max(1, epochs[n-1] - epochs[0])))
    let yr = int(float(plotH - 1) * (1.0 - (values[i] - vMin) / (vMax - vMin)))
    let x = xc.clamp(0, plotW - 1)
    let y = yr.clamp(0, plotH - 1)
    grid[y][x] = '*'

  # Draw chart
  var dashes = ""
  for _ in 0 ..< 54 - title.len:
    dashes.add("─")
  echo ""
  echo fmt"┌─ {title} {dashes}┐"
  for r in 0 ..< plotH:
    let yVal = vMin + (vMax - vMin) * float(plotH - 1 - r) / float(plotH - 1)
    var line = fmt"│{yVal:6.2f} │"
    for c in 0 ..< plotW:
      line.add(grid[r][c])
    line.add("│")
    echo line

  # Bottom border
  var bottom = "│       └"
  for c in 0 ..< plotW:
    bottom.add("─")
  bottom.add("│")
  echo bottom

  # Epoch label
  var label = fmt"│       epochs {epochs[0]}-{epochs[n-1]}"
  while label.len < plotW + 10:
    label.add(" ")
  label.add("│")
  echo label
  echo "└───────────────────────────────────────────────────────┘"


proc plotAccuracyLoss*(
  epochs: seq[int];
  accs: seq[float];
  losses: seq[float32]
) =
  ## Plot both test accuracy (as %) and average loss.
  plotLineChart("Accuracy %", epochs, accs.mapIt(100.0 * it), "%")
  plotLineChart("Avg Loss", epochs, losses.mapIt(float(it)), "")


proc printTable*(
  epochs: seq[int];
  accs: seq[float];
  losses: seq[float32];
  trainLosses: seq[float32]
) =
  ## Print a compact results table.
  echo ""
  echo "Epoch │ Train Loss │ Test Loss │ Test Acc"
  echo "──────┼────────────┼───────────┼─────────"
  for i in 0 ..< epochs.len:
    echo fmt"  {epochs[i]:2d}  │  {trainLosses[i]:.4f}  │  {losses[i]:.4f}  │ {100.0 * accs[i]:5.2f}%"