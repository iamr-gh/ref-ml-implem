## Minimal GPT-style transformer language-model experiment.
## Usage:
##   python3 download_data.py
##   nim c -d:release -r main.nim
##   ./main 64 4 2 256   # dModel heads blocks ffDim

import std/[strformat, os, strutils, times]
import data, model, tokenizer

const
  Epochs = 2
  Lr = 0.001'f32
  MaxTrainSeq = 2
  MaxEvalSeq = 1

type
  History = object
    epochs: seq[int]
    trainLoss: seq[float32]
    evalLoss: seq[float32]
    evalAcc: seq[float32]

proc add(h: var History; epoch: int; trainLoss: float32; m: Metrics) =
  h.epochs.add(epoch)
  h.trainLoss.add(trainLoss)
  h.evalLoss.add(m.loss)
  h.evalAcc.add(m.accuracy)

proc printTable(h: History) =
  echo ""
  echo "Epoch │ Train Loss │ Eval Loss │ Eval Acc"
  echo "──────┼────────────┼───────────┼─────────"
  for i in 0 ..< h.epochs.len:
    echo fmt"  {h.epochs[i]:2d}  │  {h.trainLoss[i]:.4f}  │  {h.evalLoss[i]:.4f}  │ {100.0 * h.evalAcc[i]:5.2f}%"

proc parseConfig(): tuple[dModel, heads, blocks, ffDim: int] =
  result = (DefaultDModel, DefaultHeads, DefaultBlocks, DefaultFFDim)
  let args = commandLineParams()
  if args.len >= 1: result.dModel = parseInt(args[0])
  if args.len >= 2: result.heads = parseInt(args[1])
  if args.len >= 3: result.blocks = parseInt(args[2])
  if args.len >= 4: result.ffDim = parseInt(args[3])

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

  let cfg = parseConfig()
  echo fmt"Transformer LM: dModel={cfg.dModel}, heads={cfg.heads}, blocks={cfg.blocks}, ffDim={cfg.ffDim}"

  echo "Loading GPT-2 tokenizer assets..."
  let tok = timed("load_tokenizer", proc(): Tokenizer = loadTokenizer("data"))

  echo "Tokenizing Shakespeare natively..."
  var corpus = timed("load_corpus", proc(): Corpus = loadCorpus("data/input.txt", tok, 64))

  var net = timed("init_model", proc(): Transformer =
    initTransformer(corpus.vocabSize, corpus.seqLen, cfg.dModel, cfg.heads, cfg.blocks, cfg.ffDim)
  )
  var ws = timed("init_workspace", proc(): Workspace = initWorkspace(net))
  var history: History

  proc evalSome(): Metrics =
    var totalLoss: float32 = 0
    var totalAcc: float32 = 0
    let n = min(MaxEvalSeq, corpus.len)
    for i in 0 ..< n:
      let m = net.evaluate(corpus.sequence(i), ws)
      totalLoss += m.loss
      totalAcc += m.accuracy
    result.loss = totalLoss / float32(n)
    result.accuracy = totalAcc / float32(n)

  let baseline = timed("baseline_eval", proc(): Metrics = evalSome())
  history.add(0, baseline.loss, baseline)
  echo fmt"Epoch 0: eval acc={100.0 * baseline.accuracy:5.2f}% loss={baseline.loss:.4f}"

  for epoch in 1 .. Epochs:
    timed(fmt"epoch_{epoch}_shuffle", proc() = shuffle(corpus))
    var totalLoss: float32 = 0
    let nTrain = min(MaxTrainSeq, corpus.len)
    timed(fmt"epoch_{epoch}_train", proc() =
      for i in 0 ..< nTrain:
        let m = net.trainOne(corpus.sequence(i), Lr, ws)
        totalLoss += m.loss
    )
    let trainLoss = totalLoss / float32(nTrain)
    let metrics = timed(fmt"epoch_{epoch}_eval", proc(): Metrics = evalSome())
    history.add(epoch, trainLoss, metrics)
    echo fmt"Epoch {epoch:2d}: train loss={trainLoss:.4f}  eval acc={100.0 * metrics.accuracy:5.2f}% loss={metrics.loss:.4f}"

  timed("report", proc() = printTable(history))

  let prompt = "To be"
  let promptIds = tok.encode(prompt)
  let nextId = net.bestNextToken(promptIds, ws)
  echo fmt"Prompt: {prompt.repr} -> next token id={nextId} decoded={tok.decodeId(nextId).repr}"

  echo fmt"TIMING total                {cpuTime() - totalT0:.6f}s"
