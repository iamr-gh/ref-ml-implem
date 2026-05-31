## Corpus construction for transformer language modeling.
##
## Raw text is tokenized natively with tokenizer.nim. No pre-tokenized binary or
## Python tokenizer runtime is needed.

import std/[strformat]
import tokenizer

type
  Corpus* = object
    tokens*: seq[int32]    # flat: all token ids
    seqLen*: int
    vocabSize*: int
    n*: int                 # number of sliding-window sequences
    order*: seq[int]        # shuffle indices

proc len*(c: Corpus): int = c.n

template sequence*(c: Corpus; idx: int): untyped =
  block:
    let actual = if c.order.len == 0: idx else: c.order[idx]
    c.tokens.toOpenArray(actual, actual + c.seqLen - 1)

proc loadCorpus*(textPath: string; tok: Tokenizer; seqLen: int): Corpus =
  let text = readFile(textPath)
  result.tokens = tok.encode(text)
  result.seqLen = seqLen
  result.vocabSize = tok.vocabSize
  result.n = max(0, result.tokens.len - seqLen)
  result.order = newSeq[int](result.n)
  for i in 0 ..< result.n:
    result.order[i] = i
  echo fmt"  Loaded {textPath}: {result.tokens.len} tokens, {result.n} sequences, seqLen={seqLen}, vocabSize={result.vocabSize}"

proc nextShuffleRand(seed: var uint32): uint32 {.inline.} =
  seed = seed * 1664525'u32 + 1013904223'u32
  seed

proc shuffle*(c: var Corpus) =
  ## Deterministic Fisher-Yates so Nim/Odin comparisons use identical samples.
  var seed = 246813579'u32
  for i in countdown(c.n - 1, 1):
    let j = int(nextShuffleRand(seed) mod uint32(i + 1))
    swap(c.order[i], c.order[j])
