## Shakespeare tokenized dataset loading for transformer.
##
## Binary format (little-endian):
##   [uint32 num_sequences] [uint32 seq_len] [uint32 vocab_size]
##   [int32[num_sequences * seq_len] tokens]

import std/[strformat, random]

type
  Corpus* = object
    tokens*: seq[int32]    # flat: sequence i starts at i*seqLen
    seqLen*: int
    vocabSize*: int
    n*: int                 # number of sequences
    order*: seq[int]        # shuffle indices

proc len*(c: Corpus): int = c.n

template sequence*(c: Corpus; idx: int): untyped =
  block:
    let actual = if c.order.len == 0: idx else: c.order[idx]
    let offset = actual * c.seqLen
    c.tokens.toOpenArray(offset, offset + c.seqLen - 1)

proc readU32Le(buf: string; offset: var int): uint32 {.inline.} =
  result = uint32(buf[offset].ord) or
    (uint32(buf[offset + 1].ord) shl 8) or
    (uint32(buf[offset + 2].ord) shl 16) or
    (uint32(buf[offset + 3].ord) shl 24)
  offset += 4

proc readI32Le(buf: string; offset: var int): int32 {.inline.} =
  cast[int32](readU32Le(buf, offset))

proc loadCorpus*(path: string): Corpus =
  let buf = readFile(path)
  var offset = 0

  let numSeq = readU32Le(buf, offset).int
  let seqLen = readU32Le(buf, offset).int
  let vocabSize = readU32Le(buf, offset).int

  result.n = numSeq
  result.seqLen = seqLen
  result.vocabSize = vocabSize
  result.tokens = newSeq[int32](numSeq * seqLen)
  result.order = newSeq[int](numSeq)

  for i in 0 ..< numSeq * seqLen:
    result.tokens[i] = readI32Le(buf, offset)

  for i in 0 ..< numSeq:
    result.order[i] = i

  echo fmt"  Loaded {path}: {numSeq} sequences, seqLen={seqLen}, vocabSize={vocabSize}"

proc shuffle*(c: var Corpus) =
  for i in countdown(c.n - 1, 1):
    let j = rand(i)
    swap(c.order[i], c.order[j])