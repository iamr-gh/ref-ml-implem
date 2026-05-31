## Native GPT-2 byte-level BPE tokenizer/detokenizer.
##
## Assets are produced by download_data.py as simple TSV files so the runtime has
## no Python/tiktoken dependency.

import std/[strutils, tables, os]

type
  Tokenizer* = object
    tokenToId*: Table[string, int]
    idToBytes*: seq[string]
    byteToToken*: array[256, string]
    pairRank*: Table[string, int]
    vocabSize*: int

proc fromHexDigit(c: char): int {.inline.} =
  if c in '0'..'9': ord(c) - ord('0')
  elif c in 'a'..'f': 10 + ord(c) - ord('a')
  elif c in 'A'..'F': 10 + ord(c) - ord('A')
  else: 0

proc hexToString*(hex: string): string =
  result = newString(hex.len div 2)
  var j = 0
  for i in countup(0, hex.len - 2, 2):
    result[j] = char((fromHexDigit(hex[i]) shl 4) or fromHexDigit(hex[i + 1]))
    inc j

proc pairKey(a, b: string): string {.inline.} = a & "\t" & b

proc loadTokenizer*(dir = "data"): Tokenizer =
  result.tokenToId = initTable[string, int](60000)
  result.pairRank = initTable[string, int](60000)
  result.idToBytes = @[]

  for line in lines(dir / "byte_encoder.tsv"):
    if line.len == 0: continue
    let parts = line.split('\t')
    let b = parseInt(parts[0])
    result.byteToToken[b] = hexToString(parts[1])

  for line in lines(dir / "encoder.tsv"):
    if line.len == 0: continue
    let parts = line.split('\t')
    let id = parseInt(parts[0])
    let tok = hexToString(parts[1])
    let bytes = hexToString(parts[2])
    while result.idToBytes.len <= id: result.idToBytes.add("")
    result.idToBytes[id] = bytes
    result.tokenToId[tok] = id

  for line in lines(dir / "merges.tsv"):
    if line.len == 0: continue
    let parts = line.split('\t')
    let rank = parseInt(parts[0])
    result.pairRank[pairKey(hexToString(parts[1]), hexToString(parts[2]))] = rank

  result.vocabSize = result.idToBytes.len

proc isSpaceByte(c: char): bool {.inline.} = c in {' ', '\n', '\r', '\t'}
proc isAlphaNumByte(c: char): bool {.inline.} =
  (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9')

proc chunkText(text: string): seq[string] =
  ## Small native approximation of GPT-2 pre-tokenization: keep a leading space
  ## attached to word/number runs, split punctuation and whitespace runs.
  var i = 0
  while i < text.len:
    var start = i
    if text[i] == ' ' and i + 1 < text.len and isAlphaNumByte(text[i + 1]):
      inc i
      while i < text.len and isAlphaNumByte(text[i]): inc i
      result.add(text[start ..< i])
    elif isAlphaNumByte(text[i]):
      while i < text.len and isAlphaNumByte(text[i]): inc i
      result.add(text[start ..< i])
    elif isSpaceByte(text[i]):
      while i < text.len and isSpaceByte(text[i]) and not (text[i] == ' ' and i + 1 < text.len and isAlphaNumByte(text[i + 1])): inc i
      result.add(text[start ..< i])
    else:
      result.add(text[i .. i])
      inc i

proc bpe(t: Tokenizer; piece: string): seq[string] =
  if piece.len == 0: return @[]
  var symbols = newSeq[string](piece.len)
  for i, ch in piece:
    symbols[i] = t.byteToToken[ord(ch)]

  while symbols.len > 1:
    var bestRank = high(int)
    var bestIdx = -1
    for i in 0 ..< symbols.len - 1:
      let key = pairKey(symbols[i], symbols[i + 1])
      if t.pairRank.hasKey(key):
        let r = t.pairRank[key]
        if r < bestRank:
          bestRank = r
          bestIdx = i
    if bestIdx < 0: break

    let merged = symbols[bestIdx] & symbols[bestIdx + 1]
    var next = newSeqOfCap[string](symbols.len - 1)
    var i = 0
    while i < symbols.len:
      if i == bestIdx:
        next.add(merged)
        i += 2
      else:
        next.add(symbols[i])
        inc i
    symbols = move(next)
  symbols

proc encode*(t: Tokenizer; text: string): seq[int32] =
  for piece in chunkText(text):
    for tok in t.bpe(piece):
      if t.tokenToId.hasKey(tok):
        result.add(int32(t.tokenToId[tok]))
      else:
        # Should not happen with GPT-2 assets, but byte fallback keeps this robust.
        for ch in tok:
          let one = t.byteToToken[ord(ch)]
          result.add(int32(t.tokenToId[one]))

proc decode*(t: Tokenizer; ids: openArray[int32]): string =
  for idv in ids:
    let id = int(idv)
    if id >= 0 and id < t.idToBytes.len:
      result.add(t.idToBytes[id])

proc decodeId*(t: Tokenizer; id: int): string =
  if id >= 0 and id < t.idToBytes.len: t.idToBytes[id] else: "<?>"
