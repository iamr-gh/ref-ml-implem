#!/usr/bin/env python3
"""Download Tiny Shakespeare and GPT-2 tokenizer assets.

No tokenization happens here. The Nim/Odin programs load these assets and perform
native byte-level GPT-2 BPE encode/decode themselves.

Outputs under data/:
  input.txt          raw Shakespeare corpus
  encoder.tsv        token_id<TAB>hex(token_string_utf8)<TAB>hex(decoded_bytes)
  byte_encoder.tsv   byte_value<TAB>hex(token_string_utf8)
  merges.tsv         rank<TAB>hex(left_token_utf8)<TAB>hex(right_token_utf8)
"""

import json
import urllib.request
from pathlib import Path

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ENCODER_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
MERGES_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    data = Path("data")
    data.mkdir(exist_ok=True)

    print("Downloading Tiny Shakespeare...")
    text = urllib.request.urlopen(SHAKESPEARE_URL).read()
    (data / "input.txt").write_bytes(text)
    print(f"  wrote data/input.txt ({len(text)} bytes)")

    print("Downloading GPT-2 encoder.json...")
    encoder = json.loads(urllib.request.urlopen(ENCODER_URL).read().decode("utf-8"))

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    print("Writing encoder.tsv...")
    with (data / "encoder.tsv").open("w", encoding="utf-8") as f:
        for token, idx in sorted(encoder.items(), key=lambda kv: kv[1]):
            decoded = bytes(byte_decoder[ch] for ch in token)
            f.write(f"{idx}\t{token.encode('utf-8').hex()}\t{decoded.hex()}\n")

    print("Writing byte_encoder.tsv...")
    with (data / "byte_encoder.tsv").open("w", encoding="utf-8") as f:
        for b in range(256):
            f.write(f"{b}\t{byte_encoder[b].encode('utf-8').hex()}\n")

    print("Downloading GPT-2 vocab.bpe...")
    merges_text = urllib.request.urlopen(MERGES_URL).read().decode("utf-8")
    print("Writing merges.tsv...")
    rank = 0
    with (data / "merges.tsv").open("w", encoding="utf-8") as f:
        for line in merges_text.splitlines():
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            f.write(f"{rank}\t{a.encode('utf-8').hex()}\t{b.encode('utf-8').hex()}\n")
            rank += 1

    print(f"Done. vocab={len(encoder)}, merges={rank}")


if __name__ == "__main__":
    main()
