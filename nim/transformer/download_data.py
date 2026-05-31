#!/usr/bin/env python3
"""Download Shakespeare text, tokenize with GPT-2 BPE, and write binary format.

Binary format (all little-endian):
  [uint32 num_sequences] [uint32 seq_len] [uint32 vocab_size]
  [int32[num_sequences * seq_len] tokens]

Each sequence is seq_len consecutive tokens from the corpus (overlapping sliding window).
"""

import struct
import urllib.request
from pathlib import Path

import tiktoken

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SEQ_LEN = 64  # context window size


def main():
    print("Downloading Shakespeare text...")
    resp = urllib.request.urlopen(URL)
    text = resp.read().decode("utf-8")
    print(f"  Corpus size: {len(text)} chars")

    print("Tokenizing with GPT-2 BPE...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    vocab_size = enc.n_vocab
    print(f"  Tokens: {len(tokens)}, vocab_size: {vocab_size}")

    # Create overlapping sequences: sequence i starts at token i
    num_sequences = max(0, len(tokens) - SEQ_LEN)
    print(f"  Sequences: {num_sequences} (seq_len={SEQ_LEN})")

    out_path = "data/shakespeare.bin"
    Path("data").mkdir(exist_ok=True)
    print(f"Writing {out_path}...")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", num_sequences, SEQ_LEN, vocab_size))
        for i in range(num_sequences):
            f.write(struct.pack(f"<{SEQ_LEN}i", *tokens[i : i + SEQ_LEN]))

    print("Done.")


if __name__ == "__main__":
    main()