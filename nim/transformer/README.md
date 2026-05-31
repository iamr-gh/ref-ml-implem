# Nim Transformer LM

A minimal GPT-style transformer language model in the same style as `nim/mnist`:
flat buffers, explicit loops, preallocated workspace, and native tokenizer/data parsing.

## Data and tokenizer assets

`download_data.py` downloads Tiny Shakespeare plus the public GPT-2 tokenizer assets
and converts them to simple TSV files. It does **not** tokenize the corpus and does
not require `tiktoken`.

```sh
python3 download_data.py
```

Outputs under `data/`:

```text
input.txt          raw Shakespeare corpus
encoder.tsv        token_id, token bytes, decoded bytes
byte_encoder.tsv   GPT-2 byte-to-unicode table
merges.tsv         GPT-2 BPE merge ranks
```

The Nim runtime loads these assets and performs byte-level GPT-2 BPE encode/decode
natively.

## Run

```sh
nim c -d:release -r main.nim
./main 8 2 1 16       # tiny smoke-test config
./main 64 4 2 256     # default-style small transformer
```

Note: GPT-2 has a 50,257-token vocabulary, so the pure-loop softmax is intentionally
capped to a tiny number of sequences by default for quick reference runs.
