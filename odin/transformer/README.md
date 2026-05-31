# Odin Transformer LM

A minimal GPT-style transformer language model in the same style as `odin/mnist`:
single-file implementation, explicit memory management, flat buffers, and native
binary parsing.

## Data

`download_data.py` downloads Tiny Shakespeare and tokenizes it with the GPT-2 tokenizer
via `tiktoken`, writing `data/shakespeare.bin`.

```sh
python3 -m venv .venv
.venv/bin/pip install tiktoken
.venv/bin/python download_data.py
```

The Odin code parses the binary format natively:

```text
uint32 num_sequences
uint32 seq_len
uint32 vocab_size
int32[num_sequences * seq_len] token_ids
```

## Run

```sh
odin run . -o:speed -- 8 2 1 16       # tiny smoke-test config
odin run . -o:speed -- 64 4 2 256     # default-style small transformer
```

Note: GPT-2 has a 50,257-token vocabulary, so the pure-loop softmax is intentionally
capped to a tiny number of sequences by default for quick reference runs.
