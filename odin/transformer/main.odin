package main

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

DEFAULT_D_MODEL :: 64
DEFAULT_HEADS :: 4
DEFAULT_BLOCKS :: 2
DEFAULT_FF_DIM :: 256
EPOCHS :: 2
LR :: f32(0.001)
MAX_TRAIN_SEQ :: 2
MAX_EVAL_SEQ :: 1

Tokenizer :: struct {
	token_to_id: map[string]int,
	id_to_bytes: []string,
	byte_to_token: [256]string,
	pair_rank: map[string]int,
	vocab_size: int,
}

hex_digit :: proc(c: byte) -> int {
	if c >= '0' && c <= '9' do return int(c - '0')
	if c >= 'a' && c <= 'f' do return 10 + int(c - 'a')
	if c >= 'A' && c <= 'F' do return 10 + int(c - 'A')
	return 0
}

hex_to_string :: proc(hex: string) -> string {
	buf := make([]byte, len(hex)/2)
	for i := 0; i < len(buf); i += 1 {
		buf[i] = byte((hex_digit(hex[2*i]) << 4) | hex_digit(hex[2*i+1]))
	}
	return string(buf)
}

pair_key :: proc(a, b: string) -> string {
	buf := make([]byte, len(a) + 1 + len(b))
	copy(buf[0:len(a)], a)
	buf[len(a)] = '\t'
	copy(buf[len(a)+1:], b)
	return string(buf)
}

load_tokenizer :: proc(dir: string = "data") -> Tokenizer {
	t: Tokenizer
	t.token_to_id = make(map[string]int, 60000)
	t.pair_rank = make(map[string]int, 60000)
	t.id_to_bytes = make([]string, 50257)

	// byte_encoder.tsv: byte<TAB>hex(token_string_utf8)
	path := fmt.tprintf("%s/byte_encoder.tsv", dir)
	bytes, ok := os.read_entire_file(path)
	if !ok { fmt.eprintf("failed to read %s\n", path); os.exit(1) }
	text := string(bytes)
	for line in strings.split_lines_iterator(&text) {
		if len(line) == 0 do continue
		parts := strings.split(line, "\t")
		if len(parts) >= 2 {
			b, _ := strconv.parse_int(parts[0])
			t.byte_to_token[b] = hex_to_string(parts[1])
		}
		delete(parts)
	}
	delete(bytes)

	// encoder.tsv: id<TAB>hex(token_string_utf8)<TAB>hex(decoded_bytes)
	path = fmt.tprintf("%s/encoder.tsv", dir)
	bytes, ok = os.read_entire_file(path)
	if !ok { fmt.eprintf("failed to read %s\n", path); os.exit(1) }
	text = string(bytes)
	max_id := 0
	for line in strings.split_lines_iterator(&text) {
		if len(line) == 0 do continue
		parts := strings.split(line, "\t")
		if len(parts) >= 3 {
			id, _ := strconv.parse_int(parts[0])
			tok := hex_to_string(parts[1])
			decoded := hex_to_string(parts[2])
			t.id_to_bytes[id] = decoded
			t.token_to_id[tok] = id
			if id > max_id do max_id = id
		}
		delete(parts)
	}
	t.vocab_size = max_id + 1
	delete(bytes)

	// merges.tsv: rank<TAB>hex(left)<TAB>hex(right)
	path = fmt.tprintf("%s/merges.tsv", dir)
	bytes, ok = os.read_entire_file(path)
	if !ok { fmt.eprintf("failed to read %s\n", path); os.exit(1) }
	text = string(bytes)
	for line in strings.split_lines_iterator(&text) {
		if len(line) == 0 do continue
		parts := strings.split(line, "\t")
		if len(parts) >= 3 {
			rank, _ := strconv.parse_int(parts[0])
			a := hex_to_string(parts[1])
			b := hex_to_string(parts[2])
			t.pair_rank[pair_key(a, b)] = rank
		}
		delete(parts)
	}
	delete(bytes)
	return t
}

is_space_byte :: proc(c: byte) -> bool { return c == ' ' || c == '\n' || c == '\r' || c == '\t' }
is_alnum_byte :: proc(c: byte) -> bool { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') }

bpe_piece :: proc(t: Tokenizer, piece: string) -> [dynamic]string {
	syms := make([dynamic]string)
	for i in 0 ..< len(piece) {
		append(&syms, t.byte_to_token[int(piece[i])])
	}

	for len(syms) > 1 {
		best_rank := max(int)
		best_idx := -1
		for i in 0 ..< len(syms)-1 {
			key := pair_key(syms[i], syms[i+1])
			if r, ok := t.pair_rank[key]; ok {
				if r < best_rank { best_rank = r; best_idx = i }
			}
		}
		if best_idx < 0 do break

		merged := strings.concatenate({syms[best_idx], syms[best_idx+1]})
		next := make([dynamic]string)
		i := 0
		for i < len(syms) {
			if i == best_idx {
				append(&next, merged)
				i += 2
			} else {
				append(&next, syms[i])
				i += 1
			}
		}
		delete(syms)
		syms = next
	}
	return syms
}

encode_append_piece :: proc(t: Tokenizer, piece: string, out: ^[dynamic]i32) {
	syms := bpe_piece(t, piece)
	defer delete(syms)
	for s in syms {
		if id, ok := t.token_to_id[s]; ok {
			append(out, i32(id))
		} else {
			for i in 0 ..< len(s) {
				one := t.byte_to_token[int(s[i])]
				append(out, i32(t.token_to_id[one]))
			}
		}
	}
}

encode :: proc(t: Tokenizer, text: string) -> [dynamic]i32 {
	out := make([dynamic]i32)
	i := 0
	for i < len(text) {
		start := i
		if text[i] == ' ' && i+1 < len(text) && is_alnum_byte(text[i+1]) {
			i += 1
			for i < len(text) && is_alnum_byte(text[i]) do i += 1
			encode_append_piece(t, text[start:i], &out)
		} else if is_alnum_byte(text[i]) {
			for i < len(text) && is_alnum_byte(text[i]) do i += 1
			encode_append_piece(t, text[start:i], &out)
		} else if is_space_byte(text[i]) {
			for i < len(text) && is_space_byte(text[i]) && !(text[i] == ' ' && i+1 < len(text) && is_alnum_byte(text[i+1])) do i += 1
			encode_append_piece(t, text[start:i], &out)
		} else {
			encode_append_piece(t, text[i:i+1], &out)
			i += 1
		}
	}
	return out
}

decode_id :: proc(t: Tokenizer, id: int) -> string {
	if id >= 0 && id < len(t.id_to_bytes) do return t.id_to_bytes[id]
	return "<?>"
}

Corpus :: struct {
	tokens: []i32,   // flat: sequence i starts at i*seq_len
	seq_len: int,
	vocab_size: int,
	order: []int,
	n: int,
}

Linear :: struct {
	in_dim: int,
	out_dim: int,
	w: []f32, // row-major: w[i*out_dim + j]
	b: []f32,
}

Block :: struct {
	q, k, v, o: Linear,
	ff1, ff2: Linear,
}

Transformer :: struct {
	vocab_size, seq_len, d_model, n_heads, d_head, n_blocks, ff_dim: int,
	token_emb: []f32, // vocab_size * d_model
	pos_emb: []f32,   // seq_len * d_model
	blocks: []Block,
	head: Linear,      // d_model -> vocab_size
}

Workspace :: struct {
	x: []f32,       // seq_len * d_model
	x2: []f32,      // seq_len * d_model
	q, k, v: []f32,
	att: []f32,     // n_heads * seq_len * seq_len
	ff: []f32,      // seq_len * ff_dim
	logits: []f32,  // seq_len * vocab_size
	probs: []f32,   // vocab_size scratch
	grad_h: []f32,  // d_model scratch
}

Metrics :: struct {
	loss: f32,
	accuracy: f32,
}

History :: struct {
	epochs: [dynamic]int,
	train_loss: [dynamic]f32,
	eval_loss: [dynamic]f32,
	eval_acc: [dynamic]f32,
}

seconds_since :: proc(t0: time.Tick) -> f64 {
	return time.duration_seconds(time.tick_since(t0))
}

print_timing :: proc(label: string, seconds: f64) {
	fmt.printf("TIMING %-20s %.6fs\n", label, seconds)
}

load_corpus :: proc(path: string, tok: Tokenizer, seq_len: int) -> Corpus {
	bytes, ok := os.read_entire_file(path)
	if !ok {
		fmt.eprintf("failed to read %s\n", path)
		os.exit(1)
	}
	text := string(bytes)
	encoded := encode(tok, text)

	c: Corpus
	c.seq_len = seq_len
	c.vocab_size = tok.vocab_size
	c.n = max(0, len(encoded) - seq_len)
	c.tokens = make([]i32, len(encoded))
	copy(c.tokens, encoded[:])
	c.order = make([]int, c.n)
	for i in 0 ..< c.n {
		c.order[i] = i
	}
	delete(encoded)
	delete(bytes)

	fmt.printf("  Loaded %s: %d tokens, %d sequences, seqLen=%d, vocabSize=%d\n", path, len(c.tokens), c.n, seq_len, c.vocab_size)
	return c
}

free_corpus :: proc(c: Corpus) {
	delete(c.tokens)
	delete(c.order)
}

shuffle_corpus :: proc(c: ^Corpus) {
	seed := u32(246813579)
	i := c.n - 1
	for i > 0 {
		seed = seed * 1664525 + 1013904223
		j := int(seed % u32(i + 1))
		c.order[i], c.order[j] = c.order[j], c.order[i]
		i -= 1
	}
}

sequence :: proc(c: Corpus, idx: int) -> []i32 {
	actual := c.order[idx]
	return c.tokens[actual:actual+c.seq_len]
}

relu :: proc(x: f32) -> f32 {
	return x if x > 0 else 0
}

init_linear :: proc(in_dim, out_dim: int, relu_scale: bool = true) -> Linear {
	// Deprecated wrapper kept for readability; deterministic init uses
	// init_linear_seeded below.
	seed := u32(123456789)
	return init_linear_seeded(in_dim, out_dim, &seed, relu_scale)
}

next_rand :: proc(seed: ^u32) -> f32 {
	seed^ = seed^ * 1664525 + 1013904223
	return f32(seed^ >> 8) / f32(1 << 24)
}

rand_range :: proc(seed: ^u32, lo, hi: f32) -> f32 {
	return lo + (hi - lo) * next_rand(seed)
}

init_linear_seeded :: proc(in_dim, out_dim: int, seed: ^u32, relu_scale: bool = true) -> Linear {
	l: Linear
	l.in_dim = in_dim
	l.out_dim = out_dim
	l.w = make([]f32, in_dim * out_dim)
	l.b = make([]f32, out_dim)
	scale: f32
	if relu_scale {
		scale = f32(math.sqrt_f32(2.0 / f32(in_dim)))
	} else {
		scale = f32(math.sqrt_f32(1.0 / f32(in_dim)))
	}
	for i in 0 ..< len(l.w) {
		l.w[i] = rand_range(seed, -scale, scale)
	}
	return l
}

free_linear :: proc(l: Linear) {
	delete(l.w)
	delete(l.b)
}

init_transformer :: proc(vocab_size, seq_len, d_model, n_heads, n_blocks, ff_dim: int) -> Transformer {
	if d_model % n_heads != 0 {
		fmt.eprintln("d_model must be divisible by n_heads")
		os.exit(1)
	}

	m: Transformer
	m.vocab_size = vocab_size
	m.seq_len = seq_len
	m.d_model = d_model
	m.n_heads = n_heads
	m.d_head = d_model / n_heads
	m.n_blocks = n_blocks
	m.ff_dim = ff_dim
	seed := u32(123456789)

	m.token_emb = make([]f32, vocab_size * d_model)
	m.pos_emb = make([]f32, seq_len * d_model)
	emb_scale := f32(0.02)
	for i in 0 ..< len(m.token_emb) do m.token_emb[i] = rand_range(&seed, -emb_scale, emb_scale)
	for i in 0 ..< len(m.pos_emb) do m.pos_emb[i] = rand_range(&seed, -emb_scale, emb_scale)

	m.blocks = make([]Block, n_blocks)
	for i in 0 ..< n_blocks {
		m.blocks[i].q = init_linear_seeded(d_model, d_model, &seed, false)
		m.blocks[i].k = init_linear_seeded(d_model, d_model, &seed, false)
		m.blocks[i].v = init_linear_seeded(d_model, d_model, &seed, false)
		m.blocks[i].o = init_linear_seeded(d_model, d_model, &seed, false)
		m.blocks[i].ff1 = init_linear_seeded(d_model, ff_dim, &seed, true)
		m.blocks[i].ff2 = init_linear_seeded(ff_dim, d_model, &seed, false)
	}
	m.head = init_linear_seeded(d_model, vocab_size, &seed, false)
	return m
}

free_transformer :: proc(m: Transformer) {
	delete(m.token_emb)
	delete(m.pos_emb)
	for b in m.blocks {
		free_linear(b.q)
		free_linear(b.k)
		free_linear(b.v)
		free_linear(b.o)
		free_linear(b.ff1)
		free_linear(b.ff2)
	}
	delete(m.blocks)
	free_linear(m.head)
}

init_workspace :: proc(m: Transformer) -> Workspace {
	ws: Workspace
	ws.x = make([]f32, m.seq_len * m.d_model)
	ws.x2 = make([]f32, m.seq_len * m.d_model)
	ws.q = make([]f32, m.seq_len * m.d_model)
	ws.k = make([]f32, m.seq_len * m.d_model)
	ws.v = make([]f32, m.seq_len * m.d_model)
	ws.att = make([]f32, m.n_heads * m.seq_len * m.seq_len)
	ws.ff = make([]f32, m.seq_len * m.ff_dim)
	ws.logits = make([]f32, m.seq_len * m.vocab_size)
	ws.probs = make([]f32, m.vocab_size)
	ws.grad_h = make([]f32, m.d_model)
	return ws
}

free_workspace :: proc(ws: Workspace) {
	delete(ws.x)
	delete(ws.x2)
	delete(ws.q)
	delete(ws.k)
	delete(ws.v)
	delete(ws.att)
	delete(ws.ff)
	delete(ws.logits)
	delete(ws.probs)
	delete(ws.grad_h)
}

linear_forward :: proc(inp: []f32, out: []f32, l: Linear, rows: int) {
	for r in 0 ..< rows {
		in_base := r * l.in_dim
		out_base := r * l.out_dim
		for j in 0 ..< l.out_dim {
			s := l.b[j]
			for i in 0 ..< l.in_dim {
				s += inp[in_base + i] * l.w[i * l.out_dim + j]
			}
			out[out_base + j] = s
		}
	}
}

softmax_into :: proc(logits: []f32, probs: []f32, n: int) {
	max_v := logits[0]
	for i in 1 ..< n {
		if logits[i] > max_v do max_v = logits[i]
	}
	sum: f32
	for i in 0 ..< n {
		e := math.exp(logits[i] - max_v)
		probs[i] = e
		sum += e
	}
	for i in 0 ..< n {
		probs[i] /= sum
	}
}

forward :: proc(m: Transformer, tokens: []i32, ws: ^Workspace) {
	T := min(len(tokens), m.seq_len)
	C := m.d_model

	for t in 0 ..< T {
		tok := int(tokens[t])
		for c in 0 ..< C {
			ws.x[t*C + c] = m.token_emb[tok*C + c] + m.pos_emb[t*C + c]
		}
	}

	for bi in 0 ..< m.n_blocks {
		blk := m.blocks[bi]
		linear_forward(ws.x, ws.q, blk.q, T)
		linear_forward(ws.x, ws.k, blk.k, T)
		linear_forward(ws.x, ws.v, blk.v, T)

		for h in 0 ..< m.n_heads {
			head_off := h * m.d_head
			for tq in 0 ..< T {
				max_score := f32(-1.0e30)
				for tk in 0 ..< T {
					att_idx := (h * m.seq_len + tq) * m.seq_len + tk
					if tk > tq {
						ws.att[att_idx] = 0
					} else {
						score: f32
						for d in 0 ..< m.d_head {
							score += ws.q[tq*C + head_off + d] * ws.k[tk*C + head_off + d]
						}
						score /= f32(math.sqrt_f32(f32(m.d_head)))
						ws.att[att_idx] = score
						if score > max_score do max_score = score
					}
				}
				sum: f32
				for tk in 0 ..= tq {
					att_idx := (h * m.seq_len + tq) * m.seq_len + tk
					e := math.exp(ws.att[att_idx] - max_score)
					ws.att[att_idx] = e
					sum += e
				}
				for tk in 0 ..= tq {
					att_idx := (h * m.seq_len + tq) * m.seq_len + tk
					ws.att[att_idx] /= sum
				}
			}

			for tq in 0 ..< T {
				for d in 0 ..< m.d_head {
					s: f32
					for tk in 0 ..= tq {
						att_idx := (h * m.seq_len + tq) * m.seq_len + tk
						s += ws.att[att_idx] * ws.v[tk*C + head_off + d]
					}
					ws.x2[tq*C + head_off + d] = s
				}
			}
		}

		linear_forward(ws.x2, ws.q, blk.o, T)
		for i in 0 ..< T*C {
			ws.x[i] += ws.q[i]
		}

		linear_forward(ws.x, ws.ff, blk.ff1, T)
		for i in 0 ..< T * m.ff_dim do ws.ff[i] = relu(ws.ff[i])
		linear_forward(ws.ff, ws.x2, blk.ff2, T)
		for i in 0 ..< T*C {
			ws.x[i] += ws.x2[i]
		}
	}

	linear_forward(ws.x, ws.logits, m.head, T)
}

train_one :: proc(m: ^Transformer, tokens: []i32, lr: f32, ws: ^Workspace) -> Metrics {
	forward(m^, tokens, ws)
	T := min(len(tokens), m.seq_len) - 1
	C := m.d_model
	V := m.vocab_size
	correct := 0
	total_loss: f32

	for t in 0 ..< T {
		y := int(tokens[t + 1])
		log_base := t * V
		softmax_into(ws.logits[log_base:log_base+V], ws.probs, V)
		total_loss += -math.ln(max(ws.probs[y], f32(1e-7)))

		pred := 0
		best := ws.probs[0]
		for j in 1 ..< V {
			if ws.probs[j] > best {
				best = ws.probs[j]
				pred = j
			}
		}
		if pred == y do correct += 1

		for c in 0 ..< C do ws.grad_h[c] = 0

		for j in 0 ..< V {
			g := ws.probs[j]
			if j == y do g -= 1
			for c in 0 ..< C {
				idx := c * V + j
				ws.grad_h[c] += m.head.w[idx] * g
				m.head.w[idx] -= lr * ws.x[t*C + c] * g
			}
			m.head.b[j] -= lr * g
		}

		tok := int(tokens[t])
		for c in 0 ..< C {
			m.token_emb[tok*C + c] -= lr * ws.grad_h[c]
			m.pos_emb[t*C + c] -= lr * ws.grad_h[c]
		}
	}

	return Metrics{loss = total_loss / f32(T), accuracy = f32(correct) / f32(T)}
}

evaluate_one :: proc(m: Transformer, tokens: []i32, ws: ^Workspace) -> Metrics {
	forward(m, tokens, ws)
	T := min(len(tokens), m.seq_len) - 1
	V := m.vocab_size
	correct := 0
	total_loss: f32
	for t in 0 ..< T {
		y := int(tokens[t + 1])
		log_base := t * V
		softmax_into(ws.logits[log_base:log_base+V], ws.probs, V)
		total_loss += -math.ln(max(ws.probs[y], f32(1e-7)))
		pred := 0
		best := ws.probs[0]
		for j in 1 ..< V {
			if ws.probs[j] > best {
				best = ws.probs[j]
				pred = j
			}
		}
		if pred == y do correct += 1
	}
	return Metrics{loss = total_loss / f32(T), accuracy = f32(correct) / f32(T)}
}

best_next_token :: proc(m: Transformer, tokens: []i32, ws: ^Workspace) -> int {
	forward(m, tokens, ws)
	t := min(len(tokens), m.seq_len) - 1
	base := t * m.vocab_size
	best_id := 0
	best := ws.logits[base]
	for j in 1 ..< m.vocab_size {
		if ws.logits[base + j] > best {
			best = ws.logits[base + j]
			best_id = j
		}
	}
	return best_id
}

eval_some :: proc(m: Transformer, c: Corpus, ws: ^Workspace) -> Metrics {
	n := min(MAX_EVAL_SEQ, c.n)
	total_loss: f32
	total_acc: f32
	for i in 0 ..< n {
		mt := evaluate_one(m, sequence(c, i), ws)
		total_loss += mt.loss
		total_acc += mt.accuracy
	}
	return Metrics{loss = total_loss / f32(n), accuracy = total_acc / f32(n)}
}

history_add :: proc(h: ^History, epoch: int, train_loss: f32, eval: Metrics) {
	append(&h.epochs, epoch)
	append(&h.train_loss, train_loss)
	append(&h.eval_loss, eval.loss)
	append(&h.eval_acc, eval.accuracy)
}

print_table :: proc(h: History) {
	fmt.println("\nEpoch | Train Loss | Eval Loss | Eval Acc")
	fmt.println("------|------------|-----------|---------")
	for i in 0 ..< len(h.epochs) {
		fmt.printf("%5d | %10.4f | %9.4f | %7.2f%%\n", h.epochs[i], h.train_loss[i], h.eval_loss[i], 100 * h.eval_acc[i])
	}
}

free_history :: proc(h: History) {
	delete(h.epochs)
	delete(h.train_loss)
	delete(h.eval_loss)
	delete(h.eval_acc)
}

parse_config :: proc() -> (int, int, int, int) {
	d_model := DEFAULT_D_MODEL
	heads := DEFAULT_HEADS
	blocks := DEFAULT_BLOCKS
	ff_dim := DEFAULT_FF_DIM
	args := os.args
	if len(args) > 1 {
		v, ok := strconv.parse_int(args[1]); if ok do d_model = v
	}
	if len(args) > 2 {
		v, ok := strconv.parse_int(args[2]); if ok do heads = v
	}
	if len(args) > 3 {
		v, ok := strconv.parse_int(args[3]); if ok do blocks = v
	}
	if len(args) > 4 {
		v, ok := strconv.parse_int(args[4]); if ok do ff_dim = v
	}
	return d_model, heads, blocks, ff_dim
}

main :: proc() {
	total_t0 := time.tick_now()
	d_model, heads, blocks, ff_dim := parse_config()
	fmt.printf("Transformer LM: dModel=%d, heads=%d, blocks=%d, ffDim=%d\n", d_model, heads, blocks, ff_dim)
	fmt.println("Loading GPT-2 tokenizer assets...")

	t0 := time.tick_now()
	tok := load_tokenizer("data")
	print_timing("load_tokenizer", seconds_since(t0))

	fmt.println("Tokenizing Shakespeare natively...")
	t0 = time.tick_now()
	corpus := load_corpus("data/input.txt", tok, 64)
	print_timing("load_corpus", seconds_since(t0))
	defer free_corpus(corpus)

	t0 = time.tick_now()
	net := init_transformer(corpus.vocab_size, corpus.seq_len, d_model, heads, blocks, ff_dim)
	print_timing("init_model", seconds_since(t0))
	defer free_transformer(net)

	t0 = time.tick_now()
	ws := init_workspace(net)
	print_timing("init_workspace", seconds_since(t0))
	defer free_workspace(ws)

	history: History
	history.epochs = make([dynamic]int)
	history.train_loss = make([dynamic]f32)
	history.eval_loss = make([dynamic]f32)
	history.eval_acc = make([dynamic]f32)
	defer free_history(history)

	t0 = time.tick_now()
	baseline := eval_some(net, corpus, &ws)
	print_timing("baseline_eval", seconds_since(t0))
	history_add(&history, 0, baseline.loss, baseline)
	fmt.printf("Epoch 0: eval acc=%5.2f%% loss=%.4f\n", 100 * baseline.accuracy, baseline.loss)

	for epoch in 1 ..= EPOCHS {
		label: string
		label = fmt.tprintf("epoch_%d_shuffle", epoch)
		t0 = time.tick_now()
		shuffle_corpus(&corpus)
		print_timing(label, seconds_since(t0))

		n_train := min(MAX_TRAIN_SEQ, corpus.n)
		total_loss: f32
		label = fmt.tprintf("epoch_%d_train", epoch)
		t0 = time.tick_now()
		for i in 0 ..< n_train {
			mt := train_one(&net, sequence(corpus, i), LR, &ws)
			total_loss += mt.loss
		}
		train_loss := total_loss / f32(n_train)
		print_timing(label, seconds_since(t0))

		label = fmt.tprintf("epoch_%d_eval", epoch)
		t0 = time.tick_now()
		metrics := eval_some(net, corpus, &ws)
		print_timing(label, seconds_since(t0))
		history_add(&history, epoch, train_loss, metrics)
		fmt.printf("Epoch %2d: train loss=%.4f  eval acc=%5.2f%% loss=%.4f\n", epoch, train_loss, 100 * metrics.accuracy, metrics.loss)
	}

	t0 = time.tick_now()
	print_table(history)
	print_timing("report", seconds_since(t0))

	prompt := "To be"
	prompt_ids := encode(tok, prompt)
	next_id := best_next_token(net, prompt_ids[:], &ws)
	fmt.printf("Prompt: %q -> next token id=%d decoded=%q\n", prompt, next_id, decode_id(tok, next_id))
	delete(prompt_ids)

	print_timing("total", seconds_since(total_t0))
}
