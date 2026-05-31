package main

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:time"
import rand "core:math/rand"

INPUT_DIM :: 28 * 28
OUTPUT_DIM :: 10
EPOCHS :: 10
LR :: f32(0.01)

Dataset :: struct {
	xs: [][]f32,
	ys: []int,
}

Layer :: struct {
	in_dim: int,
	out_dim: int,
	w: []f32, // row-major: w[input*out_dim + output]
	b: []f32,
}

MLP :: struct {
	layers: []Layer,
}

Metrics :: struct {
	loss: f32,
	accuracy: f32,
}

History :: struct {
	epochs: [dynamic]int,
	train_loss: [dynamic]f32,
	test: [dynamic]Metrics,
}

seconds_since :: proc(t0: time.Tick) -> f64 {
	return time.duration_seconds(time.tick_since(t0))
}

print_timing :: proc(label: string, seconds: f64) {
	fmt.printf("TIMING %-20s %.6fs\n", label, seconds)
}

read_u32_le :: proc(data: []byte, offset: ^int) -> u32 {
	i := offset^
	v := u32(data[i]) | (u32(data[i+1]) << 8) | (u32(data[i+2]) << 16) | (u32(data[i+3]) << 24)
	offset^ += 4
	return v
}

read_f32_le :: proc(data: []byte, offset: ^int) -> f32 {
	u := read_u32_le(data, offset)
	return transmute(f32)u
}

load_mnist :: proc(path: string) -> Dataset {
	bytes, ok := os.read_entire_file(path)
	if !ok {
		fmt.eprintf("failed to read %s\n", path)
		os.exit(1)
	}
	defer delete(bytes)

	offset := 0
	num_images := int(read_u32_le(bytes, &offset))
	rows := int(read_u32_le(bytes, &offset))
	cols := int(read_u32_le(bytes, &offset))
	if rows != 28 || cols != 28 {
		fmt.eprintf("expected 28x28 images, got %dx%d\n", rows, cols)
		os.exit(1)
	}

	ds: Dataset
	ds.xs = make([][]f32, num_images)
	ds.ys = make([]int, num_images)

	for i in 0 ..< num_images {
		ds.xs[i] = make([]f32, INPUT_DIM)
		for p in 0 ..< INPUT_DIM {
			ds.xs[i][p] = read_f32_le(bytes, &offset)
		}
	}

	for i in 0 ..< num_images {
		ds.ys[i] = int(bytes[offset])
		offset += 1
	}

	return ds
}

free_dataset :: proc(ds: Dataset) {
	for x in ds.xs {
		delete(x)
	}
	delete(ds.xs)
	delete(ds.ys)
}

shuffle_dataset :: proc(ds: ^Dataset) {
	for i := len(ds.xs) - 1; i > 0; i -= 1 {
		j := rand.int_max(i + 1)
		ds.xs[i], ds.xs[j] = ds.xs[j], ds.xs[i]
		ds.ys[i], ds.ys[j] = ds.ys[j], ds.ys[i]
	}
}

init_layer :: proc(in_dim, out_dim: int) -> Layer {
	l: Layer
	l.in_dim = in_dim
	l.out_dim = out_dim
	l.w = make([]f32, in_dim * out_dim)
	l.b = make([]f32, out_dim)

	scale := f32(math.sqrt_f32(2.0 / f32(in_dim)))
	for i in 0 ..< len(l.w) {
		l.w[i] = rand.float32_range(-scale, scale)
	}
	return l
}

init_mlp :: proc(input_dim, output_dim: int, hidden: []int) -> MLP {
	dims := make([]int, len(hidden) + 2)
	defer delete(dims)
	dims[0] = input_dim
	for h, i in hidden {
		if h <= 0 {
			fmt.eprintln("hidden layer sizes must be positive")
			os.exit(1)
		}
		dims[i + 1] = h
	}
	dims[len(dims) - 1] = output_dim

	m: MLP
	m.layers = make([]Layer, len(dims) - 1)
	for i in 0 ..< len(m.layers) {
		m.layers[i] = init_layer(dims[i], dims[i+1])
	}
	return m
}

free_mlp :: proc(m: MLP) {
	for l in m.layers {
		delete(l.w)
		delete(l.b)
	}
	delete(m.layers)
}

relu :: proc(x: f32) -> f32 {
	return x if x > 0 else 0
}

softmax :: proc(x: []f32) {
	max_v := x[0]
	for v in x {
		if v > max_v do max_v = v
	}
	sum: f32
	for i in 0 ..< len(x) {
		x[i] = math.exp(x[i] - max_v)
		sum += x[i]
	}
	for i in 0 ..< len(x) {
		x[i] /= sum
	}
}

cross_entropy :: proc(probs: []f32, y: int) -> f32 {
	eps := f32(1e-7)
	return -math.ln(max(probs[y], eps))
}

argmax :: proc(x: []f32) -> int {
	best := 0
	best_v := x[0]
	for i in 1 ..< len(x) {
		if x[i] > best_v {
			best = i
			best_v = x[i]
		}
	}
	return best
}

forward :: proc(m: MLP, x: []f32) -> [][]f32 {
	acts := make([][]f32, len(m.layers) + 1)
	acts[0] = make([]f32, len(x))
	copy(acts[0], x)

	for li in 0 ..< len(m.layers) {
		layer := m.layers[li]
		prev := acts[li]
		out := make([]f32, layer.out_dim)

		for j in 0 ..< layer.out_dim {
			s := layer.b[j]
			for i in 0 ..< layer.in_dim {
				s += prev[i] * layer.w[i * layer.out_dim + j]
			}
			out[j] = s
		}

		if li < len(m.layers) - 1 {
			for j in 0 ..< len(out) do out[j] = relu(out[j])
		} else {
			softmax(out)
		}
		acts[li + 1] = out
	}
	return acts
}

free_activations :: proc(acts: [][]f32) {
	for a in acts do delete(a)
	delete(acts)
}

predict_probs :: proc(m: MLP, x: []f32) -> []f32 {
	acts := forward(m, x)
	probs := make([]f32, len(acts[len(acts)-1]))
	copy(probs, acts[len(acts)-1])
	free_activations(acts)
	return probs
}

train_one :: proc(m: ^MLP, x: []f32, y: int, lr: f32) -> f32 {
	acts := forward(m^, x)
	defer free_activations(acts)

	last := acts[len(acts)-1]
	loss := cross_entropy(last, y)

	delta := make([]f32, len(last))
	defer delete(delta)
	copy(delta, last)
	delta[y] -= 1

	for li := len(m.layers) - 1; li >= 0; li -= 1 {
		layer := &m.layers[li]
		prev := acts[li]

		prev_delta: []f32
		if li > 0 {
			prev_delta = make([]f32, layer.in_dim)
			for i in 0 ..< layer.in_dim {
				s: f32
				for j in 0 ..< layer.out_dim {
					s += layer.w[i * layer.out_dim + j] * delta[j]
				}
				prev_delta[i] = s if prev[i] > 0 else 0
			}
		}

		for i in 0 ..< layer.in_dim {
			for j in 0 ..< layer.out_dim {
				layer.w[i * layer.out_dim + j] -= lr * prev[i] * delta[j]
			}
		}
		for j in 0 ..< layer.out_dim {
			layer.b[j] -= lr * delta[j]
		}

		if li > 0 {
			delete(delta)
			delta = prev_delta
		}
	}
	return loss
}

train_epoch :: proc(m: ^MLP, ds: Dataset, lr: f32) -> f32 {
	total: f32
	for i in 0 ..< len(ds.xs) {
		total += train_one(m, ds.xs[i], ds.ys[i], lr)
	}
	return total / f32(len(ds.xs))
}

evaluate :: proc(m: MLP, ds: Dataset) -> Metrics {
	correct := 0
	total_loss: f32
	for i in 0 ..< len(ds.xs) {
		probs := predict_probs(m, ds.xs[i])
		total_loss += cross_entropy(probs, ds.ys[i])
		if argmax(probs) == ds.ys[i] do correct += 1
		delete(probs)
	}
	return Metrics{loss = total_loss / f32(len(ds.xs)), accuracy = f32(correct) / f32(len(ds.xs))}
}

history_add :: proc(h: ^History, epoch: int, train_loss: f32, test: Metrics) {
	append(&h.epochs, epoch)
	append(&h.train_loss, train_loss)
	append(&h.test, test)
}

print_table :: proc(h: History) {
	fmt.println("\nEpoch | Train Loss | Test Loss | Test Acc")
	fmt.println("------|------------|-----------|---------")
	for i in 0 ..< len(h.epochs) {
		fmt.printf("%5d | %10.4f | %9.4f | %7.2f%%\n", h.epochs[i], h.train_loss[i], h.test[i].loss, 100 * h.test[i].accuracy)
	}
}

free_history :: proc(h: History) {
	delete(h.epochs)
	delete(h.train_loss)
	delete(h.test)
}

parse_hidden :: proc() -> []int {
	args := os.args
	if len(args) <= 1 {
		hidden := make([]int, 1)
		hidden[0] = 128
		return hidden
	}
	if len(args) == 2 && (args[1] == "none" || args[1] == "0" || args[1] == "linear") {
		return make([]int, 0)
	}

	hidden := make([]int, len(args)-1)
	for arg, i in args[1:] {
		h, ok := strconv.parse_int(arg)
		if !ok || h <= 0 {
			fmt.eprintf("invalid hidden size: %s\n", arg)
			os.exit(1)
		}
		hidden[i] = h
	}
	return hidden
}

main :: proc() {
	total_t0 := time.tick_now()
	hidden := parse_hidden()
	defer delete(hidden)

	fmt.printf("Model: input=%d, hidden=%v, output=%d\n", INPUT_DIM, hidden, OUTPUT_DIM)
	fmt.println("Loading data...")

	t0 := time.tick_now()
	train := load_mnist("../../nim/mnist/data/train.bin")
	print_timing("load_train", seconds_since(t0))
	defer free_dataset(train)

	t0 = time.tick_now()
	test := load_mnist("../../nim/mnist/data/test.bin")
	print_timing("load_test", seconds_since(t0))
	defer free_dataset(test)

	fmt.printf("  train: %d samples\n", len(train.xs))
	fmt.printf("  test:  %d samples\n", len(test.xs))

	t0 = time.tick_now()
	net := init_mlp(INPUT_DIM, OUTPUT_DIM, hidden)
	print_timing("init_model", seconds_since(t0))
	defer free_mlp(net)

	history: History
	history.epochs = make([dynamic]int)
	history.train_loss = make([dynamic]f32)
	history.test = make([dynamic]Metrics)
	defer free_history(history)

	t0 = time.tick_now()
	baseline := evaluate(net, test)
	print_timing("baseline_eval", seconds_since(t0))
	history_add(&history, 0, baseline.loss, baseline)
	fmt.printf("\nEpoch 0: test acc=%5.2f%% loss=%.4f\n", 100 * baseline.accuracy, baseline.loss)

	for epoch in 1 ..= EPOCHS {
		label: string

		label = fmt.tprintf("epoch_%d_shuffle", epoch)
		t0 = time.tick_now()
		shuffle_dataset(&train)
		print_timing(label, seconds_since(t0))

		label = fmt.tprintf("epoch_%d_train", epoch)
		t0 = time.tick_now()
		train_loss := train_epoch(&net, train, LR)
		print_timing(label, seconds_since(t0))

		label = fmt.tprintf("epoch_%d_eval", epoch)
		t0 = time.tick_now()
		metrics := evaluate(net, test)
		print_timing(label, seconds_since(t0))
		history_add(&history, epoch, train_loss, metrics)
		fmt.printf("Epoch %2d: train loss=%.4f  test acc=%5.2f%% loss=%.4f\n", epoch, train_loss, 100 * metrics.accuracy, metrics.loss)
	}

	t0 = time.tick_now()
	print_table(history)
	print_timing("report", seconds_since(t0))
	print_timing("total", seconds_since(total_t0))
}