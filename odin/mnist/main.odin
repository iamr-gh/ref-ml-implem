package main
import "core:fmt"
import "core:math"
import "core:mem"

// for now, I am going to imagine, I got the data into a 30 x 30 f32 format
image :: struct {
	data: [30][30]f32,
}

transpose :: proc(x: ^[$A][$B]$V, xT: ^[B][A]V) {
	for i in 0 ..< A {
		for j in 0 ..< B {
			xT[i][j] = x[j][i]
		}
	}
}

// probably should softmax?
softmax :: proc(vec: ^[1][$A]$V) {
	// there's a math trick often done with subtracting max before exponentiating

	sum := V(0)
	// traditionally slow part
	for num in vec[0] {
		sum += math.exp(num)
	}

	for i in 0 ..< A {
		vec[0][i] = math.exp(vec[0][i]) / sum
	}
}


matmul :: proc(a: [$A][$B]$V, b: [B][$C]V, o: ^[A][C]V) {
	for i in 0 ..< A {
		for j in 0 ..< C {
			sum := V(0)
			for k in 0 ..< B {
				sum += a[i][k] * b[k][j]
			}
			o[i][j] = sum
		}
	}
}

// we're going to implement a basic neural network initially, no CNN no nothing
// straight to softmax without relu rn -- TODO: add relu

flatten :: proc(img: image, flat: ^[1][30 * 30]f32) {
	for i in 0 ..< len(img.data) {
		for j in 0 ..< len(img.data[0]) {
			flat[0][(i * len(img.data[0])) + j] = img.data[i][j]
		}
	}
}

forward :: proc(
	weights: [$INPUT_D][$OUTPUT_D]f32,
	input: [1][INPUT_D]f32,
	output: ^[1][OUTPUT_D]f32,
) {
	matmul(input, weights, output)
	softmax(output)
}

// output is 0-9
predict :: proc(img: image, weights: [$INPUT_D][$OUTPUT_D]f32) -> int {
	// flattening
	input := [1][INPUT_D]f32{}
	flatten(img, &input)

	output := [1][OUTPUT_D]f32{}

	forward(weights, input, &output)

	max_v: f32
	max_i: int
	max_v = output[0][max_i]


	for v, i in output[0] {
		if v > max_v {
			max_i = i
			max_v = v
		}
	}

	return max_i
}


// this is not the full backprop, need to think about a define that further

backward :: proc(
	grad: ^[$INPUT_D][$OUTPUT_D]f32,
	input: [1][INPUT_D]f32,
	dldy: [1][OUTPUT_D]f32, // previous gradient step
) {
	// full gradient is input^T @ (y - y)
	// note, do not need to explicitly transpose, can change indexing over top
	// this is a core understanding about layout -> zero copy formatting
	for i in 0 ..< INPUT_D {
		for j in 0 ..< OUTPUT_D {
			grad[i][j] = input[0][i] * dldy[0][j]
		}
	}

}

train :: proc(xs: []image, ys: []int, weights: ^[$INPUT_D][$OUTPUT_D]f32, lr: f32, batch: int) {
	// do gradient descent on batches (eventually)
	grad := [INPUT_D][OUTPUT_D]f32{}


	// can predefine and reuse buffers
	input := [1][INPUT_D]f32{}
	y_hat := [1][OUTPUT_D]f32{}

	// can start with just one
	for i in 0 ..< len(xs) {
		img := xs[i]
		y := ys[i]

		y_one_hot := [1][OUTPUT_D]f32{}
		y_one_hot[0][y] = 1

		// clear gradients
		mem.zero(&grad, size_of(grad))
		flatten(img, &input)

		forward(weights^, input, &y_hat)
		backward(&grad, input, y_hat - y_one_hot)

		// update step
		for i in 0 ..< INPUT_D {
			for j in 0 ..< OUTPUT_D {
				weights[i][j] -= grad[i][j] * lr
			}
		}
	}

}

main :: proc() {
	img := image{}


	for i in 0 ..< len(img.data) {
		for j in 0 ..< len(img.data[0]) {
			img.data[i][j] = 1
		}
	}


	weights := [30 * 30][9]f32{}
	weights[10][8] = 1

	// assuming prepad with 1 for bias
	pred := predict(img, weights)
	fmt.printf("{}", pred)

	xs := [1]image{img}
	ys := [1]int{0}

	train(xs[:], ys[:], &weights, 0.1, 0)

	// will need to add some basic pieces to pull the data in
}
