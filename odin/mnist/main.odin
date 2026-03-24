package main
import "core:fmt"
import "core:math"

// for now, I am going to imagine, I got the data into a 30 x 30 f32 format
image :: struct {
	data: [30][30]f32,
}

// probably should softmax?
softmax :: proc(vec: ^[1][$A]$V) {
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
// dense with Relu

// output is 0-9
predict :: proc(img: image, weights: [$INPUT_D][$OUTPUT_D]f32) -> int {
	// no hidden layer, just a matrix and relu
	input := [1][INPUT_D]f32{}

	// flattening
	for i in 0 ..< len(img.data) {
		for j in 0 ..< len(img.data[0]) {
			input[0][(i * len(img.data[0])) + j] = img.data[i][j]
		}
	}

	output := [1][OUTPUT_D]f32{}
	matmul(input, weights, &output)
	softmax(&output)

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

// back prop means
// start at 1 with loss
// and then go backwards and accumulate the gradients
backward :: proc(img: image, weights: [$INPUT_D][$OUTPUT_D]f32, grad: ^[INPUT_D][OUTPUT_D]f32) {
	// in this case, the first level weights are just the gradient
	// no they really aren't, it's actually the input vector summed in a particular way?
	for i in len(weights) {
		for j in len(weights[0]) {
			grad[i][j] = weights[i][j]
		}
	}

}

train :: proc(data: []image, weights: ^[$INPUT_D][$OUTPUT_D]f32, lr: f32, batch: int) {
	// do gradient descent on batches (eventually)
	grad := [INPUT_D][OUTPUT_D]f32{}

	// can start with just one
	for img in data {
		// clear gradients
		mem.zero(&grad, size_of(grad))

		// forward and backwards pass
		// need to adjust and refine how loss is calculated
		loss := 0
		backward(img, weights, &grad)

		// update step
		for i in 0 ..< INPUT_D {
			for j in 0 ..< OUTPUT_D {
				weights[i][j] += grad[i][j] * lr
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

	// will need to add some basic pieces to pull the data in
}
