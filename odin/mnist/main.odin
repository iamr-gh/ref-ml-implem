package main
import "core:fmt"

// for now, I am going to imagine, I got the data into a 30 x 30 f32 format
image :: struct {
	data: [30][30]f32,
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
run_model :: proc(img: image, weights: [$INPUT_D][$OUTPUT_D]f32) -> int {
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

	// fmt.printf("i:{}, w:{}, o:{}", input, weights, output)

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
	pred := run_model(img, weights)
	fmt.printf("{}", pred)

	// will need to add some basic pieces to pull the data in
}
