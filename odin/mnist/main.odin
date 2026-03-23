package main

// for now, I am going to imagine, I got the data into a 30 x 30 f32 format
image :: struct {
	data: [30][30]f32,
}

matmul :: proc(a: [$A][$B]$V, b: [B][$C]V, o: [A][C]V) {
	for i in 0 ..< A {
		for j in 0 ..< C {
			sum := 0
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
run_model :: proc(img: image, $INPUT_D, $OUTPUT_D: int) -> int {
	// no hidden layer, just a matrix and relu
	weights := [INPUT_D][OUTPUT_D]f32{}

	input := [1][INPUT_D + 1]f32{}
	input[0][INPUT_D] = 1 // bias

	// can figure out how to collapse or remove this copy with a view
	// transmute wasn't doing what I wanted it to
	for i in 0 ..< len(img.data) {
		for j in 0 ..< len(img.data[0]) {
			input[i * len(img.data[0]) + j] = img.data[i][j]
		}
	}

	output := [1][OUTPUT_D]f32{}
	matmul(input, weights, outputs)

	max_v, max_i: int
	max_v = output[max_i]
	for v, i in output {
		if v > max_v {
			max_i = i
			max_v = v
		}
	}

	return max_i
}


main :: proc() {
	// using the standard library, implement the basic script shown in mnist dataset

	// each image looks to be 30 x 30
	// ok apparently this is hard to do in full odin?
	// might need to spend some time with the raw data types, which I don't feel like doing tonight

	// pull in the data, do a simple fully connected dense nn first(900 inputs, 10 outputs), and then train and go from there
}
