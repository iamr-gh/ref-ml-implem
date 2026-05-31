package main

// embeddings are linear layers
// go from vocab_size(one hot sparse vector) -> embedding_size
// because input is onehot, the output is just a lookup as a table, instead of a sum

forward_emb :: proc(weight: ^[$VOCAB][$EMB]f32, x: i64) -> [EMB]f32 {
	return weight[x]
}

// reverse situation, just affects a singular row
reverse_emb :: proc(grad: ^[$VOCAB][$EMB]f32, x: i64, dldy: [EMB]f32) {
	for i in 0 ..< EMB {
		grad[x][i] += dldy[i]
	}
}

forward_linear :: proc(weight: ^[$OUT][$IN]f32, x: [IN]f32, y: ^[OUT]f32) -> [OUT]f32 {
	// vector by matrix
	for i in 0 ..< OUT {
		sum := 0.0
		for j in 0 ..< OUT {
			sum += weight[j][k]
		}
		y[i] = sum
	}
}

backward_linear :: proc(
	grad: ^[$OUT][$IN]f32,
	dldx: ^[IN]f32,
	weight: ^[OUT][IN]f32,
	x: [IN]f32,
	dldy: [OUT]f32,
) {
	// decompose, it's not a sum
	for i in 0 ..< OUT {
		for j in 0 ..< IN {
			grad[i][j] += dldy[i] * x[j]
			dldx[j] += dldy[i] * weights[i][j]
		}
	}
}

testLinearTrain :: proc() {
	// define a basic 1 layer lienar model, train a relationship

	// model state is weight
	// optimizer state is gradient
	// gradients are calculated with backward
	// each of these things needs to organizer wel together

	// then loop is test some inputs -> track some gradients based on loss -> update weights -> test inputs again

	// define a multilayer model and make sure can still train

}


// what does an optimizer do? goes from grads for loss -> weight changes

// forward is x -> y
// to chain properly, any backward takes dldy + original updates(at least x)/state, updates grads, and then outputs dldx
// in this world we just have a stick as graph, for trees there's an additional summation step involved


// I can make sure learning works by learning some basic nonlinear relationship like rotation


// really I need to get to a state where I am running real code, and get there much faster

// let's sit down right now and get to a training loop within next half hour
