package main

import "core:fmt"
import "core:math"
import la "core:math/linalg"

// can do alias  :: package if needed, like using in cpp

softmax_last_dim :: proc(m: ^$M/matrix[$R, $C]$T) {
	for r in 0 ..< R {
		// 1. Find max in this row for numerical stability
		max_val: T = m[r, 0]
		for c in 1 ..< C {
			max_val = max(max_val, m[r, c])
		}

		// 2. Compute exp(x - max) and the sum of exponents
		row_sum: T = 0
		for c in 0 ..< C {
			m[r, c] = math.exp(m[r, c] - max_val)
			row_sum += m[r, c]
		}

		// 3. Divide by the sum to get probabilities
		for c in 0 ..< C {
			m[r, c] /= row_sum
		}
	}
}

// attention is the

single_attn :: proc(
	x: matrix[$L, $E]f32,
	w_q: matrix[E, $D]f32,
	w_k: matrix[E, D]f32,
	w_v: matrix[E, $DV]f32,
) -> matrix[L, DV]f32 {
	d := f32(D)

	q := x * w_q
	k := x * w_k
	v := x * w_v

	s := (1 / math.sqrt(d)) * (q * la.transpose(k))

	softmax_last_dim(&s)

	return s * v
}


// def batch_single_attn(
//     x: Float[Tensor, "b seq_len emb_dim"],
//     W_q: Float[Tensor, "emb_dim d"],
//     W_k: Float[Tensor, "emb_dim d"],
//     W_v: Float[Tensor, "emb_dim d_v"],
// ) -> Float[Tensor, "b seq_len d_v"]:


// def batch_mha(
//     x: Float[Tensor, "b seq_len emb_dim"],
//     W_q: Float[Tensor, "h emb_dim d"],
//     W_k: Float[Tensor, "h emb_dim d"],
//     W_v: Float[Tensor, "h emb_dim d_v"],
// ) -> Float[Tensor, "b seq_len h_d_v"]:


// I should do the full classic mnist training


main :: proc() {
	x := matrix[2, 3]f32{}
	w_q := matrix[3, 4]f32{}
	w_k := matrix[3, 4]f32{}
	w_v := matrix[3, 4]f32{}

	y := single_attn(x, w_q, w_k, w_v)

	fmt.printf("{}", y)
}
