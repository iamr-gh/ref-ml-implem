package tensor

// we can have runtime managed shape easily
// compile time shape is nicer?
// 3 or 4d is not bad

// product of an array
prod :: proc(arr: [$S]$V) -> V {
	num := V(1)
	for val in arr {
		num *= val
	}
	return num
}

// I don't think full generality is really needed in almost all uses
// and under this formulation
Tensor4 :: struct($A: int, $B: int, $C: int, $D: int) {
	data: [A * B * C * D]f32,
}

make_1d :: proc($A: int) -> Tensor4(1, 1, 1, A) {
	return Tensor4(1, 1, 1, A){}
}

make_2d :: proc($A, $B: int) -> Tensor4(1, 1, A, B) {
	return Tensor4(1, 1, A, B){}
}

make_3d :: proc($A, $B, $C: int) -> Tensor4(1, A, B, C) {
	return Tensor4(1, A, B, C){}
}

make_4d :: proc($A, $B, $C, $D: int) -> Tensor4(A, B, C, D) {
	return Tensor4(A, B, C, D){}
}

tensor :: proc {
	make_1d,
	make_2d,
	make_3d,
	make_4d,
}


index1 :: proc(t: Tensor4($A, $B, $C, $D), i: int) -> f32 {
	return t.data[i]
}

index2 :: proc(t: Tensor4($A, $B, $C, $D), i, j: int) -> f32 {
	return t.data[(D * i) + j]
}

index3 :: proc(t: Tensor4($A, $B, $C, $D), i, j, k: int) -> f32 {
	return t.data[(C * D * i) + (D * j) + k]
}

index4 :: proc(t: Tensor4($A, $B, $C, $D), i, j, k, l: int) -> f32 {
	return t.data[(B * C * D * i) + (C * D * j) + (D * k) + l]
}

// similar slicing can be defined
i :: proc {
	index1,
	index2,
	index3,
	index4,
}

set4 :: proc(t: ^Tensor4($A, $B, $C, $D), i, j, k, l: int, val: f32) {
	t.data[(B * C * D * i) + (C * D * j) + (D * k) + l] = val
}


Axis_0 :: distinct int
Axis_1 :: distinct int

AXIS_0: Axis_0 : 0
AXIS_1: Axis_1 : 1

unsqueeze_0 :: proc(t: Tensor4(1, $A, $B, $C), _: Axis_0) -> Tensor4(1, A, B, C) {
	return t
}

unsqueeze_1 :: proc(t: Tensor4(1, $A, $B, $C), _: Axis_1) -> Tensor4(A, 1, B, C) {
	out := Tensor4(A, 1, B, C){}
	out.data = t.data
	return out
}

unsqueeze :: proc {
	unsqueeze_0,
	unsqueeze_1,
}

matmul_impl :: proc(
	out: ^Tensor4($AO, $BO, $C, $E),
	t1: Tensor4($A1, $B1, C, $D),
	t2: Tensor4($A2, $B2, D, E),
) {
	for a in 0 ..< AO {
		a1 := a
		a2 := a
		if A1 == 1 {
			a1 = 0
		}
		if A2 == 1 {
			a2 = 0
		}

		for b in 0 ..< BO {
			b1 := b
			b2 := b
			if B1 == 1 {
				b1 = 0
			}
			if B2 == 1 {
				b2 = 0
			}

			for c in 0 ..< C {
				for e in 0 ..< E {
					sum: f32 = 0
					for d in 0 ..< D {
						sum += i(t1, a1, b1, c, d) * i(t2, a2, b2, d, e)
					}
					set4(out, a, b, c, e, sum)
				}
			}
		}
	}
}

matmul_same :: proc(t1: Tensor4($A, $B, $C, $D), t2: Tensor4(A, B, D, $E)) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul_a_left :: proc(
	t1: Tensor4(1, $B, $C, $D),
	t2: Tensor4($A, B, D, $E),
) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul_a_right :: proc(
	t1: Tensor4($A, $B, $C, $D),
	t2: Tensor4(1, B, D, $E),
) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul_b_left :: proc(
	t1: Tensor4($A, 1, $C, $D),
	t2: Tensor4(A, $B, D, $E),
) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul_b_right :: proc(
	t1: Tensor4($A, $B, $C, $D),
	t2: Tensor4(A, 1, D, $E),
) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul_ab_left :: proc(
	t1: Tensor4(1, 1, $C, $D),
	t2: Tensor4($A, $B, D, $E),
) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul_ab_right :: proc(
	t1: Tensor4($A, $B, $C, $D),
	t2: Tensor4(1, 1, D, $E),
) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul_a_right_b_left :: proc(
	t1: Tensor4($A, 1, $C, $D),
	t2: Tensor4(1, $B, D, $E),
) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul_a_left_b_right :: proc(
	t1: Tensor4(1, $B, $C, $D),
	t2: Tensor4($A, 1, D, $E),
) -> Tensor4(A, B, C, E) {
	out := Tensor4(A, B, C, E){}
	matmul_impl(&out, t1, t2)
	return out
}

matmul :: proc {
	matmul_same,
	matmul_a_left,
	matmul_a_right,
	matmul_b_left,
	matmul_b_right,
	matmul_ab_left,
	matmul_ab_right,
	matmul_a_right_b_left,
	matmul_a_left_b_right,
}
