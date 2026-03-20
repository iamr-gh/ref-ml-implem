package main

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


index1 :: proc(t: Tensor4($A, $B, $C, $D), i: int) {
	return t.data[i]
}

index2 :: proc(t: Tensor4($A, $B, $C, $D), i, j: int) {
	return t.data[(D * i) + j]
}

index3 :: proc(t: Tensor4($A, $B, $C, $D), i, j, k: int) {
	return t.data[(C * D * i) + (D * j) + k]
}

index4 :: proc(t: Tensor4($A, $B, $C, $D), i, j, k, l: int) {
	return t.data[(B * C * D * i) + (C * D * j) + (D * k) + l]
}

// similar slicing can be defined
i :: proc {
	index1,
	index2,
	index3,
	index4,
}

matmul :: proc(t1: Tensor4($A, $B, $C, $D), t2: Tensor4(A, B, D, $E)) -> Tensor4(A, B, C, E) {

	// todo
}
