package hnsw

import "github.com/nrlacerda/fraud-detection-api/internal/config"

// Dist computes the squared integer Euclidean distance between a quantized
// query and a stored vector.
//
// Rankings produced by this function are identical to rankings produced by
// true (float) Euclidean distance over the same quantized values — no sqrt,
// no dequantization. The result fits comfortably in int32:
//
//	max per-dim diff = 255  →  squared = 65025
//	14 dims          →  max sum = 910 350  (well under 2^31)
//
// Hot path: called ~100–300× per /fraud-score request. The 14-element loop
// is fully unrolled and accepts array pointers to give the compiler every
// chance to keep operands in registers and skip slice bounds checks.
//
//go:nosplit
func Dist(q *[config.VectorDim]int16, v *[config.VectorDim]uint8) int32 {
	d0 := int32(q[0]) - int32(v[0])
	d1 := int32(q[1]) - int32(v[1])
	d2 := int32(q[2]) - int32(v[2])
	d3 := int32(q[3]) - int32(v[3])
	d4 := int32(q[4]) - int32(v[4])
	d5 := int32(q[5]) - int32(v[5])
	d6 := int32(q[6]) - int32(v[6])
	d7 := int32(q[7]) - int32(v[7])
	d8 := int32(q[8]) - int32(v[8])
	d9 := int32(q[9]) - int32(v[9])
	d10 := int32(q[10]) - int32(v[10])
	d11 := int32(q[11]) - int32(v[11])
	d12 := int32(q[12]) - int32(v[12])
	d13 := int32(q[13]) - int32(v[13])

	return d0*d0 + d1*d1 + d2*d2 + d3*d3 +
		d4*d4 + d5*d5 + d6*d6 + d7*d7 +
		d8*d8 + d9*d9 + d10*d10 + d11*d11 +
		d12*d12 + d13*d13
}

// Dist2 computes squared integer Euclidean distance between two stored uint8
// vectors. Used during graph construction (neighbor-vs-neighbor distance for
// the heuristic neighbor selector), avoiding the need to materialize an
// int16 view of either operand. Same int32 result range as Dist.
//
//go:nosplit
func Dist2(a, b *[config.VectorDim]uint8) int32 {
	d0 := int32(a[0]) - int32(b[0])
	d1 := int32(a[1]) - int32(b[1])
	d2 := int32(a[2]) - int32(b[2])
	d3 := int32(a[3]) - int32(b[3])
	d4 := int32(a[4]) - int32(b[4])
	d5 := int32(a[5]) - int32(b[5])
	d6 := int32(a[6]) - int32(b[6])
	d7 := int32(a[7]) - int32(b[7])
	d8 := int32(a[8]) - int32(b[8])
	d9 := int32(a[9]) - int32(b[9])
	d10 := int32(a[10]) - int32(b[10])
	d11 := int32(a[11]) - int32(b[11])
	d12 := int32(a[12]) - int32(b[12])
	d13 := int32(a[13]) - int32(b[13])

	return d0*d0 + d1*d1 + d2*d2 + d3*d3 +
		d4*d4 + d5*d5 + d6*d6 + d7*d7 +
		d8*d8 + d9*d9 + d10*d10 + d11*d11 +
		d12*d12 + d13*d13
}
