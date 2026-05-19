// Package hnsw implements a Hierarchical Navigable Small World graph for
// approximate k-nearest-neighbor search over 14-d uint8-quantized vectors.
//
// This file contains only the quantization primitives. Distance, graph,
// query, and persistence live in sibling files.
package hnsw

import "github.com/nrlacerda/fraud-detection-api/internal/config"

// Quantization scaling per dimension.
//
// The reference vectors in resources/references.json are already normalized
// to [0,1] (with a -1 sentinel at indices 5 and 6 when last_transaction was
// null). We scale dims 5 and 6 over the range [-1, 1] so the sentinel maps
// cleanly to byte 0 and any normalized value lands in [128, 255] — a hard
// integer gap separates "missing" from "present". All other dims scale
// over [0, 1].
//
// Using fixed (rather than data-observed) per-dim ranges keeps quantization
// deterministic across rebuilds and lets us hard-code the scaling on the
// hot path with no lookup table.

// QuantizeVector scales a float32 reference vector into a uint8 storage
// vector. Used at index build time only.
//
// Pre-conditions: v[i] ∈ [0,1] for i ∉ {5,6}; v[i] ∈ {-1} ∪ [0,1] for
// i ∈ {5,6}. Inputs outside these ranges are clamped to the byte range.
func QuantizeVector(v *[config.VectorDim]float32, out *[config.VectorDim]uint8) {
	out[0] = qStd(v[0])
	out[1] = qStd(v[1])
	out[2] = qStd(v[2])
	out[3] = qStd(v[3])
	out[4] = qStd(v[4])
	out[5] = qSentinel(v[5])
	out[6] = qSentinel(v[6])
	out[7] = qStd(v[7])
	out[8] = qStd(v[8])
	out[9] = qStd(v[9])
	out[10] = qStd(v[10])
	out[11] = qStd(v[11])
	out[12] = qStd(v[12])
	out[13] = qStd(v[13])
}

// QuantizeQuery scales a float32 query vector into an int16 query vector.
//
// int16 (not uint8) is used at the query side so the difference (qi - vi)
// during distance computation does not underflow. Cost is one extra byte
// per dim — 28 bytes per query, irrelevant.
func QuantizeQuery(v *[config.VectorDim]float32, out *[config.VectorDim]int16) {
	out[0] = int16(qStd(v[0]))
	out[1] = int16(qStd(v[1]))
	out[2] = int16(qStd(v[2]))
	out[3] = int16(qStd(v[3]))
	out[4] = int16(qStd(v[4]))
	out[5] = int16(qSentinel(v[5]))
	out[6] = int16(qSentinel(v[6]))
	out[7] = int16(qStd(v[7]))
	out[8] = int16(qStd(v[8]))
	out[9] = int16(qStd(v[9]))
	out[10] = int16(qStd(v[10]))
	out[11] = int16(qStd(v[11]))
	out[12] = int16(qStd(v[12]))
	out[13] = int16(qStd(v[13]))
}

// qStd quantizes a value expected in [0,1] to a byte in [0,255].
func qStd(x float32) uint8 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 255
	}
	// +0.5 for round-to-nearest (Go's float→int truncates toward zero).
	return uint8(x*255.0 + 0.5)
}

// qSentinel quantizes a value in {-1} ∪ [0,1] to a byte in [0,255].
// Range is [-1, 1] → [0, 255]; the value -1 maps exactly to 0.
func qSentinel(x float32) uint8 {
	if x <= -1 {
		return 0
	}
	if x >= 1 {
		return 255
	}
	// (x + 1) / 2 maps [-1,1] → [0,1]; multiply by 255 and round.
	return uint8((x+1.0)*127.5 + 0.5)
}
