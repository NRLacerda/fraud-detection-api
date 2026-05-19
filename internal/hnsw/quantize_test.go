package hnsw

import (
	"testing"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// TestQuantize_StdDimEndpoints — 0 → 0 and 1 → 255 on every non-sentinel dim.
func TestQuantize_StdDimEndpoints(t *testing.T) {
	cases := []struct {
		in   float32
		want uint8
	}{
		{0.0, 0},
		{1.0, 255},
		{0.5, 128},
		{0.25, 64},
		{-0.1, 0},  // clamp below
		{1.5, 255}, // clamp above
	}
	for _, c := range cases {
		if got := qStd(c.in); got != c.want {
			t.Errorf("qStd(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

// TestQuantize_SentinelDimEndpoints — sentinel -1 must map to byte 0,
// and any normalized [0,1] value must land in [128, 255] (the integer gap
// from the sentinel is what makes 'missing' distinct from 'present').
func TestQuantize_SentinelDimEndpoints(t *testing.T) {
	cases := []struct {
		in   float32
		want uint8
	}{
		{-1.0, 0},   // sentinel
		{0.0, 128},  // (0+1)/2*255 = 127.5 → 128 after rounding
		{1.0, 255},
		{0.5, 191},  // (0.5+1)/2*255 = 191.25 → 191
		{-2.0, 0},   // clamp below
		{2.0, 255},  // clamp above
	}
	for _, c := range cases {
		if got := qSentinel(c.in); got != c.want {
			t.Errorf("qSentinel(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

// TestQuantize_SentinelIntegerGap is the load-bearing property of our
// sentinel encoding: -1 (missing) and 0 (zero minutes since last tx) MUST
// quantize to different bytes, and the gap should be large.
func TestQuantize_SentinelIntegerGap(t *testing.T) {
	missing := qSentinel(-1.0)
	zero := qSentinel(0.0)
	if missing == zero {
		t.Fatalf("sentinel -1 and value 0 must not collide; both = %d", missing)
	}
	if zero-missing < 100 {
		t.Errorf("integer gap between sentinel and zero is %d, want ≥ 100", zero-missing)
	}
}

// TestQuantizeVector_LegitExample is the worked legit example from
// DETECTION_RULES.md, quantized. Spot-checks the 14-d output.
func TestQuantizeVector_LegitExample(t *testing.T) {
	v := [config.VectorDim]float32{
		0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1,
		0.0292, 0.15, 0, 1, 0, 0.15, 0.006,
	}
	var q [config.VectorDim]uint8
	QuantizeVector(&v, &q)

	// Spot checks:
	// - dim 0: 0.0041 * 255 ≈ 1.04 → 1
	if q[0] != 1 {
		t.Errorf("q[0] = %d, want 1", q[0])
	}
	// - dim 5 (sentinel): -1 → 0
	if q[5] != 0 {
		t.Errorf("q[5] sentinel = %d, want 0", q[5])
	}
	if q[6] != 0 {
		t.Errorf("q[6] sentinel = %d, want 0", q[6])
	}
	// - dim 10: 1.0 → 255 (card_present true)
	if q[10] != 255 {
		t.Errorf("q[10] = %d, want 255", q[10])
	}
	// - dim 11: 0.0 → 0 (known merchant)
	if q[11] != 0 {
		t.Errorf("q[11] = %d, want 0", q[11])
	}
}

func TestQuantizeQuery_MatchesQuantizeVector(t *testing.T) {
	// Same input through both quantizers should produce identical byte values
	// (the int16 query type only matters for distance subtraction).
	v := [config.VectorDim]float32{
		0.5, 0.25, 1.0, 0.7826, 0.3333, -1, 0.5,
		0.0292, 0.15, 0, 1, 0, 0.15, 0.006,
	}
	var qv [config.VectorDim]uint8
	var qq [config.VectorDim]int16
	QuantizeVector(&v, &qv)
	QuantizeQuery(&v, &qq)
	for i := range config.VectorDim {
		if int16(qv[i]) != qq[i] {
			t.Errorf("dim %d: QuantizeVector=%d, QuantizeQuery=%d", i, qv[i], qq[i])
		}
	}
}

func TestQuantize_ZeroAlloc(t *testing.T) {
	v := [config.VectorDim]float32{0.1, 0.2, 0.3, 0.4, 0.5, -1, -1, 0.7, 0.8, 0, 1, 0, 0.5, 0.1}
	var qv [config.VectorDim]uint8
	var qq [config.VectorDim]int16

	if allocs := testing.AllocsPerRun(1000, func() {
		QuantizeVector(&v, &qv)
	}); allocs != 0 {
		t.Errorf("QuantizeVector: %v allocs/op, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(1000, func() {
		QuantizeQuery(&v, &qq)
	}); allocs != 0 {
		t.Errorf("QuantizeQuery: %v allocs/op, want 0", allocs)
	}
}

func BenchmarkQuantizeQuery(b *testing.B) {
	v := [config.VectorDim]float32{0.1, 0.2, 0.3, 0.4, 0.5, -1, -1, 0.7, 0.8, 0, 1, 0, 0.5, 0.1}
	var q [config.VectorDim]int16
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		QuantizeQuery(&v, &q)
	}
}
