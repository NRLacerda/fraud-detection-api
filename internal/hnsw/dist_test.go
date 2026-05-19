package hnsw

import (
	"math/rand"
	"sort"
	"testing"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// TestDist_Identical: distance from a vector to itself must be 0.
func TestDist_Identical(t *testing.T) {
	q := [config.VectorDim]int16{
		10, 20, 30, 40, 50, 60, 70,
		80, 90, 100, 110, 120, 130, 140,
	}
	v := [config.VectorDim]uint8{
		10, 20, 30, 40, 50, 60, 70,
		80, 90, 100, 110, 120, 130, 140,
	}
	if got := Dist(&q, &v); got != 0 {
		t.Errorf("Dist(self,self) = %d, want 0", got)
	}
}

// TestDist_OneDimensionDifference verifies the squared semantics.
func TestDist_OneDimensionDifference(t *testing.T) {
	q := [config.VectorDim]int16{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	v := [config.VectorDim]uint8{255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	// diff = -255 → squared = 65025
	if got := Dist(&q, &v); got != 65025 {
		t.Errorf("got %d, want 65025", got)
	}
}

// TestDist_AllMaxDifference: every dimension at max distance.
func TestDist_AllMaxDifference(t *testing.T) {
	q := [config.VectorDim]int16{}
	v := [config.VectorDim]uint8{}
	for i := range config.VectorDim {
		q[i] = 0
		v[i] = 255
	}
	// 14 dims × 65025 = 910350
	if got := Dist(&q, &v); got != 910350 {
		t.Errorf("got %d, want 910350", got)
	}
}

// TestDist_RankingMatchesNaiveFloat: on 100 random pairs (one fixed query,
// 100 different reference vectors), the order-by-Dist must equal the
// order-by-naive-float-Euclidean computed independently. This is the
// load-bearing property — actual numeric values can differ; ranks cannot.
func TestDist_RankingMatchesNaiveFloat(t *testing.T) {
	const refs = 100
	rng := rand.New(rand.NewSource(42))

	var q [config.VectorDim]int16
	for i := range config.VectorDim {
		q[i] = int16(rng.Intn(256))
	}

	type scored struct {
		id      int
		intDist int32
		fltDist float64
	}
	results := make([]scored, refs)
	for r := range refs {
		var v [config.VectorDim]uint8
		var fSum float64
		for i := range config.VectorDim {
			v[i] = uint8(rng.Intn(256))
			d := float64(q[i]) - float64(v[i])
			fSum += d * d
		}
		results[r] = scored{
			id:      r,
			intDist: Dist(&q, &v),
			fltDist: fSum,
		}
	}

	// Sort once by int distance, once by float distance — orders must match.
	byInt := make([]int, refs)
	byFlt := make([]int, refs)
	for i := range refs {
		byInt[i] = i
		byFlt[i] = i
	}
	sort.Slice(byInt, func(a, b int) bool {
		return results[byInt[a]].intDist < results[byInt[b]].intDist
	})
	sort.Slice(byFlt, func(a, b int) bool {
		return results[byFlt[a]].fltDist < results[byFlt[b]].fltDist
	})

	for i := range refs {
		if byInt[i] != byFlt[i] {
			t.Errorf("rank %d: int order id=%d, float order id=%d",
				i, byInt[i], byFlt[i])
		}
	}
}

// TestDist_ZeroAlloc: a benchmark via AllocsPerRun (cheap, no -benchmem).
func TestDist_ZeroAlloc(t *testing.T) {
	q := [config.VectorDim]int16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	v := [config.VectorDim]uint8{14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	allocs := testing.AllocsPerRun(1000, func() {
		_ = Dist(&q, &v)
	})
	if allocs != 0 {
		t.Errorf("Dist allocs/op = %v, want 0", allocs)
	}
}

func BenchmarkDist(b *testing.B) {
	q := [config.VectorDim]int16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	v := [config.VectorDim]uint8{14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	b.ReportAllocs()
	b.ResetTimer()
	var sink int32
	for i := 0; i < b.N; i++ {
		sink ^= Dist(&q, &v)
	}
	runtime_KeepAlive(sink)
}

// runtime_KeepAlive is a local stand-in to keep the benchmark sink live
// without importing runtime.
//
//go:noinline
func runtime_KeepAlive(int32) {}
