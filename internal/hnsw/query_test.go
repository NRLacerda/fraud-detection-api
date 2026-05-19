package hnsw

import (
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// buildTestGraph builds a small graph of N random uint8 vectors for query
// tests. Returns the finalized Graph and a slice of the raw vectors.
func buildTestGraph(t testing.TB, N uint32, seed int64) *Graph {
	t.Helper()
	const M = 4
	const M0 = 8
	const efC = 200

	b := NewBuilder(N, M, M0, efC, seed)
	rng := rand.New(rand.NewSource(seed + 1))
	for id := uint32(0); id < N; id++ {
		var v [config.VectorDim]uint8
		for d := range config.VectorDim {
			v[d] = uint8(rng.Intn(256))
		}
		label := LabelLegit
		if rng.Intn(4) == 0 {
			label = LabelFraud
		}
		b.Insert(id, &v, label)
	}
	b.Finalize()
	return b.G
}

func loadProductionGraphForBenchmark(b *testing.B) *Graph {
	b.Helper()

	path := filepath.Join("..", "..", "resources", "hnsw.bin")
	if _, err := os.Stat(path); err != nil {
		b.Skipf("production index not available at %s: %v", path, err)
	}
	g, err := LoadMmap(path)
	if err != nil {
		b.Fatalf("load production index: %v", err)
	}
	return g
}

// bruteForceTop5 computes the exact 5 nearest node IDs to q by scanning all
// nodes. Used as ground truth for recall measurement.
func bruteForceTop5(g *Graph, q *[config.VectorDim]int16) [5]uint32 {
	type pair struct {
		id   uint32
		dist int32
	}
	pairs := make([]pair, g.N)
	for i := range g.N {
		pairs[i] = pair{i, Dist(q, g.VectorAt(i))}
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		if pairs[i].dist != pairs[j].dist {
			return pairs[i].dist < pairs[j].dist
		}
		return pairs[i].id < pairs[j].id
	})
	var out [5]uint32
	for i := range 5 {
		out[i] = pairs[i].id
	}
	return out
}

// TestQueryFast5_RecallVsBruteForce sanity-checks that the search finds a
// solid fraction of the true top-5 against brute force.
//
// Threshold note: random uniform uint8 vectors are a worst case for any ANN
// algorithm — 14-d uniform random distances concentrate, so "neighborhoods"
// are weak and HNSW with queryEf=10 plateaus around 80–85% recall@5. Real
// transaction data has natural cluster structure (MCC bucket, time-of-day,
// amount band) and recalls significantly higher with the same parameters.
// The production-data recall@5 check is in cmd/build-index.
func TestQueryFast5_RecallVsBruteForce(t *testing.T) {
	const N = uint32(1000)
	g := buildTestGraph(t, N, 11)
	slot := NewVisitSlot(N)

	rng := rand.New(rand.NewSource(99))
	const Q = 200
	totalHits := 0
	for q := 0; q < Q; q++ {
		var qVec [config.VectorDim]int16
		for d := range config.VectorDim {
			qVec[d] = int16(rng.Intn(256))
		}

		var got [5]uint32
		g.QueryFast5(&qVec, slot, &got)
		truth := bruteForceTop5(g, &qVec)

		truthSet := map[uint32]struct{}{}
		for _, id := range truth {
			truthSet[id] = struct{}{}
		}
		for _, id := range got {
			if _, ok := truthSet[id]; ok {
				totalHits++
			}
		}
	}

	// Random uniform uint8 baseline. 75% is a safe floor; my measured run
	// hits ~81%. Production data is expected ≥95%.
	const minHits = Q * 5 * 75 / 100
	if totalHits < minHits {
		t.Errorf("recall too low: %d/%d hits (need >= %d)", totalHits, Q*5, minHits)
	}
}

// TestQueryFast5_ZeroAlloc: after slot is constructed, repeated queries must
// not allocate any heap memory. This is the headline guarantee.
func TestQueryFast5_ZeroAlloc(t *testing.T) {
	const N = uint32(500)
	g := buildTestGraph(t, N, 7)
	slot := NewVisitSlot(N)

	rng := rand.New(rand.NewSource(123))
	var qVec [config.VectorDim]int16
	for d := range config.VectorDim {
		qVec[d] = int16(rng.Intn(256))
	}
	var out [5]uint32

	// Warm up to populate any one-time state.
	g.QueryFast5(&qVec, slot, &out)

	allocs := testing.AllocsPerRun(200, func() {
		g.QueryFast5(&qVec, slot, &out)
	})
	if allocs != 0 {
		t.Errorf("QueryFast5 must be zero-alloc, got %.2f allocs/op", allocs)
	}
}

// TestQueryFast5_ResultsDistinct: the 5 returned IDs must all be distinct.
func TestQueryFast5_ResultsDistinct(t *testing.T) {
	const N = uint32(500)
	g := buildTestGraph(t, N, 5)
	slot := NewVisitSlot(N)

	rng := rand.New(rand.NewSource(101))
	for q := 0; q < 100; q++ {
		var qVec [config.VectorDim]int16
		for d := range config.VectorDim {
			qVec[d] = int16(rng.Intn(256))
		}
		var got [5]uint32
		g.QueryFast5(&qVec, slot, &got)

		seen := map[uint32]struct{}{}
		for _, id := range got {
			if _, dup := seen[id]; dup {
				t.Errorf("duplicate id %d in result %v", id, got)
				break
			}
			seen[id] = struct{}{}
		}
	}
}

// TestQueryFast5_GenerationWrap exercises the uint8 generation-counter
// wraparound path (every 256 queries, slot.marks must be re-zeroed). Recall
// must remain consistent across the wrap.
func TestQueryFast5_GenerationWrap(t *testing.T) {
	const N = uint32(300)
	g := buildTestGraph(t, N, 4)
	slot := NewVisitSlot(N)

	rng := rand.New(rand.NewSource(2))
	var qVec [config.VectorDim]int16
	for d := range config.VectorDim {
		qVec[d] = int16(rng.Intn(256))
	}
	var first [5]uint32
	g.QueryFast5(&qVec, slot, &first)

	// Run 300 more queries on the same slot — straddling the 256 wrap.
	var noise [config.VectorDim]int16
	for q := 0; q < 300; q++ {
		for d := range config.VectorDim {
			noise[d] = int16(rng.Intn(256))
		}
		var got [5]uint32
		g.QueryFast5(&noise, slot, &got)
		_ = got
	}

	// Repeat the original query — result must match (graph is read-only).
	var again [5]uint32
	g.QueryFast5(&qVec, slot, &again)
	if again != first {
		t.Errorf("query result drifted across generation wrap: %v vs %v", first, again)
	}
}

// BenchmarkQueryFast5 measures ns/op and allocs/op for the hot query path.
func BenchmarkQueryFast5(b *testing.B) {
	const N = uint32(10000)
	g := buildTestGraph(b, N, 3)
	slot := NewVisitSlot(N)

	rng := rand.New(rand.NewSource(0))
	var qVec [config.VectorDim]int16
	for d := range config.VectorDim {
		qVec[d] = int16(rng.Intn(256))
	}
	var out [5]uint32

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.QueryFast5(&qVec, slot, &out)
	}
}

func BenchmarkQueryFast5ProductionIndex(b *testing.B) {
	g := loadProductionGraphForBenchmark(b)
	slot := NewVisitSlot(g.N)

	var qVec [config.VectorDim]int16
	for d := range config.VectorDim {
		qVec[d] = int16((d + 1) * 17)
	}
	var out [5]uint32

	// Warm the slot and the graph pages touched by this query.
	for i := 0; i < 100; i++ {
		g.QueryFast5(&qVec, slot, &out)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.QueryFast5(&qVec, slot, &out)
	}
}
