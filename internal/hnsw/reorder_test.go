package hnsw

import (
	"math/rand"
	"testing"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// TestReorder_PreservesStructure: after Reorder, the same set of (a,b) edges
// must exist (modulo relabeling); BFS from the new entry must still reach
// every previously-reachable node; entry point's vector is unchanged.
func TestReorder_PreservesStructure(t *testing.T) {
	const N = uint32(200)
	const M = 4
	const M0 = 8

	b := NewBuilder(N, M, M0, 30, 19)
	rng := rand.New(rand.NewSource(3))
	for id := uint32(0); id < N; id++ {
		var v [config.VectorDim]uint8
		for d := range config.VectorDim {
			v[d] = uint8(rng.Intn(256))
		}
		b.Insert(id, &v, LabelLegit)
	}
	b.Finalize()
	g := b.G

	// Snapshot edges by vector pair (each vector is unique by construction
	// at N=200 with 14×256 = 3584 distinct bytes per dim; collisions very
	// unlikely with random uint8s).
	oldEdges := map[[2]uint32]struct{}{}
	for id := uint32(0); id < N; id++ {
		for _, nb := range g.Layer0Neighbors(id) {
			if nb < 0 {
				continue
			}
			// Canonical (low,high) tuple for undirected edge.
			a, c := id, uint32(nb)
			if a > c {
				a, c = c, a
			}
			oldEdges[[2]uint32{a, c}] = struct{}{}
		}
	}
	oldEntryVec := *g.VectorAt(g.EntryPoint)

	g.Reorder()

	// Entry point's vector must be the same bytes after reorder (just at a
	// different node id — by construction, new entry point id is 0).
	if g.EntryPoint != 0 {
		t.Errorf("after Reorder, EntryPoint should be 0, got %d", g.EntryPoint)
	}
	if *g.VectorAt(g.EntryPoint) != oldEntryVec {
		t.Errorf("entry point vector changed across reorder")
	}

	// BFS from new entry reaches every node.
	visited := make([]bool, N)
	queue := []uint32{g.EntryPoint}
	visited[g.EntryPoint] = true
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, nb := range g.Layer0Neighbors(cur) {
			if nb < 0 {
				continue
			}
			if !visited[nb] {
				visited[nb] = true
				queue = append(queue, uint32(nb))
			}
		}
	}
	for id := range visited {
		if !visited[id] {
			t.Errorf("node %d unreachable after Reorder", id)
		}
	}

	// Upper layer node lists must remain sorted ascending (binary search
	// depends on this).
	for L := 1; L <= int(g.MaxLayer); L++ {
		nodes := g.upperNodesOf(L)
		for i := 1; i < len(nodes); i++ {
			if nodes[i-1] >= nodes[i] {
				t.Errorf("layer %d not sorted: %d then %d", L, nodes[i-1], nodes[i])
				break
			}
		}
	}
}
