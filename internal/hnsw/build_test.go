package hnsw

import (
	"math/rand"
	"testing"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// TestBuilder_SmokeSmallGraph builds a 200-node random graph and asserts
// basic structural properties: the entry point lives on the top layer, every
// node has at least one layer-0 neighbor, neighbor counts respect M / M0,
// and every node is reachable from the entry point via layer-0 BFS.
func TestBuilder_SmokeSmallGraph(t *testing.T) {
	const N = uint32(200)
	const M = 4
	const M0 = 8
	const efC = 50

	b := NewBuilder(N, M, M0, efC, 1)
	rng := rand.New(rand.NewSource(7))

	for id := uint32(0); id < N; id++ {
		var v [config.VectorDim]uint8
		for d := range config.VectorDim {
			v[d] = uint8(rng.Intn(256))
		}
		label := LabelLegit
		if rng.Intn(3) == 0 {
			label = LabelFraud
		}
		b.Insert(id, &v, label)
	}
	b.Finalize()
	g := b.G

	// (1) Entry point lives on top layer.
	if g.MaxLayer >= 1 {
		idx := g.upperLocalIndex(g.EntryPoint, int(g.MaxLayer))
		if idx < 0 {
			t.Errorf("entry point %d should exist on top layer %d", g.EntryPoint, g.MaxLayer)
		}
	}

	// (2) Every node (except possibly the first inserted, which had no peers)
	// has at least one layer-0 neighbor.
	noNeighbor := 0
	for id := uint32(0); id < N; id++ {
		if g.Conn0Cnt[id] == 0 {
			noNeighbor++
		}
		if int(g.Conn0Cnt[id]) > M0 {
			t.Errorf("node %d: layer-0 degree %d > M0=%d", id, g.Conn0Cnt[id], M0)
		}
	}
	// At most 1 isolated node is tolerable (the first inserted before any
	// peer existed). In practice it gets connections-in from later inserts.
	if noNeighbor > 1 {
		t.Errorf("%d isolated nodes at layer 0 (expected ≤ 1)", noNeighbor)
	}

	// (3) Upper-layer degrees ≤ M.
	for L := 1; L <= int(g.MaxLayer); L++ {
		nodes := g.upperNodesOf(L)
		for i := range nodes {
			slot := int(g.UpperOff[L-1]) + i
			if int(g.UpperCnt[slot]) > M {
				t.Errorf("layer %d slot %d: degree %d > M=%d",
					L, slot, g.UpperCnt[slot], M)
			}
		}
	}

	// (4) Reachability: BFS from entry point over layer 0 must reach every
	// node. If not, the graph has disconnected components and queries can
	// silently miss results.
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
	unreached := 0
	for id := range visited {
		if !visited[id] {
			unreached++
		}
	}
	if unreached > 0 {
		t.Errorf("BFS from entry point reached %d/%d nodes (%d unreachable)",
			int(N)-unreached, N, unreached)
	}

	// (5) Labels round-trip via IsFraud.
	frauds := 0
	for id := uint32(0); id < N; id++ {
		if g.IsFraud(id) {
			frauds++
		}
	}
	if frauds == 0 || frauds == int(N) {
		t.Errorf("labels degenerate: %d frauds of %d", frauds, N)
	}
}

// TestBuilder_Determinism: two builds with the same seed produce identical
// graphs. This matters because we want repeatable rebuilds — a stochastic
// build makes A/B tests on parameter tweaks impossible to interpret.
func TestBuilder_Determinism(t *testing.T) {
	const N = uint32(100)
	build := func() *Graph {
		b := NewBuilder(N, 4, 8, 20, 42)
		rng := rand.New(rand.NewSource(99))
		for id := uint32(0); id < N; id++ {
			var v [config.VectorDim]uint8
			for d := range config.VectorDim {
				v[d] = uint8(rng.Intn(256))
			}
			b.Insert(id, &v, LabelLegit)
		}
		b.Finalize()
		return b.G
	}
	g1 := build()
	g2 := build()

	if g1.EntryPoint != g2.EntryPoint {
		t.Errorf("entry points differ: %d vs %d", g1.EntryPoint, g2.EntryPoint)
	}
	if g1.MaxLayer != g2.MaxLayer {
		t.Errorf("max layers differ: %d vs %d", g1.MaxLayer, g2.MaxLayer)
	}
	for i := range g1.Conn0 {
		if g1.Conn0[i] != g2.Conn0[i] {
			t.Errorf("Conn0[%d] diverges: %d vs %d", i, g1.Conn0[i], g2.Conn0[i])
			break
		}
	}
}
