package hnsw

import (
	"sort"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// Reorder permutes node IDs so that nodes reachable in BFS order from the
// entry point are stored contiguously in memory. This is purely a layout
// optimization — it does not change which nodes are nearest neighbors of any
// query, only the page/cache locality of the graph traversal during query.
//
// Tradeoff: BFS over a 3M-node graph allocates O(N) bytes for the queue and
// the permutation arrays. The win is realised during query, where neighbors
// of a node tend to land on the same or adjacent cache lines, dropping the
// effective memory-stall cost of each hop. With M0=8 and 14-byte vectors
// (112 bytes for a full row of 8 neighbor vectors), several neighbor fetches
// land on the same 64-byte cache line.
//
// Must be called after Builder.Finalize and before Save. Nodes unreachable
// from the entry point are placed at the end of the new ordering.
func (g *Graph) Reorder() {
	N := int(g.N)
	if N == 0 {
		return
	}

	// oldToNew[oldID] = newID
	oldToNew := make([]int32, N)
	for i := range oldToNew {
		oldToNew[i] = -1
	}

	// BFS from entry point over layer 0.
	queue := make([]uint32, 0, N)
	queue = append(queue, g.EntryPoint)
	oldToNew[g.EntryPoint] = 0
	next := int32(1)
	head := 0
	for head < len(queue) {
		cur := queue[head]
		head++
		for _, nb := range g.Layer0Neighbors(cur) {
			if nb < 0 {
				continue
			}
			if oldToNew[nb] != -1 {
				continue
			}
			oldToNew[nb] = next
			next++
			queue = append(queue, uint32(nb))
		}
	}

	// Any disconnected nodes get appended at the end.
	for old := range N {
		if oldToNew[old] == -1 {
			oldToNew[old] = next
			next++
		}
	}

	// newToOld[newID] = oldID
	newToOld := make([]uint32, N)
	for old := range N {
		newToOld[oldToNew[old]] = uint32(old)
	}

	// Build new buffers in permuted order.
	newVectors := make([]uint8, len(g.Vectors))
	newConn0 := make([]int32, len(g.Conn0))
	newConn0Cnt := make([]uint8, len(g.Conn0Cnt))
	newLabels := make([]uint8, len(g.Labels))

	for newID := range N {
		oldID := newToOld[newID]

		// Vector: copy 14 bytes.
		copy(newVectors[newID*config.VectorDim:],
			g.Vectors[int(oldID)*config.VectorDim:int(oldID)*config.VectorDim+config.VectorDim])

		// Layer-0 neighbors: copy + remap IDs.
		oldOff := int(oldID) * g.M0
		newOff := newID * g.M0
		cnt := int(g.Conn0Cnt[oldID])
		for i := range cnt {
			oldNb := g.Conn0[oldOff+i]
			if oldNb < 0 {
				newConn0[newOff+i] = -1
				continue
			}
			newConn0[newOff+i] = oldToNew[oldNb]
		}
		for i := cnt; i < g.M0; i++ {
			newConn0[newOff+i] = -1
		}
		newConn0Cnt[newID] = uint8(cnt)

		// Labels (bit-packed).
		if g.IsFraud(oldID) {
			nID := uint32(newID)
			newLabels[nID>>3] |= 1 << (nID & 7)
		}
	}

	// Upper layers: each UpperNodes[i] is an old node ID — remap. Then re-sort
	// per-layer node lists ascending by new ID (with their cnt/conn rows
	// moved alongside) so binary search still works on the loaded graph.
	if int(g.MaxLayer) >= 1 {
		// Remap node IDs and connections in place.
		for i := range g.UpperNodes {
			g.UpperNodes[i] = oldToNew[g.UpperNodes[i]]
		}
		for i := range g.UpperConn {
			if g.UpperConn[i] >= 0 {
				g.UpperConn[i] = oldToNew[g.UpperConn[i]]
			}
		}
		// Re-sort each layer's node list by new ID (and move cnt/conn rows
		// along with it). Simple insertion sort per layer — typically O(N log N)
		// nodes per layer for the top layers, fine offline.
		for L := 1; L <= int(g.MaxLayer); L++ {
			start := int(g.UpperOff[L-1])
			end := int(g.UpperOff[L])
			sortUpperLayer(g, start, end)
		}
	}

	// Update entry point and swap.
	g.EntryPoint = uint32(oldToNew[g.EntryPoint])
	g.Vectors = newVectors
	g.Conn0 = newConn0
	g.Conn0Cnt = newConn0Cnt
	g.Labels = newLabels
}

// sortUpperLayer sorts the upper-layer slice [start:end) ascending by
// UpperNodes value, moving UpperCnt and UpperConn rows in lockstep.
//
// This is build-time only, so extra temporary memory is a good trade for
// avoiding quadratic reorder time on layer 1.
func sortUpperLayer(g *Graph, start, end int) {
	n := end - start
	if n <= 1 {
		return
	}

	M := g.M
	order := make([]int, n)
	for i := range order {
		order[i] = start + i
	}
	sort.Slice(order, func(i, j int) bool {
		return g.UpperNodes[order[i]] < g.UpperNodes[order[j]]
	})

	nodes := make([]int32, n)
	cnts := make([]uint8, n)
	conns := make([]int32, n*M)
	for newI, oldSlot := range order {
		nodes[newI] = g.UpperNodes[oldSlot]
		cnts[newI] = g.UpperCnt[oldSlot]
		copy(conns[newI*M:newI*M+M], g.UpperConn[oldSlot*M:oldSlot*M+M])
	}

	copy(g.UpperNodes[start:end], nodes)
	copy(g.UpperCnt[start:end], cnts)
	copy(g.UpperConn[start*M:end*M], conns)
}
