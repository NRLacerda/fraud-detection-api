package hnsw

import "github.com/nrlacerda/fraud-detection-api/internal/config"

// queryEf is the search width used at layer 0 for /fraud-score queries.
// Top-5 is extracted from the queryEf nearest.
const queryEf = 10

// VisitSlot holds a single worker's pre-allocated query buffers. One slot per
// concurrent worker (GOMAXPROCS=2 in production → two slots). Reusable across
// many queries; reusing the slot is what makes QueryFast5 allocation-free.
//
// Memory: marks is one byte per node — at N=3M that's 3 MB per slot. Wider
// counters (uint16/uint32) burn more RSS for no real win since clears are
// O(N) and amortize away. Wraparound clear cost: ~3 MB of writes every 255
// queries → ~300 μs spike amortized to ~1 μs/query at memory bandwidth.
type VisitSlot struct {
	// marks[id] == gen means id has been visited in the current query.
	// gen is bumped each query; on wrap to 0, marks is zeroed and gen = 1.
	marks []uint8
	gen   uint8

	// candidates min-heap, manual binary heap over a pre-sized slice.
	cands []distItem

	// Result set: queryEf nearest seen so far, kept as parallel arrays plus
	// a tracked "current worst" so admission is O(1) and only the worst-find
	// after a replacement is O(queryEf).
	resIDs    [queryEf]uint32
	resDists  [queryEf]int32
	resCnt    int
	resWorstD int32
	resWorstI int
}

// NewVisitSlot sizes the working memory for a graph of N nodes. Allocates
// once; reusable for the lifetime of the worker.
func NewVisitSlot(N uint32) *VisitSlot {
	return &VisitSlot{
		marks: make([]uint8, N),
		// Loose upper bound on simultaneous candidates: each pop expands by up
		// to M0=8 neighbors; the algorithm caps exploration at ~efC iterations
		// at build but only ~queryEf at query. 1024 is far beyond what we'll
		// ever push during a queryEf=10 search.
		cands: make([]distItem, 0, 1024),
	}
}

// QueryFast5 returns the 5 nearest node IDs to q. Zero-allocation: all
// working memory is owned by slot.
func (g *Graph) QueryFast5(q *[config.VectorDim]int16, slot *VisitSlot, out *[5]uint32) {
	// Phase 1: greedy single-best descent through the upper layers.
	entry := g.EntryPoint
	for L := int(g.MaxLayer); L >= 1; L-- {
		entry = g.greedyDescendUpper(q, entry, L)
	}
	// Phase 2: ef-search at layer 0 populates slot's result set.
	g.searchLayer0Fast(q, entry, slot)
	// Phase 3: pick the 5 smallest from the result set.
	g.top5FromResults(slot, out)
}

// greedyDescendUpper hill-climbs on layer L until no neighbor of the current
// node is strictly closer to q. Read-only against the CSR-ish layout.
func (g *Graph) greedyDescendUpper(q *[config.VectorDim]int16, entry uint32, L int) uint32 {
	current := entry
	bestDist := Dist(q, g.VectorAt(current))
	for {
		nbs := g.upperNeighbors(current, L)
		improved := false
		for _, nb := range nbs {
			if nb < 0 {
				continue
			}
			d := Dist(q, g.VectorAt(uint32(nb)))
			if d < bestDist {
				bestDist = d
				current = uint32(nb)
				improved = true
			}
		}
		if !improved {
			return current
		}
	}
}

// searchLayer0Fast runs the HNSW ef-search at layer 0 starting from entry.
// Writes the top queryEf nodes into slot's result arrays.
func (g *Graph) searchLayer0Fast(q *[config.VectorDim]int16, entry uint32, slot *VisitSlot) {
	// Bump visited generation; wrap-safe.
	slot.gen++
	if slot.gen == 0 {
		for i := range slot.marks {
			slot.marks[i] = 0
		}
		slot.gen = 1
	}
	slot.cands = slot.cands[:0]
	slot.resCnt = 0

	d0 := Dist(q, g.VectorAt(entry))
	slot.marks[entry] = slot.gen

	slot.cands = pushMin(slot.cands, distItem{entry, d0})
	slot.resIDs[0] = entry
	slot.resDists[0] = d0
	slot.resCnt = 1
	slot.resWorstD = d0
	slot.resWorstI = 0

	for len(slot.cands) > 0 {
		c := slot.cands[0]
		slot.cands = popMin(slot.cands)

		// Standard termination from HNSW Algorithm 2: stop once the closest
		// unexplored candidate is farther than our worst kept result. The
		// result set may still be smaller than queryEf when this fires (the
		// admission rule below ensures it always grows up to queryEf if there
		// is something to add).
		if c.dist > slot.resWorstD {
			break
		}

		nbs := g.Layer0Neighbors(c.id)
		for _, nb := range nbs {
			if nb < 0 {
				continue
			}
			n := uint32(nb)
			if slot.marks[n] == slot.gen {
				continue
			}
			slot.marks[n] = slot.gen
			d := Dist(q, g.VectorAt(n))

			if slot.resCnt < queryEf {
				slot.resIDs[slot.resCnt] = n
				slot.resDists[slot.resCnt] = d
				if d > slot.resWorstD || slot.resCnt == 0 {
					slot.resWorstD = d
					slot.resWorstI = slot.resCnt
				}
				slot.resCnt++
				slot.cands = pushMin(slot.cands, distItem{n, d})
			} else if d < slot.resWorstD {
				slot.resIDs[slot.resWorstI] = n
				slot.resDists[slot.resWorstI] = d
				// Re-scan to find the new worst.
				worstD := slot.resDists[0]
				worstI := 0
				for i := 1; i < queryEf; i++ {
					if slot.resDists[i] > worstD {
						worstD = slot.resDists[i]
						worstI = i
					}
				}
				slot.resWorstD = worstD
				slot.resWorstI = worstI
				slot.cands = pushMin(slot.cands, distItem{n, d})
			}
		}
	}
}

// top5FromResults picks the 5 smallest-distance entries from slot's result
// set. Selection (50 compares max) is cheaper than sorting a 10-element slice.
func (g *Graph) top5FromResults(slot *VisitSlot, out *[5]uint32) {
	var taken [queryEf]bool
	for pick := range 5 {
		bestI := -1
		var bestD int32
		for i := 0; i < slot.resCnt; i++ {
			if taken[i] {
				continue
			}
			if bestI < 0 || slot.resDists[i] < bestD {
				bestI = i
				bestD = slot.resDists[i]
			}
		}
		if bestI < 0 {
			out[pick] = 0
			continue
		}
		out[pick] = slot.resIDs[bestI]
		taken[bestI] = true
	}
}

// pushMin inserts x into a binary min-heap stored as h; returns the updated
// header. Manual implementation to skip container/heap's interface{} boxing.
func pushMin(h []distItem, x distItem) []distItem {
	h = append(h, x)
	i := len(h) - 1
	for i > 0 {
		parent := (i - 1) >> 1
		if h[parent].dist <= h[i].dist {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
	return h
}

// popMin removes and discards the root (smallest) element. Caller reads h[0]
// before calling. Returns updated header.
func popMin(h []distItem) []distItem {
	n := len(h)
	if n <= 1 {
		return h[:0]
	}
	h[0] = h[n-1]
	h = h[:n-1]
	n--
	i := 0
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		smallest := l
		r := l + 1
		if r < n && h[r].dist < h[l].dist {
			smallest = r
		}
		if h[i].dist <= h[smallest].dist {
			break
		}
		h[i], h[smallest] = h[smallest], h[i]
		i = smallest
	}
	return h
}
