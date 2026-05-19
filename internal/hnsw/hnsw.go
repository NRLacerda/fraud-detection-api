package hnsw

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// LabelFraud and LabelLegit are stored in the labels bitset (1 bit per node).
const (
	LabelLegit uint8 = 0
	LabelFraud uint8 = 1
)

// Graph is the runtime form of the HNSW index. It can be freshly built via
// Builder, persisted via persist.go, and loaded back read-only for serving.
//
// All slices are kept flat and contiguous to be mmap-friendly: a loader can
// point the slice headers directly at byte ranges of a memory-mapped file
// without copying.
type Graph struct {
	N  uint32 // number of nodes
	M  int    // max connections per upper layer
	M0 int    // max connections at layer 0

	// vectors[id*VectorDim : (id+1)*VectorDim] is node id's stored vector.
	Vectors []uint8

	// conn0[id*M0 : id*M0+conn0Cnt[id]] is node id's layer-0 neighbors.
	Conn0    []int32
	Conn0Cnt []uint8

	// Labels is a packed bitset; bit (id & 7) of byte (id >> 3) is 1 for fraud.
	Labels []uint8

	// Upper layers (layers ≥ 1) — CSR-ish layout.
	//
	// For layer L ∈ [1, MaxLayer]:
	//   slice of nodes living on L: UpperNodes[UpperOff[L-1] : UpperOff[L]]
	//     (sorted ascending by node id, enabling binary search for membership)
	//   for the i-th node on layer L (i = local index):
	//     global slot = UpperOff[L-1] + i
	//     neighbor count = UpperCnt[global slot]
	//     neighbors      = UpperConn[global slot*M : global slot*M + UpperCnt[..]]
	//
	// UpperOff has length MaxLayer+1; UpperOff[0]=0 always.
	UpperOff   []int32
	UpperNodes []int32
	UpperCnt   []uint8
	UpperConn  []int32

	EntryPoint uint32
	MaxLayer   int32

	// raw is the underlying byte buffer (file contents or mmap'd region)
	// when the Graph was created via Load / LoadMmap. Keeping the reference
	// prevents Go's GC from reclaiming the storage behind the unsafe-derived
	// int32 slice views. Nil for graphs built in-memory.
	raw []byte
}

// Raw returns the underlying byte buffer if the graph was loaded from disk,
// or nil for an in-memory build.
func (g *Graph) Raw() []byte { return g.raw }

// VectorAt returns a pointer to the 14-byte vector slot for node id.
//
//go:inline
func (g *Graph) VectorAt(id uint32) *[config.VectorDim]uint8 {
	off := uintptr(id) * config.VectorDim
	return (*[config.VectorDim]uint8)(g.Vectors[off : off+config.VectorDim])
}

// Layer0Neighbors returns node id's layer-0 neighbor slice (length up to M0).
//
//go:inline
func (g *Graph) Layer0Neighbors(id uint32) []int32 {
	cnt := int(g.Conn0Cnt[id])
	off := int(id) * g.M0
	return g.Conn0[off : off+cnt]
}

// IsFraud reports whether node id is labeled fraud.
//
//go:inline
func (g *Graph) IsFraud(id uint32) bool {
	return g.Labels[id>>3]&(1<<(id&7)) != 0
}

// upperNodesOf returns the sorted node-id slice on a given layer L (L >= 1).
// Returns nil if layer is out of range.
func (g *Graph) upperNodesOf(L int) []int32 {
	if L < 1 || L > int(g.MaxLayer) {
		return nil
	}
	return g.UpperNodes[g.UpperOff[L-1]:g.UpperOff[L]]
}

// upperLocalIndex returns the local index of nodeID on layer L (i.e., its
// position in upperNodesOf(L)), or -1 if not present. Layer L >= 1.
func (g *Graph) upperLocalIndex(nodeID uint32, L int) int {
	nodes := g.upperNodesOf(L)
	if len(nodes) == 0 {
		return -1
	}
	// Binary search — nodes are sorted ascending.
	lo, hi := 0, len(nodes)
	target := int32(nodeID)
	for lo < hi {
		mid := (lo + hi) >> 1
		if nodes[mid] < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(nodes) && nodes[lo] == target {
		return lo
	}
	return -1
}

// upperNeighbors returns nodeID's neighbor slice at layer L (L >= 1).
// Returns nil if nodeID does not exist on that layer.
func (g *Graph) upperNeighbors(nodeID uint32, L int) []int32 {
	local := g.upperLocalIndex(nodeID, L)
	if local < 0 {
		return nil
	}
	slot := int(g.UpperOff[L-1]) + local
	cnt := int(g.UpperCnt[slot])
	off := slot * g.M
	return g.UpperConn[off : off+cnt]
}

// ---------------------------------------------------------------------------
// Builder: build-time wrapper around Graph.
//
// The Builder owns:
//   - per-node assigned level (top layer)
//   - sparse map-of-maps for upper-layer connections during insertion
//   - heap buffers and a generation-based visited tracker
//
// Builder.Finalize() flattens the build-time structures into the Graph's
// CSR-ish runtime layout, after which the build state can be dropped.
// ---------------------------------------------------------------------------

type Builder struct {
	G *Graph

	// levels[id] is the top layer assigned to node id at insertion time.
	levels []int8

	// upperConns[L-1] is a map nodeID -> []int32 of neighbors at layer L,
	// for L ∈ [1, MaxLayer]. Allocated lazily per layer.
	upperConns []map[uint32][]int32

	// efC is efConstruction (search width during inserts).
	efC int
	// mL is the level-assignment scaling factor = 1 / ln(M).
	mL float64

	// rng is the random source for level assignment. Seeded for determinism.
	rng *rand.Rand

	// visited: generation-counter array used during insert searches.
	visited []uint32
	visGen  uint32

	// Pre-allocated working slices reused across inserts to limit allocs.
	candHeap   minHeap
	resHeap    maxHeap
	tmpCands   []distItem
	tmpResults []distItem

	// inserted is the count of nodes added so far (for first-insert handling).
	inserted uint32
}

// NewBuilder constructs a Builder with capacity for n nodes and the given
// graph parameters. The Builder seeds its RNG with seed for determinism.
func NewBuilder(n uint32, m, m0, efConstruction int, seed int64) *Builder {
	g := &Graph{
		N:        n,
		M:        m,
		M0:       m0,
		Vectors:  make([]uint8, int(n)*config.VectorDim),
		Conn0:    make([]int32, int(n)*m0),
		Conn0Cnt: make([]uint8, n),
		Labels:   make([]uint8, (n+7)/8),
	}
	// Fill Conn0 with -1 sentinel so unused slots are obvious in dumps and
	// any accidental over-read is identifiable.
	for i := range g.Conn0 {
		g.Conn0[i] = -1
	}

	b := &Builder{
		G:        g,
		levels:   make([]int8, n),
		efC:      efConstruction,
		mL:       1.0 / math.Log(float64(m)),
		rng:      rand.New(rand.NewSource(seed)),
		visited:  make([]uint32, n),
		visGen:   0,
		candHeap: make(minHeap, 0, efConstruction*2),
		resHeap:  make(maxHeap, 0, efConstruction*2),
	}
	return b
}

// SetLabel marks node id with the given label (LabelFraud or LabelLegit).
func (b *Builder) SetLabel(id uint32, label uint8) {
	if label == LabelFraud {
		b.G.Labels[id>>3] |= 1 << (id & 7)
	} else {
		b.G.Labels[id>>3] &^= 1 << (id & 7)
	}
}

// Insert adds a new node with the given quantized vector and fraud label.
// Node ids must be inserted in order 0, 1, 2, ... b.G.N-1.
func (b *Builder) Insert(id uint32, v *[config.VectorDim]uint8, label uint8) {
	// Store vector.
	copy(b.G.Vectors[uintptr(id)*config.VectorDim:], v[:])
	b.SetLabel(id, label)

	// Pick a level: floor(-ln(U) * mL). Cap at 31 (we use int8 for storage,
	// and any real run with M=4 / N=3M tops out around level 10).
	u := b.rng.Float64()
	if u == 0 {
		u = 1e-300 // avoid log(0)
	}
	level := int8(-math.Log(u) * b.mL)
	if level > 31 {
		level = 31
	}
	b.levels[id] = level

	// First insertion: just become the entry point. Register on every upper
	// layer it lives on so Finalize includes it (even with no neighbors).
	if b.inserted == 0 {
		b.G.EntryPoint = id
		b.G.MaxLayer = int32(level)
		b.ensureLayers(int(level))
		b.registerOnUpperLayers(id, int(level))
		b.inserted++
		return
	}

	// Build the int16 query view of this vector for distance computation.
	var q [config.VectorDim]int16
	for i := range config.VectorDim {
		q[i] = int16(v[i])
	}

	// Phase 1: greedy descent from MaxLayer down to level+1 — single-best
	// neighbor at each layer. Below level we start ef-expansion.
	currentEntry := b.G.EntryPoint
	maxL := int(b.G.MaxLayer)
	for L := maxL; L > int(level); L-- {
		currentEntry = b.greedyDescend(&q, currentEntry, L)
	}

	// Phase 2: from min(level, maxL) down to 0, run searchLayer with ef =
	// efC, pick neighbors via heuristic, connect bidirectionally, prune
	// each neighbor's connections if it now exceeds capacity.
	topL := int(level)
	if maxL < topL {
		topL = maxL
	}

	// Register this node on every upper layer it lives on (idempotent),
	// even if some loop iterations leave it with zero neighbors there.
	b.registerOnUpperLayers(id, int(level))

	for L := topL; L >= 0; L-- {
		cands := b.searchLayer(&q, currentEntry, b.efC, L)
		mLayer := b.G.M
		if L == 0 {
			mLayer = b.G.M0
		}
		selected := b.selectNeighborsHeuristic(cands, mLayer)
		b.connectBidirectional(id, selected, L)

		// Use the closest neighbor of the search as the entry point for the
		// next layer down.
		if len(cands) > 0 {
			// cands isn't sorted ascending — find the nearest.
			best := cands[0]
			for _, c := range cands[1:] {
				if c.dist < best.dist {
					best = c
				}
			}
			currentEntry = best.id
		}
	}

	// Promote entry point if this node has a higher level.
	if int(level) > maxL {
		b.G.EntryPoint = id
		b.G.MaxLayer = int32(level)
		b.ensureLayers(int(level))
	}

	b.inserted++
}

// registerOnUpperLayers ensures nodeID has a (possibly empty) entry in the
// upperConns map for every layer in [1, level]. This guarantees Finalize
// includes the node in each layer's node list even when no edges land there.
func (b *Builder) registerOnUpperLayers(nodeID uint32, level int) {
	if level < 1 {
		return
	}
	b.ensureLayers(level)
	for L := 1; L <= level; L++ {
		if _, ok := b.upperConns[L-1][nodeID]; !ok {
			b.upperConns[L-1][nodeID] = nil
		}
	}
}

// ensureLayers makes sure upperConns has slots for layers 1..upTo.
func (b *Builder) ensureLayers(upTo int) {
	for len(b.upperConns) < upTo {
		b.upperConns = append(b.upperConns, make(map[uint32][]int32))
	}
}

// greedyDescend takes a single entry and returns the closest node it can
// find on layer L by greedy hill-climbing (no ef expansion).
func (b *Builder) greedyDescend(q *[config.VectorDim]int16, entry uint32, L int) uint32 {
	current := entry
	bestDist := Dist(q, b.G.VectorAt(current))
	for {
		neighbors := b.neighborsAt(current, L)
		improved := false
		for _, nb := range neighbors {
			if nb < 0 {
				continue
			}
			d := Dist(q, b.G.VectorAt(uint32(nb)))
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

// neighborsAt returns the current neighbor list of nodeID on layer L.
// For layer 0 it slices into Conn0; for L >= 1 it reads from the build-time
// map.
func (b *Builder) neighborsAt(nodeID uint32, L int) []int32 {
	if L == 0 {
		cnt := int(b.G.Conn0Cnt[nodeID])
		off := int(nodeID) * b.G.M0
		return b.G.Conn0[off : off+cnt]
	}
	if L-1 >= len(b.upperConns) {
		return nil
	}
	return b.upperConns[L-1][nodeID]
}

// resetVisited bumps the generation counter; on wrap-around it zeros the
// marks array.
func (b *Builder) resetVisited() {
	b.visGen++
	if b.visGen == 0 {
		for i := range b.visited {
			b.visited[i] = 0
		}
		b.visGen = 1
	}
}

// searchLayer returns up to ef nearest nodes to q reachable from entry on
// layer L. Output slice is reused (b.tmpResults) — callers must copy if they
// need to retain it across further searches.
func (b *Builder) searchLayer(q *[config.VectorDim]int16, entry uint32, ef int, L int) []distItem {
	b.resetVisited()
	b.visited[entry] = b.visGen

	d0 := Dist(q, b.G.VectorAt(entry))
	b.candHeap = b.candHeap[:0]
	b.resHeap = b.resHeap[:0]
	heap.Push(&b.candHeap, distItem{entry, d0})
	heap.Push(&b.resHeap, distItem{entry, d0})

	for b.candHeap.Len() > 0 {
		c := heap.Pop(&b.candHeap).(distItem)
		worst := b.resHeap[0] // max
		if c.dist > worst.dist {
			break
		}
		neighbors := b.neighborsAt(c.id, L)
		for _, nb := range neighbors {
			if nb < 0 {
				continue
			}
			n := uint32(nb)
			if b.visited[n] == b.visGen {
				continue
			}
			b.visited[n] = b.visGen
			d := Dist(q, b.G.VectorAt(n))
			worst = b.resHeap[0]
			if b.resHeap.Len() < ef || d < worst.dist {
				heap.Push(&b.candHeap, distItem{n, d})
				heap.Push(&b.resHeap, distItem{n, d})
				if b.resHeap.Len() > ef {
					heap.Pop(&b.resHeap)
				}
			}
		}
	}

	// Copy the result out (caller will mutate/sort).
	b.tmpResults = b.tmpResults[:0]
	for _, it := range b.resHeap {
		b.tmpResults = append(b.tmpResults, it)
	}
	return b.tmpResults
}

// selectNeighborsHeuristic implements Algorithm 4 of the HNSW paper: from a
// candidate set, keep up to m items in order of distance from the query, but
// only admit a candidate if no already-kept neighbor is closer to it than
// the query is.
//
// Each candidate carries its query-distance in distItem.dist (populated by
// searchLayer / pruneIfOverCapacity), so the query vector itself isn't a
// parameter — c.dist already encodes the dist-to-query comparison.
//
// Writes the chosen items in-place at the head of cands and returns the
// slice header truncated to the chosen count. Caller-owned buffer; the
// returned slice shares cands' backing storage.
func (b *Builder) selectNeighborsHeuristic(cands []distItem, m int) []distItem {
	// Sort by distance ascending (cheapest candidate first).
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })

	if len(cands) <= m {
		return cands
	}

	kept := cands[:0]
	for _, c := range cands {
		if len(kept) >= m {
			break
		}
		good := true
		cVec := b.G.VectorAt(c.id)
		for _, k := range kept {
			// dist(c, k) — distance from candidate to an already-kept neighbor.
			// If c is closer to k than to the query (c.dist), drop it.
			if Dist2(cVec, b.G.VectorAt(k.id)) < c.dist {
				good = false
				break
			}
		}
		if good {
			kept = append(kept, c)
		}
	}
	return kept
}

// connectBidirectional adds edges (newID <-> n) at layer L for each n in
// selected, then prunes any neighbor whose connection list now exceeds the
// per-layer capacity using the same heuristic.
func (b *Builder) connectBidirectional(newID uint32, selected []distItem, L int) {
	// Edge: newID -> selected (count up to mLayer)
	for _, s := range selected {
		b.addEdge(newID, s.id, L)
	}
	// Edge: each selected -> newID (then prune over-capacity)
	for _, s := range selected {
		b.addEdge(s.id, newID, L)
		b.pruneIfOverCapacity(s.id, L)
	}
}

// addEdge adds a directed edge from src to dst at layer L.
//
// Layer 0: writes into Conn0. If src is already at the M0 cap, the new edge
// is merged with the existing M0 via the heuristic selector — the new edge
// MUST be admitted so that newly-inserted nodes are reachable from their
// neighbors. A previously-kept edge may be evicted in the process.
//
// Upper layers: writes into the per-layer map (unbounded during build).
// Capacity is enforced afterwards by pruneIfOverCapacity.
func (b *Builder) addEdge(src, dst uint32, L int) {
	if L == 0 {
		cnt := b.G.Conn0Cnt[src]
		if int(cnt) < b.G.M0 {
			// Room available — append.
			b.G.Conn0[int(src)*b.G.M0+int(cnt)] = int32(dst)
			b.G.Conn0Cnt[src] = cnt + 1
			return
		}
		// At capacity: merge-and-prune.
		b.relinkLayer0(src, dst)
		return
	}
	b.ensureLayers(L)
	b.upperConns[L-1][src] = append(b.upperConns[L-1][src], int32(dst))
}

// relinkLayer0 is the over-capacity path of addEdge for layer 0. It rebuilds
// src's layer-0 connection list as the heuristic-selected best M0 of
// (existing M0 ∪ {newDst}). Always leaves Conn0Cnt[src] == M0.
func (b *Builder) relinkLayer0(src, newDst uint32) {
	srcVec := b.G.VectorAt(src)
	off := int(src) * b.G.M0
	cnt := int(b.G.Conn0Cnt[src])

	// Avoid duplicate edges: if newDst is already among current neighbors,
	// nothing to do.
	for i := range cnt {
		if b.G.Conn0[off+i] == int32(newDst) {
			return
		}
	}

	cands := make([]distItem, 0, cnt+1)
	for i := range cnt {
		nb := uint32(b.G.Conn0[off+i])
		cands = append(cands, distItem{nb, Dist2(srcVec, b.G.VectorAt(nb))})
	}
	cands = append(cands, distItem{newDst, Dist2(srcVec, b.G.VectorAt(newDst))})

	kept := b.selectNeighborsHeuristic(cands, b.G.M0)
	for i, k := range kept {
		b.G.Conn0[off+i] = int32(k.id)
	}
	for i := len(kept); i < b.G.M0; i++ {
		b.G.Conn0[off+i] = -1
	}
	b.G.Conn0Cnt[src] = uint8(len(kept))
}

// pruneIfOverCapacity trims the neighbor list of nodeID on layer L back to
// the per-layer capacity, keeping neighbors selected by the heuristic.
func (b *Builder) pruneIfOverCapacity(nodeID uint32, L int) {
	limit := b.G.M
	if L == 0 {
		limit = b.G.M0
	}

	var current []int32
	if L == 0 {
		cnt := int(b.G.Conn0Cnt[nodeID])
		off := int(nodeID) * b.G.M0
		current = b.G.Conn0[off : off+cnt]
	} else {
		if L-1 >= len(b.upperConns) {
			return
		}
		current = b.upperConns[L-1][nodeID]
	}
	if len(current) <= limit {
		return
	}

	// Compute distance from nodeID to each current neighbor using two-uint8
	// Dist2 (no int16 view needed).
	src := b.G.VectorAt(nodeID)
	cands := make([]distItem, len(current))
	for i, n := range current {
		cands[i] = distItem{uint32(n), Dist2(src, b.G.VectorAt(uint32(n)))}
	}
	kept := b.selectNeighborsHeuristic(cands, limit)

	// Write back.
	if L == 0 {
		off := int(nodeID) * b.G.M0
		for i, k := range kept {
			b.G.Conn0[off+i] = int32(k.id)
		}
		// Fill remaining slots with -1 sentinels.
		for i := len(kept); i < b.G.M0; i++ {
			b.G.Conn0[off+i] = -1
		}
		b.G.Conn0Cnt[nodeID] = uint8(len(kept))
		return
	}
	out := make([]int32, len(kept))
	for i, k := range kept {
		out[i] = int32(k.id)
	}
	b.upperConns[L-1][nodeID] = out
}

// Finalize flattens the build-time upperConns map-of-maps into the Graph's
// runtime CSR-ish layout. After Finalize, the Builder's mutable state is no
// longer needed and the Graph alone is sufficient for Save / Query.
func (b *Builder) Finalize() {
	maxL := int(b.G.MaxLayer)
	if maxL == 0 {
		// No upper layers. Set up empty slices and return.
		b.G.UpperOff = []int32{0}
		b.G.UpperNodes = nil
		b.G.UpperCnt = nil
		b.G.UpperConn = nil
		return
	}

	// Per-layer node ID lists, sorted ascending.
	layerNodes := make([][]uint32, maxL) // layerNodes[L-1]
	for L := 1; L <= maxL; L++ {
		if L-1 >= len(b.upperConns) {
			continue
		}
		m := b.upperConns[L-1]
		ids := make([]uint32, 0, len(m))
		for id := range m {
			ids = append(ids, id)
		}
		sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
		layerNodes[L-1] = ids
	}

	// CSR offsets: UpperOff[L] = cumulative node count through layer L.
	off := make([]int32, maxL+1)
	off[0] = 0
	total := 0
	for L := 1; L <= maxL; L++ {
		total += len(layerNodes[L-1])
		off[L] = int32(total)
	}

	nodes := make([]int32, total)
	cnts := make([]uint8, total)
	conns := make([]int32, total*b.G.M)
	for i := range conns {
		conns[i] = -1
	}

	slot := 0
	for L := 1; L <= maxL; L++ {
		ids := layerNodes[L-1]
		layerMap := b.upperConns[L-1]
		for _, id := range ids {
			nodes[slot] = int32(id)
			nb := layerMap[id]
			n := len(nb)
			if n > b.G.M {
				n = b.G.M
			}
			cnts[slot] = uint8(n)
			copy(conns[slot*b.G.M:slot*b.G.M+n], nb[:n])
			slot++
		}
	}

	b.G.UpperOff = off
	b.G.UpperNodes = nodes
	b.G.UpperCnt = cnts
	b.G.UpperConn = conns
}

// ---------------------------------------------------------------------------
// Heap support for build-time search.
// ---------------------------------------------------------------------------

type distItem struct {
	id   uint32
	dist int32
}

type minHeap []distItem

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)         { *h = append(*h, x.(distItem)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

type maxHeap []distItem

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)         { *h = append(*h, x.(distItem)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
