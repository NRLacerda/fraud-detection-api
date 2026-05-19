// Package buildindex contains the offline pipeline that converts
// resources/references.json into an hnsw.bin file: stream-decode, quantize,
// insert into the HNSW builder, finalize, reorder, save. It also exposes a
// recall-sanity helper used after a build to detect regressions in either
// the build or the query path.
//
// All hot-path code (vectorize, quantize, query) lives elsewhere — this
// package is only ever called from cmd/build-index and from its own tests.
package buildindex

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
	"github.com/nrlacerda/fraud-detection-api/internal/hnsw"
)

// Record is one entry from references.json.
type Record struct {
	Vector [config.VectorDim]float32 `json:"vector"`
	Label  string                    `json:"label"`
}

// StreamRecords decodes a top-level JSON array of records from r, calling fn
// for each record in document order. Returns the first non-nil error from fn
// or from the decoder. Memory cost is O(1) in the number of records — the
// decoder reads incrementally.
func StreamRecords(r io.Reader, fn func(idx uint32, rec *Record) error) error {
	dec := json.NewDecoder(r)
	tok, err := dec.Token()
	if err != nil {
		return fmt.Errorf("read open token: %w", err)
	}
	if d, ok := tok.(json.Delim); !ok || d != '[' {
		return fmt.Errorf("expected top-level JSON array, got %v", tok)
	}

	var idx uint32
	var rec Record
	for dec.More() {
		rec = Record{}
		if err := dec.Decode(&rec); err != nil {
			return fmt.Errorf("decode record %d: %w", idx, err)
		}
		if err := fn(idx, &rec); err != nil {
			return err
		}
		idx++
	}
	return nil
}

// BuildIndex streams records from r, quantizing each vector and inserting
// into builder b. It returns the number of inserted records. progress is
// called every progressEvery records (skipped if progress is nil or
// progressEvery is 0).
//
// Caller is responsible for constructing the Builder with the right N, and
// for calling Finalize / Reorder / Save afterwards.
func BuildIndex(r io.Reader, b *hnsw.Builder, progress func(done uint32), progressEvery uint32) (uint32, error) {
	var inserted uint32
	err := StreamRecords(r, func(idx uint32, rec *Record) error {
		if idx >= b.G.N {
			return fmt.Errorf("record %d exceeds builder capacity %d", idx, b.G.N)
		}

		var qv [config.VectorDim]uint8
		hnsw.QuantizeVector(&rec.Vector, &qv)

		label := hnsw.LabelLegit
		if rec.Label == "fraud" {
			label = hnsw.LabelFraud
		}
		b.Insert(idx, &qv, label)

		if progress != nil && progressEvery > 0 && (idx+1)%progressEvery == 0 {
			progress(idx + 1)
		}
		inserted = idx + 1
		return nil
	})
	return inserted, err
}

// MeasureRecallAt5 queries the graph with the stored vectors of the given
// query node IDs (promoted to int16) and compares the top-5 against
// brute-force ground truth. Returns mean recall@5 in [0, 1].
//
// Note: querying a stored ref against the index includes self (distance 0)
// in both the truth and the approximate result — that's a free hit. A score
// of, say, 0.97 means the approximate algorithm found ~4.85 of the 5 true
// nearest including self.
func MeasureRecallAt5(g *hnsw.Graph, slot *hnsw.VisitSlot, queries []uint32) float64 {
	if len(queries) == 0 {
		return 0
	}
	var totalHits int
	for _, qid := range queries {
		v := g.VectorAt(qid)
		var q [config.VectorDim]int16
		for i := range config.VectorDim {
			q[i] = int16(v[i])
		}

		var got [5]uint32
		g.QueryFast5(&q, slot, &got)
		truth := bruteTop5(g, &q)

		var truthSet [5]uint32
		copy(truthSet[:], truth[:])
		for _, id := range got {
			for _, t := range truthSet {
				if id == t {
					totalHits++
					break
				}
			}
		}
	}
	return float64(totalHits) / float64(len(queries)*5)
}

// bruteTop5 returns the 5 node IDs with smallest Dist(q, ·) in g. O(N) scan
// + linear-scan worst-tracking — faster than sorting the full slice.
func bruteTop5(g *hnsw.Graph, q *[config.VectorDim]int16) [5]uint32 {
	var ids [5]uint32
	var dists [5]int32
	cnt := 0
	var worstD int32
	var worstI int

	for i := range g.N {
		d := hnsw.Dist(q, g.VectorAt(i))
		if cnt < 5 {
			ids[cnt] = i
			dists[cnt] = d
			if cnt == 0 || d > worstD {
				worstD = d
				worstI = cnt
			}
			cnt++
			continue
		}
		if d < worstD {
			ids[worstI] = i
			dists[worstI] = d
			worstD = dists[0]
			worstI = 0
			for k := 1; k < 5; k++ {
				if dists[k] > worstD {
					worstD = dists[k]
					worstI = k
				}
			}
		}
	}
	for i := 1; i < 5; i++ {
		j := i
		for j > 0 && dists[j-1] > dists[j] {
			dists[j-1], dists[j] = dists[j], dists[j-1]
			ids[j-1], ids[j] = ids[j], ids[j-1]
			j--
		}
	}
	return ids
}
