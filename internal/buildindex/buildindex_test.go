package buildindex

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
	"github.com/nrlacerda/fraud-detection-api/internal/hnsw"
)

// makeReferencesJSON synthesizes a JSON array of N reference records with
// random in-range values. Returns the encoded bytes and the parsed records
// so the test can cross-check decode results.
func makeReferencesJSON(t testing.TB, n int, seed int64) ([]byte, []Record) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	recs := make([]Record, n)
	for i := range recs {
		var v [config.VectorDim]float32
		for d := range config.VectorDim {
			if d == 5 || d == 6 {
				// 25% null sentinel (-1), else [0,1].
				if rng.Intn(4) == 0 {
					v[d] = -1
				} else {
					v[d] = rng.Float32()
				}
				continue
			}
			v[d] = rng.Float32()
		}
		label := "legit"
		if rng.Intn(5) == 0 {
			label = "fraud"
		}
		recs[i] = Record{Vector: v, Label: label}
	}
	buf, err := json.Marshal(recs)
	if err != nil {
		t.Fatalf("marshal references: %v", err)
	}
	return buf, recs
}

// TestStreamRecords_DecodesAllInOrder confirms the streaming decoder yields
// each record exactly once in source order with intact values.
func TestStreamRecords_DecodesAllInOrder(t *testing.T) {
	const N = 50
	raw, want := makeReferencesJSON(t, N, 1)

	var got []Record
	err := StreamRecords(bytes.NewReader(raw), func(idx uint32, rec *Record) error {
		if int(idx) != len(got) {
			t.Errorf("idx %d at position %d", idx, len(got))
		}
		got = append(got, *rec)
		return nil
	})
	if err != nil {
		t.Fatalf("StreamRecords: %v", err)
	}
	if len(got) != N {
		t.Fatalf("got %d records, want %d", len(got), N)
	}
	for i := range got {
		if got[i].Label != want[i].Label {
			t.Errorf("record %d label %q vs %q", i, got[i].Label, want[i].Label)
		}
		if got[i].Vector != want[i].Vector {
			t.Errorf("record %d vector mismatch", i)
		}
	}
}

// TestStreamRecords_RejectsNonArray returns an error for malformed input.
func TestStreamRecords_RejectsNonArray(t *testing.T) {
	err := StreamRecords(strings.NewReader(`{"oops":1}`), func(uint32, *Record) error {
		t.Fatal("fn should not be called for non-array input")
		return nil
	})
	if err == nil {
		t.Fatal("expected error on non-array input, got nil")
	}
}

// TestBuildIndex_EndToEnd builds a small index from JSON, saves it, loads it
// back, and runs a recall sanity check — exercising the entire offline path
// (decode → quantize → insert → finalize → reorder → save → load → query).
func TestBuildIndex_EndToEnd(t *testing.T) {
	const N = 600
	raw, _ := makeReferencesJSON(t, N, 11)

	b := hnsw.NewBuilder(N, 4, 8, 200, 17)
	inserted, err := BuildIndex(bytes.NewReader(raw), b, nil, 0)
	if err != nil {
		t.Fatalf("BuildIndex: %v", err)
	}
	if inserted != N {
		t.Fatalf("inserted %d records, want %d", inserted, N)
	}
	b.Finalize()
	b.G.Reorder()

	out := filepath.Join(t.TempDir(), "test.bin")
	if _, err := b.G.Save(out); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := hnsw.Load(out)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.N != N {
		t.Errorf("loaded N=%d, want %d", loaded.N, N)
	}

	// Recall sanity. Real (correlated) data should easily clear 0.80 even on
	// our small synthetic set — and querying with a stored vector gives a
	// guaranteed self-hit (1/5 free).
	slot := hnsw.NewVisitSlot(loaded.N)
	rng := rand.New(rand.NewSource(99))
	const Q = 60
	queries := make([]uint32, Q)
	for i := range queries {
		queries[i] = uint32(rng.Intn(int(loaded.N)))
	}
	got := MeasureRecallAt5(loaded, slot, queries)
	if got < 0.80 {
		t.Errorf("recall@5 too low: %.3f", got)
	}
}

func TestBuildIndex_RejectsTooManyRecords(t *testing.T) {
	const records = 10
	raw, _ := makeReferencesJSON(t, records, 21)
	b := hnsw.NewBuilder(records-1, 4, 8, 50, 1)

	inserted, err := BuildIndex(bytes.NewReader(raw), b, nil, 0)
	if err == nil {
		t.Fatal("expected capacity error, got nil")
	}
	if inserted != records-1 {
		t.Fatalf("inserted %d records before error, want %d", inserted, records-1)
	}
}

// TestBruteTop5_FindsSelf querying with a stored vector must always rank
// that node first (distance 0).
func TestBruteTop5_FindsSelf(t *testing.T) {
	const N = 100
	b := hnsw.NewBuilder(N, 4, 8, 50, 1)
	rng := rand.New(rand.NewSource(2))
	for id := range uint32(N) {
		var v [config.VectorDim]uint8
		for d := range config.VectorDim {
			v[d] = uint8(rng.Intn(256))
		}
		b.Insert(id, &v, hnsw.LabelLegit)
	}
	b.Finalize()

	for _, qid := range []uint32{0, 17, 50, 99} {
		v := b.G.VectorAt(qid)
		var q [config.VectorDim]int16
		for i := range config.VectorDim {
			q[i] = int16(v[i])
		}
		got := bruteTop5(b.G, &q)
		if got[0] != qid {
			t.Errorf("bruteTop5 for qid=%d returned %v, expected self first", qid, got)
		}
	}
}
