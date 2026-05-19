// build-index converts resources/references.json into hnsw.bin.
//
// Run locally (not on the submission server):
//
//	go run ./cmd/build-index \
//	    -in resources/references.json \
//	    -out resources/hnsw.bin
//
// The output file is baked into the Docker image and mmap'd at runtime.
package main

import (
	"flag"
	"log"
	"math/rand"
	"os"
	"runtime"
	"time"

	"github.com/nrlacerda/fraud-detection-api/internal/buildindex"
	"github.com/nrlacerda/fraud-detection-api/internal/hnsw"
)

func main() {
	in := flag.String("in", "resources/references.json", "input references JSON")
	out := flag.String("out", "resources/hnsw.bin", "output index path")
	n := flag.Uint("n", 3_000_000, "expected record count (Builder pre-sizes for this)")
	m := flag.Int("m", 6, "M (upper-layer max connections)")
	m0 := flag.Int("m0", 12, "M0 (layer-0 max connections)")
	efC := flag.Int("efc", 200, "efConstruction")
	seed := flag.Int64("seed", 42, "build RNG seed")
	recallQ := flag.Int("recall-queries", 200, "recall sanity queries (0 = skip)")
	recallMin := flag.Float64("recall-min", 0.95, "fail if mean recall@5 below this")
	flag.Parse()

	log.Printf("build-index: in=%s out=%s N=%d M=%d M0=%d efC=%d seed=%d goroutines=%d",
		*in, *out, *n, *m, *m0, *efC, *seed, runtime.NumCPU())

	f, err := os.Open(*in)
	if err != nil {
		log.Fatalf("open input: %v", err)
	}
	defer f.Close()

	start := time.Now()
	b := hnsw.NewBuilder(uint32(*n), *m, *m0, *efC, *seed)

	const progressEvery = uint32(50_000)
	inserted, err := buildindex.BuildIndex(f, b, func(done uint32) {
		elapsed := time.Since(start)
		rate := float64(done) / elapsed.Seconds()
		log.Printf("  inserted %d  (%.0f/s, %s elapsed)",
			done, rate, elapsed.Round(time.Second))
	}, progressEvery)
	if err != nil {
		log.Fatalf("BuildIndex: %v", err)
	}
	if inserted != uint32(*n) {
		log.Fatalf("record count mismatch: inserted %d, expected %d; pass -n with the exact references count", inserted, *n)
	}

	log.Printf("insert done in %s; finalizing...", time.Since(start).Round(time.Second))
	b.Finalize()

	log.Printf("reordering for cache locality...")
	b.G.Reorder()

	log.Printf("saving to %s...", *out)
	nBytes, err := b.G.Save(*out)
	if err != nil {
		log.Fatalf("Save: %v", err)
	}
	log.Printf("wrote %d bytes (%.1f MB) in %s total",
		nBytes, float64(nBytes)/1e6, time.Since(start).Round(time.Second))

	if *recallQ <= 0 {
		log.Printf("done (recall check skipped).")
		return
	}

	log.Printf("re-loading hnsw.bin for recall sanity...")
	loaded, err := hnsw.Load(*out)
	if err != nil {
		log.Fatalf("Load round-trip: %v", err)
	}

	log.Printf("measuring recall@5 over %d sampled queries...", *recallQ)
	rng := rand.New(rand.NewSource(*seed + 1))
	queries := make([]uint32, *recallQ)
	for i := range queries {
		queries[i] = uint32(rng.Intn(int(loaded.N)))
	}
	slot := hnsw.NewVisitSlot(loaded.N)

	rs := time.Now()
	mean := buildindex.MeasureRecallAt5(loaded, slot, queries)
	log.Printf("recall@5 = %.4f over %d queries (took %s)",
		mean, len(queries), time.Since(rs).Round(time.Second))

	if mean < *recallMin {
		log.Fatalf("recall too low: %.4f < %.4f — refusing to ship this index",
			mean, *recallMin)
	}
	log.Printf("done. recall@5=%.4f total=%s",
		mean, time.Since(start).Round(time.Second))
}
