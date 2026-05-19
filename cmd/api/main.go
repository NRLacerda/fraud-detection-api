// api is the fraud-score HTTP server. It mmap-loads hnsw.bin and serves two
// endpoints on the configured listen address:
//
//	GET  /ready         → "ok"
//	POST /fraud-score   → { approved, fraud_score }
//
// Configuration is environment-driven so the same binary runs in dev and
// inside the production container:
//
//	INDEX_PATH        path to hnsw.bin              (default resources/hnsw.bin)
//	LISTEN_ADDR       fasthttp listen address       (default :8080)
//	GOMAXPROCS        runtime parallelism           (default 2)
//
// nginx (port 9999) round-robins between two instances of this binary at
// :8080. The two instances share the hnsw.bin file; on Linux the kernel page
// cache lets both mmap views of the same inode share physical pages.
package main

import (
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"time"

	"github.com/valyala/fasthttp"

	"github.com/nrlacerda/fraud-detection-api/internal/api"
	"github.com/nrlacerda/fraud-detection-api/internal/hnsw"
)

func main() {
	indexPath := envOr("INDEX_PATH", "resources/hnsw.bin")
	listenAddr := envOr("LISTEN_ADDR", ":8080")

	if v := os.Getenv("GOMAXPROCS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			runtime.GOMAXPROCS(n)
		}
	} else {
		runtime.GOMAXPROCS(2)
	}

	// Loosen GC: the hot path is allocation-free by design, so the only
	// allocations come from JSON decoding (small) and pool top-ups. Raising
	// GOGC reduces collection frequency and trims tail latency.
	debug.SetGCPercent(400)

	log.Printf("loading index %s ...", indexPath)
	loadStart := time.Now()
	g, err := hnsw.LoadMmap(indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	log.Printf("loaded N=%d M=%d M0=%d maxLayer=%d entry=%d in %s",
		g.N, g.M, g.M0, g.MaxLayer, g.EntryPoint,
		time.Since(loadStart).Round(time.Millisecond))

	warmPages(g)

	h := api.NewHandler(g)
	server := &fasthttp.Server{
		Handler:                       h.Handle,
		Name:                          "fraud-detection-api",
		ReadTimeout:                   2 * time.Second,
		WriteTimeout:                  2 * time.Second,
		IdleTimeout:                   30 * time.Second,
		MaxRequestBodySize:            16 << 10,
		Concurrency:                   1024,
		DisableHeaderNamesNormalizing: true,
		NoDefaultServerHeader:         true,
		NoDefaultContentType:          true,
		// fasthttp parses date strings into the connection's Date header;
		// disabling it shaves a small per-conn cost.
		DisableKeepalive: false,
	}

	log.Printf("listening on %s (GOMAXPROCS=%d)", listenAddr, runtime.GOMAXPROCS(0))
	if err := server.ListenAndServe(listenAddr); err != nil {
		log.Fatalf("ListenAndServe: %v", err)
	}
}

func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

// warmPages touches every byte of the loaded graph so that all mmap pages are
// resident before we accept traffic. First-request latency then looks like
// steady-state latency.
func warmPages(g *hnsw.Graph) {
	start := time.Now()
	var checksum uint64
	for _, b := range g.Vectors {
		checksum += uint64(b)
	}
	for _, b := range g.Conn0Cnt {
		checksum += uint64(b)
	}
	for _, v := range g.Conn0 {
		checksum += uint64(uint32(v))
	}
	for _, b := range g.Labels {
		checksum += uint64(b)
	}
	for _, v := range g.UpperOff {
		checksum += uint64(uint32(v))
	}
	for _, v := range g.UpperNodes {
		checksum += uint64(uint32(v))
	}
	for _, b := range g.UpperCnt {
		checksum += uint64(b)
	}
	for _, v := range g.UpperConn {
		checksum += uint64(uint32(v))
	}
	log.Printf("warmed pages (checksum %#x) in %s",
		checksum, time.Since(start).Round(time.Millisecond))
}
