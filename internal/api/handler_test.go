package api

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/valyala/fasthttp"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
	"github.com/nrlacerda/fraud-detection-api/internal/hnsw"
	"github.com/nrlacerda/fraud-detection-api/internal/vectorize"
)

// buildTinyGraph creates an in-memory HNSW graph with N labeled nodes; every
// 4th node is labeled fraud. Used to exercise both branches of fraud_score.
func buildTinyGraph(tb testing.TB, N uint32, seed int64) *hnsw.Graph {
	tb.Helper()
	b := hnsw.NewBuilder(N, 4, 8, 50, seed)
	rng := rand.New(rand.NewSource(seed + 1))
	for id := range N {
		var v [config.VectorDim]uint8
		for d := range config.VectorDim {
			v[d] = uint8(rng.Intn(256))
		}
		label := hnsw.LabelLegit
		if id%4 == 0 {
			label = hnsw.LabelFraud
		}
		b.Insert(id, &v, label)
	}
	b.Finalize()
	return b.G
}

func loadProductionGraphForBenchmark(b *testing.B) *hnsw.Graph {
	b.Helper()

	path := filepath.Join("..", "..", "resources", "hnsw.bin")
	if _, err := os.Stat(path); err != nil {
		b.Skipf("production index not available at %s: %v", path, err)
	}
	g, err := hnsw.LoadMmap(path)
	if err != nil {
		b.Fatalf("load production index: %v", err)
	}
	return g
}

// callHandler routes one request through h via an in-memory RequestCtx.
// Returns the captured status code and response body bytes.
func callHandler(t testing.TB, h *Handler, method, uri string, body []byte) (int, []byte) {
	t.Helper()
	var ctx fasthttp.RequestCtx
	ctx.Request.SetRequestURI(uri)
	ctx.Request.Header.SetMethod(method)
	if body != nil {
		ctx.Request.SetBody(body)
	}
	h.Handle(&ctx)
	return ctx.Response.StatusCode(), append([]byte(nil), ctx.Response.Body()...)
}

const samplePayload = `{
  "id": "tx-test-1",
  "transaction": {
    "amount": 384.88,
    "installments": 3,
    "requested_at": "2026-03-11T20:23:35Z"
  },
  "customer": {
    "avg_amount": 769.76,
    "tx_count_24h": 3,
    "known_merchants": ["MERC-001", "MERC-009"]
  },
  "merchant": {
    "id": "MERC-001",
    "mcc": "5912",
    "avg_amount": 298.95
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 13.71
  },
  "last_transaction": {
    "timestamp": "2026-03-11T14:58:35Z",
    "km_from_current": 18.86
  }
}`

const samplePayloadNullLast = `{
  "id": "tx-test-2",
  "transaction": {
    "amount": 100.0,
    "installments": 1,
    "requested_at": "2026-03-11T20:23:35Z"
  },
  "customer": {
    "avg_amount": 200.0,
    "tx_count_24h": 1,
    "known_merchants": []
  },
  "merchant": {
    "id": "MERC-X",
    "mcc": "5411",
    "avg_amount": 100.0
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 2.0
  },
  "last_transaction": null
}`

func TestHandler_Ready(t *testing.T) {
	g := buildTinyGraph(t, 200, 1)
	h := NewHandler(g)
	code, body := callHandler(t, h, fasthttp.MethodGet, "/ready", nil)
	if code != 200 {
		t.Errorf("status %d, want 200", code)
	}
	if string(body) != "ok" {
		t.Errorf("body %q, want %q", body, "ok")
	}
}

func TestHandler_NotFound(t *testing.T) {
	g := buildTinyGraph(t, 200, 1)
	h := NewHandler(g)
	code, _ := callHandler(t, h, fasthttp.MethodGet, "/whatever", nil)
	if code != fasthttp.StatusNotFound {
		t.Errorf("status %d, want 404", code)
	}
}

func TestHandler_WrongMethod(t *testing.T) {
	g := buildTinyGraph(t, 200, 1)
	h := NewHandler(g)
	code, _ := callHandler(t, h, fasthttp.MethodGet, "/fraud-score", []byte(samplePayload))
	if code != fasthttp.StatusMethodNotAllowed {
		t.Errorf("status %d, want 405", code)
	}
}

func TestHandler_FraudScore_ParsesAndReturnsResponse(t *testing.T) {
	g := buildTinyGraph(t, 200, 1)
	h := NewHandler(g)
	code, body := callHandler(t, h, fasthttp.MethodPost, "/fraud-score", []byte(samplePayload))
	if code != 200 {
		t.Fatalf("status %d, want 200", code)
	}
	s := string(body)
	if !strings.Contains(s, `"approved":`) || !strings.Contains(s, `"fraud_score":`) {
		t.Errorf("body missing required keys: %q", s)
	}
}

func TestHandler_FraudScore_NullLastTransaction(t *testing.T) {
	g := buildTinyGraph(t, 200, 1)
	h := NewHandler(g)
	code, body := callHandler(t, h, fasthttp.MethodPost, "/fraud-score", []byte(samplePayloadNullLast))
	if code != 200 {
		t.Fatalf("status %d, want 200", code)
	}
	if !bytes.Contains(body, []byte(`"fraud_score":`)) {
		t.Errorf("missing fraud_score: %q", body)
	}
}

func TestHandler_FraudScore_BadJSONFallsBackToFraud(t *testing.T) {
	g := buildTinyGraph(t, 200, 1)
	h := NewHandler(g)
	code, body := callHandler(t, h, fasthttp.MethodPost, "/fraud-score", []byte(`{nope`))
	if code != 200 {
		t.Fatalf("status %d, want 200 (never 5xx)", code)
	}
	// Default-to-fraud: should equal responseTable[5].
	if !bytes.Equal(body, respFallback) {
		t.Errorf("bad-JSON body = %q, want fallback %q", body, respFallback)
	}
}

func TestParseStrictRequest_MatchesSonicVectorization(t *testing.T) {
	for name, payload := range map[string]string{
		"with-last-transaction": samplePayload,
		"null-last-transaction": samplePayloadNullLast,
	} {
		t.Run(name, func(t *testing.T) {
			var strictReq vectorize.Request
			if err := parseStrictRequest([]byte(payload), &strictReq); err != nil {
				t.Fatalf("parseStrictRequest: %v", err)
			}

			var sonicReq vectorize.Request
			if err := sonic.Unmarshal([]byte(payload), &sonicReq); err != nil {
				t.Fatalf("sonic.Unmarshal: %v", err)
			}
			sonicReq.HasLastTransaction = !sonicReq.LastTransaction.Timestamp.IsZero()

			var strictVec, sonicVec [config.VectorDim]float32
			vectorize.Vectorize(&strictReq, &strictVec)
			vectorize.Vectorize(&sonicReq, &sonicVec)
			if strictVec != sonicVec {
				t.Fatalf("vector mismatch\nstrict=%v\nsonic =%v", strictVec, sonicVec)
			}
		})
	}
}

func TestParseStrictRequest_ChallengeDatasetVectors(t *testing.T) {
	if os.Getenv("FRAUD_VALIDATE_TEST_DATA") != "1" {
		t.Skip("set FRAUD_VALIDATE_TEST_DATA=1 to validate strict parser against test/test-data.json")
	}

	path := filepath.Join("..", "..", "test", "test-data.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("test dataset not available at %s: %v", path, err)
	}

	var dataset struct {
		Entries []struct {
			Request json.RawMessage `json:"request"`
		} `json:"entries"`
	}
	if err := sonic.Unmarshal(raw, &dataset); err != nil {
		t.Fatalf("decode test-data.json: %v", err)
	}

	for i, entry := range dataset.Entries {
		var strictReq vectorize.Request
		if err := parseStrictRequest(entry.Request, &strictReq); err != nil {
			t.Fatalf("entry %d strict parse: %v", i, err)
		}

		var sonicReq vectorize.Request
		if err := sonic.Unmarshal(entry.Request, &sonicReq); err != nil {
			t.Fatalf("entry %d sonic parse: %v", i, err)
		}
		sonicReq.HasLastTransaction = !sonicReq.LastTransaction.Timestamp.IsZero()

		var strictVec, sonicVec [config.VectorDim]float32
		vectorize.Vectorize(&strictReq, &strictVec)
		vectorize.Vectorize(&sonicReq, &sonicVec)
		if strictVec != sonicVec {
			t.Fatalf("entry %d vector mismatch\nstrict=%v\nsonic =%v", i, strictVec, sonicVec)
		}
	}
}

// TestHandler_FraudScore_ResponseTableShape checks that the precomputed
// response strings are exactly what we promise: 6 variants keyed by fraud
// count, with approved flipping at count==3 (score 0.6).
func TestHandler_FraudScore_ResponseTableShape(t *testing.T) {
	for count := range 6 {
		b := responseTable[count]
		approved := count < 3
		if approved && !bytes.Contains(b, []byte(`"approved":true`)) {
			t.Errorf("count=%d: expected approved=true, got %q", count, b)
		}
		if !approved && !bytes.Contains(b, []byte(`"approved":false`)) {
			t.Errorf("count=%d: expected approved=false, got %q", count, b)
		}
	}
}

// TestHandler_FraudScore_LowAllocOnHotPath checks the per-request allocation
// count for /fraud-score. Sonic does allocate a small number of bytes (mostly
// for parsed strings — Merchant.ID, KnownMerchants slice growth), but the
// path must NOT allocate per-call working buffers (those come from the pool).
//
// Threshold note: zero allocs is unattainable while we use a generic JSON
// decoder; the goal is "single-digit allocs/op" so the pool and pre-sized
// buffers are doing their job. If this regresses sharply (e.g., > 20), the
// pool or response path likely broke.
func TestHandler_FraudScore_LowAllocOnHotPath(t *testing.T) {
	g := buildTinyGraph(t, 200, 1)
	h := NewHandler(g)
	body := []byte(samplePayload)

	// Warm the pool.
	for i := 0; i < 10; i++ {
		var ctx fasthttp.RequestCtx
		ctx.Request.SetRequestURI("/fraud-score")
		ctx.Request.Header.SetMethod(fasthttp.MethodPost)
		ctx.Request.SetBody(body)
		h.Handle(&ctx)
	}

	allocs := testing.AllocsPerRun(200, func() {
		var ctx fasthttp.RequestCtx
		ctx.Request.SetRequestURI("/fraud-score")
		ctx.Request.Header.SetMethod(fasthttp.MethodPost)
		ctx.Request.SetBody(body)
		h.Handle(&ctx)
	})
	// RequestCtx + Request setup themselves allocate; what matters is that
	// per-request overhead doesn't grow unboundedly. 30 is generous but
	// catches catastrophic regressions.
	if allocs > 30 {
		t.Errorf("allocs/op = %.1f, expected <= 30 (vectorize+query path stayed pool-backed?)", allocs)
	}
}

func BenchmarkHandlerFraudScoreProductionIndex(b *testing.B) {
	g := loadProductionGraphForBenchmark(b)
	h := NewHandler(g)
	body := []byte(samplePayload)

	// Warm the pool and mmap pages touched by this query shape.
	for i := 0; i < 100; i++ {
		var ctx fasthttp.RequestCtx
		ctx.Request.SetRequestURI("/fraud-score")
		ctx.Request.Header.SetMethod(fasthttp.MethodPost)
		ctx.Request.SetBody(body)
		h.Handle(&ctx)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var ctx fasthttp.RequestCtx
		ctx.Request.SetRequestURI("/fraud-score")
		ctx.Request.Header.SetMethod(fasthttp.MethodPost)
		ctx.Request.SetBody(body)
		h.Handle(&ctx)
	}
}

func BenchmarkFraudScoreStagesProductionIndex(b *testing.B) {
	g := loadProductionGraphForBenchmark(b)
	body := []byte(samplePayload)
	slot := hnsw.NewVisitSlot(g.N)

	var req vectorize.Request
	var vec [config.VectorDim]float32
	var qvec [config.VectorDim]int16
	var out [5]uint32
	var ctx fasthttp.RequestCtx
	ctx.Request.SetRequestURI("/fraud-score")
	ctx.Request.Header.SetMethod(fasthttp.MethodPost)

	var unmarshalDur, vectorizeDur, quantizeDur, queryDur, classifyDur, writeDur time.Duration

	// Warm code paths and pages before timing.
	for i := 0; i < 100; i++ {
		req.Reset()
		_ = parseStrictRequest(body, &req)
		vectorize.Vectorize(&req, &vec)
		hnsw.QuantizeQuery(&vec, &qvec)
		g.QueryFast5(&qvec, slot, &out)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req.Reset()

		start := time.Now()
		if err := parseStrictRequest(body, &req); err != nil {
			b.Fatal(err)
		}
		unmarshalDur += time.Since(start)

		start = time.Now()
		vectorize.Vectorize(&req, &vec)
		vectorizeDur += time.Since(start)

		start = time.Now()
		hnsw.QuantizeQuery(&vec, &qvec)
		quantizeDur += time.Since(start)

		start = time.Now()
		g.QueryFast5(&qvec, slot, &out)
		queryDur += time.Since(start)

		start = time.Now()
		fraudCount := 0
		for _, id := range out {
			if g.IsFraud(id) {
				fraudCount++
			}
		}
		classifyDur += time.Since(start)

		start = time.Now()
		ctx.Response.Reset()
		writeJSON(&ctx, responseTable[fraudCount])
		writeDur += time.Since(start)
	}

	b.StopTimer()
	reportStageMetric(b, "unmarshal", unmarshalDur)
	reportStageMetric(b, "vectorize", vectorizeDur)
	reportStageMetric(b, "quantize", quantizeDur)
	reportStageMetric(b, "query", queryDur)
	reportStageMetric(b, "classify", classifyDur)
	reportStageMetric(b, "write", writeDur)
}

func reportStageMetric(b *testing.B, name string, d time.Duration) {
	b.Helper()
	b.ReportMetric(float64(d.Nanoseconds())/float64(b.N), name+"_ns/op")
}
