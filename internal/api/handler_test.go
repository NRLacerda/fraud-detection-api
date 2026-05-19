package api

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"

	"github.com/valyala/fasthttp"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
	"github.com/nrlacerda/fraud-detection-api/internal/hnsw"
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
