// Package api is the fraud-score HTTP handler. The hot path here matches the
// invariants from project memory:
//
//   - zero allocation in steady state — all per-request buffers come from a
//     sync.Pool of workBuf objects sized for the worst case
//   - never return 5xx or timeout — JSON parse failures and any other error
//     fall through to a precomputed "definitely fraud" response so the
//     client always gets a well-formed answer fast
//
// The handler is split out of cmd/api so it can be tested with an in-memory
// graph (cmd/main is the wiring layer — config loading + ListenAndServe).
package api

import (
	"sync"

	"github.com/valyala/fasthttp"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
	"github.com/nrlacerda/fraud-detection-api/internal/hnsw"
	"github.com/nrlacerda/fraud-detection-api/internal/vectorize"
)

// Responses are precomputed and chosen by the fraud count (0–5 hits in the
// 5-NN result). Picking by index avoids any string formatting on the hot
// path. fraud_score is fraudCount/5; approved is fraud_score < 0.6
// (i.e., fraudCount < 3).
var responseTable = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1}`),
}

// respFallback is what we return on any parse error or unexpected failure.
// Default-to-fraud honors the "never let a malformed request glide through"
// requirement while still satisfying the never-5xx rule.
var respFallback = responseTable[5]

// readyBody is the body served for GET /ready.
var readyBody = []byte("ok")

// jsonContentType is the body content type, kept as []byte so fasthttp's
// SetContentTypeBytes doesn't need to copy from a string each call.
var jsonContentType = []byte("application/json")

// workBuf is the per-request scratch space. Pooled. Sized once at slot
// creation; reused unchanged across requests.
type workBuf struct {
	req  vectorize.Request
	vec  [config.VectorDim]float32
	qvec [config.VectorDim]int16
	out  [5]uint32
	slot *hnsw.VisitSlot
}

// Handler is the constructed /fraud-score + /ready router. Build once at
// startup, then pass to fasthttp.
type Handler struct {
	g    *hnsw.Graph
	pool sync.Pool
}

// NewHandler returns a Handler that serves /ready and /fraud-score against g.
// Workers (workBuf) are lazily allocated by the pool — there is one logical
// worker per concurrent in-flight request.
func NewHandler(g *hnsw.Graph) *Handler {
	h := &Handler{g: g}
	h.pool.New = func() any {
		return &workBuf{
			slot: hnsw.NewVisitSlot(g.N),
		}
	}
	return h
}

// Handle implements fasthttp.RequestHandler.
func (h *Handler) Handle(ctx *fasthttp.RequestCtx) {
	path := ctx.Path()
	// Match by length first to avoid a full byte compare on the common case.
	if len(path) == 6 && string(path) == "/ready" {
		if ctx.IsGet() {
			ctx.SetStatusCode(fasthttp.StatusOK)
			ctx.SetContentTypeBytes(jsonContentType)
			ctx.SetBody(readyBody)
			return
		}
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}
	if len(path) == 12 && string(path) == "/fraud-score" {
		if ctx.IsPost() {
			h.handleScore(ctx)
			return
		}
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}
	ctx.SetStatusCode(fasthttp.StatusNotFound)
}

// handleScore is the /fraud-score path. Never errors out to the client; any
// parse / decode failure falls through to respFallback so the response is
// always a well-formed JSON body with a deterministic latency profile.
func (h *Handler) handleScore(ctx *fasthttp.RequestCtx) {
	w := h.pool.Get().(*workBuf)
	defer h.pool.Put(w)

	w.req.Reset()

	body := ctx.PostBody()
	if err := parseStrictRequest(body, &w.req); err != nil {
		writeJSON(ctx, respFallback)
		return
	}

	vectorize.Vectorize(&w.req, &w.vec)
	hnsw.QuantizeQuery(&w.vec, &w.qvec)
	h.g.QueryFast5(&w.qvec, w.slot, &w.out)

	fraudCount := 0
	for _, id := range w.out {
		if h.g.IsFraud(id) {
			fraudCount++
		}
	}
	writeJSON(ctx, responseTable[fraudCount])
}

func writeJSON(ctx *fasthttp.RequestCtx, body []byte) {
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentTypeBytes(jsonContentType)
	ctx.SetBody(body)
}
