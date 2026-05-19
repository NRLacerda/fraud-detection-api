# Project Plan — Rinha de Backend 2026

Fraud detection API in **Go** using HNSW (Hierarchical Navigable Small World) for approximate vector similarity search with uint8 quantization.

---

## Challenge constraints

| Constraint | Value |
|------------|-------|
| Total CPU across all services | ≤ 1 CPU |
| Total RAM across all services | ≤ 350 MB |
| Minimum topology | 1 load balancer + 2 API instances |
| Load balancer rule | Round-robin only, no business logic |
| Port | 9999 |
| Image platform | linux/amd64 |
| Network mode | bridge (no host, no privileged) |
| Branches | `main` (source), `submission` (docker-compose only) |
| License | MIT |
| Deadline | 2026-06-05T23:59:59-03:00 |

### Test environment

Mac Mini Late 2014, 2.6 GHz Intel Core i5 (x86_64), 8 GB RAM, Ubuntu 24.04.

### Resource allocation

| Service | CPU | Memory |
|---------|-----|--------|
| nginx | 0.10 | 6 MB |
| api1 | 0.45 | 172 MB |
| api2 | 0.45 | 172 MB |
| **Total** | **1.00** | **350 MB** |

### Scoring formula

```
final_score = score_p99 + score_det   (range: -6000 to +6000)

score_p99 = 1000 × log10(1000 / max(p99_ms, 1))
  p99 ≤ 1ms   → +3000 (cap)
  p99 > 2000ms → -3000 (floor)

score_det:
  E = 1×FP + 3×FN + 5×Err
  failure_rate = (FP + FN + Err) / N
  failure_rate > 15% → -3000 (hard cutoff)
```

**Key rule:** Never return HTTP 5xx. On any internal error or timeout approaching 2000ms, return `{"approved": false, "fraud_score": 1.0}` (treat as fraud). A false negative (weight 3) is always better than an HTTP error (weight 5).

---

## API contract

### `GET /ready`

Returns `HTTP 200` once the HNSW index is loaded and the API is ready to serve. Loading takes ~10–30s at startup.

### `POST /fraud-score`

**Request:**
```json
{
  "id": "tx-3576980410",
  "transaction": {
    "amount": 384.88,
    "installments": 3,
    "requested_at": "2026-03-11T20:23:35Z"
  },
  "customer": {
    "avg_amount": 769.76,
    "tx_count_24h": 3,
    "known_merchants": ["MERC-009", "MERC-001", "MERC-001"]
  },
  "merchant": {
    "id": "MERC-001",
    "mcc": "5912",
    "avg_amount": 298.95
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 13.7090520965
  },
  "last_transaction": {
    "timestamp": "2026-03-11T14:58:35Z",
    "km_from_current": 18.8626479774
  }
}
```

`last_transaction` may be `null`.

**Response:**
```json
{ "approved": false, "fraud_score": 1.0 }
```

`fraud_score ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` — fraction of fraud among the 5 nearest neighbors.  
`approved = fraud_score < 0.6`

---

## Vectorization — 14 dimensions

All values normalized to [0.0, 1.0] via `clamp(x) = min(max(x, 0.0), 1.0)`.  
Indices 5 and 6 use sentinel **-1** when `last_transaction` is null — not clamped.

| idx | dimension | formula |
|-----|-----------|---------|
| 0 | amount | `clamp(transaction.amount / 10000)` |
| 1 | installments | `clamp(transaction.installments / 12)` |
| 2 | amount_vs_avg | `clamp((transaction.amount / customer.avg_amount) / 10)` |
| 3 | hour_of_day | `hour(requested_at UTC) / 23` |
| 4 | day_of_week | `weekday(requested_at UTC) / 6` (Mon=0, Sun=6) |
| 5 | minutes_since_last_tx | `clamp(minutes_diff / 1440)` or **-1** if null |
| 6 | km_from_last_tx | `clamp(last_transaction.km_from_current / 1000)` or **-1** if null |
| 7 | km_from_home | `clamp(terminal.km_from_home / 1000)` |
| 8 | tx_count_24h | `clamp(customer.tx_count_24h / 20)` |
| 9 | is_online | `1` if online else `0` |
| 10 | card_present | `1` if card present else `0` |
| 11 | unknown_merchant | `1` if merchant.id NOT in known_merchants else `0` |
| 12 | mcc_risk | lookup table (default `0.5`) |
| 13 | merchant_avg_amount | `clamp(merchant.avg_amount / 10000)` |

### MCC risk table

| MCC | Risk |
|-----|------|
| 5411 | 0.15 |
| 5812 | 0.30 |
| 5912 | 0.20 |
| 5944 | 0.45 |
| 7801 | 0.80 |
| 7802 | 0.75 |
| 7995 | 0.85 |
| 4511 | 0.35 |
| 5311 | 0.25 |
| 5999 | 0.50 |

---

## Implementation

### Approach: HNSW with uint8 quantization

Vectors are quantized from float32 to uint8 at build time using per-dimension min/max scaling. Distance is computed as **squared integer Euclidean** — no sqrt, no dequantization in the hot path. Rankings are identical to true Euclidean distance.

```
dist(q, v) = Σ (q[d] - v[d])²   for d in 0..13
```

Query vectors are quantized to int16 once per request; all neighbor distance calls reuse the pre-quantized form.

### Memory layout (N ≈ 2M nodes, M=6, M0=12)

| Structure | Size |
|-----------|------|
| `vectors [N×14]uint8` | ~28 MB |
| `conn0 [N×12]int32` | ~96 MB |
| `conn0cnt [N]uint8` | ~2 MB |
| `upperConns` (sparse map) | ~30 MB |
| `visitPool` (2 slots × [N]uint32) | ~16 MB |
| misc / runtime | ~15 MB |
| **Total per instance** | **~187 MB** |

### Index build (`cmd/build-index`)

Reads `resources/references.json`, computes per-dimension min/max for quantization, builds the HNSW graph, and saves `resources/hnsw.bin`. This runs once locally; the binary file is baked into the Docker image.

Build parameters: `M=6`, `efConstruction=200`.

### Query path (`QueryFast5`)

Uses `ef=5` (fixed, matching k) with a specialised layer-0 search that avoids heap overhead by tracking the worst slot in a fixed 5-element array. Upper layers use single-best-result traversal (`searchUpperLayer`) with no results heap.

### Default fraud response

Any request that fails to parse, or any internal error, returns `{"approved": false, "fraud_score": 1.0}` immediately — never HTTP 5xx. The 2s `WriteTimeout` on the HTTP server also enforces this as a hard ceiling.

### Concurrency

`GOMAXPROCS=2`. Two concurrent query slots (`visitPool` of size 2), one per API goroutine. Each slot owns a `[N]uint32` visited-mark array with a generation counter — no per-query allocation, no locking.

---

## Libraries

| Library | Role |
|---------|------|
| `github.com/valyala/fasthttp v1.55.0` | HTTP server (zero-alloc request handling) |
| `github.com/bytedance/sonic v1.15.1` | Fast JSON unmarshal for request payloads |
| Standard library only for HNSW, vectorization, and binary serialization | — |

---

## Project structure

```
rinha-de-backend-2026-NRLacerda/
├── cmd/
│   ├── api/main.go          # entry point: load index, start fasthttp server
│   └── build-index/main.go  # offline: references.json → hnsw.bin
├── internal/
│   ├── hnsw/
│   │   ├── hnsw.go          # HNSW graph: insert, query, quantization
│   │   ├── persist.go       # binary serialization (SaveWithLabels / Load)
│   │   ├── reorder.go       # BFS reorder for cache locality
│   │   └── dist_generic.go  # squared integer distance (pure Go)
│   ├── vectorize/
│   │   ├── vectorize.go     # payload → [14]float32
│   │   └── vectorize_test.go
│   └── handler/
│       └── handler.go       # /ready, /fraud-score handlers
├── resources/
│   └── hnsw.bin             # pre-built index baked into Docker image (gitignored)
├── Dockerfile               # multi-stage: builder (cross-compile amd64) + debian:slim
├── docker-compose.yml       # local dev only
├── nginx.conf               # round-robin upstream api1:8080 / api2:8080
├── go.mod
└── go.sum
```

### Submission branch

```
submission/
├── docker-compose.yml   # references ghcr.io/nrlacerda/rinha-de-backend-2026-api:latest
└── nginx.conf
```
