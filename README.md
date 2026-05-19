# Fraud Detection API

High-performance fraud scoring service for the Rinha de Backend 2026 challenge.

The service receives card transaction payloads, turns each payload into the challenge's normalized 14-dimensional vector, searches a prebuilt HNSW index of reference transactions, and returns an approval decision plus fraud score.

The main engineering goal is to stay fast under the official limits: one load balancer, two API instances, 1 total CPU, and 350 MB total memory.

## Objective

For each `POST /fraud-score` request:

1. Parse the fixed challenge JSON payload.
2. Vectorize it using the 14 dimensions from `docs/DETECTION_RULES.md`.
3. Quantize the query vector.
4. Search the 5 nearest neighbors in the HNSW reference index.
5. Compute `fraud_score = fraud_neighbors / 5`.
6. Return `approved = fraud_score < 0.6`.

The service also exposes `GET /ready` and must be reachable on port `9999`.

## API

### `GET /ready`

Returns HTTP `200` with body `ok` after the process has loaded the index and is ready.

### `POST /fraud-score`

Returns:

```json
{
  "approved": false,
  "fraud_score": 1
}
```

The complete request payload contract is in `docs/API.md`.

## Runtime Architecture

```text
client -> nginx:9999 -> api1:8080
                    -> api2:8080
```

Services:

- `nginx`: round-robin load balancer only. It does not inspect payloads or apply fraud logic.
- `api1`, `api2`: identical Go API processes.
- Each API process loads `/hnsw.bin` inside the container.

Compose resource limits:

```text
nginx: 0.10 CPU, 6 MB
api1:  0.45 CPU, 172 MB
api2:  0.45 CPU, 172 MB
total: 1.00 CPU, 350 MB
```

## Software Choices

### Go

Chosen for a small static binary, predictable memory behavior, cheap goroutines when needed, and good low-level control over slices, binary formats, and unsafe zero-copy views.

### fasthttp

Chosen instead of `net/http` for lower overhead on the hot path. The API surface is tiny and fixed, so `fasthttp` is a good fit.

### nginx

Chosen as the load balancer because the challenge requires at least one load balancer and two API instances. nginx is configured to proxy only; all fraud logic stays in the Go API.

### Custom HNSW

The vector search is implemented in `internal/hnsw` instead of using an external vector database. This avoids:

- extra containers
- network hops
- background services
- generic index metadata
- memory overhead outside our control

The tradeoff is more implementation responsibility, but the index format and query path can be shaped exactly to the challenge.

### Strict Payload Parser

The handler uses `parseStrictRequest` instead of a general JSON decoder on the hot path.

This parser is intentionally narrow:

- assumes the exact challenge schema
- assumes the challenge field order
- rejects escaped strings
- handles simple positive numbers used by the dataset
- handles `last_transaction: null`
- parses fixed UTC timestamps like `2026-03-11T20:23:35Z`

This is not a general JSON parser. It exists only to reduce per-request work for this challenge.

`github.com/bytedance/sonic` remains useful in tests and validation, but the runtime scoring path no longer depends on generic JSON unmarshalling.

## What We Dropped For Performance

Several normal production conveniences were intentionally avoided:

- No database.
- No external vector search service.
- No request logging on the hot path.
- No generic router.
- No middleware chain.
- No response serialization per request.
- No dynamic config file reads at runtime.
- No JSON decoding through reflection or generic decoder dispatch on the hot path.
- No storing vectors as `float32` in the runtime index.
- No per-request heap allocation for HNSW work buffers.
- No HTTP 5xx on parse errors; malformed payloads return the precomputed fraud fallback.

These choices make the code less general, but the challenge workload is fixed and rewards latency and memory discipline.

## Vectorization

The challenge vector has 14 dimensions. Most dimensions are normalized into `[0, 1]`.

Dimensions 5 and 6 are special:

- `minutes_since_last_tx`
- `km_from_last_tx`

When `last_transaction` is `null`, both use sentinel value `-1`.

The vectorization code is in `internal/vectorize`.

## Quantization

The raw normalized reference vector would be:

```text
14 dimensions * float32 = 14 * 4 bytes = 56 bytes per vector
```

This implementation stores reference vectors as:

```text
14 dimensions * uint8 = 14 bytes per vector
```

That is a 4x reduction for the vector storage itself, or 75% less memory than `float32`.

For 3,000,000 vectors:

```text
float32 vectors: 3,000,000 * 56 bytes = 168,000,000 bytes
uint8 vectors:   3,000,000 * 14 bytes =  42,000,000 bytes
saved:                                      126,000,000 bytes
```

Standard dimensions use:

```text
[0, 1] -> [0, 255]
```

The two sentinel-aware dimensions use:

```text
[-1, 1] -> [0, 255]
```

So:

- `-1` maps exactly to `0`
- present normalized values in `[0, 1]` map roughly into `[128, 255]`
- missing and present values remain clearly separated after quantization

Queries are quantized to `int16`, not `uint8`, so distance calculation can do `qi - vi` without unsigned underflow:

```text
14 dimensions * int16 = 28 bytes per query
```

That query-side cost is negligible because it is per in-flight request, not per indexed vector.

## Memory Budget

The runtime index file currently lives at:

```text
resources/hnsw.bin
```

The local artifact size is about:

```text
162,358,496 bytes ~= 155 MiB
```

That must fit inside each API service limit of `172 MB`, together with:

- Go binary
- small heap
- request buffers
- visit marks
- stack/runtime overhead

The index is compact because:

- vectors are `uint8`, not `float32`
- labels are a bitset, 1 bit per node
- layer-0 connection counts are `uint8`
- graph connections are flat `int32` arrays
- upper layers use a compact CSR-like layout
- the persisted binary format has slice-ready sections

On Linux, the API uses `mmap` for `hnsw.bin`. This avoids copying the whole index into the Go heap and lets the kernel share file-backed pages between the two API processes when possible. The Windows test environment falls back to `os.ReadFile`, so local Windows memory profiles are not representative of Linux container memory behavior.

## HNSW Layout

Runtime graph layout:

- `Vectors`: `[N * 14]uint8`
- `Conn0Cnt`: `[N]uint8`
- `Conn0`: `[N * M0]int32`
- `Labels`: packed bitset
- `UpperOff`: offsets per upper layer
- `UpperNodes`: sorted node IDs per upper layer
- `UpperCnt`: neighbor counts
- `UpperConn`: flat upper-layer connections

Current index parameters used by the build path:

```text
M:  6
M0: 12
efConstruction: 200
queryEf: 10
KNN: 5
```

The query path:

1. Greedy descent through upper layers.
2. Bounded layer-0 search with `queryEf = 10`.
3. Select the best 5 results.
4. Count fraud labels among those 5.

## Runtime Hot Path

The scoring path is:

```text
fasthttp ctx
-> strict parser
-> vectorize
-> quantize query
-> HNSW QueryFast5
-> fraud bitset lookups
-> precomputed JSON response
```

The response body is selected from six prebuilt JSON byte slices:

```text
0 frauds -> {"approved":true,"fraud_score":0}
1 fraud  -> {"approved":true,"fraud_score":0.2}
2 frauds -> {"approved":true,"fraud_score":0.4}
3 frauds -> {"approved":false,"fraud_score":0.6}
4 frauds -> {"approved":false,"fraud_score":0.8}
5 frauds -> {"approved":false,"fraud_score":1}
```

## Current Stage Timing

Measured with:

```powershell
go test ./internal/api -run '^$' -bench BenchmarkFraudScoreStagesProductionIndex -benchmem -benchtime=10s
```

Recent local result:

```text
total:       3115 ns/op
query:       2156 ns/op
unmarshal:    756 ns/op
vectorize:     85 ns/op
write:         64 ns/op
classify:      12 ns/op
quantize:      12 ns/op
allocs:         0 B/op, 0 allocs/op in the staged method body
```

The biggest cost is HNSW query. The second biggest is strict parsing. Vectorization and quantization are already small.

## Project Layout

```text
cmd/api             HTTP server entry point
cmd/build-index     offline HNSW index builder
internal/api        handlers, strict parser, handler benchmarks
internal/vectorize  request model and 14-dimensional vectorization
internal/hnsw       HNSW graph, quantization, persistence, mmap, query
internal/config     normalization constants and MCC risk table
docs/               challenge docs and project notes
test/               k6 smoke/load tests and fixture
```

## Data And Index

Raw challenge reference dataset:

```text
resources/references.json
```

Generated index:

```text
resources/hnsw.bin
```

Both are intentionally ignored by Git because they are large.

Rebuild the index locally:

```powershell
go run .\cmd\build-index -in .\resources\references.json -out .\resources\hnsw.bin
```

## Running Locally

Run the API directly:

```powershell
$env:INDEX_PATH = "resources/hnsw.bin"
$env:LISTEN_ADDR = ":9999"
go run .\cmd\api
```

Run with Docker Compose:

```powershell
docker compose up --build
```

Check readiness:

```powershell
curl http://localhost:9999/ready
```

## Tests

Run Go tests:

```powershell
$env:GOCACHE = Join-Path (Get-Location) ".gocache"
go test ./...
```

Validate the strict parser against the full k6 fixture:

```powershell
$env:FRAUD_VALIDATE_TEST_DATA = "1"
go test ./internal/api -run TestParseStrictRequest_ChallengeDatasetVectors
```

Run k6 smoke:

```powershell
k6 run .\test\smoke.js
```

Run full local k6:

```powershell
k6 run .\test\test.js
```

The k6 result is written to:

```text
test/results.json
```

## Docker Image

The submission branch references:

```text
ghcr.io/nrlacerda/fraud-detection-api:latest
```

Build and push:

```powershell
docker buildx build --platform linux/amd64 -t ghcr.io/nrlacerda/fraud-detection-api:latest --push .
```

The image must be public because the challenge runner pulls without repository-specific credentials.

## Challenge Constraints

Required by Rinha de Backend 2026:

- Expose port `9999`.
- Provide `GET /ready`.
- Provide `POST /fraud-score`.
- Use at least one load balancer and two API instances.
- Load balancer must not perform fraud detection logic.
- Total service limits must not exceed 1 CPU and 350 MB RAM.
- Images must be public and compatible with `linux/amd64`.
- Docker network mode must be `bridge`.
- `host` networking and privileged containers are not allowed.
- Repository must use the MIT license.
- `submission` branch must contain only runtime files needed to run the test.
- Test payload lookup is not allowed.

## Submission Branch

The `submission` branch contains only:

```text
LICENSE
docker-compose.yml
info.json
nginx.conf
```

It intentionally excludes source code, tests, fixtures, and build artifacts.

## Documentation Map

- `docs/API.md`: endpoint and payload contract.
- `docs/ARCHITECTURE.md`: infrastructure limits.
- `docs/DATASET.md`: reference data files.
- `docs/DETECTION_RULES.md`: vector dimensions and fraud scoring logic.
- `docs/EVALUATION.md`: k6 load test and scoring formula.
- `docs/SUBMISSION.md`: participation and branch requirements.
- `docs/VECTOR_SEARCH.md`: vector search explanation.

## Latest Local k6 Benchmark

Last recorded full k6 run against the local API:

```text
p99:          1.87 ms
failure rate: 1.34%
http errors:  0
final score:  3282.67
```

Local benchmark numbers depend on machine, Docker state, index artifact, operating system, and load environment.
