# syntax=docker/dockerfile:1.7

# ---------- builder ----------
# Compile static linux/amd64 binaries and bake hnsw.bin from references.json.
FROM --platform=linux/amd64 golang:1.23-bookworm AS builder

WORKDIR /src

# Cache dependencies separately from source for faster iterative rebuilds.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64

RUN go build -trimpath -ldflags='-s -w' -o /out/api ./cmd/api && \
    go build -trimpath -ldflags='-s -w' -o /out/build-index ./cmd/build-index

# Bake the production index. Recall sanity is enforced here — if the index
# regresses, the image build fails fast instead of shipping a bad binary.
RUN /out/build-index \
        -in /src/resources/references.json \
        -out /out/hnsw.bin \
        -m 6 \
        -m0 12 \
        -efc 200 \
        -recall-queries 200 \
        -recall-min 0.95

# ---------- runtime ----------
# distroless/static is a minimal scratch-like image with CA certs + tzdata.
FROM --platform=linux/amd64 gcr.io/distroless/static-debian12:nonroot AS runtime

COPY --from=builder /out/api       /api
COPY --from=builder /out/hnsw.bin  /hnsw.bin

ENV INDEX_PATH=/hnsw.bin \
    LISTEN_ADDR=:8080 \
    GOMAXPROCS=2

EXPOSE 8080
ENTRYPOINT ["/api"]
