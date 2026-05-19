# syntax=docker/dockerfile:1.7

# ---------- builder ----------
# Compile a static linux/amd64 API binary.
FROM --platform=linux/amd64 golang:1.23-bookworm AS builder

WORKDIR /src

# Cache dependencies separately from source for faster iterative rebuilds.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64

RUN go build -trimpath -ldflags='-s -w' -o /out/api ./cmd/api

# ---------- runtime ----------
# distroless/static is a minimal scratch-like image with CA certs + tzdata.
FROM --platform=linux/amd64 gcr.io/distroless/static-debian12:nonroot AS runtime

COPY --from=builder /out/api       /api
COPY resources/hnsw.bin            /hnsw.bin

ENV INDEX_PATH=/hnsw.bin \
    LISTEN_ADDR=:8080 \
    GOMAXPROCS=2

EXPOSE 8080
ENTRYPOINT ["/api"]
