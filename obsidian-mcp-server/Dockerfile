# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM rust:1.82-slim AS builder

WORKDIR /build

# Cache dependencies by copying manifests first
COPY Cargo.toml Cargo.lock ./

# Create stub src so cargo can resolve the dependency graph
RUN mkdir src && echo 'fn main(){}' > src/main.rs && echo '' > src/lib.rs

RUN cargo fetch --locked

# Now copy real source and build
COPY src ./src
RUN cargo build --release --locked

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --uid 1000 --user-group --no-create-home appuser

COPY --from=builder /build/target/release/obsidian-mcp-server /usr/local/bin/obsidian-mcp-server

USER appuser

ENV VAULT_PATH=/vault \
    BIND_ADDR=0.0.0.0:8080 \
    RUST_LOG=obsidian_mcp_server=info

EXPOSE 8080

# No HEALTHCHECK here — liveness and readiness are handled by Kubernetes
# httpGet probes on GET /health and GET /ready respectively.

ENTRYPOINT ["/usr/local/bin/obsidian-mcp-server"]
