# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM rust:slim AS builder

WORKDIR /build

# Cache compiled dependency artifacts by building stubs first.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && echo 'fn main(){}' > src/main.rs \
    && echo '' > src/lib.rs \
    && cargo build --release --locked \
    && rm -f  target/release/obsidian-mcp-server \
              target/release/deps/obsidian_mcp_server* \
              target/release/deps/libobsidian_mcp_server* \
    && rm -rf target/release/.fingerprint/obsidian-mcp-server-*

# Now copy real source and rebuild — only our crate is recompiled.
COPY src ./src
RUN touch src/main.rs src/lib.rs \
    && cargo build --release --locked

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim

# Non-root user
RUN useradd --uid 1000 --user-group --no-create-home appuser

COPY --from=builder /build/target/release/obsidian-mcp-server /usr/local/bin/obsidian-mcp-server

USER appuser

ENV VAULT_PATH=/vault \
    BIND_ADDR=0.0.0.0:8080 \
    RUST_LOG=obsidian_mcp_server=info

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/obsidian-mcp-server"]
