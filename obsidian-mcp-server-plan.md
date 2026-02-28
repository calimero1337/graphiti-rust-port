# Obsidian MCP Server — Rust Implementation Plan

## Overview

Build a standalone MCP server in Rust that provides Claude Code-style file editing tools for an Obsidian vault. The server runs in Kubernetes, accesses the vault via a filesystem bind mount (PVC), and exposes the MCP protocol over Streamable HTTP transport (with legacy SSE backward compatibility).

**Replaces**: The current `obsidian-claude-code-mcp` Node.js plugin + socat architecture.

---

## Architecture

```
┌─────────────────┐     Streamable HTTP      ┌──────────────────────┐
│  Claude Code /   │  ◄──────────────────────► │  obsidian-mcp-server │
│  Claude Desktop  │    POST /mcp (JSON-RPC)  │  (Rust, axum)        │
│                  │    GET  /mcp (SSE stream) │                      │
└─────────────────┘                           │  ┌────────────────┐  │
                                              │  │ Tool Registry   │  │
                                              │  │ ┌────────────┐ │  │
                                              │  │ │ view       │ │  │
                                              │  │ │ create     │ │  │
                                              │  │ │ str_replace│ │  │
                                              │  │ │ insert     │ │  │
                                              │  │ │ undo_edit  │ │  │
                                              │  │ │ get_files  │ │  │
                                              │  │ │ search     │ │  │
                                              │  │ │ delete     │ │  │
                                              │  │ │ rename     │ │  │
                                              │  │ └────────────┘ │  │
                                              │  └────────┬───────┘  │
                                              │           │          │
                                              │  ┌────────▼───────┐  │
                                              │  │ VaultFs Layer   │  │
                                              │  │ (tokio::fs)     │  │
                                              │  └────────┬───────┘  │
                                              └───────────┼──────────┘
                                                          │
                                              ┌───────────▼──────────┐
                                              │  PVC / Bind Mount    │
                                              │  /vault (read/write) │
                                              └──────────────────────┘
```

### Key Design Decisions

1. **`rmcp` v0.17.x** — Official Rust MCP SDK with `#[tool]` macros and built-in Streamable HTTP server support via axum integration.
2. **Streamable HTTP transport** (MCP spec 2025-03-26) — Single `/mcp` endpoint supporting POST (JSON-RPC) and GET (SSE stream). Replaces the deprecated dual-endpoint SSE transport.
3. **Filesystem-based vault access** — No Obsidian API dependency. Reads/writes the vault directory directly. This works because Obsidian watches for file changes and syncs automatically.
4. **Undo stack** — Per-file in-memory undo (single level), matching the reference `mcp-server-text-editor` behavior.
5. **Path security** — All paths validated and sandboxed to the vault root. No `..` traversal, no symlink escape.

---

## Project Structure

```
obsidian-mcp-server/
├── Cargo.toml
├── Cargo.lock
├── Dockerfile
├── kubernetes/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   └── networkpolicy.yaml
├── src/
│   ├── main.rs                    # Entry point, config, server bootstrap
│   ├── config.rs                  # Configuration (env vars, defaults)
│   ├── server.rs                  # MCP ServerHandler impl
│   ├── vault/
│   │   ├── mod.rs
│   │   ├── fs.rs                  # VaultFs: sandboxed filesystem operations
│   │   ├── path.rs                # Path validation & normalization
│   │   └── undo.rs                # Per-file undo stack
│   └── tools/
│       ├── mod.rs                 # Tool router registration
│       ├── view.rs                # view tool
│       ├── create.rs              # create tool
│       ├── str_replace.rs         # str_replace tool
│       ├── insert.rs              # insert tool
│       ├── undo_edit.rs           # undo_edit tool
│       ├── get_workspace_files.rs # get_workspace_files tool
│       ├── search.rs              # search tool (grep-like)
│       ├── delete.rs              # delete tool
│       └── rename.rs              # rename/move tool
└── tests/
    ├── integration/
    │   ├── mod.rs
    │   ├── tools_test.rs          # Tool-level integration tests
    │   ├── transport_test.rs      # HTTP endpoint tests
    │   └── security_test.rs       # Path traversal / sandbox tests
    └── helpers/
        └── mod.rs                 # Test vault setup utilities
```

---

## Crate Dependencies

```toml
[package]
name = "obsidian-mcp-server"
version = "0.1.0"
edition = "2024"

[dependencies]
# MCP SDK
rmcp = { version = "0.17", features = [
    "server",
    "transport-streamable-http-server",
    "transport-streamable-http-server-session",
] }

# Async runtime & utilities
tokio = { version = "1", features = ["full"] }
tokio-util = { version = "0.7", features = ["rt"] }

# HTTP framework (required by rmcp Streamable HTTP)
axum = "0.8"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "1"

# Error handling
thiserror = "2"

# Logging / Tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# File watching (for vault change detection)
notify = "8"

# Regex (for search tool and get_workspace_files pattern matching)
regex = "1"

# Glob (for pattern matching)
globset = "0.4"

[dev-dependencies]
tempfile = "3"
reqwest = { version = "0.12", features = ["json"] }
tokio-test = "0.4"
assert_fs = "1"
```

---

## Module Specifications

### `src/config.rs` — Configuration

```
struct Config {
    vault_path: PathBuf,       // VAULT_PATH env var, default "/vault"
    bind_addr: SocketAddr,     // BIND_ADDR env var, default "0.0.0.0:8080"
    log_level: String,         // RUST_LOG env var, default "info"
    max_file_size: usize,      // MAX_FILE_SIZE env var, default 10MB
    undo_stack_limit: usize,   // UNDO_STACK_LIMIT env var, default 100
}
```

- Parse from environment variables with sensible defaults
- Validate `vault_path` exists and is a directory at startup
- Fail fast with clear error messages if misconfigured

### `src/vault/path.rs` — Path Validation

```
fn normalize_and_validate(vault_root: &Path, user_path: &str) -> Result<PathBuf, VaultError>
```

**Security requirements:**
- Strip leading `/` from user path
- Reject paths containing `..` segments
- Reject paths containing null bytes
- Resolve the joined path (`vault_root.join(cleaned_path)`) and call `canonicalize()`
- Verify the canonical path starts with the canonical `vault_root` (symlink escape prevention)
- Return `VaultError::PathTraversal` if any check fails

### `src/vault/fs.rs` — VaultFs

Sandboxed filesystem abstraction over `tokio::fs`.

```
struct VaultFs {
    root: PathBuf,            // Canonical vault root
    undo_stack: UndoStack,    // Per-file undo
    max_file_size: usize,
}
```

**Methods:**

| Method | Description |
|--------|-------------|
| `read_file(path) -> Result<String>` | Read file contents, validate size |
| `write_file(path, content) -> Result<()>` | Write file, save previous to undo stack |
| `create_file(path, content) -> Result<()>` | Create new file (error if exists), create parent dirs |
| `delete_file(path) -> Result<()>` | Delete file, save to undo stack |
| `rename_file(from, to) -> Result<()>` | Move/rename file |
| `list_dir(path) -> Result<Vec<String>>` | List immediate children of a directory |
| `list_all_files() -> Result<Vec<String>>` | Recursively list all files relative to root |
| `file_exists(path) -> Result<bool>` | Check existence |
| `is_dir(path) -> Result<bool>` | Check if path is a directory |
| `undo(path) -> Result<String>` | Restore previous version from undo stack |

All methods call `normalize_and_validate()` internally before any filesystem access.

### `src/vault/undo.rs` — Undo Stack

```
struct UndoStack {
    entries: DashMap<PathBuf, Vec<String>>,  // file path -> stack of previous contents
    limit: usize,
}
```

- In-memory, not persisted across restarts
- Bounded per-file stack (default 100 entries)
- Thread-safe via `DashMap` (or `tokio::sync::RwLock<HashMap>`)

### `src/tools/` — MCP Tool Implementations

Each tool is a method on the `ObsidianMcpServer` struct annotated with `#[tool]`.

#### `view` — Read file or list directory

```
#[tool(description = "Read file contents or list directory. Paths are relative to vault root.")]
async fn view(
    &self,
    #[tool(param)] path: String,
    #[tool(param)] view_range: Option<Vec<i64>>,  // [start_line, end_line], 1-indexed, -1 = EOF
) -> Result<CallToolResult, McpError>
```

**Behavior:**
- If path is a directory → list immediate children
- If path is a file → read contents with line numbers (`N: content`)
- If `view_range` provided → return only lines in `[start, end]` (1-indexed, -1 = EOF)
- Error if file/dir not found
- Error if `view_range` has invalid bounds

#### `create` — Create a new file

```
#[tool(description = "Create a new file in the vault. Fails if file already exists.")]
async fn create(
    &self,
    #[tool(param)] path: String,
    #[tool(param)] file_text: String,
) -> Result<CallToolResult, McpError>
```

**Behavior:**
- Error if file already exists (use `str_replace` to modify)
- Create parent directories as needed
- Write content

#### `str_replace` — Replace text in a file

```
#[tool(description = "Replace exact text in a file. old_str must match exactly one location.")]
async fn str_replace(
    &self,
    #[tool(param)] path: String,
    #[tool(param)] old_str: String,
    #[tool(param)] new_str: String,
) -> Result<CallToolResult, McpError>
```

**Behavior:**
- Read file, count occurrences of `old_str`
- Error if 0 matches: "No match found for replacement text"
- Error if >1 matches: "Found N matches. Provide more specific text."
- Replace the single occurrence, save undo, write back

#### `insert` — Insert text at line number

```
#[tool(description = "Insert text at a specific line number. Line 0 = beginning of file.")]
async fn insert(
    &self,
    #[tool(param)] path: String,
    #[tool(param)] insert_line: i64,
    #[tool(param)] new_str: String,
) -> Result<CallToolResult, McpError>
```

**Behavior:**
- Read file, split into lines
- Validate `insert_line` in `[0, line_count]`
- Splice new lines at position, save undo, write back

#### `undo_edit` — Revert last edit

```
#[tool(description = "Undo the last edit to a file.")]
async fn undo_edit(
    &self,
    #[tool(param)] path: String,
) -> Result<CallToolResult, McpError>
```

**Behavior:**
- Pop from undo stack for this file
- Error if no undo history
- Write restored content

#### `get_workspace_files` — List vault files

```
#[tool(description = "List all files in the vault, optionally filtered by regex.")]
async fn get_workspace_files(
    &self,
    #[tool(param)] pattern: Option<String>,
) -> Result<CallToolResult, McpError>
```

**Behavior:**
- List all files recursively
- If pattern provided, compile as regex and filter
- Return newline-separated file list

#### `search` — Search file contents

```
#[tool(description = "Search for text across vault files. Returns matching lines with context.")]
async fn search(
    &self,
    #[tool(param)] query: String,
    #[tool(param)] path: Option<String>,       // Scope to subdirectory
    #[tool(param)] regex: Option<bool>,         // Treat query as regex
    #[tool(param)] context_lines: Option<u32>,  // Lines of context (default 2)
) -> Result<CallToolResult, McpError>
```

**Behavior:**
- Walk files (optionally scoped to `path`)
- Search for `query` (literal or regex) in each file
- Return matches with file path, line number, and context lines

#### `delete` — Delete a file

```
#[tool(description = "Delete a file from the vault.")]
async fn delete(
    &self,
    #[tool(param)] path: String,
) -> Result<CallToolResult, McpError>
```

#### `rename` — Rename/move a file

```
#[tool(description = "Rename or move a file within the vault.")]
async fn rename(
    &self,
    #[tool(param)] old_path: String,
    #[tool(param)] new_path: String,
) -> Result<CallToolResult, McpError>
```

---

## Error Handling Strategy

### Error Type Hierarchy

```
#[derive(Debug, thiserror::Error)]
pub(crate) enum VaultError {
    #[error("Path traversal attempt: {0}")]
    PathTraversal(String),

    #[error("Path not found: {0}")]
    NotFound(String),

    #[error("File already exists: {0}")]
    AlreadyExists(String),

    #[error("Not a file: {0}")]
    NotAFile(String),

    #[error("Not a directory: {0}")]
    NotADirectory(String),

    #[error("File too large: {size} bytes (max: {max})")]
    FileTooLarge { size: usize, max: usize },

    #[error("Invalid line range: {0}")]
    InvalidRange(String),

    #[error("No match found for replacement text in {path}")]
    NoMatch { path: String },

    #[error("Found {count} matches in {path}. Provide more specific text.")]
    MultipleMatches { path: String, count: usize },

    #[error("No undo history for: {0}")]
    NoUndoHistory(String),

    #[error("Invalid regex: {0}")]
    InvalidRegex(String),

    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },
}
```

### Error Mapping to MCP

- `VaultError` → `CallToolResult` with `isError: true` and descriptive text content
- All tool methods return `Result<CallToolResult, McpError>` — panics are impossible
- Use `?` operator throughout with `From<VaultError> for McpError` conversion
- Tool-level errors (bad input, no match) use `isError: true` in the result (not protocol-level errors)
- Protocol-level errors (invalid JSON-RPC, unknown method) handled by `rmcp` framework

### No-Panic Guarantees

- No `unwrap()` or `expect()` on fallible operations
- All array indexing uses `.get()` with proper error handling
- All user input validated before use
- All filesystem operations wrapped in `Result`

---

## Transport & Server Setup

### `src/main.rs`

```
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize tracing (JSON format for K8s log aggregation)
    // 2. Load Config from env
    // 3. Validate vault path exists
    // 4. Create ObsidianMcpServer instance
    // 5. Build axum Router:
    //    - POST/GET /mcp  → StreamableHttpService (MCP protocol)
    //    - GET /health     → health check (200 OK)
    //    - GET /ready      → readiness check (vault accessible)
    //    - GET /metrics    → Prometheus metrics (optional)
    // 6. Bind to configured address
    // 7. Graceful shutdown on SIGTERM/SIGINT
}
```

### `src/server.rs` — ServerHandler

```
#[derive(Clone)]
pub struct ObsidianMcpServer {
    vault: Arc<VaultFs>,
    tool_router: ToolRouter<Self>,
}

#[tool_handler]
impl ServerHandler for ObsidianMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            name: "obsidian-mcp-server".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            instructions: Some("MCP server for Obsidian vault file operations".into()),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}
```

---

## Kubernetes Manifests

### Deployment (`kubernetes/deployment.yaml`)

Following the established patterns from `business_intelligence_plattform/kubernetes/api-gateway/`:

- **Image**: `obsidian-mcp-server:latest` (multi-stage build)
- **Replicas**: 1 (single writer to avoid conflicts)
- **Security context**: `runAsNonRoot: true`, `runAsUser: 1001`, `readOnlyRootFilesystem: true`
- **Volume mounts**:
  - `/vault` — PVC with the Obsidian vault (readwrite)
  - `/tmp` — emptyDir for tokio temp files
- **Probes**:
  - Liveness: `GET /health` (server is running)
  - Readiness: `GET /ready` (vault directory accessible)
- **Resources**: requests 50m CPU / 64Mi memory, limits 500m CPU / 256Mi memory
- **Env vars**: `VAULT_PATH`, `BIND_ADDR`, `RUST_LOG`, `MAX_FILE_SIZE`

### Service (`kubernetes/service.yaml`)

- **Type**: ClusterIP (internal access only, or LoadBalancer if external access needed)
- **Port**: 8080 → 8080
- **Session affinity**: ClientIP (important for MCP session statefulness)

### PVC (`kubernetes/pvc.yaml`)

- Mount the existing Obsidian vault PVC or create a new one
- Access mode: ReadWriteOnce (single pod writer)
- Storage class: whatever is available in the cluster

### ConfigMap (`kubernetes/configmap.yaml`)

- Optional configuration overrides

### NetworkPolicy (`kubernetes/networkpolicy.yaml`)

- Allow ingress on port 8080 from Claude Code / Claude Desktop pods
- Deny all other ingress

### Dockerfile

```dockerfile
# Stage 1: Build
FROM rust:1.85-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/obsidian-mcp-server /app/obsidian-mcp-server
RUN useradd -m -u 1001 appuser
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s CMD ["/app/obsidian-mcp-server", "--health-check"]
ENV RUST_LOG=info VAULT_PATH=/vault BIND_ADDR=0.0.0.0:8080
CMD ["/app/obsidian-mcp-server"]
```

---

## Edge Cases & Safety

| Edge Case | Handling |
|-----------|----------|
| Path traversal (`../../etc/passwd`) | Rejected by `normalize_and_validate()` — both `..` check and canonical prefix check |
| Symlink escape (vault contains symlink pointing outside) | `canonicalize()` resolves symlinks; prefix check catches escapes |
| Binary files | `read_file` validates UTF-8; returns error for non-UTF-8 files |
| Empty file | Returns empty content with 0 lines (not an error) |
| Concurrent writes to same file | Serialized via per-file lock in `VaultFs` (prevents data corruption) |
| Very large files | Rejected if > `MAX_FILE_SIZE` (default 10MB) |
| Unicode file names | Handled natively by `std::path::PathBuf` |
| `.obsidian/` config directory | Accessible by default (can be restricted via config if needed) |
| Vault not mounted (PVC missing) | Readiness probe fails → pod not in service |
| Undo after server restart | Stack is in-memory → undo history lost (documented behavior) |
| `str_replace` with empty `old_str` | Rejected (0-length match not meaningful) |
| `view_range` out of bounds | Error with line count information |
| File has no trailing newline | Preserved as-is (no normalization) |
| CRLF line endings | Handled — split on `\n`, `\r` preserved within lines |
| Null bytes in path | Rejected by path validation |
| Concurrent MCP sessions | Each session gets its own `ServerHandler` instance via factory; `VaultFs` is shared via `Arc` |

---

## Testing Strategy

### Unit Tests (in each module)

| Module | Tests |
|--------|-------|
| `vault::path` | Valid paths, `..` traversal, symlink escape, null bytes, leading `/`, empty path, unicode |
| `vault::fs` | Read existing, read missing, create new, create existing (error), write, delete, rename, list_dir, list_all |
| `vault::undo` | Push/pop, empty stack error, multi-file isolation, stack limit enforcement |
| `tools::view` | File view, dir listing, view_range valid, view_range invalid, missing file |
| `tools::str_replace` | Single match, no match, multiple matches, empty old_str |
| `tools::insert` | Insert at 0, insert at end, insert middle, invalid line |
| `tools::create` | New file, existing file error, nested dirs created |
| `tools::search` | Literal match, regex match, scoped to dir, no results |

### Integration Tests (`tests/integration/`)

- **`tools_test.rs`**: Stand up a temp vault, create `ObsidianMcpServer`, call tools via the `rmcp` test client, verify results
- **`transport_test.rs`**: Start the full HTTP server, send JSON-RPC requests via `reqwest`, verify MCP protocol compliance (initialize → tools/list → tools/call)
- **`security_test.rs`**: Attempt path traversal attacks via every tool, verify all are rejected

### Test Helpers (`tests/helpers/`)

- `create_temp_vault()` — Creates a temp directory with sample markdown files
- `start_test_server()` — Boots the MCP server on a random port, returns client handle

---

## Subtask Plan

```yaml
max_parallel_subtasks: 3
subtasks:
  - id: "01"
    title: "Scaffold project: Cargo.toml, directory structure, placeholder modules with doc comments"
    group: 1
    agent: "rust-coding-agent"
    tags: [scaffold]

  - id: "02"
    title: "Implement config.rs — Config struct, env var parsing, validation"
    group: 2
    depends_on: ["01"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]

  - id: "03"
    title: "Implement vault::path — normalize_and_validate with traversal/symlink protection"
    group: 2
    depends_on: ["01"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]

  - id: "04"
    title: "Implement vault::undo — UndoStack with bounded per-file history"
    group: 2
    depends_on: ["01"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]

  - id: "05"
    title: "Implement vault::fs — VaultFs with all sandboxed file operations"
    group: 3
    depends_on: ["03", "04"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]

  - id: "06"
    title: "Implement core tools: view, create, str_replace, insert, undo_edit"
    group: 4
    depends_on: ["05"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]

  - id: "07"
    title: "Implement workspace tools: get_workspace_files, search, delete, rename"
    group: 4
    depends_on: ["05"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]

  - id: "08"
    title: "Implement server.rs — ServerHandler with tool_router, and main.rs with axum + Streamable HTTP"
    group: 5
    depends_on: ["06", "07"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]

  - id: "09"
    title: "Write integration tests: tools_test, transport_test, security_test"
    group: 6
    depends_on: ["08"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]

  - id: "10"
    title: "Create Dockerfile (multi-stage) and Kubernetes manifests (deployment, service, pvc, configmap, networkpolicy, kustomization)"
    group: 5
    depends_on: ["01"]
    agent: "devops-agent"
    tags: [devops]

  - id: "11"
    title: "Final review: clippy, formatting, compile check, run all tests, verify K8s manifests"
    group: 7
    depends_on: ["09", "10"]
    agent: "rust-coding-agent"
    tags: [tdd-refactor]
```

---

## Notes

- **No `get_current_file` tool**: This tool requires the Obsidian plugin API to know which file is active in the editor. Since we're filesystem-only, this tool is omitted. Clients can use `get_workspace_files` to discover files.
- **`search` tool added**: Not in the original Node.js plugin but extremely useful for vault navigation without Obsidian's UI. Provides grep-like search across all vault files.
- **`delete` and `rename` tools added**: The original Node.js plugin lacked these, requiring the `obsidian_api` escape hatch. We implement them properly.
- **No `obsidian_api` tool**: This was a Node.js-specific escape hatch for executing arbitrary code against the Obsidian API. Not applicable to a standalone server.
- **Session affinity**: MCP's Streamable HTTP transport uses `Mcp-Session-Id` headers for session management. K8s service must use `sessionAffinity: ClientIP` to route all requests from a client to the same pod.
- **Single replica recommended**: Running multiple replicas introduces file write conflicts. If HA is needed, consider a leader-election sidecar or distributed file locking.
