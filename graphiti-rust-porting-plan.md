# Graphiti Rust Porting Plan

**Version**: 1.0
**Date**: 2026-02-27
**Source**: [getzep/graphiti](https://github.com/getzep/graphiti) v0.28.1
**License**: Apache 2.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Crate Dependencies](#3-crate-dependencies)
4. [Module Structure](#4-module-structure)
5. [Data Models](#5-data-models)
6. [Graph Database Backends](#6-graph-database-backends)
7. [LLM Client Layer](#7-llm-client-layer)
8. [Embedder and Cross-Encoder Clients](#8-embedder-and-cross-encoder-clients)
9. [Core Algorithms](#9-core-algorithms)
10. [Search Subsystem](#10-search-subsystem)
11. [Error Handling](#11-error-handling)
12. [Async and Concurrency Model](#12-async-and-concurrency-model)
13. [Prompt System](#13-prompt-system)
14. [Observability](#14-observability)
15. [Public API Surface](#15-public-api-surface)
16. [Implementation Phases](#16-implementation-phases)
17. [Critical Challenges and Design Decisions](#17-critical-challenges-and-design-decisions)
18. [Testing Strategy](#18-testing-strategy)
19. [Out of Scope](#19-out-of-scope)

---

## 1. Executive Summary

Graphiti is a real-time, temporally-aware knowledge graph library for AI agent memory. It ingests "episodes" (messages, documents, JSON) and uses LLMs to extract entities and relationships, storing them in a graph database with bi-temporal indexing. Its core value proposition is enabling precise point-in-time historical queries over an evolving knowledge graph.

### Why Port to Rust?

| Goal | Rationale |
|---|---|
| Performance | Rust eliminates Python GIL constraints; critical for concurrent LLM call orchestration |
| Memory efficiency | Lower RAM footprint for embedding vectors and graph traversal |
| WASM/embedded targets | Rust can compile to WASM for browser/edge deployments and embed into other Rust services |
| Type safety | Rust's type system eliminates whole classes of runtime errors present in the Python version |
| Native bindings | Kuzu has official Rust bindings; eliminating language boundary overhead |

### Scope

This plan covers a **faithful port** of `graphiti_core` to Rust. The FastAPI REST server and MCP server are secondary targets treated as thin wrappers over the core library.

---

## 2. Architecture Overview

### Python Architecture (Source)

```
Graphiti (facade)
  ├── GraphDriver (abstract, 4 backends)
  │     ├── Neo4jDriver
  │     ├── FalkorDBDriver
  │     ├── KuzuDriver
  │     └── NeptuneDriver
  ├── LLMClient (abstract, 6 providers)
  ├── EmbedderClient (abstract, 4 providers)
  ├── CrossEncoderClient (abstract)
  ├── SearchEngine
  │     └── SearchUtils (BM25 + vector + BFS + rerankers)
  └── Maintenance algorithms
        ├── node_operations (entity resolution)
        ├── edge_operations (temporal edge management)
        └── community_operations (label propagation + summarization)
```

### Rust Target Architecture

```
graphiti (workspace)
  ├── graphiti-core/         # Core library crate
  │     ├── lib.rs           # Public API (Graphiti struct)
  │     ├── models/          # Data types (nodes, edges)
  │     ├── driver/          # Graph DB trait + implementations
  │     ├── llm/             # LLM client trait + implementations
  │     ├── embedder/        # Embedder trait + implementations
  │     ├── search/          # Search pipeline
  │     ├── algorithms/      # Entity resolution, temporal edges, communities
  │     ├── prompts/         # Prompt templates (Rust strings/structs)
  │     └── error.rs         # Error types
  ├── graphiti-server/       # Axum REST server (Phase 3)
  └── graphiti-mcp/          # MCP server (Phase 4)
```

### Key Mapping: Python → Rust

| Python Concept | Rust Equivalent |
|---|---|
| `BaseModel` (Pydantic) | `struct` + `serde::Serialize/Deserialize` + `validator` crate |
| `ABC` abstract class | `trait` |
| `asyncio.gather` | `tokio::join!` / `futures::future::join_all` |
| `asyncio.Semaphore` | `tokio::sync::Semaphore` |
| `tenacity` retry | `backoff` crate or custom `tokio` loop |
| `typing.Optional[T]` | `Option<T>` |
| `datetime` | `chrono::DateTime<Utc>` |
| `list[T]` | `Vec<T>` |
| `dict[str, Any]` | `serde_json::Value` or typed `HashMap<String, Value>` |
| `uuid.uuid4()` | `uuid::Uuid::new_v4()` |
| `@dataclass` / Pydantic | `#[derive(Debug, Clone, Serialize, Deserialize)]` |
| Exception hierarchy | `thiserror::Error` enum |
| `logging` | `tracing` crate |

---

## 3. Crate Dependencies

### `Cargo.toml` (workspace root)

```toml
[workspace]
members = ["graphiti-core", "graphiti-server"]
resolver = "2"

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
anyhow = "1"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4", "serde"] }
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }
async-trait = "0.1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
futures = "0.3"
tokio-retry = "0.3"
dotenvy = "0.15"
```

### `graphiti-core/Cargo.toml`

```toml
[package]
name = "graphiti-core"
version = "0.1.0"
edition = "2021"

[dependencies]
# Async runtime
tokio = { workspace = true }
futures = { workspace = true }

# Serialization
serde = { workspace = true }
serde_json = { workspace = true }

# Error handling
thiserror = { workspace = true }
anyhow = { workspace = true }

# Types
chrono = { workspace = true }
uuid = { workspace = true }

# HTTP (LLM/embedder API calls)
reqwest = { workspace = true }

# Trait objects in async contexts
async-trait = { workspace = true }

# Logging/tracing
tracing = { workspace = true }

# Retry logic
tokio-retry = { workspace = true }

# Config / env
dotenvy = { workspace = true }

# Graph database backends (feature-gated)
neo4rs = { version = "0.8", optional = true }
kuzu = { version = "0.11", optional = true }

# Math / embeddings
nalgebra = { version = "0.33", optional = true }   # cosine similarity

# OpenTelemetry (optional)
opentelemetry = { version = "0.27", optional = true }
tracing-opentelemetry = { version = "0.28", optional = true }

[features]
default = ["neo4j"]
neo4j = ["dep:neo4rs"]
kuzu = ["dep:kuzu"]
falkordb = []              # Uses reqwest (Redis protocol via HTTP or bolt-compat)
neptune = []               # Uses reqwest + AWS SigV4
embeddings = ["dep:nalgebra"]
otel = ["dep:opentelemetry", "dep:tracing-opentelemetry"]
```

### Key Crate Notes

| Crate | Version | Purpose | Notes |
|---|---|---|---|
| `neo4rs` | 0.8 | Neo4j Bolt protocol | Async; has `Graph`, `query`, `BoltType` |
| `kuzu` | 0.11 | KuzuDB embedded | Official Rust bindings; synchronous C FFI — wrap in `spawn_blocking` |
| `reqwest` | 0.12 | HTTP client | For LLM/embedder APIs + FalkorDB/Neptune |
| `nalgebra` | 0.33 | Cosine similarity | `DVector<f32>` dot products |
| `backoff` or `tokio-retry` | latest | Retry with exponential backoff | Match Python tenacity behavior |
| `schemars` | 0.8 | JSON Schema generation | Generate JSON schema for LLM structured output prompts |
| `validator` | 0.19 | Field validation | Validate `group_id` pattern `[a-zA-Z0-9_-]+` |
| `md-5` | 0.10 | LLM response caching | MD5 key for disk cache |

---

## 4. Module Structure

```
graphiti-core/src/
├── lib.rs                    # Re-exports; Graphiti struct; build_indices_and_constraints
├── graphiti.rs               # Graphiti facade: add_episode, search, etc.
├── types.rs                  # GraphitiClients bundle, shared type aliases
├── error.rs                  # GraphitiError enum (thiserror)
├── helpers.rs                # semaphore_gather equivalent, parse_db_date, etc.
│
├── models/
│   ├── mod.rs
│   ├── node.rs               # Node trait, EpisodicNode, EntityNode, CommunityNode, SagaNode
│   ├── edge.rs               # Edge trait, EntityEdge, EpisodicEdge, CommunityEdge, etc.
│   ├── episode.rs            # EpisodeType enum, RawEpisode
│   └── search.rs             # SearchResults, AddEpisodeResults, SearchFilters
│
├── driver/
│   ├── mod.rs                # GraphDriver trait, GraphProvider enum
│   ├── neo4j.rs              # Neo4jDriver (feature = "neo4j")
│   ├── kuzu.rs               # KuzuDriver (feature = "kuzu")
│   ├── falkordb.rs           # FalkorDBDriver (feature = "falkordb")
│   ├── neptune.rs            # NeptuneDriver (feature = "neptune")
│   └── session.rs            # GraphDriverSession, Transaction traits
│
├── llm/
│   ├── mod.rs                # LLMClient trait, LLMConfig, ModelSize
│   ├── openai.rs             # OpenAIClient
│   ├── openai_generic.rs     # OpenAI-compatible (Ollama, vLLM)
│   ├── anthropic.rs          # AnthropicClient
│   ├── gemini.rs             # GeminiClient
│   ├── groq.rs               # GroqClient
│   ├── cache.rs              # MD5-keyed disk cache
│   └── token_tracker.rs      # TokenUsageTracker
│
├── embedder/
│   ├── mod.rs                # EmbedderClient trait
│   ├── openai.rs
│   ├── azure_openai.rs
│   ├── gemini.rs
│   └── voyage.rs
│
├── cross_encoder/
│   ├── mod.rs                # CrossEncoderClient trait
│   └── local.rs              # Local model via ONNX/candle (optional)
│
├── prompts/
│   ├── mod.rs                # Message type, prompt rendering
│   ├── extract_nodes.rs      # Node extraction prompts
│   ├── extract_edges.rs      # Edge extraction prompts
│   ├── dedupe_nodes.rs
│   ├── dedupe_edges.rs
│   └── summarize_nodes.rs
│
├── search/
│   ├── mod.rs
│   ├── config.rs             # SearchConfig, EdgeSearchConfig, reranker types
│   ├── recipes.rs            # COMBINED_HYBRID_SEARCH_RRF, etc.
│   ├── filters.rs            # SearchFilters
│   ├── engine.rs             # Main search orchestration
│   └── utils.rs              # BM25, vector search, BFS, RRF, MMR implementations
│
├── algorithms/
│   ├── mod.rs
│   ├── node_ops.rs           # Entity resolution (3-stage dedup)
│   ├── edge_ops.rs           # Temporal edge management
│   ├── community_ops.rs      # Label propagation + hierarchical summarization
│   ├── dedup.rs              # Shared dedup helpers
│   └── graph_data_ops.rs     # Bulk save/delete helpers
│
└── utils/
    ├── mod.rs
    ├── datetime.rs           # DateTime parsing, UTC conversions
    ├── chunking.rs           # Content chunking for large documents
    ├── bulk.rs               # Bulk episode ingestion helpers
    └── validation.rs         # group_id validation regex
```

---

## 5. Data Models

### 5.1 Node Trait

```rust
// models/node.rs
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::driver::GraphDriver;
use crate::error::GraphitiError;

#[async_trait]
pub trait Node: Send + Sync {
    fn uuid(&self) -> &str;
    fn name(&self) -> &str;
    fn group_id(&self) -> &str;
    fn labels(&self) -> &[String];
    fn created_at(&self) -> DateTime<Utc>;

    async fn save(&self, driver: &dyn GraphDriver) -> Result<(), GraphitiError>;
    async fn delete(&self, driver: &dyn GraphDriver) -> Result<(), GraphitiError>;
}
```

### 5.2 EpisodicNode

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub source: EpisodeType,
    pub content: String,
    pub valid_at: DateTime<Utc>,
    pub source_description: String,
    pub entity_edges: Vec<String>,    // UUIDs of extracted entity edges
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeType {
    Message,
    Json,
    Text,
}
```

### 5.3 EntityNode

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub name_embedding: Option<Vec<f32>>,
    pub summary: String,
    pub attributes: serde_json::Value,   // Flexible: user-defined Pydantic → JSON Value
}
```

**Design note**: Python uses user-defined Pydantic models for custom entity attributes. In Rust, we store these as `serde_json::Value` during runtime and provide a `TypedAttributes<T: DeserializeOwned>` wrapper for compile-time access. An optional `schemars`-based schema generation allows injecting attribute schemas into LLM prompts.

### 5.4 EntityEdge (Bi-temporal)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEdge {
    pub uuid: String,
    pub group_id: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    pub created_at: DateTime<Utc>,
    pub name: String,
    pub fact: String,
    pub fact_embedding: Option<Vec<f32>>,
    pub episodes: Vec<String>,              // Episode UUIDs this edge appeared in
    pub attributes: serde_json::Value,
    // Bi-temporal fields
    pub valid_at: Option<DateTime<Utc>>,    // When fact became true in the world
    pub invalid_at: Option<DateTime<Utc>>,  // When fact was superseded/contradicted
    pub expired_at: Option<DateTime<Utc>>,  // Administrative expiry
}
```

**Critical invariant**: `invalid_at` must be distinguished from `expired_at` in all query logic. Historical queries filter: `valid_at <= query_time AND (invalid_at IS NULL OR invalid_at > query_time)`.

### 5.5 Other Edge Types

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEdge {
    pub uuid: String,
    pub group_id: String,
    pub source_node_uuid: String,   // EpisodicNode UUID
    pub target_node_uuid: String,   // EntityNode UUID
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityEdge {
    pub uuid: String,
    pub group_id: String,
    pub source_node_uuid: String,   // CommunityNode UUID
    pub target_node_uuid: String,   // EntityNode UUID
    pub created_at: DateTime<Utc>,
}
```

### 5.6 Search and Result Types

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchResults {
    pub edges: Vec<EntityEdge>,
    pub edge_reranker_scores: Vec<f32>,
    pub nodes: Vec<EntityNode>,
    pub node_reranker_scores: Vec<f32>,
    pub episodes: Vec<EpisodicNode>,
    pub episode_reranker_scores: Vec<f32>,
    pub communities: Vec<CommunityNode>,
    pub community_reranker_scores: Vec<f32>,
}

impl SearchResults {
    pub fn merge(results: Vec<SearchResults>) -> SearchResults {
        // Merge and deduplicate by UUID, averaging scores
        todo!()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddEpisodeResults {
    pub episode: EpisodicNode,
    pub episodic_edges: Vec<EpisodicEdge>,
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<EntityEdge>,
    pub communities: Vec<CommunityNode>,
    pub community_edges: Vec<CommunityEdge>,
}
```

---

## 6. Graph Database Backends

### 6.1 GraphDriver Trait

```rust
// driver/mod.rs
use async_trait::async_trait;
use serde_json::Value;

use crate::error::GraphitiError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphProvider {
    Neo4j,
    FalkorDB,
    Kuzu,
    Neptune,
}

pub type QueryResult = Vec<std::collections::HashMap<String, Value>>;

#[async_trait]
pub trait GraphDriver: Send + Sync {
    fn provider(&self) -> GraphProvider;
    fn fulltext_syntax(&self) -> &str;
    fn default_group_id(&self) -> &str;

    async fn execute_query(
        &self,
        cypher: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult, GraphitiError>;

    async fn build_indices_and_constraints(
        &self,
        delete_existing: bool,
    ) -> Result<(), GraphitiError>;

    async fn close(&self) -> Result<(), GraphitiError>;
    async fn delete_all_indexes(&self) -> Result<(), GraphitiError>;

    // Transaction support
    async fn begin_transaction(&self) -> Result<Box<dyn GraphTransaction>, GraphitiError>;
}

#[async_trait]
pub trait GraphTransaction: Send + Sync {
    async fn execute_query(
        &mut self,
        cypher: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult, GraphitiError>;
    async fn commit(self: Box<Self>) -> Result<(), GraphitiError>;
    async fn rollback(self: Box<Self>) -> Result<(), GraphitiError>;
}
```

### 6.2 Neo4j Backend

```rust
// driver/neo4j.rs
use neo4rs::{Graph, query};

pub struct Neo4jDriver {
    graph: Graph,
    default_group_id: String,
}

impl Neo4jDriver {
    pub async fn new(uri: &str, user: &str, password: &str) -> Result<Self, GraphitiError> {
        let config = neo4rs::ConfigBuilder::default()
            .uri(uri)
            .user(user)
            .password(password)
            .build()?;
        let graph = Graph::connect(config).await?;
        Ok(Self { graph, default_group_id: String::new() })
    }
}

#[async_trait]
impl GraphDriver for Neo4jDriver {
    fn provider(&self) -> GraphProvider { GraphProvider::Neo4j }
    fn fulltext_syntax(&self) -> &str { "db.index.fulltext" }
    // ... implement execute_query by converting serde_json::Value to BoltType
}
```

**Key challenge**: `neo4rs` uses `BoltType` for parameters. Write a `json_to_bolt(value: &Value) -> BoltType` conversion function covering all JSON primitive and array types.

### 6.3 Kuzu Backend

```rust
// driver/kuzu.rs
// kuzu-rs bindings are synchronous (C FFI).
// All Kuzu operations must be wrapped in tokio::task::spawn_blocking.

pub struct KuzuDriver {
    db_path: String,
    default_group_id: String,
    // Cannot store kuzu::Database directly (not Send); use a channel-based worker or Arc<Mutex<>>
    db: Arc<tokio::sync::Mutex<kuzu::Database>>,
}

#[async_trait]
impl GraphDriver for KuzuDriver {
    async fn execute_query(&self, cypher: &str, params: HashMap<String, Value>)
        -> Result<QueryResult, GraphitiError>
    {
        let db = self.db.clone();
        let cypher = cypher.to_string();
        tokio::task::spawn_blocking(move || {
            let db = db.blocking_lock();
            let conn = kuzu::Connection::new(&db)?;
            let stmt = conn.prepare(&cypher)?;
            // bind params...
            let result = conn.execute(&stmt, params_as_kuzu)?;
            // convert to QueryResult...
            Ok(result)
        }).await?
    }

    // Kuzu-specific: EntityEdge stored as intermediate RelatesToNode_
    // Query generation must use triple-node pattern for entity edges
}
```

**Critical Kuzu note**: EntityEdge objects cannot be stored as relationship properties. Use the triple-node pattern: `(Entity)-[:RELATES_TO]->(RelatesToNode_)-[:RELATES_TO]->(Entity)`. All edge queries targeting Kuzu must use `if driver.provider() == GraphProvider::Kuzu` branches.

### 6.4 FalkorDB Backend

FalkorDB uses a Redis-compatible wire protocol. Options:
- Use a Redis client crate (`redis-rs`) with FalkorDB's GRAPH commands
- Use the FalkorDB Python client's REST mode via HTTP (simpler but slower)

**Recommended approach**: Use `redis` crate with FalkorDB's GRAPH.QUERY command:

```rust
// driver/falkordb.rs
use redis::aio::MultiplexedConnection;

pub struct FalkorDBDriver {
    connection: Arc<tokio::sync::Mutex<MultiplexedConnection>>,
    default_graph: String,
}

impl FalkorDBDriver {
    pub async fn execute_graph_query(&self, graph: &str, query: &str) -> Result<QueryResult, GraphitiError> {
        let mut conn = self.connection.lock().await;
        let result: redis::Value = redis::cmd("GRAPH.QUERY")
            .arg(graph)
            .arg(query)
            .query_async(&mut *conn)
            .await?;
        // Parse FalkorDB result format...
        Ok(parse_falkordb_result(result)?)
    }
}
```

**Multi-group fanout**: FalkorDB stores each group_id as a separate graph. Implement `execute_for_groups(group_ids: &[String], query: &str)` that fans out calls and merges `SearchResults`.

### 6.5 Neptune Backend

Amazon Neptune uses OpenSearch Serverless for full-text search. Implementation:
- HTTP requests to Neptune endpoint (bolt or HTTP API) with AWS SigV4 signing
- OpenSearch HTTP API for full-text indexing
- Use `aws-sigv4` crate for request signing

```rust
// driver/neptune.rs
// Vectors stored as comma-separated strings (no native vector type)
// Conversion: "[x IN split(e.fact_embedding, ',') | toFloat(x)]"
pub struct NeptuneDriver {
    neptune_endpoint: String,
    opensearch_endpoint: String,
    region: String,
    http_client: reqwest::Client,
}
```

---

## 7. LLM Client Layer

### 7.1 LLMClient Trait

```rust
// llm/mod.rs
use async_trait::async_trait;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    Small,
    Medium,
}

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub model: String,              // "medium" model name
    pub small_model: String,        // "small" model name
    pub api_key: String,
    pub base_url: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,       // "system", "user", "assistant"
    pub content: String,
}

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn generate_response(
        &self,
        messages: &[Message],
        response_schema: Option<&serde_json::Value>,   // JSON Schema for structured output
        max_tokens: Option<u32>,
        model_size: ModelSize,
    ) -> Result<Value, LLMError>;

    fn config(&self) -> &LLMConfig;
    fn token_tracker(&self) -> Option<&TokenUsageTracker> { None }
}
```

**Structured output approach**: Python appends the Pydantic schema as JSON to the last user message. In Rust, use `schemars::schema_for!()` to generate JSON schemas from Rust types (which implement `JsonSchema`), then serialize and append to the prompt in the same format.

### 7.2 OpenAI Client

```rust
// llm/openai.rs
pub struct OpenAIClient {
    config: LLMConfig,
    http_client: reqwest::Client,
    cache: Option<LLMCache>,
    token_tracker: Arc<Mutex<TokenUsageTracker>>,
}

impl OpenAIClient {
    async fn call_with_retry(
        &self,
        payload: &Value,
    ) -> Result<Value, LLMError> {
        let max_attempts = 4u32;
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            match self.http_client
                .post("https://api.openai.com/v1/chat/completions")
                .bearer_auth(&self.config.api_key)
                .json(payload)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => return Ok(resp.json().await?),
                Ok(resp) if resp.status().as_u16() == 429 => {
                    // Rate limit: exponential backoff
                    if attempt >= max_attempts { return Err(LLMError::RateLimit); }
                    let wait = exponential_backoff(attempt, 5, 120);
                    tokio::time::sleep(wait).await;
                }
                Ok(resp) if resp.status().is_server_error() => {
                    if attempt >= max_attempts { return Err(LLMError::ServerError(resp.status().as_u16())); }
                    let wait = exponential_backoff(attempt, 5, 120);
                    tokio::time::sleep(wait).await;
                }
                Err(e) => return Err(LLMError::Network(e.to_string())),
                Ok(resp) => return Err(LLMError::ApiError(resp.status().as_u16())),
            }
        }
    }
}
```

### 7.3 Response Parsing and Structured Output

```rust
/// Append JSON schema to prompt for structured output
fn inject_response_schema(messages: &mut Vec<Message>, schema: &Value) {
    let instruction = format!(
        "Respond with a JSON object in the following format:\n{}",
        serde_json::to_string_pretty(schema).unwrap()
    );
    if let Some(last) = messages.last_mut() {
        last.content.push_str(&format!("\n\n{}", instruction));
    }
}

/// Extract JSON from LLM response text (handle markdown code fences)
fn extract_json_from_response(text: &str) -> Result<Value, serde_json::Error> {
    let text = text.trim();
    // Strip ```json ... ``` fences if present
    let json_str = if let Some(inner) = text.strip_prefix("```json") {
        inner.trim_end_matches("```").trim()
    } else if let Some(inner) = text.strip_prefix("```") {
        inner.trim_end_matches("```").trim()
    } else {
        text
    };
    serde_json::from_str(json_str)
}
```

### 7.4 LLM Response Schema Generation

Use `schemars` to generate JSON schemas from Rust types:

```rust
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ExtractedEntity {
    name: String,
    entity_type_id: u32,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ExtractedEntities {
    extracted_entities: Vec<ExtractedEntity>,
}

// Usage:
let schema = schema_for!(ExtractedEntities);
let schema_json = serde_json::to_value(schema).unwrap();
inject_response_schema(&mut messages, &schema_json);
```

### 7.5 LLM Cache

```rust
// llm/cache.rs
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct LLMCache {
    store: Arc<Mutex<HashMap<String, Value>>>,  // In-memory; disk cache is optional extension
}

impl LLMCache {
    pub fn cache_key(messages: &[Message], schema: Option<&Value>) -> String {
        use md5::{Md5, Digest};
        let input = serde_json::to_string(&(messages, schema)).unwrap();
        format!("{:x}", Md5::new().chain_update(input.as_bytes()).finalize())
    }

    pub async fn get(&self, key: &str) -> Option<Value> {
        self.store.lock().await.get(key).cloned()
    }

    pub async fn set(&self, key: String, value: Value) {
        self.store.lock().await.insert(key, value);
    }
}
```

---

## 8. Embedder and Cross-Encoder Clients

### 8.1 EmbedderClient Trait

```rust
// embedder/mod.rs
#[async_trait]
pub trait EmbedderClient: Send + Sync {
    async fn create_embedding(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;
    async fn create_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError>;
}
```

### 8.2 OpenAI Embedder

```rust
pub struct OpenAIEmbedder {
    config: EmbedderConfig,
    http_client: reqwest::Client,
}

impl OpenAIEmbedder {
    async fn call_embeddings_api(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        let payload = json!({
            "model": self.config.model,  // "text-embedding-3-small"
            "input": texts,
        });
        let resp = self.http_client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.config.api_key)
            .json(&payload)
            .send()
            .await?
            .json::<Value>()
            .await?;
        // Parse resp["data"][*]["embedding"] as Vec<Vec<f32>>
        todo!()
    }
}
```

### 8.3 Cosine Similarity

```rust
// utils/math.rs
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}
```

---

## 9. Core Algorithms

### 9.1 Episode Ingestion Pipeline (`add_episode`)

```rust
// graphiti.rs
impl Graphiti {
    pub async fn add_episode(
        &self,
        name: String,
        episode_body: String,
        source_description: String,
        reference_time: DateTime<Utc>,
        source: EpisodeType,
        group_id: String,
        entity_types: Option<EntityTypeMap>,
        edge_types: Option<EdgeTypeMap>,
        custom_extraction_instructions: String,
        update_communities: bool,
        saga_name: Option<String>,
    ) -> Result<AddEpisodeResults, GraphitiError> {
        // 1. Validate group_id
        validate_group_id(&group_id)?;

        // 2. Retrieve prior episodes for context
        let prior_episodes = self.retrieve_episodes(reference_time, 3, Some(&[group_id.clone()]), None, None).await?;

        // 3. Extract nodes (LLM call)
        let extracted_nodes = extract_nodes(
            &*self.clients.llm,
            &episode_body,
            &prior_episodes,
            &source_description,
            &entity_types,
            &custom_extraction_instructions,
            &source,
        ).await?;

        // 4. Resolve extracted nodes (3-stage dedup)
        let resolved_nodes = resolve_extracted_nodes(
            &*self.clients.llm,
            &*self.clients.embedder,
            &*self.clients.driver,
            extracted_nodes,
            &group_id,
        ).await?;

        // 5. Extract attributes and summaries (parallel)
        let (nodes_with_attrs, name_embeddings) = tokio::join!(
            extract_node_attributes(&*self.clients.llm, &resolved_nodes, &entity_types),
            generate_name_embeddings(&*self.clients.embedder, &resolved_nodes),
        );

        // 6. Extract edges (LLM call)
        let extracted_edges = extract_edges(
            &*self.clients.llm,
            &episode_body,
            &resolved_nodes,
            &prior_episodes,
            &source_description,
            &edge_types,
            &custom_extraction_instructions,
        ).await?;

        // 7. Resolve extracted edges (temporal dedup)
        let resolved_edges = resolve_extracted_edges(
            &*self.clients.llm,
            &*self.clients.embedder,
            &*self.clients.driver,
            extracted_edges,
            &group_id,
        ).await?;

        // 8. Save all entities
        let episode_node = EpisodicNode { /* ... */ };
        // Save in parallel
        let _ = tokio::join!(
            episode_node.save(&*self.clients.driver),
            save_entity_nodes(&*self.clients.driver, &nodes_with_attrs),
            save_entity_edges(&*self.clients.driver, &resolved_edges),
        );

        // 9. Save episodic edges (MENTIONS relationships)
        // 10. Link saga
        // 11. Optionally build communities

        Ok(AddEpisodeResults { /* ... */ })
    }
}
```

### 9.2 Entity Resolution (Node Deduplication)

```rust
// algorithms/node_ops.rs

/// Three-stage entity resolution funnel
pub async fn resolve_extracted_nodes(
    llm: &dyn LLMClient,
    embedder: &dyn EmbedderClient,
    driver: &dyn GraphDriver,
    extracted: Vec<ExtractedEntity>,
    group_id: &str,
) -> Result<Vec<EntityNode>, GraphitiError> {
    let mut results = Vec::new();

    // Generate embeddings for all extracted names in parallel
    let names: Vec<String> = extracted.iter().map(|e| e.name.clone()).collect();
    let embeddings = embedder.create_embeddings(&names).await?;

    // Process each entity
    let tasks: Vec<_> = extracted.iter().zip(embeddings.iter())
        .map(|(entity, embedding)| {
            let driver = driver;
            let llm = llm;
            async move {
                // Stage 1: Hybrid search for candidates
                let candidates = hybrid_node_search(driver, &entity.name, group_id, 10).await?;

                // Stage 2: Cosine similarity threshold match
                const SIMILARITY_THRESHOLD: f32 = 0.95;
                if let Some(match_) = candidates.iter()
                    .filter_map(|c| c.name_embedding.as_ref().map(|e| (c, e)))
                    .find(|(_, emb)| cosine_similarity(embedding, emb) >= SIMILARITY_THRESHOLD)
                {
                    return Ok(ResolvedNode::Existing(match_.0.clone()));
                }

                // Stage 3: LLM deduplication for ambiguous cases
                if !candidates.is_empty() {
                    if let Some(existing) = llm_deduplicate_node(llm, entity, &candidates).await? {
                        return Ok(ResolvedNode::Existing(existing));
                    }
                }

                // Create new entity node
                Ok(ResolvedNode::New(EntityNode {
                    uuid: Uuid::new_v4().to_string(),
                    name: entity.name.clone(),
                    group_id: group_id.to_string(),
                    labels: vec!["Entity".to_string()],
                    created_at: Utc::now(),
                    name_embedding: Some(embedding.clone()),
                    summary: String::new(),
                    attributes: serde_json::Value::Null,
                }))
            }
        }).collect();

    // Execute in parallel with semaphore
    let resolved = semaphore_gather(tasks, self.max_coroutines).await?;
    Ok(resolved.into_iter().map(|r| r.into_entity_node()).collect())
}
```

### 9.3 Temporal Edge Management

```rust
// algorithms/edge_ops.rs

pub async fn resolve_extracted_edges(
    llm: &dyn LLMClient,
    embedder: &dyn EmbedderClient,
    driver: &dyn GraphDriver,
    extracted: Vec<ExtractedEdge>,
    group_id: &str,
) -> Result<Vec<EntityEdge>, GraphitiError> {
    let mut resolved = Vec::new();

    for edge in extracted {
        // Generate fact embedding
        let fact_embedding = embedder.create_embedding(&edge.fact).await?;

        // Search for existing edges between same endpoints
        let existing_between = query_edges_between_nodes(
            driver, &edge.source_uuid, &edge.target_uuid, group_id
        ).await?;

        // Hybrid semantic search for similar facts
        let similar_edges = hybrid_edge_search(driver, &fact_embedding, group_id, 10).await?;

        // Combine and deduplicate candidate set
        let candidates: Vec<EntityEdge> = merge_and_dedup(existing_between, similar_edges);

        // LLM determines which existing edges represent the same fact
        let (matches, contradictions) = llm_classify_edges(llm, &edge, &candidates).await?;

        // Merge into existing edge OR create new
        if let Some(existing) = matches.first() {
            // Update existing: extend valid_at window
            let mut updated = existing.clone();
            if let (Some(new_valid), Some(old_valid)) = (edge.valid_at, updated.valid_at) {
                updated.valid_at = Some(new_valid.min(old_valid));
            }
            resolved.push(updated);
        } else {
            // Create new edge
            resolved.push(EntityEdge {
                uuid: Uuid::new_v4().to_string(),
                fact_embedding: Some(fact_embedding),
                valid_at: edge.valid_at,
                invalid_at: None,
                expired_at: None,
                // ... other fields
            });
        }

        // Invalidate contradicted edges: set invalid_at = new edge's valid_at
        for contradiction in contradictions {
            invalidate_edge(driver, &contradiction.uuid, edge.valid_at).await?;
        }
    }

    Ok(resolved)
}
```

### 9.4 Community Detection (Label Propagation)

```rust
// algorithms/community_ops.rs
use std::collections::HashMap;

pub async fn build_communities(
    llm: &dyn LLMClient,
    embedder: &dyn EmbedderClient,
    driver: &dyn GraphDriver,
    group_ids: Option<&[String]>,
) -> Result<(Vec<CommunityNode>, Vec<CommunityEdge>), GraphitiError> {
    // Fetch all entity nodes and their adjacency
    let nodes = fetch_all_entity_nodes(driver, group_ids).await?;
    let edges = fetch_all_entity_edges(driver, group_ids).await?;

    // Build adjacency list
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for edge in &edges {
        adjacency.entry(edge.source_node_uuid.clone())
            .or_default().push(edge.target_node_uuid.clone());
        adjacency.entry(edge.target_node_uuid.clone())
            .or_default().push(edge.source_node_uuid.clone());
    }

    // Label propagation
    let mut community_map: HashMap<String, String> = nodes.iter()
        .map(|n| (n.uuid.clone(), n.uuid.clone()))  // Initialize: each node is its own community
        .collect();

    let mut converged = false;
    while !converged {
        converged = true;
        let node_order: Vec<String> = {
            let mut v: Vec<String> = community_map.keys().cloned().collect();
            // Randomize order (use thread_rng or seeded rng)
            use rand::seq::SliceRandom;
            v.shuffle(&mut rand::thread_rng());
            v
        };
        for node_id in node_order {
            if let Some(neighbors) = adjacency.get(&node_id) {
                let neighbor_communities: Vec<String> = neighbors.iter()
                    .filter_map(|n| community_map.get(n))
                    .cloned()
                    .collect();
                if let Some(most_common) = plurality_community(&neighbor_communities) {
                    if community_map[&node_id] != most_common {
                        community_map.insert(node_id, most_common);
                        converged = false;
                    }
                }
            }
        }
    }

    // Group nodes by community
    let clusters = invert_community_map(&community_map, &nodes);

    // For each cluster: hierarchical summarization → community node
    let community_nodes_futures: Vec<_> = clusters.iter()
        .map(|(_, members)| summarize_community(llm, embedder, members))
        .collect();

    let community_nodes = semaphore_gather(community_nodes_futures, Some(10)).await?;

    // Build CommunityEdge (HAS_MEMBER) for each node in each cluster
    let community_edges = build_community_edges(&clusters, &community_nodes);

    Ok((community_nodes, community_edges))
}

/// Tree-based hierarchical summarization: pair summaries, reduce recursively
async fn hierarchical_summarize(llm: &dyn LLMClient, summaries: Vec<String>) -> Result<String, GraphitiError> {
    if summaries.is_empty() { return Ok(String::new()); }
    if summaries.len() == 1 { return Ok(summaries.into_iter().next().unwrap()); }

    // Pair up and summarize in parallel
    let pairs: Vec<(String, String)> = summaries.chunks(2)
        .map(|chunk| match chunk {
            [a, b] => (a.clone(), b.clone()),
            [a] => (a.clone(), String::new()),
            _ => unreachable!(),
        })
        .collect();

    let summarized: Vec<String> = futures::future::join_all(
        pairs.iter().map(|(a, b)| summarize_pair(llm, a, b))
    ).await.into_iter().collect::<Result<Vec<_>, _>>()?;

    // Recurse
    Box::pin(hierarchical_summarize(llm, summarized)).await
}
```

---

## 10. Search Subsystem

### 10.1 SearchConfig Types

```rust
// search/config.rs
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub edge_config: Option<EdgeSearchConfig>,
    pub node_config: Option<NodeSearchConfig>,
    pub episode_config: Option<EpisodeSearchConfig>,
    pub community_config: Option<CommunitySearchConfig>,
}

#[derive(Debug, Clone)]
pub struct EdgeSearchConfig {
    pub search_methods: Vec<EdgeSearchMethod>,
    pub reranker: EdgeReranker,
    pub limit: usize,
    pub min_score: f32,
}

#[derive(Debug, Clone)]
pub enum EdgeSearchMethod {
    Bm25,
    CosineSimilarity,
    BfsTraversal,
}

#[derive(Debug, Clone)]
pub enum EdgeReranker {
    Rrf,
    Mmr { lambda: f32 },
    NodeDistance,
    EpisodeMentions,
    CrossEncoder,
}
```

### 10.2 RRF (Reciprocal Rank Fusion)

```rust
// search/utils.rs
pub fn rrf_rerank(ranked_lists: &[Vec<(String, usize)>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for list in ranked_lists {
        for (uuid, rank) in list {
            *scores.entry(uuid.clone()).or_default() += 1.0 / (*rank as f32 + k);
        }
    }
    let mut result: Vec<(String, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}
```

### 10.3 MMR (Maximal Marginal Relevance)

```rust
pub fn mmr_rerank(
    query_embedding: &[f32],
    candidates: &[(String, Vec<f32>)],  // (uuid, embedding)
    lambda: f32,
    limit: usize,
) -> Vec<String> {
    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();

    while selected.len() < limit && !remaining.is_empty() {
        let next = remaining.iter().max_by(|&&a, &&b| {
            let relevance_a = cosine_similarity(query_embedding, &candidates[a].1);
            let relevance_b = cosine_similarity(query_embedding, &candidates[b].1);
            let max_sim_a = selected.iter()
                .map(|&s| cosine_similarity(&candidates[s].1, &candidates[a].1))
                .fold(f32::NEG_INFINITY, f32::max);
            let max_sim_b = selected.iter()
                .map(|&s| cosine_similarity(&candidates[s].1, &candidates[b].1))
                .fold(f32::NEG_INFINITY, f32::max);
            let score_a = lambda * relevance_a - (1.0 - lambda) * max_sim_a;
            let score_b = lambda * relevance_b - (1.0 - lambda) * max_sim_b;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        }).copied().unwrap();
        selected.push(next);
        remaining.retain(|&i| i != next);
    }

    selected.iter().map(|&i| candidates[i].0.clone()).collect()
}
```

---

## 11. Error Handling

```rust
// error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphitiError {
    #[error("Edge not found: {uuid}")]
    EdgeNotFound { uuid: String },

    #[error("Edges not found for query")]
    EdgesNotFound,

    #[error("No edges found for group IDs: {group_ids:?}")]
    GroupsEdgesNotFound { group_ids: Vec<String> },

    #[error("No nodes found for group IDs: {group_ids:?}")]
    GroupsNodesNotFound { group_ids: Vec<String> },

    #[error("Node not found: {uuid}")]
    NodeNotFound { uuid: String },

    #[error("Search reranker error: {0}")]
    SearchReranker(String),

    #[error("Entity type validation error: {0}")]
    EntityTypeValidation(String),

    #[error("Invalid group ID '{group_id}': must match [a-zA-Z0-9_-]+")]
    GroupIdValidation { group_id: String },

    #[error("Graph database error: {0}")]
    Driver(#[from] DriverError),

    #[error("LLM error: {0}")]
    Llm(#[from] LLMError),

    #[error("Embedder error: {0}")]
    Embedder(#[from] EmbedderError),

    #[error("Serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("Rate limit exceeded")]
    RateLimit,
    #[error("Server error: HTTP {0}")]
    ServerError(u16),
    #[error("API error: HTTP {0}")]
    ApiError(u16),
    #[error("Network error: {0}")]
    Network(String),
    #[error("JSON parse error: {0}")]
    JsonParse(#[from] serde_json::Error),
    #[error("Invalid response schema")]
    InvalidSchema,
}
```

---

## 12. Async and Concurrency Model

### 12.1 Semaphore-bounded Parallel Execution

```rust
// helpers.rs
use futures::future::Future;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Execute futures concurrently with optional semaphore bound
pub async fn semaphore_gather<F, T, E>(
    futures: Vec<F>,
    max_concurrent: Option<usize>,
) -> Vec<Result<T, E>>
where
    F: Future<Output = Result<T, E>> + Send + 'static,
    T: Send + 'static,
    E: Send + 'static,
{
    match max_concurrent {
        None => futures::future::join_all(futures).await,
        Some(n) => {
            let semaphore = Arc::new(Semaphore::new(n));
            let tasks: Vec<_> = futures.into_iter().map(|fut| {
                let sem = semaphore.clone();
                tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    fut.await
                })
            }).collect();
            futures::future::join_all(tasks)
                .await
                .into_iter()
                .map(|r| r.expect("task panicked"))
                .collect()
        }
    }
}
```

### 12.2 Multi-Group FalkorDB Fanout

```rust
// FalkorDB requires per-group execution. Macro or trait method:
impl Graphiti {
    async fn execute_for_groups<F, Fut, R>(
        &self,
        group_ids: &[String],
        operation: F,
    ) -> Result<Vec<R>, GraphitiError>
    where
        F: Fn(String) -> Fut + Clone + Send + 'static,
        Fut: Future<Output = Result<R, GraphitiError>> + Send,
        R: Send + 'static,
    {
        let tasks: Vec<_> = group_ids.iter().map(|gid| {
            let op = operation.clone();
            let gid = gid.clone();
            async move { op(gid).await }
        }).collect();
        semaphore_gather(tasks, self.max_coroutines).await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
    }
}
```

### 12.3 Sequential Episode Processing Guarantee

Document clearly in the API that `add_episode` must not be called concurrently for the same `group_id`. Provide `add_episode_bulk` as the correct batch API. Consider adding a per-group mutex in `Graphiti` to enforce this:

```rust
pub struct Graphiti {
    // ...
    group_locks: Arc<tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl Graphiti {
    async fn get_group_lock(&self, group_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.group_locks.lock().await;
        locks.entry(group_id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }
}
```

---

## 13. Prompt System

Prompts in Python are string templates with structured JSON context injection. In Rust, implement as format strings or a minimal template system:

```rust
// prompts/extract_nodes.rs
pub struct ExtractNodesPrompt {
    pub episode_content: String,
    pub episode_type: EpisodeType,
    pub previous_episodes: Vec<String>,
    pub source_description: String,
    pub entity_types_schema: Option<serde_json::Value>,
    pub custom_instructions: String,
    pub language_instruction: String,
}

impl ExtractNodesPrompt {
    pub fn render(&self) -> Vec<Message> {
        vec![
            Message {
                role: "system".to_string(),
                content: self.system_prompt(),
            },
            Message {
                role: "user".to_string(),
                content: self.user_prompt(),
            },
        ]
    }

    fn system_prompt(&self) -> String {
        format!(
            "You are an expert knowledge graph builder. Extract named entities from the provided content.\n\
             {}\n\
             {}",
            self.language_instruction,
            self.custom_instructions,
        )
    }

    fn user_prompt(&self) -> String {
        // Format context from previous episodes, then current episode
        let context = self.previous_episodes.iter()
            .enumerate()
            .map(|(i, ep)| format!("Previous episode {}: {}", i + 1, ep))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "Previous context:\n{}\n\nCurrent episode ({}): {}\nSource: {}",
            context, self.episode_type.as_str(), self.episode_content, self.source_description
        )
    }
}
```

All prompt types follow the same pattern: render to `Vec<Message>`, then pass to `LLMClient::generate_response` with the appropriate JSON schema.

---

## 14. Observability

### 14.1 Tracing with `tracing` crate

Replace Python's `logging` + OpenTelemetry with Rust's `tracing` ecosystem:

```rust
// In graphiti.rs
use tracing::{debug, warn, info, instrument};

impl Graphiti {
    #[instrument(skip(self, episode_body), fields(group_id = %group_id, name = %name))]
    pub async fn add_episode(&self, name: String, episode_body: String, ...) -> Result<...> {
        debug!("Starting episode ingestion");
        // ...
        info!(nodes_extracted = nodes.len(), edges_extracted = edges.len(), "Episode ingested");
        Ok(result)
    }
}
```

### 14.2 OpenTelemetry (Optional)

Use `opentelemetry` + `tracing-opentelemetry` crates to bridge `tracing` spans to OTEL exporters. This is a feature-gated dependency.

### 14.3 Token Tracking

```rust
// llm/token_tracker.rs
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct TokenUsageTracker {
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    pub total_tokens: AtomicU64,
}

impl TokenUsageTracker {
    pub fn record(&self, prompt: u64, completion: u64) {
        self.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        self.completion_tokens.fetch_add(completion, Ordering::Relaxed);
        self.total_tokens.fetch_add(prompt + completion, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.prompt_tokens.load(Ordering::Relaxed),
            self.completion_tokens.load(Ordering::Relaxed),
            self.total_tokens.load(Ordering::Relaxed),
        )
    }
}
```

---

## 15. Public API Surface

```rust
// lib.rs / graphiti.rs

pub struct Graphiti {
    clients: Arc<GraphitiClients>,
    max_coroutines: Option<usize>,
    store_raw_episode_content: bool,
    token_tracker: Arc<TokenUsageTracker>,
}

pub struct GraphitiClients {
    pub driver: Arc<dyn GraphDriver>,
    pub llm: Arc<dyn LLMClient>,
    pub embedder: Arc<dyn EmbedderClient>,
    pub cross_encoder: Option<Arc<dyn CrossEncoderClient>>,
}

pub struct GraphitiBuilder {
    // Optional driver injection (preferred)
    driver: Option<Arc<dyn GraphDriver>>,
    // Legacy Neo4j shorthand
    neo4j_uri: Option<String>,
    neo4j_user: Option<String>,
    neo4j_password: Option<String>,
    // LLM/embedder (default: OpenAI via env OPENAI_API_KEY)
    llm_client: Option<Arc<dyn LLMClient>>,
    embedder: Option<Arc<dyn EmbedderClient>>,
    cross_encoder: Option<Arc<dyn CrossEncoderClient>>,
    max_coroutines: Option<usize>,
    store_raw_episode_content: bool,
}

impl GraphitiBuilder {
    pub fn new() -> Self { /* defaults */ }
    pub fn driver(mut self, d: Arc<dyn GraphDriver>) -> Self { self.driver = Some(d); self }
    pub fn neo4j(mut self, uri: &str, user: &str, password: &str) -> Self { /* ... */ }
    pub fn llm_client(mut self, c: Arc<dyn LLMClient>) -> Self { /* ... */ }
    pub fn embedder(mut self, e: Arc<dyn EmbedderClient>) -> Self { /* ... */ }
    pub fn max_coroutines(mut self, n: usize) -> Self { /* ... */ }
    pub async fn build(self) -> Result<Graphiti, GraphitiError> { /* ... */ }
}

impl Graphiti {
    // Setup
    pub async fn build_indices_and_constraints(&self, delete_existing: bool) -> Result<(), GraphitiError>;

    // Write
    pub async fn add_episode(&self, params: AddEpisodeParams) -> Result<AddEpisodeResults, GraphitiError>;
    pub async fn add_episode_bulk(&self, episodes: Vec<RawEpisode>, params: BulkParams) -> Result<AddBulkEpisodeResults, GraphitiError>;
    pub async fn remove_episode(&self, episode_uuid: &str) -> Result<(), GraphitiError>;

    // Read
    pub async fn search(&self, query: &str, params: SearchParams) -> Result<Vec<EntityEdge>, GraphitiError>;
    pub async fn search_with_config(&self, query: &str, config: SearchConfig, params: SearchParams) -> Result<SearchResults, GraphitiError>;
    pub async fn retrieve_episodes(&self, params: RetrieveEpisodesParams) -> Result<Vec<EpisodicNode>, GraphitiError>;
    pub async fn get_nodes_and_edges_by_episode(&self, episode_uuids: &[String]) -> Result<SearchResults, GraphitiError>;

    // Maintenance
    pub async fn build_communities(&self, params: BuildCommunitiesParams) -> Result<(Vec<CommunityNode>, Vec<CommunityEdge>), GraphitiError>;
    pub async fn close(&self) -> Result<(), GraphitiError>;

    // Token tracking
    pub fn token_tracker(&self) -> Arc<TokenUsageTracker>;
}
```

**Builder pattern** is preferred over Python's constructor with optional parameters. `AddEpisodeParams`, `SearchParams`, etc. are structs with `Default` implementations.

---

## 16. Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
**Goal**: Core types compile; Neo4j backend works; single LLM call succeeds.

- [ ] Workspace setup with `graphiti-core` crate
- [ ] Error types (`error.rs`)
- [ ] Data models: `EpisodicNode`, `EntityNode`, `CommunityNode`, `EntityEdge`, all edge types
- [ ] `GraphDriver` trait + `Neo4jDriver` implementation
  - `execute_query` with `BoltType` conversion
  - `build_indices_and_constraints`
  - Transaction support
- [ ] `LLMClient` trait + `OpenAIClient` (no structured output yet, just raw JSON)
- [ ] `EmbedderClient` trait + `OpenAIEmbedder`
- [ ] Basic `Graphiti` struct with `GraphitiBuilder`
- [ ] `build_indices_and_constraints` public method
- [ ] Integration test: connect to Neo4j, build indices

**Deliverable**: Can connect to Neo4j and build graph schema.

---

### Phase 2: Episode Ingestion (Weeks 4-7)
**Goal**: `add_episode` works end-to-end with Neo4j + OpenAI.

- [ ] Prompt system: `ExtractNodesPrompt`, `ExtractEdgesPrompt`
- [ ] JSON Schema generation with `schemars` for structured LLM output
- [ ] Response parsing: `extract_json_from_response`
- [ ] `extract_nodes` algorithm
- [ ] Cosine similarity utilities
- [ ] Hybrid node search (BM25 + vector)
- [ ] `resolve_extracted_nodes` (3-stage dedup)
- [ ] Node attribute extraction and summarization prompts
- [ ] Embedding generation for entity names
- [ ] `extract_edges` algorithm
- [ ] `resolve_extracted_edges` (temporal edge management)
- [ ] Edge invalidation logic
- [ ] Save pipeline: EpisodicNode, EpisodicEdges, EntityNodes, EntityEdges
- [ ] `retrieve_episodes` query
- [ ] `semaphore_gather` implementation
- [ ] `add_episode` public method
- [ ] Token tracking in LLM calls
- [ ] LLM response caching (MD5-keyed in-memory)

**Deliverable**: Full episode ingestion pipeline working with Neo4j + OpenAI.

---

### Phase 3: Search and Retrieval (Weeks 8-10)
**Goal**: All search modes work; `search` and `search_with_config` are usable.

- [ ] `SearchConfig` types and recipes
- [ ] `SearchFilters` (date ranges, group_ids, node types)
- [ ] BM25 full-text search via Neo4j index
- [ ] Vector similarity search via Neo4j index
- [ ] BFS traversal search
- [ ] RRF reranker
- [ ] MMR reranker
- [ ] Node distance reranker
- [ ] Episode mentions reranker
- [ ] Cross-encoder reranker (optional; ONNX-based local model)
- [ ] `search` public method
- [ ] `search_with_config` public method
- [ ] `get_nodes_and_edges_by_episode` method
- [ ] `SearchResults::merge` for FalkorDB fanout

**Deliverable**: Full search pipeline working.

---

### Phase 4: Additional LLM Providers and Graph Backends (Weeks 11-13)
**Goal**: Multi-provider support; alternative graph backends.

- [ ] `AnthropicClient`
- [ ] `GeminiClient`
- [ ] `GroqClient`
- [ ] `OpenAIGenericClient` (Ollama/vLLM compatible)
- [ ] `AzureOpenAIClient`
- [ ] `KuzuDriver` (with triple-node EntityEdge pattern)
- [ ] `FalkorDBDriver` (Redis wire protocol; multi-group fanout)
- [ ] Additional embedders: Azure OpenAI, Gemini, Voyage
- [ ] `NeptuneDriver` (AWS; OpenSearch for full-text)

**Deliverable**: Full provider matrix functional.

---

### Phase 5: Communities and Bulk Ingestion (Weeks 14-15)
**Goal**: Complete algorithmic coverage.

- [ ] `SummarizeNodesPrompt`
- [ ] Hierarchical summarization (`summarize_pair`, `hierarchical_summarize`)
- [ ] Label propagation community detection
- [ ] `CommunityNode` + `CommunityEdge` persistence
- [ ] `build_communities` public method
- [ ] `add_episode_bulk` implementation
- [ ] Content chunking (`utils/chunking.rs`)

**Deliverable**: Community detection and bulk ingestion complete.

---

### Phase 6: REST Server (Week 16)
**Goal**: Axum HTTP server wrapping `graphiti-core`.

- [ ] `graphiti-server` crate with Axum
- [ ] `/add_episode` endpoint
- [ ] `/search` endpoint
- [ ] `/retrieve_episodes` endpoint
- [ ] `/build_communities` endpoint
- [ ] `/healthz` + `/readyz` endpoints
- [ ] Docker image + Docker Compose

**Deliverable**: HTTP API feature-equivalent to Python FastAPI server.

---

### Phase 7: Observability and Polish (Week 17)
**Goal**: Production-ready library.

- [ ] `tracing` integration throughout
- [ ] OpenTelemetry bridge (feature-gated)
- [ ] Telemetry opt-out mechanism
- [ ] `dotenvy` config loading
- [ ] Documentation (rustdoc)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Crates.io publication prep

---

## 17. Critical Challenges and Design Decisions

### 17.1 Pydantic → Rust Type System

**Problem**: Python uses Pydantic for both validation and dynamic LLM response schemas. Users pass custom `BaseModel` subclasses as `entity_types` to control extraction.

**Rust solution**:
- Use `schemars::JsonSchema` trait on user-defined structs
- API accepts `HashMap<String, Box<dyn JsonSchema>>` (requires object safety workarounds)
- **Simpler alternative**: Accept `HashMap<String, serde_json::Value>` (pre-serialized schemas) from users, sacrificing compile-time schema validation
- **Recommended**: Define a `EntityType` trait with `fn json_schema() -> Value` and `fn name() -> &'static str`. Users implement this for their structs using a derive macro.

### 17.2 Kuzu Synchronous FFI

**Problem**: `kuzu-rs` bindings are synchronous C FFI. All operations must be `spawn_blocking`-wrapped.

**Solution**: Thread-local Kuzu connection pool. Maintain a `rayon`-powered worker pool or `tokio::task::JoinSet` with dedicated threads. The `KuzuDriver` holds an `Arc<tokio::sync::Mutex<kuzu::Database>>` and always dispatches to `spawn_blocking`.

**Performance implication**: Kuzu operations have a thread-switch overhead. Acceptable for batch writes; may be noticeable for frequent single-edge lookups.

### 17.3 FalkorDB Wire Protocol

**Problem**: No official Rust FalkorDB client exists. The Redis GRAPH command protocol differs from standard Redis.

**Solutions** (pick one):
1. Use `redis` crate with raw GRAPH.QUERY commands (parse result format manually)
2. Use the FalkorDB REST API via `reqwest` (easier parsing, network overhead)
3. Submit a FalkorDB Rust client to the ecosystem

**Recommended**: Option 1 using `redis` crate with a custom `FalkorDBResult` parser. Document as best-effort.

### 17.4 Structured LLM Output

**Problem**: Python relies on Pydantic for structured output. OpenAI's structured output (`response_format: {type: "json_schema"}`) requires the exact JSON schema. Anthropic uses tool calls. Gemini has its own schema format.

**Solution**:
- Generate JSON schemas using `schemars` from Rust types
- Implement schema injection into prompts (fallback: append to user message)
- For OpenAI: use `response_format: {type: "json_object"}` and parse result
- For Anthropic: use tool_use with the schema as the tool's input schema
- For Gemini: use `generationConfig.responseSchema`
- Implement per-provider schema injection logic in each client

### 17.5 Async Trait Objects

**Problem**: `async fn` in traits is not directly object-safe in stable Rust (before the `async fn in traits` feature becomes fully stable).

**Solution**: Use `async-trait` crate for now. This adds a `Box<dyn Future>` allocation per call but is ergonomic. When Rust stabilizes async fn in traits fully (expected ~1.80+), migrate off `async-trait`.

### 17.6 Dynamic Dispatch for Node/Edge Saving

**Problem**: Python's `Node.save(driver)` method is polymorphic. In Rust, we need to either:
1. Use `dyn Node` with `async-trait` for the `save` method
2. Or pass concrete types directly (preferred: no dynamic dispatch)

**Recommended**: Drop the `Node` trait `save` method from the public API. Instead, implement save functions directly on `GraphDriver`:

```rust
impl Neo4jDriver {
    async fn save_entity_node(&self, node: &EntityNode) -> Result<(), GraphitiError> { ... }
    async fn save_episodic_node(&self, node: &EpisodicNode) -> Result<(), GraphitiError> { ... }
    // etc.
}
```

This avoids the async-trait-on-trait-with-associated-types complexity.

### 17.7 Neptune Embeddings as Strings

**Problem**: Neptune stores embedding vectors as comma-separated strings. The conversion `[x IN split(e.fact_embedding, ',') | toFloat(x)]` in Cypher is Neptune-specific.

**Solution**: In `NeptuneDriver`, override the `save_entity_edge` method to serialize `Vec<f32>` as a comma-joined string, and add the `split()` expression to all Neptune vector queries.

### 17.8 Multi-Language Support

Python injects a multilingual instruction into system prompts. Port this as a static string constant in `prompts/mod.rs`, injected by `LLMClient::generate_response` (or let callers inject it via `CustomExtractionInstructions`).

---

## 18. Testing Strategy

### Unit Tests

- Cosine similarity: verify against known vectors
- RRF reranking: verify score formula
- MMR reranking: verify diversity selection
- `group_id` validation regex
- JSON extraction from LLM response (with/without code fences)
- Hierarchical summarization logic (mock LLM client)
- Label propagation convergence (small graph)
- Bi-temporal filtering: `valid_at`/`invalid_at` logic

### Integration Tests

Use `testcontainers-rs` to spin up Neo4j and run full pipeline tests:

```toml
[dev-dependencies]
testcontainers = "0.23"
testcontainers-modules = { version = "0.11", features = ["neo4j"] }
```

```rust
#[tokio::test]
async fn test_add_episode_round_trip() {
    let docker = testcontainers::clients::Cli::default();
    let neo4j = docker.run(testcontainers_modules::neo4j::Neo4j::default());
    let driver = Neo4jDriver::new(&neo4j.bolt_url(), "neo4j", "neo4j").await.unwrap();
    let llm = MockLLMClient::new(); // returns predefined structured responses
    let embedder = MockEmbedder::new(); // returns random unit vectors
    let graphiti = GraphitiBuilder::new()
        .driver(Arc::new(driver))
        .llm_client(Arc::new(llm))
        .embedder(Arc::new(embedder))
        .build().await.unwrap();

    graphiti.build_indices_and_constraints(false).await.unwrap();

    let result = graphiti.add_episode(AddEpisodeParams {
        name: "test".to_string(),
        episode_body: "Alice works at Acme Corp.".to_string(),
        reference_time: Utc::now(),
        source: EpisodeType::Text,
        group_id: "test-group".to_string(),
        ..Default::default()
    }).await.unwrap();

    assert!(!result.nodes.is_empty());
    assert!(!result.edges.is_empty());
}
```

### Property-Based Tests

Use `proptest` or `quickcheck` for:
- Edge deduplication is idempotent
- Temporal invariants: `valid_at <= invalid_at` when both present
- Search results always have matching score vector lengths

---

## 19. Out of Scope

The following Python components are explicitly excluded from the initial Rust port:

| Component | Reason |
|---|---|
| **FastAPI REST server** | Covered in Phase 6 (Axum replacement); deprioritized |
| **MCP server** | Thin protocol wrapper; port after core is stable |
| **Neptune backend** | Complex AWS integration; Phase 4 stretch goal |
| **Cross-encoder reranker** | Requires ONNX/candle ML integration; optional feature |
| **Disk-based LLM cache** | In-memory cache covers most use cases; disk is an enhancement |
| **PostHog telemetry** | Consider separate opt-in crate |
| **Python `examples/`** | Port incrementally as Rust examples after library stabilizes |
| **`graphiti-ts` TypeScript port** | Out of scope entirely |
| **Schema migrations** | Port after schema stabilizes in Rust |

---

## Summary Checklist

| Milestone | Target | Status |
|---|---|---|
| Phase 1: Foundation | Week 3 | Planned |
| Phase 2: Episode Ingestion | Week 7 | Planned |
| Phase 3: Search | Week 10 | Planned |
| Phase 4: Multi-provider | Week 13 | Planned |
| Phase 5: Communities + Bulk | Week 15 | Planned |
| Phase 6: REST Server | Week 16 | Planned |
| Phase 7: Polish | Week 17 | Planned |

**Total estimated effort**: 17 weeks for a single senior Rust engineer working full-time.
**Parallelizable**: Phases 4 (LLM providers) and parts of Phase 3 (rerankers) can be parallelized across team members.
