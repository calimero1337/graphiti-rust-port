# Graphiti → Rust Port: Comprehensive Implementation Plan

> **Project**: `graphiti-rs` — A Rust port of [getzep/graphiti](https://github.com/getzep/graphiti)
> **Date**: 2026-02-27
> **Status**: Planning (Updated)

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Source Architecture Analysis](#2-source-architecture-analysis)
3. [Rust Crate Ecosystem Mapping](#3-rust-crate-ecosystem-mapping)
4. [Module-by-Module Porting Strategy](#4-module-by-module-porting-strategy)
5. [Data Model Design](#5-data-model-design)
6. [Trait Hierarchy & Abstractions](#6-trait-hierarchy--abstractions)
7. [Ingestion Pipeline Design](#7-ingestion-pipeline-design)
8. [Search Subsystem Design](#8-search-subsystem-design)
9. [LLM Client Design](#9-llm-client-design)
10. [Prompt System Design](#10-prompt-system-design)
11. [Error Handling Strategy](#11-error-handling-strategy)
12. [Testing Strategy](#12-testing-strategy)
13. [Edge Cases & Risks](#13-edge-cases--risks)
14. [Directory Structure](#14-directory-structure)
15. [Subtask Plan](#15-subtask-plan)
16. [Appendix: Crate Versions](#appendix-crate-versions)
17. [Appendix: Python-to-Rust Pattern Mapping](#appendix-python-to-rust-pattern-mapping)
18. [Appendix: Concurrency Patterns](#appendix-concurrency-patterns)

---

## 1. Executive Summary

Graphiti is a framework for constructing and querying **temporally-aware knowledge graphs** for AI agents. It provides:

- **Bi-temporal data model**: Tracks both real-world validity (`valid_at`/`invalid_at`) and graph transaction time (`created_at`/`expired_at`)
- **Incremental ingestion**: New episodes integrate without full graph recomputation
- **Hybrid retrieval**: BM25 + vector cosine similarity + graph BFS traversal
- **LLM-powered extraction**: Entity/edge extraction, deduplication, contradiction detection, and summarization
- **Multi-backend**: Neo4j, FalkorDB, Kuzu, Amazon Neptune (4 graph database providers)

The Rust port aims to deliver the same functionality with:
- **Memory safety** without garbage collection overhead
- **Predictable latency** via zero-cost async abstractions
- **Type-safe guarantees** at compile time (replacing Pydantic runtime validation)
- **High concurrency** via Tokio's work-stealing scheduler

### Scope

**Phase 1 (MVP)**: Core types, Neo4j driver, OpenAI LLM client, ingestion pipeline, hybrid search. Single-backend (Neo4j only).

**Phase 2**: Additional LLM providers (Anthropic, Gemini), additional embedders, cross-encoder reranking, bulk ingestion.

**Phase 3**: Additional graph backends (FalkorDB, Kuzu, Neptune), OpenTelemetry tracing, community detection.

### Source Metrics

| Metric | Value |
|--------|-------|
| Python files in `graphiti_core/` | ~70 |
| Lines of code (estimated) | ~15,000 |
| ABC/trait interfaces | ~15 |
| Pydantic models | ~40+ |
| LLM prompt templates | ~20 |
| Cypher query templates per backend | ~50+ |
| Graph database backends | 4 (Neo4j, FalkorDB, Kuzu, Neptune) |
| LLM providers | 6 (OpenAI, Anthropic, Gemini, Groq, custom, Ollama) |
| Embedder providers | 4 (OpenAI, Voyage, custom, Ollama) |

---

## 2. Source Architecture Analysis

### Python Source Structure (graphiti_core)

```
graphiti_core/
├── graphiti.py              # ~63KB Facade class — sole public API entry point
├── graphiti_types.py        # GraphitiClients container
├── nodes.py                 # Node hierarchy: Entity, Episodic, Community, Saga
├── edges.py                 # Edge hierarchy: Entity, Episodic, Community, HasEpisode, NextEpisode
├── graph_queries.py         # Cross-provider query generation
├── helpers.py               # semaphore_gather, lucene_sanitize, normalize_l2
├── errors.py                # Custom exceptions
├── driver/                  # GraphDriver ABC + 11 operation ABCs + 4 backend impls
│   ├── driver.py            # GraphDriver ABC with 11 operation interface attributes
│   ├── operations/          # 11 abstract operation interfaces
│   │   ├── entity_node_operations.py
│   │   ├── episodic_node_operations.py
│   │   ├── community_node_operations.py
│   │   ├── saga_node_operations.py
│   │   ├── entity_edge_operations.py
│   │   ├── episodic_edge_operations.py
│   │   ├── community_edge_operations.py
│   │   ├── has_episode_edge_operations.py
│   │   ├── next_episode_edge_operations.py
│   │   ├── search_operations.py
│   │   └── graph_operations.py
│   ├── neo4j_driver.py      # Neo4j backend
│   ├── falkordb_driver.py   # FalkorDB backend
│   ├── kuzu_driver.py       # Kuzu backend
│   └── neptune_driver.py    # Amazon Neptune backend
├── llm_client/              # LLMClient ABC + 6 provider implementations
│   ├── client.py            # LLMClient ABC
│   ├── config.py            # LLMConfig dataclass
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── gemini_client.py
│   ├── groq_client.py
│   └── ollama_client.py
├── embedder/                # EmbedderClient ABC + 4 provider implementations
│   ├── client.py            # EmbedderClient ABC
│   ├── openai.py
│   ├── voyageai.py
│   └── ollama.py
├── cross_encoder/           # CrossEncoderClient ABC + 3 implementations
│   ├── client.py
│   ├── openai.py
│   └── bge.py
├── search/                  # ~73KB search subsystem
│   ├── search.py            # 4 search functions (edge, node, episode, community)
│   ├── search_utils.py      # BM25 scoring, cosine similarity, BFS traversal
│   ├── search_config.py     # SearchConfig with 17 pre-built recipes
│   ├── search_config_recipes.py  # Pre-built recipe constants
│   └── search_filters.py    # Filter system with CNF date filters
├── prompts/                 # Versioned prompt templates for all LLM steps
│   ├── extract_nodes.py     # Entity extraction prompts (message, json, text variants)
│   ├── extract_edges.py     # Edge extraction prompts
│   ├── dedupe_nodes.py      # Node deduplication prompts
│   ├── dedupe_edges.py      # Edge deduplication & contradiction detection
│   ├── summarize_nodes.py   # Node summarization prompts
│   ├── eval.py              # Evaluation/QA prompts
│   └── models.py            # Pydantic response models for all prompts
├── utils/                   # Bulk ingestion, content chunking, datetime, text
│   ├── bulk_utils.py        # Batch operation utilities
│   ├── content_chunking.py  # Chunking strategies (JSON, text, message)
│   ├── datetime_utils.py    # Temporal parsing/formatting
│   └── maintenance/         # Core maintenance operations
│       ├── community_operations.py  # Community detection & summarization
│       ├── edge_operations.py       # Temporal invalidation, dedup helpers
│       ├── node_operations.py       # Node dedup with MinHash/LSH
│       └── graph_data_operations.py # Graph-level data operations
└── models/                  # DB query builders for nodes and edges
    ├── nodes/               # Node query model builders
    └── edges/               # Edge query model builders
```

### Key Design Patterns in Source

| Pattern | Python Implementation | Rust Equivalent |
|---------|----------------------|-----------------|
| Abstract Base Classes | `ABC` + `@abstractmethod` | `trait` + required methods |
| Pydantic Models | `BaseModel` with field validators | `struct` + `serde::Deserialize` + `validator::Validate` |
| Strategy Pattern | Constructor injection of ABC impls | Generics with trait bounds or `dyn Trait` objects |
| Async Concurrency | `asyncio.gather` + `Semaphore` | `tokio::join!` / `futures::join_all` + `tokio::sync::Semaphore` |
| Enum Dispatch | Python `match` on provider enum | Rust `match` on enum + trait objects |
| Optional Fields | `field: Type \| None = None` | `Option<T>` |
| Type Unions | `str \| int \| float` | `enum` variants or generics |
| Caching | File-based MD5 cache | `moka` async cache or `dashmap` |
| Dynamic Attributes | `dict[str, Any]` | `HashMap<String, serde_json::Value>` |
| MinHash/LSH | Custom Python implementation | `probminhash` + custom LSH |
| Structured LLM Output | Pydantic model → JSON schema → `beta.parse()` | `schemars::JsonSchema` → `response_format` → `serde_json::from_str()` |

### Critical Architecture Details

#### 2.1 Driver Operation Interface (11 Operations per Backend)

Each graph backend (Neo4j, FalkorDB, Kuzu, Neptune) implements **11 operation interfaces**:

| Operation Interface | Key Methods |
|-------------------|-------------|
| `EntityNodeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete`, `search_by_name_embedding` |
| `EpisodicNodeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete`, `get_episodes_by_group_ids` |
| `CommunityNodeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete`, `search_by_name_embedding` |
| `SagaNodeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete` |
| `EntityEdgeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete`, `search_by_embedding`, `expire` |
| `EpisodicEdgeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete` |
| `CommunityEdgeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete` |
| `HasEpisodeEdgeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete` |
| `NextEpisodeEdgeOperations` | `save`, `get_by_uuid`, `get_by_group_ids`, `delete` |
| `SearchOperations` | `bm25_search`, `vector_search`, `bfs_search`, `hybrid_search` |
| `GraphOperations` | `build_indices`, `get_graph_stats`, `clear_data`, `get_communities` |

Each Neo4j operation file generates **Cypher queries with parameter binding**. This is a significant volume — each operation file is ~80KB.

#### 2.2 Temporal Invalidation System

The temporal invalidation of edges is a **core correctness requirement**. When a new fact contradicts an existing `EntityEdge`:

1. `edge_operations.py` → `invalidate_edges()` is called
2. Existing edges with contradicting facts have `expired_at` set to current timestamp
3. The old edge is **never deleted** — it remains for historical queries
4. `valid_at` comes from LLM extraction (when fact became true)
5. `invalid_at` comes from LLM extraction (when fact ceased to be true)
6. `created_at` is set on persistence (graph transaction time)
7. `expired_at` is set on contradiction detection (edge supersession)

**Invariant**: An edge with `expired_at = Some(_)` is never deleted. The bi-temporal model (real-world time vs. graph time) must be preserved exactly.

#### 2.3 Deduplication System (3-Tier)

Node deduplication uses a 3-tier approach in `node_operations.py`:

1. **Fast pass — Exact match**: Direct name equality check
2. **Medium pass — MinHash/LSH + entropy filtering**:
   - Generate MinHash signatures for entity names
   - LSH banding to find candidate pairs
   - Entropy filtering to eliminate low-information matches (e.g., "yes", "ok")
   - Jaccard similarity threshold
3. **Slow pass — LLM comparison**: For ambiguous cases, ask the LLM if two entities refer to the same thing

Edge deduplication uses:
1. **Embedding similarity**: Cosine similarity between fact embeddings
2. **Word overlap**: Token-level overlap ratio
3. **LLM judgment**: Classify as `duplicate`, `contradiction`, or `unique`

#### 2.4 Community Detection

`community_operations.py` implements:
1. Label propagation algorithm on the entity graph
2. Community membership assignment
3. LLM-powered community summarization
4. This runs as a **background maintenance task**, separate from ingestion

---

## 3. Rust Crate Ecosystem Mapping

| Python Dependency | Rust Crate | Version | Notes |
|-------------------|-----------|---------|-------|
| `asyncio` | `tokio` | 1.x | Work-stealing async runtime |
| `pydantic` | `serde` + `validator` | 1.0 / 0.20 | Derive macros for (de)serialization + validation |
| `neo4j` (driver) | `neo4rs` | 0.8 | Bolt protocol, async, connection pooling via `deadpool` |
| `openai` | `async-openai` | 0.33 | Full API coverage, structured output, SSE streaming |
| `numpy` | `ndarray` | 0.17 | N-dimensional arrays, BLAS backend |
| `tenacity` | `backoff` | 0.4 | Exponential backoff retry |
| `httpx`/`requests` | `reqwest` | 0.13 | Async HTTP with JSON support |
| `uuid` | `uuid` | 1.x | v4/v7 generation, serde support |
| `datetime` | `chrono` | 0.4 | Timezone-aware datetime |
| N/A (graph algos) | `petgraph` | 0.8 | BFS, community detection base |
| N/A (BM25) | `bm25` | 2.3 | 3-level API: Embedder, Scorer, SearchEngine |
| N/A (MinHash) | `probminhash` + custom LSH | 0.1 | MinHash signatures; LSH built manually |
| N/A (concurrent map) | `dashmap` | 6.x | Lock-free concurrent HashMap |
| N/A (caching) | `moka` | 0.12 | Async-friendly concurrent cache with TTL |
| N/A (JSON Schema) | `schemars` | 0.8 | Generate JSON schemas from Rust types for LLM structured output |

### Crate Validation Notes

**neo4rs 0.8**: Stable, async on Tokio. `Graph` is `Clone + Send + Sync` (Arc-wrapped). Built-in connection pooling (default 16 max). Transaction support with `Txn` handle. Serde integration via `Row::to::<T>()`. **Risk**: v0.8 is Bolt 4.x only; v0.9.0-rc.9 adds Bolt 5.x support but is still RC. Element IDs (Neo4j 5.x) require `unstable-bolt-protocol-impl-v2` feature flag.

**async-openai 0.33**: Latest (Feb 2026). Full OpenAI API coverage. Builder pattern for requests. Automatic retry with exponential backoff on rate limits. `byot` feature for custom types. Azure OpenAI support. **Key for structured output**: JSON schema response format works with schemars-generated schemas.

**bm25 2.3**: Three abstraction levels — `Embedder` (sparse vectors), `Scorer` (pre-embedded), `SearchEngine` (full in-memory). Multilingual tokenization with stemming. Custom `Tokenizer` trait. This matches Graphiti's needs perfectly — use `Scorer` level for the hybrid search pipeline.

**Performance note**: If `ndarray` cosine similarity becomes a bottleneck, `simsimd` crate offers SIMD-accelerated (AVX2/AVX-512/NEON) similarity at up to 200x speedup.

### Crates NOT Needed (Rust provides natively)

| Python Need | Rust Native Feature |
|-------------|-------------------|
| `python-dotenv` | `dotenvy` crate or `std::env` |
| Type hints | Built into the type system |
| `dataclasses` | `struct` with derive macros |
| `json` | `serde_json` |
| `re` (regex) | `regex` crate |
| `collections.defaultdict` | `HashMap::entry().or_default()` |
| `functools.lru_cache` | `moka` or manual caching |

---

## 4. Module-by-Module Porting Strategy

### 4.1 Core Types (`nodes.rs`, `edges.rs`)

**Source**: `nodes.py`, `edges.py`

Port the node and edge type hierarchies as Rust structs with trait implementations.

**Key decisions**:
- Use `enum` for node/edge variants (not trait objects) for the top-level hierarchy
- Each variant holds its specific struct
- `serde` for serialization, `validator` for field constraints
- Embeddings as `Option<Vec<f32>>` (not `Vec<f64>` — most embedding APIs return f32)
- Dynamic attributes as `HashMap<String, serde_json::Value>` (matching Python's `dict[str, Any]`)

### 4.2 Driver Layer (`driver/`)

**Source**: `driver/driver.py`, `driver/operations/*.py`, `driver/neo4j_driver.py`

**Phase 1**: Neo4j only. Define traits for all 11 operation interfaces, implement for Neo4j.

**Key decisions**:
- `GraphDriver` as an async trait (Rust 2024 native async traits with `Send` bound)
- Operation traits as separate trait definitions (not associated types)
- Neo4j implementation using `neo4rs::Graph` with connection pooling
- Cypher query templates as `const &str` with parameter binding via `neo4rs::query().param()`
- Each operation impl file maps 1:1 to the Python source
- `neo4rs::Row` → custom struct deserialization using serde

**Porting volume**: 11 operation files × ~10 methods each × ~5-20 lines of Cypher per method = ~1,000+ Cypher query templates. This is the largest porting effort.

### 4.3 LLM Client (`llm_client/`)

**Source**: `llm_client/client.py`, `llm_client/openai_client.py`

**Phase 1**: OpenAI only via `async-openai`.

**Key decisions**:
- `LlmClient` trait with `generate_response<T: DeserializeOwned + JsonSchema>()` method
- Structured output via `schemars::JsonSchema` derive → JSON schema → OpenAI `response_format`
- `async-openai` builder pattern: `CreateChatCompletionRequestArgs::default().model().messages().build()?`
- Retry logic via `backoff` crate or custom implementation
- Response caching via `moka::future::Cache` with MD5 key of serialized messages
- `ModelSize` enum for routing between large/small models

### 4.4 Embedder (`embedder/`)

**Source**: `embedder/client.py`, `embedder/openai.py`

**Phase 1**: OpenAI embeddings only.

**Key decisions**:
- `EmbedderClient` trait with `create()` and `create_batch()` methods
- `async-openai` embeddings API: `CreateEmbeddingRequestArgs::default().model().input().build()?`
- Embedding dimension configurable at runtime (not compile time)
- Truncation to configured dimension matches Python behavior
- L2 normalization via `ndarray` or inline computation

### 4.5 Search (`search/`)

**Source**: `search/search.py`, `search_utils.py`, `search_config.py`, `search_filters.py`

**Key decisions**:
- Search config as builder pattern structs
- Pre-built recipes as `const` or `lazy_static` instances (17 recipes)
- BM25 via the `bm25` crate `Scorer` level for pre-embedded scoring
- Cosine similarity via `ndarray` dot product (upgrade to `simsimd` if needed)
- BFS via database Cypher queries (not in-memory petgraph)
- RRF, MMR, cross-encoder reranking as enum-dispatched functions
- Filters generate Cypher WHERE clauses, not in-memory filtering

### 4.6 Prompts (`prompts/`)

**Source**: `prompts/*.py` (7 modules, ~20 prompt functions)

**Key decisions**:
- Prompt templates as `const &str` with `format!()` interpolation
- Each prompt module as a Rust module with public functions
- Response models as `#[derive(Deserialize, JsonSchema)]` structs (critical for structured LLM output)
- `schemars::schema_for!()` generates JSON Schema at build time or runtime
- Message type: simple `struct Message { role: Role, content: String }`

### 4.7 Utilities (`utils/`)

**Source**: `utils/bulk_utils.py`, `utils/content_chunking.py`, `utils/datetime_utils.py`

**Key decisions**:
- `semaphore_gather` → `tokio::sync::Semaphore` + `futures::future::join_all`
- Content chunking as a standalone module with strategy enum
- UnionFind as a generic data structure in `utils/union_find.rs`
- Lucene sanitization as a simple string replacement function
- `normalize_l2` as inline function (or `ndarray` operation)

### 4.8 Maintenance Operations (`utils/maintenance/`)

**Source**: `community_operations.py`, `edge_operations.py`, `node_operations.py`, `graph_data_operations.py`

**Key decisions**:
- Community detection (label propagation) as a pure function on adjacency data via `petgraph`
- Edge/node operations as async functions taking `&GraphitiClients`
- Temporal invalidation (`expired_at` setting) as explicit state transitions
- MinHash/LSH deduplication as custom implementation using `probminhash`
- Entropy filtering for low-information entity names

---

## 5. Data Model Design

### 5.1 Node Types

```rust
// nodes/mod.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NodeKind {
    Entity(EntityNode),
    Episodic(EpisodicNode),
    Community(CommunityNode),
    Saga(SagaNode),
}

// nodes/episodic.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpisodeType { Message, Json, Text }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub source: EpisodeType,
    pub source_description: String,
    pub content: String,
    pub valid_at: DateTime<Utc>,
    pub entity_edges: Vec<String>,
}

// nodes/entity.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub name_embedding: Option<Vec<f32>>,
    pub summary: String,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

// nodes/community.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub name_embedding: Option<Vec<f32>>,
    pub summary: String,
}

// nodes/saga.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SagaNode {
    pub uuid: String,
    pub name: String,
    pub group_id: String,
    pub labels: Vec<String>,
    pub created_at: DateTime<Utc>,
}
```

### 5.2 Edge Types

```rust
// edges/mod.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EdgeKind {
    Entity(EntityEdge),
    Episodic(EpisodicEdge),
    Community(CommunityEdge),
    HasEpisode(HasEpisodeEdge),
    NextEpisode(NextEpisodeEdge),
}

// edges/entity.rs — THE CRITICAL BI-TEMPORAL EDGE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEdge {
    pub uuid: String,
    pub group_id: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    pub created_at: DateTime<Utc>,
    // Bi-temporal fields
    pub name: String,
    pub fact: String,
    pub fact_embedding: Option<Vec<f32>>,
    pub episodes: Vec<String>,
    // Real-world time
    pub valid_at: Option<DateTime<Utc>>,
    pub invalid_at: Option<DateTime<Utc>>,
    // Graph transaction time
    pub expired_at: Option<DateTime<Utc>>,
    // Dynamic attributes
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

// edges/episodic.rs — MENTIONS relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEdge {
    pub uuid: String,
    pub group_id: String,
    pub source_node_uuid: String,
    pub target_node_uuid: String,
    pub created_at: DateTime<Utc>,
}

// (similar simple structs for CommunityEdge, HasEpisodeEdge, NextEpisodeEdge)
```

### 5.3 Design Decision: Enum vs Trait Object

Use **enums** for the node/edge hierarchies rather than trait objects:

**Rationale**:
- Closed set of variants (no user extension expected)
- Zero-cost pattern matching vs. vtable dispatch
- Better ergonomics with `serde` (tagged enum serialization)
- Exhaustive match checking at compile time

### 5.4 Bi-Temporal Model

The temporal model on `EntityEdge` is critical and must be preserved exactly:

| Field | Meaning | Set When |
|-------|---------|----------|
| `valid_at` | When the fact became true in reality | Edge creation (from LLM extraction) |
| `invalid_at` | When the fact ceased to be true in reality | Edge creation (from LLM extraction) |
| `created_at` | When the edge was recorded in the graph | Edge persistence |
| `expired_at` | When a contradicting edge superseded this one | Contradiction detection |

**Invariant**: An edge with `expired_at = Some(_)` is **never deleted** — it remains for historical queries. Only `expired_at` is set; the edge stays in the graph.

---

## 6. Trait Hierarchy & Abstractions

### 6.1 Graph Driver

```rust
// driver/mod.rs

/// Core graph database driver trait.
/// Each backend (Neo4j, FalkorDB, etc.) implements this.
#[trait_variant::make(Send)]
pub trait GraphDriver: Send + Sync {
    // Core operations
    async fn execute_query(
        &self, query: &str, params: HashMap<String, serde_json::Value>
    ) -> Result<Vec<Record>>;
    async fn close(&self) -> Result<()>;
    async fn build_indices_and_constraints(&self, group_ids: &[String]) -> Result<()>;

    // Operation interfaces — returns trait objects for each operation category
    fn entity_node_ops(&self) -> &dyn EntityNodeOps;
    fn episode_node_ops(&self) -> &dyn EpisodeNodeOps;
    fn community_node_ops(&self) -> &dyn CommunityNodeOps;
    fn saga_node_ops(&self) -> &dyn SagaNodeOps;
    fn entity_edge_ops(&self) -> &dyn EntityEdgeOps;
    fn episodic_edge_ops(&self) -> &dyn EpisodicEdgeOps;
    fn community_edge_ops(&self) -> &dyn CommunityEdgeOps;
    fn has_episode_edge_ops(&self) -> &dyn HasEpisodeEdgeOps;
    fn next_episode_edge_ops(&self) -> &dyn NextEpisodeEdgeOps;
    fn search_ops(&self) -> &dyn SearchOps;
    fn graph_ops(&self) -> &dyn GraphOps;
}
```

### 6.2 Operation Traits (example)

```rust
// driver/traits.rs

#[trait_variant::make(Send)]
pub trait EntityNodeOps: Send + Sync {
    async fn save(&self, node: &EntityNode) -> Result<EntityNode>;
    async fn get_by_uuid(&self, uuid: &str) -> Result<EntityNode>;
    async fn get_by_group_ids(
        &self, group_ids: &[String], limit: Option<usize>
    ) -> Result<Vec<EntityNode>>;
    async fn delete(&self, uuid: &str) -> Result<()>;
    async fn search_by_name_embedding(
        &self, embedding: &[f32], group_ids: &[String], limit: usize
    ) -> Result<Vec<EntityNode>>;
}

#[trait_variant::make(Send)]
pub trait EntityEdgeOps: Send + Sync {
    async fn save(&self, edge: &EntityEdge) -> Result<EntityEdge>;
    async fn get_by_uuid(&self, uuid: &str) -> Result<EntityEdge>;
    async fn get_by_group_ids(
        &self, group_ids: &[String], limit: Option<usize>
    ) -> Result<Vec<EntityEdge>>;
    async fn delete(&self, uuid: &str) -> Result<()>;
    async fn search_by_embedding(
        &self, embedding: &[f32], group_ids: &[String], limit: usize
    ) -> Result<Vec<EntityEdge>>;
    async fn expire(&self, uuid: &str, expired_at: DateTime<Utc>) -> Result<()>;
}

#[trait_variant::make(Send)]
pub trait SearchOps: Send + Sync {
    async fn bm25_search(
        &self, query: &str, group_ids: &[String], limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchResult>>;
    async fn vector_search(
        &self, embedding: &[f32], group_ids: &[String], limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SearchResult>>;
    async fn bfs_search(
        &self, center_uuid: &str, group_ids: &[String], limit: usize, depth: usize,
    ) -> Result<Vec<SearchResult>>;
}

// ... (similar patterns for all 11 operation traits)
```

### 6.3 LLM Client

```rust
// llm_client/mod.rs

#[derive(Debug, Clone, Copy)]
pub enum ModelSize {
    Small,  // gpt-4.1-nano equivalent — simple classification tasks
    Medium, // gpt-4.1-mini equivalent — complex extraction/reasoning
}

#[trait_variant::make(Send)]
pub trait LlmClient: Send + Sync {
    /// Generate a structured response deserialized into type T.
    /// T must implement both DeserializeOwned (for parsing) and JsonSchema (for schema generation).
    async fn generate_response<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: &[Message],
        model_size: ModelSize,
    ) -> Result<T>;

    /// Generate a plain text response.
    async fn generate_text(
        &self,
        messages: &[Message],
        model_size: ModelSize,
    ) -> Result<String>;
}
```

### 6.4 Embedder

```rust
// embedder/mod.rs

#[trait_variant::make(Send)]
pub trait EmbedderClient: Send + Sync {
    async fn create(&self, input: &str) -> Result<Vec<f32>>;
    async fn create_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embedding_dim(&self) -> usize;
}
```

### 6.5 Cross-Encoder

```rust
// cross_encoder/mod.rs

#[trait_variant::make(Send)]
pub trait CrossEncoderClient: Send + Sync {
    async fn rank(&self, query: &str, candidates: &[String]) -> Result<Vec<f32>>;
}
```

### 6.6 Clients Container

```rust
// types.rs

pub struct GraphitiClients {
    pub driver: Arc<dyn GraphDriver>,
    pub llm_client: Arc<dyn LlmClient>,
    pub embedder: Arc<dyn EmbedderClient>,
    pub cross_encoder: Option<Arc<dyn CrossEncoderClient>>,
}
```

Use `Arc<dyn Trait>` for shared ownership across concurrent tasks. The cross-encoder is optional since not all search configurations use it.

---

## 7. Ingestion Pipeline Design

### 7.1 Single Episode Flow

```
add_episode(&self, episode: EpisodeInput) -> Result<EpisodeResult>

    1. Content Chunking
       - should_chunk() heuristic (entity density estimation)
       - ChunkStrategy::Json | ChunkStrategy::Text | ChunkStrategy::Message
       - Overlap between chunks to catch boundary entities

    2. Entity Extraction (LLM)
       - prompts::extract_nodes::extract_{message,json,text}(context)
       - Returns Vec<ExtractedEntity>

    3. Edge Extraction (LLM)
       - prompts::extract_edges::edge(context)
       - Returns Vec<ExtractedEdge>

    4. Node Deduplication (3-tier)
       a. Fast: exact name match against existing graph entities
       b. Medium: MinHash/LSH + entropy filtering + Jaccard similarity
       c. Slow: LLM comparison for ambiguous cases
       - Returns (resolved_nodes, uuid_map, duplicate_pairs)

    5. Edge Deduplication (LLM)
       - Detect duplicate facts (identical information)
       - Detect contradicted facts (mutually exclusive)
       - Set expired_at on contradicted edges (TEMPORAL INVALIDATION)

    6. Embedding Generation
       - embedder.create_batch() for entity name embeddings
       - embedder.create_batch() for fact embeddings

    7. Persistence
       - Save nodes and edges via driver operations
       - Create episodic edges (MENTIONS)
       - Maintain episode sequencing (NEXT_EPISODE)

    8. Community Update
       - Determine community membership for new entities
       - Update or create community nodes and edges
```

### 7.2 Bulk Episode Flow

```
add_episode_bulk(&self, episodes: Vec<EpisodeInput>) -> Result<Vec<EpisodeResult>>

    1. Parallel extraction via semaphore-bounded tasks
    2. Two-pass node dedup: graph resolution, then batch-internal (UnionFind)
    3. Edge dedup: embedding similarity + word overlap + LLM
    4. Batch persistence with transaction management
```

### 7.3 Concurrency Model

```rust
// Pattern: semaphore-bounded concurrent LLM calls
use tokio::sync::Semaphore;
use futures::future::join_all;
use std::sync::Arc;

pub async fn semaphore_gather<T, F, Fut>(
    semaphore: &Arc<Semaphore>,
    items: Vec<T>,
    f: F,
) -> Vec<Result<F::Output>>
where
    F: Fn(T) -> Fut + Clone,
    Fut: Future<Output = Result<_>>,
{
    let futures = items.into_iter().map(|item| {
        let sem = semaphore.clone();
        let func = f.clone();
        async move {
            let _permit = sem.acquire().await.map_err(|_| GraphitiError::Internal)?;
            func(item).await
        }
    });
    join_all(futures).await
}
```

- Default semaphore permits: 20 (matching Python default)
- `Arc<Semaphore>` shared across the `Graphiti` instance
- Used for all LLM calls, embedding calls, and driver queries

---

## 8. Search Subsystem Design

### 8.1 Search Entry Point

```rust
pub async fn search(
    clients: &GraphitiClients,
    query: &str,
    group_ids: &[String],
    config: SearchConfig,
    filters: Option<SearchFilters>,
    center_node_uuid: Option<&str>,
) -> Result<SearchResults>
```

### 8.2 Four Search Pipelines

| Pipeline | Retrieval Methods | Rerankers |
|----------|---------|-----------|
| `edge_search` | BM25, cosine similarity, BFS | RRF, node_distance, episode_mentions, MMR, cross_encoder |
| `node_search` | BM25, cosine similarity, BFS | RRF, node_distance, episode_mentions, MMR, cross_encoder |
| `episode_search` | BM25 | RRF, cross_encoder |
| `community_search` | BM25, cosine similarity | RRF, MMR, cross_encoder |

### 8.3 Reranking Algorithms

**RRF (Reciprocal Rank Fusion)**:
```
score(doc) = Σ 1/(rank_i + k)  for each ranked list i
```

**MMR (Maximal Marginal Relevance)**:
```
MMR = λ * sim(query, doc) + (λ - 1) * max_sim(doc, selected)
```
Uses L2-normalized dot products for pairwise similarity matrix.

**Cross-Encoder**: External model scoring via `CrossEncoderClient` trait.

**Node Distance**: Shortest-path distance from center node (RRF pre-sort, then proximity rerank).

**Episode Mentions**: Count of MENTIONS relationships (frequency-based ranking).

### 8.4 Filter System

```rust
pub struct SearchFilters {
    pub node_labels: Option<Vec<String>>,
    pub edge_types: Option<Vec<String>>,
    pub valid_at: Option<Vec<Vec<DateFilter>>>,      // CNF: OR of ANDs
    pub invalid_at: Option<Vec<Vec<DateFilter>>>,
    pub created_at: Option<Vec<Vec<DateFilter>>>,
    pub expired_at: Option<Vec<Vec<DateFilter>>>,
    pub edge_uuids: Option<Vec<String>>,
    pub property_filters: Option<Vec<PropertyFilter>>,
}

pub struct DateFilter {
    pub date: Option<DateTime<Utc>>,
    pub comparison_operator: ComparisonOperator,
}

pub enum ComparisonOperator {
    Eq, Ne, Gt, Lt, Gte, Lte, IsNull, IsNotNull,
}

pub struct PropertyFilter {
    pub key: String,
    pub value: serde_json::Value,
    pub operator: ComparisonOperator,
}
```

### 8.5 Search Config Recipes

Port the ~17 pre-built recipes as `const fn` or `lazy_static` instances:

```rust
pub const COMBINED_HYBRID_SEARCH_RRF: SearchConfig = SearchConfig { ... };
pub const EDGE_HYBRID_SEARCH_CROSS_ENCODER: SearchConfig = SearchConfig { ... };
pub const NODE_HYBRID_SEARCH_RRF: SearchConfig = SearchConfig { ... };
// ... 14 more recipes
```

---

## 9. LLM Client Design

### 9.1 OpenAI Implementation

```rust
pub struct OpenAiClient {
    client: async_openai::Client<OpenAIConfig>,
    model: String,           // e.g., "gpt-4o"
    small_model: String,     // e.g., "gpt-4.1-nano"
    temperature: f32,
    max_tokens: u32,
    cache: Option<moka::future::Cache<String, String>>,
}

impl LlmClient for OpenAiClient {
    async fn generate_response<T>(&self, messages, model_size) -> Result<T>
    where T: DeserializeOwned + JsonSchema
    {
        // 1. Clean input (strip invalid Unicode)
        // 2. Check cache (MD5 of serialized messages)
        // 3. Select model based on ModelSize enum
        // 4. Generate JSON Schema from T via schemars::schema_for!()
        // 5. Build request with response_format: { type: "json_schema", json_schema: ... }
        // 6. Call async-openai with CreateChatCompletionRequestArgs builder
        // 7. Deserialize response content into T via serde_json::from_str()
        // 8. Retry with exponential backoff on transient errors
        // 9. Cache successful response
    }
}
```

### 9.2 Structured Output

Use `schemars::JsonSchema` derive macro to generate JSON schemas from Rust types. Pass the schema to OpenAI's structured output API.

```rust
#[derive(Deserialize, JsonSchema)]
pub struct ExtractedEntities {
    pub extracted_entities: Vec<ExtractedEntity>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type_id: i32,
}

// At call site:
let schema = schemars::schema_for!(ExtractedEntities);
// Pass schema to OpenAI response_format parameter
```

### 9.3 Model Size Routing

```rust
pub enum ModelSize {
    Small,  // Simple tasks: classification, yes/no, entity type resolution
    Medium, // Complex tasks: extraction, summarization, contradiction detection
}
```

### 9.4 Retry Strategy

- Max 4 attempts with random exponential backoff (via `backoff` crate)
- Retry on: HTTP 5xx, rate limit (429), JSON parse errors
- Fail fast on: authentication errors (401), refusal errors
- Append error context to retry messages (so LLM can self-correct)

---

## 10. Prompt System Design

### 10.1 Architecture

Each prompt module is a Rust module exposing public functions that return `Vec<Message>`:

```rust
// prompts/extract_nodes.rs
pub fn extract_message(context: &ExtractNodeContext) -> Vec<Message> { ... }
pub fn extract_json(context: &ExtractNodeContext) -> Vec<Message> { ... }
pub fn extract_text(context: &ExtractNodeContext) -> Vec<Message> { ... }
pub fn classify_nodes(context: &ClassifyContext) -> Vec<Message> { ... }
pub fn extract_attributes(context: &AttributeContext) -> Vec<Message> { ... }
pub fn extract_summary(context: &SummaryContext) -> Vec<Message> { ... }
```

### 10.2 Prompt Templates

Use `format!()` with inline `const` strings:

```rust
const EXTRACT_MESSAGE_SYSTEM: &str = r#"You are an expert at extracting entities from conversational messages..."#;

pub fn extract_message(ctx: &ExtractNodeContext) -> Vec<Message> {
    vec![
        Message::system(format!(
            "{}\n\n{}",
            EXTRACT_MESSAGE_SYSTEM,
            ctx.entity_types_desc()
        )),
        Message::user(format!(
            "PREVIOUS MESSAGES:\n{}\n\nCURRENT MESSAGE:\n{}",
            ctx.previous, ctx.current
        )),
    ]
}
```

### 10.3 Response Models

All LLM response types need both `Deserialize` and `JsonSchema`:

```rust
// prompts/models.rs

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractedEntities {
    pub extracted_entities: Vec<ExtractedEntity>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type_id: i32,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractedEdges {
    pub extracted_edges: Vec<ExtractedEdge>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractedEdge {
    pub source_entity_name: String,
    pub target_entity_name: String,
    pub relation_type: String,
    pub fact: String,
    pub valid_at: Option<String>,
    pub invalid_at: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct NodeResolution {
    pub uuid: Option<String>,
    pub is_duplicate: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EdgeDuplicateResult {
    pub is_duplicate: bool,
    pub is_contradiction: bool,
}
```

### 10.4 Prompt Modules

| Module | Functions | Response Type |
|--------|-----------|---------------|
| `extract_nodes` | `extract_message`, `extract_json`, `extract_text`, `classify_nodes`, `extract_attributes`, `extract_summary`, `extract_summaries_batch` | `ExtractedEntities`, `EntitySummary`, etc. |
| `extract_edges` | `edge`, `extract_attributes` | `ExtractedEdges` |
| `dedupe_nodes` | `node`, `nodes`, `node_list` | `NodeResolutions` |
| `dedupe_edges` | `resolve_edge` | `EdgeDuplicateResult` |
| `summarize_nodes` | `summarize_pair`, `summarize_context`, `summary_description` | `Summary`, `SummaryDescription` |
| `eval` | `query_expansion`, `qa_prompt`, `eval_prompt`, `eval_add_episode_results` | `QueryExpansion`, `QAResponse`, etc. |

---

## 11. Error Handling Strategy

### 11.1 Error Type Hierarchy

```rust
#[derive(Debug, thiserror::Error)]
pub enum GraphitiError {
    #[error("Neo4j error: {0}")]
    Driver(#[from] neo4rs::Error),

    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    #[error("Embedder error: {0}")]
    Embedder(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("Rate limited")]
    RateLimit,

    #[error("Model refused to respond")]
    Refusal,

    #[error("Empty response from LLM")]
    EmptyResponse,

    #[error("Authentication failed")]
    Authentication,

    #[error("API error: HTTP {status} — {message}")]
    Api { status: u16, message: String },
}
```

### 11.2 Result Type

```rust
pub type Result<T> = std::result::Result<T, GraphitiError>;
```

### 11.3 Error Handling Strategy

- **Public API**: Returns `GraphitiError` — consumers can pattern-match
- **Internal functions**: Use `anyhow::Context` for rich error propagation
- **LLM calls**: `LlmError` variants enable retry decisions (retry on `RateLimit`, fail fast on `Authentication`)
- **Driver errors**: Auto-convert via `#[from] neo4rs::Error`

---

## 12. Testing Strategy

### 12.1 Test Layers

| Layer | What | How | Tools |
|-------|------|-----|-------|
| **Unit** | Pure functions (BM25 scoring, RRF, MMR, cosine similarity, Lucene sanitize, UnionFind, content chunking) | Direct function calls with known inputs/outputs | `#[cfg(test)]` modules |
| **Integration** | Neo4j driver operations, LLM client calls | Testcontainers for Neo4j, mock HTTP server for LLM | `testcontainers-rs`, `wiremock` |
| **Prompt** | Prompt template generation | Snapshot testing of generated prompts | `insta` crate |
| **Pipeline** | End-to-end ingestion and search | Full stack with Neo4j testcontainer + LLM mock | Custom test harness |
| **Property** | Temporal invariants, dedup idempotency | Randomized inputs checking properties | `proptest` |

### 12.2 Key Test Cases

**Temporal Invariants**:
- Adding a contradicting fact sets `expired_at` on the old edge but does not delete it
- `valid_at` / `invalid_at` are preserved through deduplication
- Point-in-time queries return correct temporal slice
- An edge with `expired_at` set is still returned in historical queries

**Deduplication**:
- Exact name match resolves to existing node
- Fuzzy match (high Jaccard similarity) resolves correctly
- Different entities with similar names are NOT merged
- UnionFind path compression produces correct equivalence classes
- Low-entropy names (e.g., "yes", "ok") are filtered by entropy threshold

**Search**:
- BM25 returns results for keyword matches
- Cosine similarity returns semantically similar results
- RRF correctly merges multiple ranked lists
- MMR produces diverse result sets
- Filters correctly restrict temporal and label dimensions
- Empty queries return empty results gracefully

**Edge Cases**:
- Empty input text produces no entities/edges
- Unicode/emoji in entity names
- Very long input requiring chunking
- Concurrent episode ingestion doesn't corrupt graph state
- LLM returns invalid JSON → retry with error context
- Network timeout → exponential backoff

### 12.3 Test Infrastructure

```toml
[dev-dependencies]
tokio-test = "0.4"
testcontainers = "0.23"
wiremock = "0.6"
insta = { version = "1", features = ["json"] }
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }
```

---

## 13. Edge Cases & Risks

### 13.1 Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `neo4rs` 0.8 lacks Bolt 5.x features (element IDs) | High | Evaluate 0.9.0-rc.9; use string UUIDs instead of element IDs |
| `async-openai` structured output API may differ from Python `beta.parse` | Medium | Verify JSON schema generation matches; test with actual API calls |
| MinHash/LSH ecosystem immature in Rust | Medium | Implement custom LSH on top of `probminhash`; test against Python baseline |
| Embedding dimension mismatch between providers | Low | Runtime dimension config (not compile-time) |
| Neo4j DateTime serialization differences | Medium | Custom `chrono` ↔ Neo4j `BoltDateTime` conversion layer |
| Large Cypher query string management (~1000+ templates) | Medium | Organize as `const &str` per operation file; consider `include_str!()` for long queries |
| `neo4rs` documentation coverage is only 19.88% | Medium | Read source code; rely on integration tests to validate behavior |

### 13.2 Semantic Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Prompt behavior drift | LLM prompts tuned for Python output format may need adjustment for Rust JSON schemas | Snapshot-test all prompts; compare extraction results with Python baseline |
| Temporal logic correctness | Bi-temporal invalidation is subtle; bugs here corrupt knowledge graph integrity | Property-based testing of temporal invariants |
| Dedup false positives | Incorrect entity merging destroys graph structure | Conservative thresholds; extensive test coverage |
| Concurrency bugs | Race conditions in parallel dedup/persistence | Use atomic operations; careful `Arc<Mutex<>>` usage; test with high concurrency |

### 13.3 Scope Risks

| Risk | Mitigation |
|------|------------|
| 4 graph backends is too much for initial port | Phase 1: Neo4j only. Trait design ensures backends can be added later. |
| 6 LLM providers is too much | Phase 1: OpenAI only. Trait design allows extension. |
| Community detection is complex | Phase 3. Label propagation is algorithmically straightforward; LLM summarization reuses existing client. |
| 11 operation interfaces × ~10 methods each = ~110 trait methods | Macro-generate common CRUD patterns; focus on correctness over cleverness. |

### 13.4 Crate-Specific Risks

| Crate | Risk | Mitigation |
|-------|------|------------|
| `neo4rs` | v0.9 RC may not stabilize | Pin to 0.8; test compatibility |
| `async-openai` | Structured output schema format | Test with schemars output against OpenAI API |
| `bm25` | Tokenization differences from Python | Compare BM25 scores with Python baseline on same corpus |
| `probminhash` | LSH banding must be custom-built | Implement LSH hash banding on top of MinHash signatures |
| `schemars` | Generated schemas may not match OpenAI expectations | Validate schemas with OpenAI's schema validator; add `#[schemars(rename_all = "camelCase")]` if needed |

---

## 14. Directory Structure

```
graphiti-rs/
├── Cargo.toml
├── PORTING_PLAN.md
├── README.md
├── src/
│   ├── lib.rs                          # Public API: re-export Graphiti struct
│   ├── graphiti.rs                     # Facade: add_episode, search, build_communities
│   ├── types.rs                        # GraphitiClients, config types, EpisodeInput
│   ├── errors.rs                       # GraphitiError, LlmError
│   │
│   ├── nodes/
│   │   ├── mod.rs                      # NodeKind enum, common Node methods
│   │   ├── entity.rs                   # EntityNode
│   │   ├── episodic.rs                 # EpisodicNode, EpisodeType
│   │   ├── community.rs               # CommunityNode
│   │   └── saga.rs                     # SagaNode
│   │
│   ├── edges/
│   │   ├── mod.rs                      # EdgeKind enum, common Edge methods
│   │   ├── entity.rs                   # EntityEdge (bi-temporal)
│   │   ├── episodic.rs                 # EpisodicEdge (MENTIONS)
│   │   ├── community.rs               # CommunityEdge (HAS_MEMBER)
│   │   ├── has_episode.rs             # HasEpisodeEdge
│   │   └── next_episode.rs           # NextEpisodeEdge
│   │
│   ├── driver/
│   │   ├── mod.rs                      # GraphDriver trait
│   │   ├── traits.rs                   # 11 operation trait definitions
│   │   ├── record.rs                   # Record parsing utilities (neo4rs Row → structs)
│   │   └── neo4j/
│   │       ├── mod.rs                  # Neo4jDriver struct, GraphDriver impl
│   │       ├── entity_node_ops.rs      # EntityNodeOps impl
│   │       ├── episode_node_ops.rs     # EpisodeNodeOps impl
│   │       ├── community_node_ops.rs   # CommunityNodeOps impl
│   │       ├── saga_node_ops.rs        # SagaNodeOps impl
│   │       ├── entity_edge_ops.rs      # EntityEdgeOps impl
│   │       ├── episodic_edge_ops.rs    # EpisodicEdgeOps impl
│   │       ├── community_edge_ops.rs   # CommunityEdgeOps impl
│   │       ├── has_episode_edge_ops.rs # HasEpisodeEdgeOps impl
│   │       ├── next_episode_edge_ops.rs # NextEpisodeEdgeOps impl
│   │       ├── search_ops.rs           # SearchOps impl
│   │       ├── graph_ops.rs            # GraphOps impl
│   │       └── queries.rs             # Cypher query template constants
│   │
│   ├── llm_client/
│   │   ├── mod.rs                      # LlmClient trait, LlmConfig, ModelSize
│   │   ├── openai.rs                   # OpenAI implementation
│   │   ├── cache.rs                    # Response caching (moka)
│   │   └── retry.rs                    # Retry logic with backoff
│   │
│   ├── embedder/
│   │   ├── mod.rs                      # EmbedderClient trait, EmbedderConfig
│   │   └── openai.rs                   # OpenAI embeddings implementation
│   │
│   ├── cross_encoder/
│   │   ├── mod.rs                      # CrossEncoderClient trait
│   │   └── openai.rs                   # OpenAI reranker implementation
│   │
│   ├── search/
│   │   ├── mod.rs                      # search() entry point
│   │   ├── config.rs                   # SearchConfig, SearchMethod, RerankerType enums
│   │   ├── recipes.rs                  # 17 pre-built search configurations
│   │   ├── filters.rs                  # SearchFilters, DateFilter, PropertyFilter
│   │   ├── edge_search.rs             # Edge search pipeline
│   │   ├── node_search.rs             # Node search pipeline
│   │   ├── episode_search.rs          # Episode search pipeline
│   │   ├── community_search.rs        # Community search pipeline
│   │   └── rerankers.rs               # RRF, MMR, cross-encoder, node_distance, episode_mentions
│   │
│   ├── prompts/
│   │   ├── mod.rs                      # Message type, Role enum
│   │   ├── models.rs                   # All response model structs (Deserialize + JsonSchema)
│   │   ├── extract_nodes.rs            # Entity extraction prompt builders
│   │   ├── extract_edges.rs            # Edge extraction prompt builders
│   │   ├── dedupe_nodes.rs             # Node deduplication prompt builders
│   │   ├── dedupe_edges.rs             # Edge dedup & contradiction prompt builders
│   │   ├── summarize_nodes.rs          # Summarization prompt builders
│   │   └── eval.rs                     # Evaluation/QA prompt builders
│   │
│   ├── pipeline/
│   │   ├── mod.rs                      # Pipeline orchestration types
│   │   ├── extract.rs                  # Entity + edge extraction step
│   │   ├── dedupe.rs                   # Node + edge deduplication step
│   │   ├── persist.rs                  # Graph persistence step
│   │   └── community.rs               # Community detection + update step
│   │
│   └── utils/
│       ├── mod.rs                      # Re-exports
│       ├── chunking.rs                 # Content chunking strategies
│       ├── datetime.rs                 # Temporal utilities (parsing "2 days ago", etc.)
│       ├── text.rs                     # Text processing, Lucene sanitize
│       ├── semaphore.rs                # Bounded concurrent execution helper
│       ├── union_find.rs               # UnionFind data structure for batch dedup
│       ├── similarity.rs              # Cosine similarity, Jaccard, normalize_l2
│       └── dedup_helpers.rs           # MinHash, LSH, candidate indexing, entropy filter
│
├── tests/
│   ├── integration/
│   │   ├── neo4j_driver_test.rs       # Driver operations with testcontainers Neo4j
│   │   ├── ingestion_test.rs          # End-to-end ingestion with mock LLM
│   │   └── search_test.rs            # Search pipelines with mock data
│   └── common/
│       └── mod.rs                      # Test utilities, fixtures, mock builders
│
└── benches/
    ├── search_bench.rs                # Search pipeline benchmarks
    └── similarity_bench.rs            # Vector similarity benchmarks
```

---

## 15. Subtask Plan

```yaml
max_parallel_subtasks: 3

subtasks:
  # ── Group 1: Foundation ──────────────────────────────────────────
  - id: "01"
    title: "Implement core error types (GraphitiError, LlmError) in src/errors.rs"
    group: 1
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Port error types from errors.py. Add #[from] conversions for neo4rs::Error,
      serde_json::Error. Add anyhow::Error escape hatch. Unit test Display impls.

  - id: "02"
    title: "Implement node types (EntityNode, EpisodicNode, CommunityNode, SagaNode) with serde"
    group: 1
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Port from nodes.py. Define NodeKind enum, EpisodeType enum. All structs get
      Serialize, Deserialize, Clone, Debug. EntityNode has HashMap<String, Value> attributes.
      Test roundtrip serialization. Test tagged enum deserialization.

  - id: "03"
    title: "Implement edge types (EntityEdge with bi-temporal fields, all 5 edge variants)"
    group: 1
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Port from edges.py. EntityEdge is critical — must have valid_at, invalid_at,
      created_at, expired_at. Test temporal field serialization. Test that expired_at
      defaults to None. Test HashMap<String, Value> attributes.

  # ── Group 2: Trait Definitions + Utilities ───────────────────────
  - id: "04"
    title: "Define GraphDriver trait and 11 operation traits in src/driver/traits.rs"
    group: 2
    depends_on: ["02", "03"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Define all 11 operation traits matching Python ABCs. Use async methods with
      Send bounds. Each trait has save, get_by_uuid, get_by_group_ids, delete plus
      type-specific methods. SearchOps has bm25_search, vector_search, bfs_search.
      GraphOps has build_indices, get_graph_stats, clear_data.

  - id: "05"
    title: "Define LlmClient, EmbedderClient, CrossEncoderClient traits and GraphitiClients container"
    group: 2
    depends_on: ["01"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      LlmClient: generate_response<T: DeserializeOwned + JsonSchema>, generate_text.
      EmbedderClient: create, create_batch, embedding_dim.
      CrossEncoderClient: rank.
      GraphitiClients: Arc<dyn Trait> for all clients. ModelSize enum. Message struct.

  - id: "06"
    title: "Implement utility modules: similarity, union_find, text, datetime, semaphore"
    group: 2
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      cosine_similarity(a, b) -> f32 via ndarray dot product.
      normalize_l2(v) -> Vec<f32>.
      jaccard_similarity(a, b) -> f32.
      UnionFind<T: Eq + Hash>: new, union, find, sets.
      lucene_sanitize(s) -> String.
      parse_datetime(s) -> Option<DateTime<Utc>>.
      semaphore_gather() helper.
      All with comprehensive unit tests.

  # ── Group 3: LLM & Embedder Implementations ─────────────────────
  - id: "07"
    title: "Implement OpenAI LLM client with structured output, caching, and retry"
    group: 3
    depends_on: ["05"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      OpenAiClient struct wrapping async_openai::Client. generate_response<T> uses
      schemars::schema_for!() for JSON schema, passes to response_format.
      moka::future::Cache for MD5-keyed response caching.
      backoff crate for exponential retry on 429/5xx.
      wiremock tests for all error paths.

  - id: "08"
    title: "Implement OpenAI embedder client with batch support"
    group: 3
    depends_on: ["05"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      OpenAiEmbedder using async_openai embeddings API.
      create() for single input, create_batch() for multiple.
      L2 normalization of returned vectors.
      Dimension truncation if configured.

  - id: "09"
    title: "Port all prompt templates and response models"
    group: 3
    depends_on: ["05"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Port extract_nodes.py, extract_edges.py, dedupe_nodes.py, dedupe_edges.py,
      summarize_nodes.py, eval.py prompts.
      All response models get Deserialize + JsonSchema.
      Snapshot test all generated prompts with insta.

  # ── Group 4: Neo4j Driver ────────────────────────────────────────
  - id: "10"
    title: "Implement Neo4j driver core (connection, query execution, index building)"
    group: 4
    depends_on: ["04"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Neo4jDriver struct wrapping neo4rs::Graph.
      Connection pooling via Graph's built-in deadpool.
      execute_query with parameter binding.
      build_indices_and_constraints with Cypher CREATE INDEX/CONSTRAINT.
      Record → struct deserialization utilities.

  - id: "11"
    title: "Implement Neo4j node operations (entity, episodic, community, saga CRUD)"
    group: 4
    depends_on: ["10"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Implement EntityNodeOps, EpisodeNodeOps, CommunityNodeOps, SagaNodeOps
      for Neo4j. Each has save (MERGE + SET), get_by_uuid (MATCH + RETURN),
      get_by_group_ids, delete (DETACH DELETE). Port Cypher from Python.
      search_by_name_embedding uses Neo4j vector index.

  - id: "12"
    title: "Implement Neo4j edge operations (all 5 edge types CRUD + expire)"
    group: 4
    depends_on: ["10"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Implement all 5 edge operation traits for Neo4j.
      EntityEdgeOps.expire() sets expired_at without deleting.
      search_by_embedding for EntityEdgeOps uses vector index.
      Port all Cypher CREATE/MATCH/SET/DELETE patterns.

  - id: "13"
    title: "Implement Neo4j search and graph operations"
    group: 4
    depends_on: ["10"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      SearchOps: bm25_search (fulltext index), vector_search (vector index),
      bfs_search (variable-length path pattern).
      GraphOps: build_indices, get_graph_stats, clear_data, get_communities.
      Filter → Cypher WHERE clause generation.

  # ── Group 5: Search Subsystem ────────────────────────────────────
  - id: "14"
    title: "Implement search config, filters, and 17 search recipes"
    group: 5
    depends_on: ["04"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      SearchConfig struct with search_methods and rerankers fields.
      SearchFilters with CNF date filters, property filters.
      ComparisonOperator enum.
      17 const recipe instances matching Python search_config_recipes.py.

  - id: "15"
    title: "Implement rerankers (RRF, MMR, cross-encoder, node_distance, episode_mentions)"
    group: 5
    depends_on: ["06"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      RRF: Σ 1/(rank_i + k) fusion. Test with known ranked lists.
      MMR: λ * sim(q,d) + (λ-1) * max_sim(d, selected). Test diversity.
      cross_encoder: delegate to CrossEncoderClient.
      node_distance: shortest path scoring.
      episode_mentions: MENTIONS count scoring.
      All with unit tests on known inputs/outputs.

  - id: "16"
    title: "Implement search pipelines (edge, node, episode, community search)"
    group: 5
    depends_on: ["13", "14", "15"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Four search functions: edge_search, node_search, episode_search, community_search.
      Each dispatches to configured search methods, collects results, applies rerankers.
      Top-level search() dispatches based on SearchConfig.
      Integration test with mock driver.

  # ── Group 6: Ingestion Pipeline ──────────────────────────────────
  - id: "17"
    title: "Implement content chunking module (JSON, text, message strategies)"
    group: 6
    depends_on: ["06"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      ChunkStrategy enum: Json, Text, Message.
      should_chunk() heuristic based on content length/density.
      chunk_content() splits with overlap.
      Test boundary entity preservation across chunks.

  - id: "18"
    title: "Implement dedup helpers (MinHash, LSH, candidate indexing, entropy filter)"
    group: 6
    depends_on: ["06"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      MinHash signature generation via probminhash.
      LSH banding: hash bands of signature → candidate pairs.
      Entropy filtering: skip low-information names.
      Jaccard similarity threshold.
      Test: known duplicate pairs detected, non-duplicates not merged.

  - id: "19"
    title: "Implement extraction pipeline (node + edge extraction using LLM prompts)"
    group: 6
    depends_on: ["07", "09"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      extract_entities(): build prompt context, call LLM, parse ExtractedEntities.
      extract_edges(): build prompt context, call LLM, parse ExtractedEdges.
      Concurrent extraction via semaphore_gather.
      Test with mock LLM returning known JSON.

  - id: "20"
    title: "Implement dedup pipeline (node + edge dedup with temporal invalidation)"
    group: 6
    depends_on: ["18", "19"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      deduplicate_nodes(): 3-tier (exact → MinHash/LSH → LLM). Returns uuid_map.
      deduplicate_edges(): embedding similarity + word overlap + LLM classification.
      invalidate_edges(): set expired_at on contradicted edges (CRITICAL).
      Property test: expired_at is set, edge is not deleted.

  - id: "21"
    title: "Implement persistence pipeline (node/edge save, episodic edges, episode sequencing)"
    group: 6
    depends_on: ["11", "12"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      persist_nodes(): save resolved nodes, generate embeddings.
      persist_edges(): save edges with fact embeddings.
      create_episodic_edges(): MENTIONS relationships.
      sequence_episodes(): NEXT_EPISODE chain.

  # ── Group 7: Facade & Integration ───────────────────────────────
  - id: "22"
    title: "Implement Graphiti facade (add_episode, search, build_communities)"
    group: 7
    depends_on: ["16", "17", "19", "20", "21"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      pub struct Graphiti with GraphitiClients and Semaphore.
      add_episode(): chunk → extract → dedup → embed → persist → community.
      search(): delegate to search module.
      build_communities(): trigger community detection.
      Integration test with mock everything.

  - id: "23"
    title: "Implement community detection (label propagation) and summarization"
    group: 7
    depends_on: ["22"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Label propagation via petgraph on entity adjacency.
      Community summarization via LLM prompts.
      Save/update CommunityNode and CommunityEdge.
      Test: known graph produces expected communities.

  - id: "24"
    title: "Implement bulk ingestion pipeline"
    group: 7
    depends_on: ["22"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      add_episode_bulk(): parallel extraction, two-pass dedup with UnionFind,
      batch persistence. Transaction management for atomicity.
      Test: bulk of 10 episodes produces correct graph.

  # ── Group 8: Integration Tests ──────────────────────────────────
  - id: "25"
    title: "Write integration tests: Neo4j driver with testcontainers"
    group: 8
    depends_on: ["11", "12", "13"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Spin up Neo4j via testcontainers-rs.
      Test all CRUD operations for nodes and edges.
      Test search operations (BM25, vector, BFS).
      Test index creation.

  - id: "26"
    title: "Write integration tests: end-to-end ingestion pipeline with mock LLM"
    group: 8
    depends_on: ["22"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      wiremock for LLM API.
      testcontainers for Neo4j.
      Test: ingest a conversation, verify entities/edges in graph.
      Test: ingest contradicting fact, verify temporal invalidation.

  - id: "27"
    title: "Write integration tests: search pipeline with real Neo4j"
    group: 8
    depends_on: ["16"]
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Seed Neo4j with known test data.
      Test all 4 search pipelines.
      Test all filter combinations.
      Test RRF/MMR produce expected rankings.

  # ── Group 9: Polish ─────────────────────────────────────────────
  - id: "28"
    title: "Add benchmarks for search and similarity functions"
    group: 9
    depends_on: ["16"]
    agent: "rust-coding-agent"
    tags: [tdd-refactor]
    description: |
      criterion benchmarks for: cosine_similarity, RRF, MMR, BM25 scoring.
      Compare with baseline expectations.

  - id: "29"
    title: "Write README.md with usage examples and architecture overview"
    group: 9
    depends_on: ["22"]
    agent: "doc-agent"
    tags: [docs]

  - id: "30"
    title: "Property-based tests for temporal invariants and dedup correctness"
    group: 9
    depends_on: ["20", "22"]
    agent: "rust-coding-agent"
    tags: [tdd-refactor]
    description: |
      proptest strategies for EntityEdge with random temporal fields.
      Property: expired_at never causes deletion.
      Property: dedup is idempotent (running twice gives same result).
      Property: UnionFind equivalence classes are correct.
```

---

## Appendix: Crate Versions

```toml
[package]
name = "graphiti-rs"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }
futures = "0.3"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "0.8"

# Validation
validator = { version = "0.20", features = ["derive"] }

# HTTP
reqwest = { version = "0.13", features = ["json"] }

# Neo4j
neo4rs = "0.8"

# OpenAI
async-openai = "0.33"

# Numeric / Vector
ndarray = "0.17"

# BM25
bm25 = "2.3"

# UUID
uuid = { version = "1", features = ["v4", "v7", "serde"] }

# DateTime
chrono = { version = "0.4", features = ["serde"] }

# Graph algorithms
petgraph = { version = "0.8", features = ["serde-1"] }

# MinHash
probminhash = "0.1"

# Concurrent data structures
dashmap = "6"
moka = { version = "0.12", features = ["future"] }

# Error handling
thiserror = "2"
anyhow = "1"

# Retry
backoff = { version = "0.4", features = ["tokio"] }

# Hashing (cache keys)
md-5 = "0.10"

# Regex
regex = "1"

# Environment
dotenvy = "0.15"

# Tracing / Telemetry
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
tokio-test = "0.4"
testcontainers = "0.23"
wiremock = "0.6"
insta = { version = "1", features = ["json"] }
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "search"
harness = false

[[bench]]
name = "similarity"
harness = false
```

---

## Appendix: Python-to-Rust Pattern Mapping

### Pydantic → Serde

```python
# Python
class EntityNode(BaseModel):
    uuid: str
    name: str
    attributes: dict[str, Any] = {}
```

```rust
// Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub uuid: String,
    pub name: String,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}
```

### ABC → Trait

```python
# Python
class LLMClient(ABC):
    @abstractmethod
    async def generate_response(self, messages, response_model) -> T: ...
```

```rust
// Rust
pub trait LlmClient: Send + Sync {
    async fn generate_response<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: &[Message],
        model_size: ModelSize,
    ) -> Result<T>;
}
```

### asyncio.gather + Semaphore → Tokio

```python
# Python
async def semaphore_gather(semaphore, tasks):
    async def bounded(task):
        async with semaphore:
            return await task
    return await asyncio.gather(*[bounded(t) for t in tasks])
```

```rust
// Rust
pub async fn semaphore_gather<T, F, Fut, R>(
    semaphore: &Arc<Semaphore>,
    items: Vec<T>,
    f: F,
) -> Vec<Result<R>>
where
    F: Fn(T) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<R>> + Send,
    T: Send + 'static,
    R: Send + 'static,
{
    let futures = items.into_iter().map(|item| {
        let sem = semaphore.clone();
        let func = f.clone();
        async move {
            let _permit = sem.acquire().await.unwrap();
            func(item).await
        }
    });
    futures::future::join_all(futures).await
}
```

### Optional with default → Option<T>

```python
# Python
expired_at: datetime | None = None
```

```rust
// Rust
#[serde(skip_serializing_if = "Option::is_none")]
pub expired_at: Option<DateTime<Utc>>,
```

### Dynamic attributes → HashMap<String, Value>

```python
# Python
attributes: dict[str, Any] = {}
```

```rust
// Rust
#[serde(default)]
pub attributes: HashMap<String, serde_json::Value>,
```

---

## Appendix: Concurrency Patterns

### Pattern 1: Semaphore-bounded LLM calls

```rust
// Used for: entity extraction, edge extraction, dedup LLM calls
let semaphore = Arc::new(Semaphore::new(20)); // 20 concurrent calls

let results: Vec<Result<ExtractedEntities>> = semaphore_gather(
    &semaphore,
    chunks,
    |chunk| async move {
        llm_client.generate_response::<ExtractedEntities>(
            &extract_nodes::extract_message(&chunk),
            ModelSize::Medium,
        ).await
    },
).await;
```

### Pattern 2: JoinSet for uniform concurrent tasks

```rust
// Used for: parallel node saves, parallel edge saves
let mut set = JoinSet::new();
for node in nodes {
    let driver = driver.clone();
    set.spawn(async move {
        driver.entity_node_ops().save(&node).await
    });
}
let mut results = Vec::new();
while let Some(result) = set.join_next().await {
    results.push(result??);
}
```

### Pattern 3: Stream-based buffered processing

```rust
// Used for: bulk embedding generation
let embeddings: Vec<Vec<f32>> = stream::iter(text_chunks)
    .chunks(100) // batch size
    .then(|batch| async { embedder.create_batch(&batch).await })
    .try_collect()
    .await?;
```

### Pattern 4: Channel-based pipeline stages

```rust
// Used for: decoupling extraction from persistence in bulk mode
let (extract_tx, mut extract_rx) = mpsc::channel(100);
let (persist_tx, mut persist_rx) = mpsc::channel(100);

// Extraction stage
tokio::spawn(async move {
    while let Some(episode) = extract_rx.recv().await {
        let entities = extract(episode).await?;
        persist_tx.send(entities).await?;
    }
});

// Persistence stage
tokio::spawn(async move {
    while let Some(entities) = persist_rx.recv().await {
        persist(entities).await?;
    }
});
```
