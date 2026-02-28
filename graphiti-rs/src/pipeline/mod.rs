//! Ingestion pipeline.
//!
//! Mirrors the Python `graphiti_core.graphiti.add_episode` flow.
//!
//! The pipeline stages in order:
//! 1. **Parse** — convert raw episode to typed content (text/message/JSON)
//! 2. **Extract** — LLM extracts entity nodes and typed edges
//! 3. **Deduplicate nodes** — 3-tier: exact match → MinHash/LSH → LLM
//! 4. **Deduplicate edges** — same 3-tier pipeline
//! 5. **Resolve contradictions** — LLM invalidates superseded edges (bi-temporal)
//! 6. **Persist** — write episode, nodes, edges to graph database
//! 7. **Embed** — generate and store embeddings for new/updated nodes

// TODO: Implement Episode ingestion entry point
// TODO: Implement node deduplication (exact → MinHash/LSH → LLM arbitration)
// TODO: Implement edge deduplication and temporal invalidation
// TODO: Implement bulk ingestion (parallel episodes with semaphore)
