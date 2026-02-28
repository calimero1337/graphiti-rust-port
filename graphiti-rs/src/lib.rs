//! # graphiti-rs
//!
//! Rust port of [Graphiti](https://github.com/getzep/graphiti) — a framework for constructing
//! and querying temporally-aware knowledge graphs for AI agents.
//!
//! ## Architecture
//!
//! - **Bi-temporal data model**: Tracks real-world validity and graph transaction time
//! - **Incremental ingestion**: Episodes integrate without full graph recomputation
//! - **Hybrid retrieval**: BM25 + vector cosine similarity + graph BFS traversal
//! - **LLM-powered extraction**: Entity/edge extraction, deduplication, contradiction detection

pub mod edges;
pub mod errors;
pub mod nodes;
pub mod types;

pub mod driver;
pub mod embedder;
pub mod llm_client;
// pub mod cross_encoder;

pub mod prompts;
pub mod search;

pub mod pipeline;
pub mod utils;

// Re-export the main facade
// pub use graphiti::Graphiti;
pub use errors::{GraphitiError, Result};
