//! Prompt templates for LLM interactions.
//!
//! Mirrors the Python `graphiti_core.prompts` package (~17 template files).
//! Each submodule handles prompts for a specific pipeline stage.
//!
//! Prompts are stored as Rust string literals (not external files) for
//! compile-time inclusion and zero-cost access.

// TODO: extract_json — episode content → JSON entities/relationships
// TODO: extract_nodes — entity extraction prompt
// TODO: extract_edges — relationship extraction prompt
// TODO: dedupe_nodes — node deduplication / resolution
// TODO: dedupe_edges — edge deduplication
// TODO: resolve_edge_contradictions — temporal contradiction detection
// TODO: summarize_nodes — entity summary generation
// TODO: summarize_communities — community label propagation summary
