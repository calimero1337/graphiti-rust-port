//! Search subsystem.
//!
//! Mirrors the Python `graphiti_core.search` package. Implements:
//! - **BM25** full-text search (via the `bm25` crate)
//! - **Vector cosine similarity** search
//! - **BFS graph traversal** for neighbourhood expansion
//! - **Hybrid reranking**: RRF, MMR, cross-encoder, episode mentions, node distance
//!
//! The four search pipelines from the Python `search_utils.py`:
//! 1. `node_search` — entity node retrieval
//! 2. `edge_search` — entity edge retrieval
//! 3. `episode_search` — episodic node retrieval
//! 4. `community_search` — community node retrieval

// TODO: Implement SearchConfig and SearchPipeline
// TODO: Implement BM25Searcher
// TODO: Implement VectorSearcher
// TODO: Implement HybridReranker (RRF + MMR + cross-encoder + node-distance + episode-mentions)
// TODO: Implement the 17 pre-defined search recipes from Python search_config.py
