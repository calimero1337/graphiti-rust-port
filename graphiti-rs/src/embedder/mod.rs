//! Embedder client abstraction.
//!
//! Mirrors the Python `graphiti_core.embedder` module.
//! Provides a trait for generating vector embeddings from text.
//!
//! Phase 1 target: OpenAI `text-embedding-3-small` (1536-dim) via `async-openai`.
//! Phase 2: Voyage AI, Gemini, local models.

use crate::errors::Result;

/// A vector embedding (f32 components).
pub type Embedding = Vec<f32>;

/// Trait for text-to-vector embedding clients.
#[allow(async_fn_in_trait)]
pub trait EmbedderClient: Send + Sync {
    /// Generate an embedding for a single text string.
    async fn embed(&self, text: &str) -> Result<Embedding>;

    /// Generate embeddings for a batch of texts.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>>;

    /// Returns the dimensionality of embeddings produced by this client.
    fn dim(&self) -> usize;
}
