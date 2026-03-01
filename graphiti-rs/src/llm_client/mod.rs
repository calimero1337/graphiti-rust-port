//! LLM client abstraction.
//!
//! # Implementations
//! - [`openai::OpenAiClient`] — OpenAI GPT-4o (and variants) via `async-openai`.

pub mod openai;
//!
//! Mirrors the Python `graphiti_core.llm_client` module.
//! Provides a trait for calling language models with structured output support.
//!
//! Phase 1 target: OpenAI GPT-4o via `async-openai` with `schemars`-generated JSON schemas.
//! Phase 2: Anthropic Claude, Google Gemini, Groq, Azure OpenAI.

use crate::errors::Result;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// A chat message for the LLM conversation.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Speaker role in a chat conversation.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Trait for LLM clients supporting structured output (JSON schema).
#[allow(async_fn_in_trait)]
pub trait LlmClient: Send + Sync {
    /// Send a request and parse the response as plain text.
    async fn generate(&self, messages: &[Message]) -> Result<String>;

    /// Send a request and parse the response as a structured JSON type.
    ///
    /// Uses JSON schema derived from `T` (via `schemars`) to constrain the model output.
    async fn generate_structured<T>(&self, messages: &[Message]) -> Result<T>
    where
        T: DeserializeOwned + schemars::JsonSchema;
}
