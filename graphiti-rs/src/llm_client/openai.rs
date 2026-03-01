//! OpenAI LLM client implementation.
//!
//! Uses `async-openai` for API calls, `moka` for response caching, and
//! `backoff` for exponential-backoff retry on rate limits / transient errors.

use std::time::Duration;

use backoff::ExponentialBackoffBuilder;
use moka::future::Cache;
use serde::de::DeserializeOwned;
use serde_json::json;
use tracing::{debug, warn};

use crate::errors::{GraphitiError, LlmError, Result};

use super::{LlmClient, Message, Role};

// ── Cache configuration ───────────────────────────────────────────────────────

/// Configuration for the in-process response cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries held in memory.
    pub max_capacity: u64,
    /// How long each entry lives before eviction.
    pub ttl: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 1_000,
            ttl: Duration::from_secs(3_600), // 1 hour
        }
    }
}

// ── Client struct ─────────────────────────────────────────────────────────────

/// OpenAI LLM client implementing [`LlmClient`].
pub struct OpenAiClient {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    model: String,
    temperature: f32,
    max_tokens: u32,
    /// Keyed by `{type_name}:{md5(model + messages)}` → serialised response text.
    cache: Cache<String, String>,
}

impl OpenAiClient {
    /// Create a new client.
    ///
    /// # Arguments
    /// * `api_key` – OpenAI secret key.
    /// * `model`   – Model name (e.g. `"gpt-4o"`).
    /// * `cache_config` – Cache capacity and TTL.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        cache_config: CacheConfig,
    ) -> Self {
        let config = async_openai::config::OpenAIConfig::new().with_api_key(api_key);
        let client = async_openai::Client::with_config(config);

        let cache = Cache::builder()
            .max_capacity(cache_config.max_capacity)
            .time_to_live(cache_config.ttl)
            .build();

        Self {
            client,
            model: model.into(),
            temperature: 0.0,
            max_tokens: 8_192,
            cache,
        }
    }

    /// Override the sampling temperature (default `0.0`).
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Override the max output token limit (default `8192`).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Compute an MD5 cache key from model + message sequence.
    fn cache_key(&self, prefix: &str, messages: &[Message]) -> String {
        use md5::{Digest, Md5};
        let mut h = Md5::new();
        h.update(prefix.as_bytes());
        h.update(self.model.as_bytes());
        for m in messages {
            let role = role_str(&m.role);
            h.update(role.as_bytes());
            h.update(m.content.as_bytes());
        }
        format!("{:x}", h.finalize())
    }

    /// Serialise our [`Message`] slice into the JSON array expected by the API.
    fn messages_to_json(messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|m| {
                json!({
                    "role": role_str(&m.role),
                    "content": m.content,
                })
            })
            .collect()
    }

    /// Call the chat completions endpoint with exponential-backoff retry.
    ///
    /// Retries on [`LlmError::RateLimit`] (HTTP 429) and transient 5xx errors.
    async fn call_with_retry(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(500))
            .with_max_interval(Duration::from_secs(60))
            .with_max_elapsed_time(Some(Duration::from_secs(300)))
            .build();

        backoff::future::retry(backoff, || async {
            let outcome: std::result::Result<serde_json::Value, async_openai::error::OpenAIError> =
                self.client.chat().create_byot(request.clone()).await;

            match outcome {
                Ok(response) => Ok(response),
                Err(e) => {
                    let llm_err = map_openai_error(e);
                    match &llm_err {
                        LlmError::RateLimit => {
                            warn!("OpenAI rate limit hit — retrying with backoff");
                            Err(backoff::Error::transient(llm_err))
                        }
                        LlmError::Api { status, .. } if *status >= 500 => {
                            warn!("OpenAI transient server error ({}) — retrying", status);
                            Err(backoff::Error::transient(llm_err))
                        }
                        _ => Err(backoff::Error::permanent(llm_err)),
                    }
                }
            }
        })
        .await
        .map_err(GraphitiError::Llm)
    }

    /// Extract the assistant message text from a chat-completions response.
    fn extract_content(response: &serde_json::Value) -> Result<String> {
        response["choices"][0]["message"]["content"]
            .as_str()
            .map(ToOwned::to_owned)
            .ok_or(GraphitiError::Llm(LlmError::EmptyResponse))
    }
}

// ── LlmClient implementation ──────────────────────────────────────────────────

impl LlmClient for OpenAiClient {
    async fn generate(&self, messages: &[Message]) -> Result<String> {
        let key = self.cache_key("text", messages);

        if let Some(cached) = self.cache.get(&key).await {
            debug!("LLM cache hit (text)");
            return Ok(cached);
        }

        let request = json!({
            "model": self.model,
            "messages": Self::messages_to_json(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        });

        let response = self.call_with_retry(request).await?;
        let content = Self::extract_content(&response)?;

        self.cache.insert(key, content.clone()).await;

        Ok(content)
    }

    async fn generate_structured<T>(&self, messages: &[Message]) -> Result<T>
    where
        T: DeserializeOwned + schemars::JsonSchema,
    {
        // Include the target type name in the cache key so different T for the
        // same messages don't collide.
        let prefix = std::any::type_name::<T>();
        let key = self.cache_key(prefix, messages);

        if let Some(cached) = self.cache.get(&key).await {
            debug!("LLM cache hit (structured/{})", prefix);
            return serde_json::from_str(&cached).map_err(GraphitiError::Serialization);
        }

        // Build the JSON schema from T via schemars.
        let schema = schemars::schema_for!(T);
        let schema_value = serde_json::to_value(&schema)?;

        let request = json!({
            "model": self.model,
            "messages": Self::messages_to_json(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema_value,
                    "strict": true,
                }
            }
        });

        let response = self.call_with_retry(request).await?;
        let content = Self::extract_content(&response)?;

        self.cache.insert(key, content.clone()).await;

        serde_json::from_str(&content).map_err(GraphitiError::Serialization)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// Map an [`async_openai::error::OpenAIError`] to our [`LlmError`] domain type.
fn map_openai_error(err: async_openai::error::OpenAIError) -> LlmError {
    use async_openai::error::OpenAIError;

    match err {
        OpenAIError::ApiError(api_err) => {
            // `ApiError.status` is `Option<u16>` in async-openai ≥ 0.24.
            let status = api_err.status.unwrap_or(0);
            match status {
                401 | 403 => LlmError::Authentication,
                429 => LlmError::RateLimit,
                other => LlmError::Api {
                    status: other,
                    message: api_err.message,
                },
            }
        }
        other => LlmError::Api {
            status: 0,
            message: other.to_string(),
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a client pointing at an arbitrary base URL (mock server).
    fn client_for(base_url: &str) -> OpenAiClient {
        let config =
            async_openai::config::OpenAIConfig::new().with_api_key("test-key").with_api_base(base_url);
        let inner = async_openai::Client::with_config(config);
        OpenAiClient {
            client: inner,
            model: "gpt-4o".to_string(),
            temperature: 0.0,
            max_tokens: 512,
            cache: Cache::builder()
                .max_capacity(100)
                .time_to_live(Duration::from_secs(60))
                .build(),
        }
    }

    fn chat_completions_response(content: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        })
    }

    fn user_messages(text: &str) -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: text.to_string(),
        }]
    }

    // ── generate() ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_generate_returns_content() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("Hello, world!")),
            )
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Say hello");
        let result = client.generate(&msgs).await.expect("generate should succeed");

        assert_eq!(result, "Hello, world!");
    }

    #[tokio::test]
    async fn test_generate_uses_cache_on_second_call() {
        let server = MockServer::start().await;

        // Register the mock for exactly one request.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("cached response")),
            )
            .expect(1) // must be called exactly once
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Same question");

        let r1 = client.generate(&msgs).await.expect("first call");
        let r2 = client.generate(&msgs).await.expect("second call");

        assert_eq!(r1, "cached response");
        assert_eq!(r2, "cached response");
        // wiremock verifies the `expect(1)` on drop
    }

    #[tokio::test]
    async fn test_generate_maps_auth_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "message": "Incorrect API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            })))
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Hello");
        let err = client.generate(&msgs).await.expect_err("should fail");

        assert!(
            matches!(err, GraphitiError::Llm(LlmError::Authentication)),
            "expected Authentication, got {:?}",
            err
        );
    }

    #[tokio::test]
    async fn test_generate_retries_on_rate_limit() {
        let server = MockServer::start().await;

        // First call returns 429, second call succeeds.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "requests",
                    "code": "rate_limit_exceeded"
                }
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response("after retry")),
            )
            .mount(&server)
            .await;

        // Use a very short initial backoff so the test doesn't take forever.
        // We can't easily configure the client's backoff from outside, so we
        // rely on the default (500 ms) and accept the test is slightly slow.
        // In practice, consider exposing BackoffConfig.
        let client = client_for(&server.uri());
        let msgs = user_messages("Hello after rate limit");
        let result = client.generate(&msgs).await.expect("should succeed after retry");
        assert_eq!(result, "after retry");
    }

    // ── generate_structured() ────────────────────────────────────────────────

    #[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    #[tokio::test]
    async fn test_generate_structured_deserializes() {
        let server = MockServer::start().await;

        let body = serde_json::to_string(&Person {
            name: "Alice".to_string(),
            age: 30,
        })
        .unwrap();

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response(&body)),
            )
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Give me a person");
        let person: Person = client
            .generate_structured(&msgs)
            .await
            .expect("structured generation should succeed");

        assert_eq!(person, Person { name: "Alice".to_string(), age: 30 });
    }

    #[tokio::test]
    async fn test_generate_structured_uses_cache() {
        let server = MockServer::start().await;

        let body = serde_json::to_string(&Person {
            name: "Bob".to_string(),
            age: 25,
        })
        .unwrap();

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_completions_response(&body)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = client_for(&server.uri());
        let msgs = user_messages("Give me Bob");

        let p1: Person = client.generate_structured(&msgs).await.expect("first");
        let p2: Person = client.generate_structured(&msgs).await.expect("cached");

        assert_eq!(p1, p2);
    }

    // ── cache key ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_key_differs_by_content() {
        let client = OpenAiClient::new("key", "gpt-4o", CacheConfig::default());
        let msgs_a = user_messages("hello");
        let msgs_b = user_messages("world");
        assert_ne!(
            client.cache_key("text", &msgs_a),
            client.cache_key("text", &msgs_b)
        );
    }

    #[test]
    fn test_cache_key_differs_by_prefix() {
        let client = OpenAiClient::new("key", "gpt-4o", CacheConfig::default());
        let msgs = user_messages("hello");
        assert_ne!(
            client.cache_key("text", &msgs),
            client.cache_key("structured", &msgs)
        );
    }
}
