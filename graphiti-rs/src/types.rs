//! Shared configuration and client container types.

use serde::{Deserialize, Serialize};
use validator::Validate;

fn validate_embedding_dim(dim: usize) -> Result<(), validator::ValidationError> {
    if dim == 0 {
        return Err(validator::ValidationError::new("embedding_dim must be > 0"));
    }
    Ok(())
}

/// Central configuration loaded from environment variables.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct GraphitiConfig {
    /// Neo4j connection URI (e.g. `bolt://localhost:7687`).
    #[validate(length(min = 1))]
    pub neo4j_uri: String,

    /// Neo4j username.
    pub neo4j_user: String,

    /// Neo4j password.
    #[validate(length(min = 1))]
    pub neo4j_password: String,

    /// OpenAI API key.
    #[validate(length(min = 1))]
    pub openai_api_key: String,

    /// Embedding vector dimension (must be > 0).
    #[validate(custom(function = "validate_embedding_dim"))]
    pub embedding_dim: usize,

    /// Default LLM model name.
    pub model_name: String,

    /// Smaller/cheaper LLM model name.
    pub small_model_name: String,

    /// Optional group ID for partitioning graph data.
    pub group_id: Option<String>,
}

impl Default for GraphitiConfig {
    fn default() -> Self {
        Self {
            neo4j_uri: "bolt://localhost:7687".to_string(),
            neo4j_user: "neo4j".to_string(),
            neo4j_password: String::new(),
            openai_api_key: String::new(),
            embedding_dim: 1536,
            model_name: "gpt-4o".to_string(),
            small_model_name: "gpt-4.1-nano".to_string(),
            group_id: None,
        }
    }
}

impl GraphitiConfig {
    /// Load configuration from environment variables.
    ///
    /// Calls `dotenvy::dotenv().ok()` first (non-fatal if `.env` is absent),
    /// then reads each variable from the process environment. Required variables
    /// (`NEO4J_PASSWORD`, `OPENAI_API_KEY`) return a [`crate::GraphitiError::Validation`]
    /// error when absent or empty.
    pub fn from_env() -> crate::Result<Self> {
        dotenvy::dotenv().ok();

        let neo4j_uri = std::env::var("NEO4J_URI")
            .unwrap_or_else(|_| "bolt://localhost:7687".to_string());

        let neo4j_user = std::env::var("NEO4J_USER")
            .unwrap_or_else(|_| "neo4j".to_string());

        let neo4j_password = std::env::var("NEO4J_PASSWORD").map_err(|_| {
            crate::GraphitiError::Validation("NEO4J_PASSWORD is required".to_string())
        })?;

        let openai_api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            crate::GraphitiError::Validation("OPENAI_API_KEY is required".to_string())
        })?;

        let embedding_dim = match std::env::var("EMBEDDING_DIM") {
            Ok(val) => val.parse::<usize>().map_err(|_| {
                crate::GraphitiError::Validation(
                    "EMBEDDING_DIM must be a positive integer".to_string(),
                )
            })?,
            Err(_) => 1536,
        };

        let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "gpt-4o".to_string());

        let small_model_name = std::env::var("SMALL_MODEL_NAME")
            .unwrap_or_else(|_| "gpt-4.1-nano".to_string());

        let group_id = std::env::var("GROUP_ID").ok();

        let config = Self {
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            openai_api_key,
            embedding_dim,
            model_name,
            small_model_name,
            group_id,
        };

        config.validate().map_err(|e| {
            crate::GraphitiError::Validation(e.to_string())
        })?;

        Ok(config)
    }
}

/// Placeholder for search configuration (full implementation in search module).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchConfig {}

/// Placeholder for ingestion pipeline configuration (full implementation in pipeline module).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestionConfig {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Temporarily sets env vars for a test, restoring originals afterward.
    fn with_env<F, R>(vars: &[(&str, &str)], f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Save originals.
        let originals: Vec<(&str, Option<String>)> =
            vars.iter().map(|(k, _)| (*k, env::var(k).ok())).collect();

        // Set test values.
        for (k, v) in vars {
            env::set_var(k, v);
        }

        let result = f();

        // Restore originals.
        for (k, original) in &originals {
            match original {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }

        result
    }

    #[test]
    fn test_config_defaults() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
            ],
            || {
                // Remove optional vars in case they're set in the process env.
                env::remove_var("NEO4J_URI");
                env::remove_var("NEO4J_USER");
                env::remove_var("EMBEDDING_DIM");
                env::remove_var("MODEL_NAME");
                env::remove_var("SMALL_MODEL_NAME");
                env::remove_var("GROUP_ID");

                let config = GraphitiConfig::from_env().expect("config should load");
                assert_eq!(config.neo4j_uri, "bolt://localhost:7687");
                assert_eq!(config.neo4j_user, "neo4j");
                assert_eq!(config.embedding_dim, 1536);
                assert_eq!(config.model_name, "gpt-4o");
                assert_eq!(config.small_model_name, "gpt-4.1-nano");
                assert!(config.group_id.is_none());
            },
        );
    }

    #[test]
    fn test_config_custom_values() {
        with_env(
            &[
                ("NEO4J_URI", "bolt://db.example.com:7687"),
                ("NEO4J_USER", "admin"),
                ("NEO4J_PASSWORD", "mysecret"),
                ("OPENAI_API_KEY", "sk-real-key"),
                ("EMBEDDING_DIM", "3072"),
                ("MODEL_NAME", "gpt-4o-mini"),
                ("SMALL_MODEL_NAME", "gpt-3.5-turbo"),
                ("GROUP_ID", "team-alpha"),
            ],
            || {
                let config = GraphitiConfig::from_env().expect("config should load");
                assert_eq!(config.neo4j_uri, "bolt://db.example.com:7687");
                assert_eq!(config.neo4j_user, "admin");
                assert_eq!(config.neo4j_password, "mysecret");
                assert_eq!(config.openai_api_key, "sk-real-key");
                assert_eq!(config.embedding_dim, 3072);
                assert_eq!(config.model_name, "gpt-4o-mini");
                assert_eq!(config.small_model_name, "gpt-3.5-turbo");
                assert_eq!(config.group_id, Some("team-alpha".to_string()));
            },
        );
    }

    #[test]
    fn test_config_missing_password() {
        // Save and clear both required vars.
        let saved_pw = env::var("NEO4J_PASSWORD").ok();
        let saved_key = env::var("OPENAI_API_KEY").ok();
        env::remove_var("NEO4J_PASSWORD");
        env::remove_var("OPENAI_API_KEY");

        let result = GraphitiConfig::from_env();

        // Restore.
        if let Some(v) = saved_pw { env::set_var("NEO4J_PASSWORD", v); }
        if let Some(v) = saved_key { env::set_var("OPENAI_API_KEY", v); }

        assert!(result.is_err());
        match result.unwrap_err() {
            crate::GraphitiError::Validation(msg) => {
                assert!(msg.contains("NEO4J_PASSWORD"));
            }
            e => panic!("expected Validation error, got {:?}", e),
        }
    }

    #[test]
    fn test_config_missing_api_key() {
        let saved_key = env::var("OPENAI_API_KEY").ok();
        env::remove_var("OPENAI_API_KEY");

        // Make sure password is present.
        let saved_pw = env::var("NEO4J_PASSWORD").ok();
        env::set_var("NEO4J_PASSWORD", "secret");

        let result = GraphitiConfig::from_env();

        if let Some(v) = saved_key { env::set_var("OPENAI_API_KEY", v); } else { env::remove_var("OPENAI_API_KEY"); }
        if let Some(v) = saved_pw { env::set_var("NEO4J_PASSWORD", v); } else { env::remove_var("NEO4J_PASSWORD"); }

        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_embedding_dim() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
                ("EMBEDDING_DIM", "not-a-number"),
            ],
            || {
                let result = GraphitiConfig::from_env();
                assert!(result.is_err());
                match result.unwrap_err() {
                    crate::GraphitiError::Validation(msg) => {
                        assert!(msg.contains("EMBEDDING_DIM"));
                    }
                    e => panic!("expected Validation error, got {:?}", e),
                }
            },
        );
    }

    #[test]
    fn test_config_zero_embedding_dim() {
        with_env(
            &[
                ("NEO4J_PASSWORD", "secret"),
                ("OPENAI_API_KEY", "sk-test"),
                ("EMBEDDING_DIM", "0"),
            ],
            || {
                let result = GraphitiConfig::from_env();
                assert!(result.is_err());
            },
        );
    }

    #[test]
    fn test_search_config_default() {
        let _ = SearchConfig::default();
    }

    #[test]
    fn test_ingestion_config_default() {
        let _ = IngestionConfig::default();
    }
}
