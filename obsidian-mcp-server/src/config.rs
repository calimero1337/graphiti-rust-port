use std::{net::SocketAddr, path::PathBuf, str::FromStr};

/// Server configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the Obsidian vault directory (mounted PVC). Env: `VAULT_PATH`, default `/vault`.
    pub vault_path: PathBuf,
    /// Address to bind the HTTP server. Env: `BIND_ADDR`, default `0.0.0.0:8080`.
    pub bind_addr: SocketAddr,
    /// Maximum file size in bytes before read is rejected. Env: `MAX_FILE_SIZE`, default 10 MB.
    pub max_file_size: usize,
    /// Maximum number of undo entries per file. Env: `UNDO_STACK_LIMIT`, default 100.
    pub undo_stack_limit: usize,
}

impl Config {
    /// Load configuration from environment variables, applying sensible defaults.
    ///
    /// # Errors
    /// Returns an error if `BIND_ADDR` is set but not a valid socket address,
    /// or if numeric env vars cannot be parsed.
    pub fn from_env() -> anyhow::Result<Self> {
        let vault_path = std::env::var("VAULT_PATH").unwrap_or_else(|_| "/vault".to_string());
        let vault_path = PathBuf::from(vault_path);

        let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
        let bind_addr = SocketAddr::from_str(&bind_addr)
            .map_err(|e| anyhow::anyhow!("Invalid BIND_ADDR '{}': {}", bind_addr, e))?;

        let max_file_size = parse_env_usize("MAX_FILE_SIZE", 10 * 1024 * 1024)?;
        let undo_stack_limit = parse_env_usize("UNDO_STACK_LIMIT", 100)?;

        Ok(Config {
            vault_path,
            bind_addr,
            max_file_size,
            undo_stack_limit,
        })
    }

    /// Validate that the vault path exists and is a directory, then canonicalize it.
    ///
    /// All filesystem operations are performed via `tokio::fs` (non-blocking),
    /// which is consistent with the codebase's policy of never blocking the
    /// tokio thread pool.
    ///
    /// Canonicalization resolves symlinks and makes the stored path absolute,
    /// so that downstream path-security checks have a stable baseline.
    ///
    /// Call this after the filesystem is ready (e.g., after PVC mount).
    pub async fn validate_vault(&mut self) -> anyhow::Result<()> {
        let meta = tokio::fs::metadata(&self.vault_path).await.map_err(|e| {
            anyhow::anyhow!(
                "Cannot access vault path '{}': {}",
                self.vault_path.display(),
                e
            )
        })?;

        if !meta.is_dir() {
            anyhow::bail!(
                "Vault path is not a directory: {}",
                self.vault_path.display()
            );
        }

        self.vault_path =
            tokio::fs::canonicalize(&self.vault_path)
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Cannot canonicalize vault path '{}': {}",
                        self.vault_path.display(),
                        e
                    )
                })?;

        Ok(())
    }
}

fn parse_env_usize(name: &str, default: usize) -> anyhow::Result<usize> {
    match std::env::var(name) {
        Ok(val) => val
            .parse::<usize>()
            .map_err(|e| anyhow::anyhow!("Invalid {}: {}", name, e)),
        Err(_) => Ok(default),
    }
}
