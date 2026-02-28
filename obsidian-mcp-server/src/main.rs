mod config;
mod server;
mod vault;

use std::sync::Arc;

use axum::{Router, extract::State, http::StatusCode, response::IntoResponse};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService,
    session::local::LocalSessionManager,
};
use tokio::signal;
use tracing::{error, info};

use crate::{
    config::Config,
    server::ObsidianMcpServer,
    vault::{UndoStack, VaultFs},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Tracing ───────────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("obsidian_mcp_server=info".parse()?),
        )
        .json()
        .init();

    info!("obsidian-mcp-server starting");

    // ── Config ────────────────────────────────────────────────────────────────
    let mut config = Config::from_env().map_err(|e| {
        error!("Configuration error: {}", e);
        e
    })?;

    // validate_vault canonicalizes vault_path so the stored path is always
    // an absolute, symlink-resolved path from this point forward.
    config.validate_vault().map_err(|e| {
        error!("Vault validation failed: {}", e);
        e
    })?;

    info!(
        vault = %config.vault_path.display(),
        addr  = %config.bind_addr,
        "configuration loaded"
    );

    // ── Vault ─────────────────────────────────────────────────────────────────
    let undo_stack = Arc::new(UndoStack::new(config.undo_stack_limit));
    let vault = Arc::new(VaultFs::new(
        config.vault_path.clone(),
        undo_stack,
        config.max_file_size,
    ));

    let vault_health = vault.clone();

    // ── MCP service ───────────────────────────────────────────────────────────
    let session_manager = Arc::new(LocalSessionManager::default());
    let mcp_service = StreamableHttpService::new(
        {
            let vault = vault.clone();
            move || Ok(ObsidianMcpServer::new(vault.clone()))
        },
        session_manager,
        StreamableHttpServerConfig::default(),
    );

    // ── Axum router ───────────────────────────────────────────────────────────
    let app = Router::new()
        .route("/mcp", axum::routing::any_service(mcp_service))
        .route("/health", axum::routing::get(health_handler))
        .route("/ready", axum::routing::get(ready_handler))
        .with_state(vault_health);

    // ── Listen ────────────────────────────────────────────────────────────────
    info!(addr = %config.bind_addr, "listening");
    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("server stopped");
    Ok(())
}

/// Liveness probe — returns 200 as long as the process is running.
async fn health_handler() -> impl IntoResponse {
    StatusCode::OK
}

/// Readiness probe — returns 200 if the vault directory is accessible.
async fn ready_handler(State(vault): State<Arc<VaultFs>>) -> impl IntoResponse {
    match tokio::fs::metadata(vault.root()).await {
        Ok(meta) if meta.is_dir() => StatusCode::OK,
        _ => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// Graceful shutdown on SIGTERM or Ctrl-C.
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { info!("received Ctrl-C, shutting down"); }
        _ = terminate => { info!("received SIGTERM, shutting down"); }
    }
}
