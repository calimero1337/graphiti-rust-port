//! Graph database driver abstraction.
//!
//! Defines the [`GraphDriver`] trait that all backend implementations must satisfy,
//! plus the Neo4j implementation.

pub mod neo4j;

use crate::errors::Result;

/// Trait representing a graph database backend.
///
/// All 11 operation groups from the Python `graphiti_core.driver` are mapped here:
/// node CRUD, edge CRUD, entity ops, episode ops, community ops, saga ops,
/// search ops, batch ops, maintenance ops, index ops, and transaction ops.
///
/// Phase 1 implements only `neo4j::Neo4jDriver`. Other backends (FalkorDB, Kuzu,
/// Neptune) follow in Phase 3.
#[allow(async_fn_in_trait)]
pub trait GraphDriver: Send + Sync {
    /// Health check — verify connectivity to the database.
    async fn ping(&self) -> Result<()>;

    /// Close the connection pool / session.
    async fn close(&self) -> Result<()>;
}
