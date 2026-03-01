//! EntityEdge — bi-temporal factual relationship between EntityNodes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A factual relationship between two entity nodes, with bi-temporal metadata.
///
/// - **Valid time** (`valid_at` / `invalid_at`): when the fact was true in the real world.
/// - **Transaction time** (`created_at` / `expired_at`): when the edge exists in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEdge {
    /// Unique identifier for this edge.
    pub uuid: Uuid,
    /// UUID of the source EntityNode.
    pub source_node_uuid: Uuid,
    /// UUID of the target EntityNode.
    pub target_node_uuid: Uuid,
    /// Relationship label (e.g. "KNOWS", "WORKS_AT").
    pub name: String,
    /// Human-readable fact string.
    pub fact: String,
    /// Optional embedding vector for the fact.
    pub fact_embedding: Option<Vec<f32>>,
    /// When the fact became true in the real world (valid-time start).
    pub valid_at: Option<DateTime<Utc>>,
    /// When the fact ceased to be true in the real world (valid-time end).
    pub invalid_at: Option<DateTime<Utc>>,
    /// When this edge was created in the graph (transaction-time start).
    pub created_at: DateTime<Utc>,
    /// When this edge was superseded in the graph (transaction-time end).
    pub expired_at: Option<DateTime<Utc>>,
    /// Relevance weight (default 1.0).
    pub weight: f64,
    /// Arbitrary JSON attributes.
    pub attributes: serde_json::Value,
    /// Optional group / partition identifier.
    pub group_id: Option<String>,
}
