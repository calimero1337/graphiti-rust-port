//! Node types for the knowledge graph.
//!
//! Mirrors the Python `nodes.py` module. Four node types:
//! - [`EntityNode`] — real-world entities (people, places, concepts)
//! - [`EpisodicNode`] — ingested data episodes (messages, documents, JSON records)
//! - [`CommunityNode`] — detected community clusters
//! - [`SagaNode`] — episode groupings / sequences

pub mod community;
pub mod entity;
pub mod episodic;
pub mod saga;
