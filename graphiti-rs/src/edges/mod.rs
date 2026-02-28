//! Edge types for the knowledge graph.
//!
//! Mirrors the Python `edges.py` module. Five edge types:
//! - [`EntityEdge`] — factual relationships between entities (bi-temporal)
//! - [`EpisodicEdge`] — MENTIONS relationships (episode → entity)
//! - [`CommunityEdge`] — HAS_MEMBER relationships (community → entity)
//! - [`HasEpisodeEdge`] — HAS_EPISODE relationships (saga → episode)
//! - [`NextEpisodeEdge`] — NEXT_EPISODE sequencing (episode → episode)

pub mod community;
pub mod entity;
pub mod episodic;
pub mod has_episode;
pub mod next_episode;
