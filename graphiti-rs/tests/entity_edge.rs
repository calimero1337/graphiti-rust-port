//! Integration tests for `EntityEdge` — bi-temporal factual relationship between EntityNodes.
//!
//! These tests are written in the TDD red phase. They will fail to compile until
//! `EntityEdge` is implemented in `src/edges/entity.rs`.

use chrono::{DateTime, TimeZone, Utc};
use graphiti_rs::edges::entity::EntityEdge;
use serde_json::json;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal valid `EntityEdge` with required fields only.
fn minimal_edge() -> EntityEdge {
    EntityEdge {
        uuid: Uuid::new_v4(),
        source_node_uuid: Uuid::new_v4(),
        target_node_uuid: Uuid::new_v4(),
        name: "KNOWS".to_string(),
        fact: "Alice knows Bob".to_string(),
        fact_embedding: None,
        valid_at: None,
        invalid_at: None,
        created_at: Utc::now(),
        expired_at: None,
        weight: 1.0,
        attributes: serde_json::Value::Null,
        group_id: None,
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn test_entity_edge_construction_minimal() {
    let edge = minimal_edge();
    assert_eq!(edge.name, "KNOWS");
    assert_eq!(edge.fact, "Alice knows Bob");
    assert!(edge.valid_at.is_none());
    assert!(edge.invalid_at.is_none());
    assert!(edge.expired_at.is_none());
    assert!(edge.fact_embedding.is_none());
    assert!(edge.group_id.is_none());
    assert_eq!(edge.weight, 1.0_f64);
}

#[test]
fn test_entity_edge_construction_full() {
    let now: DateTime<Utc> = Utc::now();
    let source = Uuid::new_v4();
    let target = Uuid::new_v4();

    let edge = EntityEdge {
        uuid: Uuid::new_v4(),
        source_node_uuid: source,
        target_node_uuid: target,
        name: "WORKS_AT".to_string(),
        fact: "Alice works at Acme Corp.".to_string(),
        fact_embedding: Some(vec![0.1_f32, 0.2, 0.3]),
        valid_at: Some(Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()),
        invalid_at: Some(Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap()),
        created_at: now,
        expired_at: None,
        weight: 0.75,
        attributes: json!({ "confidence": 0.9, "source": "linkedin" }),
        group_id: Some("org_acme".to_string()),
    };

    assert_eq!(edge.source_node_uuid, source);
    assert_eq!(edge.target_node_uuid, target);
    assert!(edge.valid_at.is_some());
    assert!(edge.invalid_at.is_some());
    assert!(edge.fact_embedding.is_some());
    assert_eq!(edge.weight, 0.75);
    assert_eq!(edge.group_id.as_deref(), Some("org_acme"));
}

// ---------------------------------------------------------------------------
// Bi-temporal semantics
// ---------------------------------------------------------------------------

#[test]
fn test_valid_at_precedes_invalid_at() {
    let valid_at = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let invalid_at = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();

    let edge = EntityEdge {
        valid_at: Some(valid_at),
        invalid_at: Some(invalid_at),
        ..minimal_edge()
    };

    assert!(
        edge.valid_at.unwrap() < edge.invalid_at.unwrap(),
        "valid_at must precede invalid_at"
    );
}

#[test]
fn test_created_at_precedes_expired_at() {
    let created_at = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let expired_at = Utc.with_ymd_and_hms(2025, 6, 15, 12, 0, 0).unwrap();

    let edge = EntityEdge {
        created_at,
        expired_at: Some(expired_at),
        ..minimal_edge()
    };

    assert!(
        edge.created_at < edge.expired_at.unwrap(),
        "created_at must precede expired_at"
    );
}

#[test]
fn test_currently_valid_edge() {
    // An edge is currently valid in the real world when:
    //   valid_at <= now AND invalid_at is None (or in the future)
    let past = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let edge = EntityEdge {
        valid_at: Some(past),
        invalid_at: None,
        ..minimal_edge()
    };

    let now = Utc::now();
    let is_currently_valid = edge
        .valid_at
        .map(|vt| vt <= now)
        .unwrap_or(true)
        && edge.invalid_at.map(|ivt| ivt > now).unwrap_or(true);

    assert!(is_currently_valid);
}

#[test]
fn test_historically_invalidated_edge() {
    // An edge that was valid in the past but is now invalidated.
    let valid_at = Utc.with_ymd_and_hms(2010, 6, 1, 0, 0, 0).unwrap();
    let invalid_at = Utc.with_ymd_and_hms(2015, 6, 1, 0, 0, 0).unwrap();

    let edge = EntityEdge {
        valid_at: Some(valid_at),
        invalid_at: Some(invalid_at),
        ..minimal_edge()
    };

    let now = Utc::now();
    let is_currently_valid = edge
        .valid_at
        .map(|vt| vt <= now)
        .unwrap_or(true)
        && edge.invalid_at.map(|ivt| ivt > now).unwrap_or(true);

    assert!(!is_currently_valid, "edge should be invalidated by now");
}

#[test]
fn test_graph_expired_edge() {
    // An edge that has been superseded in the graph (transaction-time expiry).
    let created_at = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let expired_at = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();

    let edge = EntityEdge {
        created_at,
        expired_at: Some(expired_at),
        ..minimal_edge()
    };

    assert!(
        edge.expired_at.is_some(),
        "edge should be marked as expired in the graph"
    );
}

// ---------------------------------------------------------------------------
// Weight
// ---------------------------------------------------------------------------

#[test]
fn test_weight_default_is_one() {
    let edge = minimal_edge();
    assert_eq!(edge.weight, 1.0_f64);
}

#[test]
fn test_weight_custom_value() {
    let edge = EntityEdge {
        weight: 0.42,
        ..minimal_edge()
    };
    assert!((edge.weight - 0.42).abs() < f64::EPSILON);
}

#[test]
fn test_weight_zero_is_valid() {
    let edge = EntityEdge {
        weight: 0.0,
        ..minimal_edge()
    };
    assert_eq!(edge.weight, 0.0_f64);
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

#[test]
fn test_attributes_null_by_default() {
    let edge = minimal_edge();
    assert!(edge.attributes.is_null());
}

#[test]
fn test_attributes_json_object() {
    let attrs = json!({
        "confidence": 0.95,
        "source": "extraction",
        "version": 2
    });

    let edge = EntityEdge {
        attributes: attrs.clone(),
        ..minimal_edge()
    };

    assert_eq!(edge.attributes["confidence"], json!(0.95));
    assert_eq!(edge.attributes["source"], json!("extraction"));
}

#[test]
fn test_attributes_json_array() {
    let attrs = json!(["tag_a", "tag_b"]);

    let edge = EntityEdge {
        attributes: attrs,
        ..minimal_edge()
    };

    assert!(edge.attributes.is_array());
    assert_eq!(edge.attributes.as_array().unwrap().len(), 2);
}

// ---------------------------------------------------------------------------
// Serialization / Deserialization
// ---------------------------------------------------------------------------

#[test]
fn test_entity_edge_serializes_to_json() {
    let edge = minimal_edge();
    let json_str = serde_json::to_string(&edge).expect("serialization must succeed");
    assert!(json_str.contains("\"name\""));
    assert!(json_str.contains("\"fact\""));
    assert!(json_str.contains("\"weight\""));
    assert!(json_str.contains("\"created_at\""));
}

#[test]
fn test_entity_edge_roundtrips_json() {
    let original = EntityEdge {
        uuid: Uuid::new_v4(),
        source_node_uuid: Uuid::new_v4(),
        target_node_uuid: Uuid::new_v4(),
        name: "ROUNDTRIP".to_string(),
        fact: "A fact.".to_string(),
        fact_embedding: None,
        valid_at: Some(Utc.with_ymd_and_hms(2024, 3, 15, 8, 30, 0).unwrap()),
        invalid_at: None,
        created_at: Utc.with_ymd_and_hms(2024, 3, 15, 8, 30, 0).unwrap(),
        expired_at: None,
        weight: 0.9,
        attributes: json!({ "key": "value" }),
        group_id: Some("group_1".to_string()),
    };

    let json_str = serde_json::to_string(&original).expect("serialize");
    let restored: EntityEdge = serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(restored.uuid, original.uuid);
    assert_eq!(restored.name, original.name);
    assert_eq!(restored.fact, original.fact);
    assert_eq!(restored.valid_at, original.valid_at);
    assert_eq!(restored.weight, original.weight);
    assert_eq!(restored.attributes, original.attributes);
    assert_eq!(restored.group_id, original.group_id);
}

#[test]
fn test_entity_edge_deserializes_null_optionals() {
    let json_str = r#"{
        "uuid": "00000000-0000-0000-0000-000000000001",
        "source_node_uuid": "00000000-0000-0000-0000-000000000002",
        "target_node_uuid": "00000000-0000-0000-0000-000000000003",
        "name": "KNOWS",
        "fact": "A knows B",
        "fact_embedding": null,
        "valid_at": null,
        "invalid_at": null,
        "created_at": "2024-01-01T00:00:00Z",
        "expired_at": null,
        "weight": 1.0,
        "attributes": null,
        "group_id": null
    }"#;

    let edge: EntityEdge = serde_json::from_str(json_str).expect("deserialize");
    assert!(edge.valid_at.is_none());
    assert!(edge.invalid_at.is_none());
    assert!(edge.expired_at.is_none());
    assert!(edge.fact_embedding.is_none());
    assert!(edge.group_id.is_none());
}

// ---------------------------------------------------------------------------
// UUID identity
// ---------------------------------------------------------------------------

#[test]
fn test_entity_edge_has_unique_uuid() {
    let e1 = minimal_edge();
    let e2 = minimal_edge();
    assert_ne!(e1.uuid, e2.uuid, "each edge must have a unique UUID");
}

#[test]
fn test_source_and_target_are_distinct_fields() {
    let source = Uuid::new_v4();
    let target = Uuid::new_v4();

    let edge = EntityEdge {
        source_node_uuid: source,
        target_node_uuid: target,
        ..minimal_edge()
    };

    assert_ne!(edge.source_node_uuid, edge.target_node_uuid);
    assert_eq!(edge.source_node_uuid, source);
    assert_eq!(edge.target_node_uuid, target);
}
