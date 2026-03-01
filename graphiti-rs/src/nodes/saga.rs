//! SagaNode — represents a grouping/sequence of episodes.

// TODO: Implement SagaNode

#[cfg(test)]
mod tests {
    use super::SagaNode;
    use chrono::Utc;

    #[test]
    fn test_saga_node_fields() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            name: "Test Saga".to_string(),
            group_id: "group-abc".to_string(),
            labels: vec!["Saga".to_string(), "Episode".to_string()],
            created_at: now,
        };

        assert_eq!(node.uuid, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(node.name, "Test Saga");
        assert_eq!(node.group_id, "group-abc");
        assert_eq!(node.labels.len(), 2);
        assert_eq!(node.labels[0], "Saga");
    }

    #[test]
    fn test_saga_node_clone() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: "uuid-1".to_string(),
            name: "My Saga".to_string(),
            group_id: "grp-1".to_string(),
            labels: vec!["Saga".to_string()],
            created_at: now,
        };
        let cloned = node.clone();
        assert_eq!(cloned.uuid, node.uuid);
        assert_eq!(cloned.name, node.name);
    }

    #[test]
    fn test_saga_node_debug() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: "uuid-debug".to_string(),
            name: "Debug Saga".to_string(),
            group_id: "grp-debug".to_string(),
            labels: vec![],
            created_at: now,
        };
        let debug_str = format!("{node:?}");
        assert!(debug_str.contains("Debug Saga"));
    }

    #[test]
    fn test_saga_node_serde_roundtrip() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: "uuid-serde".to_string(),
            name: "Serde Saga".to_string(),
            group_id: "grp-serde".to_string(),
            labels: vec!["Saga".to_string()],
            created_at: now,
        };

        let json = serde_json::to_string(&node).expect("serialization failed");
        assert!(json.contains("Serde Saga"));
        assert!(json.contains("uuid-serde"));

        let deserialized: SagaNode =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.uuid, node.uuid);
        assert_eq!(deserialized.name, node.name);
        assert_eq!(deserialized.group_id, node.group_id);
        assert_eq!(deserialized.labels, node.labels);
    }

    #[test]
    fn test_saga_node_empty_labels() {
        let now = Utc::now();
        let node = SagaNode {
            uuid: "uuid-empty".to_string(),
            name: "Empty Labels Saga".to_string(),
            group_id: "grp-empty".to_string(),
            labels: vec![],
            created_at: now,
        };
        assert!(node.labels.is_empty());
    }
}
