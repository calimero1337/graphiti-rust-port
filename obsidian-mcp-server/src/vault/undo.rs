use std::{collections::VecDeque, path::{Path, PathBuf}};

use dashmap::DashMap;

use super::VaultError;

/// Per-file in-memory undo stack.
///
/// Thread-safe via `DashMap`. Not persisted across server restarts.
/// Each file has its own bounded history (default 100 entries).
///
/// Uses `VecDeque` so eviction of the oldest entry is O(1).
///
/// # Memory usage
/// The stack is bounded by entry count per file (`limit`), but not by total
/// bytes.  In the worst case a single file with `limit = 100` entries of
/// `MAX_FILE_SIZE = 10 MB` each can consume up to 1 GB.  Across many
/// concurrently edited files the aggregate can be higher.  For
/// memory-constrained deployments, set `UNDO_STACK_LIMIT` to a smaller value
/// (e.g. `5` or `10`) via the environment variable.
pub struct UndoStack {
    entries: DashMap<PathBuf, VecDeque<String>>,
    limit: usize,
}

impl UndoStack {
    pub fn new(limit: usize) -> Self {
        Self {
            entries: DashMap::new(),
            limit,
        }
    }

    /// Push `content` as the previous state for `path`.
    /// If the stack is full, the oldest entry is evicted (O(1)).
    pub fn push(&self, path: PathBuf, content: String) {
        let mut stack = self.entries.entry(path).or_default();
        if stack.len() >= self.limit {
            stack.pop_front(); // evict oldest — O(1)
        }
        stack.push_back(content);
    }

    /// Pop the most recent previous state for `path`.
    /// Returns `NoUndoHistory` if the stack is empty.
    pub fn pop(&self, path: &Path) -> Result<String, VaultError> {
        let mut stack = self.entries.get_mut(path).ok_or_else(|| {
            VaultError::NoUndoHistory(path.display().to_string())
        })?;
        stack.pop_back().ok_or_else(|| {
            VaultError::NoUndoHistory(path.display().to_string())
        })
    }

    /// Returns true if there is at least one undo entry for `path`.
    pub fn has_history(&self, path: &Path) -> bool {
        self.entries
            .get(path)
            .map_or(false, |s| !s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_pop() {
        let stack = UndoStack::new(10);
        let path = PathBuf::from("/vault/test.md");
        stack.push(path.clone(), "v1".into());
        stack.push(path.clone(), "v2".into());

        assert_eq!(stack.pop(&path).unwrap(), "v2");
        assert_eq!(stack.pop(&path).unwrap(), "v1");
        assert!(matches!(
            stack.pop(&path).unwrap_err(),
            VaultError::NoUndoHistory(_)
        ));
    }

    #[test]
    fn stack_limit_evicts_oldest() {
        let stack = UndoStack::new(3);
        let path = PathBuf::from("/vault/test.md");
        stack.push(path.clone(), "v1".into());
        stack.push(path.clone(), "v2".into());
        stack.push(path.clone(), "v3".into());
        stack.push(path.clone(), "v4".into()); // evicts v1

        assert_eq!(stack.pop(&path).unwrap(), "v4");
        assert_eq!(stack.pop(&path).unwrap(), "v3");
        assert_eq!(stack.pop(&path).unwrap(), "v2");
        assert!(stack.pop(&path).is_err()); // v1 was evicted
    }

    #[test]
    fn different_files_are_isolated() {
        let stack = UndoStack::new(10);
        let path_a = PathBuf::from("/vault/a.md");
        let path_b = PathBuf::from("/vault/b.md");
        stack.push(path_a.clone(), "a-content".into());
        stack.push(path_b.clone(), "b-content".into());

        assert_eq!(stack.pop(&path_a).unwrap(), "a-content");
        assert_eq!(stack.pop(&path_b).unwrap(), "b-content");
    }

    #[test]
    fn has_history_reflects_state() {
        let stack = UndoStack::new(10);
        let path = PathBuf::from("/vault/test.md");
        assert!(!stack.has_history(&path));
        stack.push(path.clone(), "v1".into());
        assert!(stack.has_history(&path));
        stack.pop(&path).unwrap();
        assert!(!stack.has_history(&path));
    }
}
