use std::{path::PathBuf, sync::Arc};

use tempfile::TempDir;

use obsidian_mcp_server::vault::{UndoStack, VaultFs};

/// Temporary vault directory with helpers for building integration tests.
pub struct TempVault {
    pub dir: TempDir,
    pub path: PathBuf,
}

impl TempVault {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().expect("create temp vault dir");
        let path = dir.path().to_path_buf();
        Self { dir, path }
    }

    /// Write a file relative to the vault root.
    pub fn write(&self, rel: &str, content: &str) -> &Self {
        let full = self.path.join(rel);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).expect("create parent dirs");
        }
        std::fs::write(&full, content).expect("write test file");
        self
    }

    /// Build a `VaultFs` over this temp vault.
    pub fn vault_fs(&self) -> Arc<VaultFs> {
        let undo = Arc::new(UndoStack::new(10));
        Arc::new(VaultFs::new(self.path.clone(), undo, 10 * 1024 * 1024))
    }
}
