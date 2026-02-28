use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::fs;

use super::{
    path::normalize_and_validate,
    undo::UndoStack,
    VaultError,
};

/// Maximum directory recursion depth for `list_all_files`.
const MAX_DEPTH: usize = 50;

/// Sandboxed filesystem abstraction over the Obsidian vault directory.
///
/// All methods accept vault-relative paths (e.g. `"notes/todo.md"`) and
/// validate them via [`normalize_and_validate`] before any filesystem access.
///
/// # Async safety
/// All filesystem operations use `tokio::fs` (non-blocking). The internal
/// `validate()` method calls `normalize_and_validate`, which uses
/// `tokio::fs::canonicalize` for existing paths — no blocking I/O on the
/// tokio thread pool.
///
/// # Path security
/// `root` must be the **canonical** (absolute, symlink-resolved) vault path.
/// Pass the value from `Config::validate_vault()`, which calls
/// `std::fs::canonicalize` at startup. The canonical root is stored once and
/// reused; it is never re-canonicalized on subsequent calls.
pub struct VaultFs {
    root: PathBuf,
    undo: Arc<UndoStack>,
    max_file_size: usize,
}

impl VaultFs {
    pub fn new(root: PathBuf, undo: Arc<UndoStack>, max_file_size: usize) -> Self {
        Self {
            root,
            undo,
            max_file_size,
        }
    }

    /// Returns a reference to the vault root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Validate `user_path` relative to the pre-canonicalized vault root.
    ///
    /// Uses `tokio::fs::canonicalize` (async, non-blocking) for existing paths.
    async fn validate(&self, user_path: &str) -> Result<PathBuf, VaultError> {
        normalize_and_validate(&self.root, user_path).await
    }

    async fn check_size(&self, path: &Path, user_path: &str) -> Result<(), VaultError> {
        let meta = fs::metadata(path)
            .await
            .map_err(|e| VaultError::io(user_path, e))?;
        let size = meta.len() as usize;
        if size > self.max_file_size {
            return Err(VaultError::FileTooLarge {
                size,
                max: self.max_file_size,
            });
        }
        Ok(())
    }

    async fn read_raw(path: &Path, user_path: &str) -> Result<String, VaultError> {
        let bytes = fs::read(path)
            .await
            .map_err(|e| VaultError::io(user_path, e))?;
        String::from_utf8(bytes).map_err(|_| VaultError::NotUtf8(user_path.to_string()))
    }

    /// Classify a path via async metadata: returns (exists, is_dir, is_file).
    async fn classify(path: &Path, user_path: &str) -> Result<(bool, bool, bool), VaultError> {
        match fs::metadata(path).await {
            Ok(meta) => Ok((true, meta.is_dir(), meta.is_file())),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok((false, false, false)),
            Err(e) => Err(VaultError::io(user_path, e)),
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Read a file's contents as a UTF-8 string.
    pub async fn read_file(&self, user_path: &str) -> Result<String, VaultError> {
        let path = self.validate(user_path).await?;
        let (exists, is_dir, _) = Self::classify(&path, user_path).await?;

        if !exists {
            return Err(VaultError::NotFound(user_path.to_string()));
        }
        if is_dir {
            return Err(VaultError::NotAFile(user_path.to_string()));
        }

        self.check_size(&path, user_path).await?;
        Self::read_raw(&path, user_path).await
    }

    /// Write content to a file, saving the previous version to the undo stack.
    ///
    /// The previous file content is read (best-effort) for the undo entry; if
    /// the file does not yet exist the undo entry is skipped.  Creates parent
    /// directories as needed.
    pub async fn write_file(&self, user_path: &str, content: String) -> Result<(), VaultError> {
        let path = self.validate(user_path).await?;

        // Best-effort: read current content for undo.  If the file doesn't
        // exist (or can't be read), we simply skip the undo entry rather than
        // failing the whole write.  We intentionally avoid a separate
        // existence check to reduce the TOCTOU window.
        if let Ok(old) = Self::read_raw(&path, user_path).await {
            self.undo.push(path.clone(), old);
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| VaultError::io(user_path, e))?;
        }

        fs::write(&path, content.as_bytes())
            .await
            .map_err(|e| VaultError::io(user_path, e))
    }

    /// Write content to a file using an already-read snapshot as the undo entry.
    ///
    /// Use this when you have already read the current file content (e.g. in
    /// `str_replace` or `insert`) to avoid a redundant re-read and its
    /// associated TOCTOU window.  The `undo_snapshot` is pushed onto the undo
    /// stack directly without a second filesystem read.
    pub async fn write_file_with_undo(
        &self,
        user_path: &str,
        undo_snapshot: String,
        new_content: String,
    ) -> Result<(), VaultError> {
        let path = self.validate(user_path).await?;

        self.undo.push(path.clone(), undo_snapshot);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| VaultError::io(user_path, e))?;
        }

        fs::write(&path, new_content.as_bytes())
            .await
            .map_err(|e| VaultError::io(user_path, e))
    }

    /// Create a new file. Returns `AlreadyExists` if the path already exists.
    /// Creates parent directories as needed.
    pub async fn create_file(&self, user_path: &str, content: &str) -> Result<(), VaultError> {
        let path = self.validate(user_path).await?;

        let (exists, _, _) = Self::classify(&path, user_path).await?;
        if exists {
            return Err(VaultError::AlreadyExists(user_path.to_string()));
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| VaultError::io(user_path, e))?;
        }

        fs::write(&path, content.as_bytes())
            .await
            .map_err(|e| VaultError::io(user_path, e))
    }

    /// Delete a file. The previous content is saved to the undo stack.
    pub async fn delete_file(&self, user_path: &str) -> Result<(), VaultError> {
        let path = self.validate(user_path).await?;

        let (exists, is_dir, _) = Self::classify(&path, user_path).await?;
        if !exists {
            return Err(VaultError::NotFound(user_path.to_string()));
        }
        if is_dir {
            return Err(VaultError::NotAFile(user_path.to_string()));
        }

        // Save to undo stack before deletion
        if let Ok(content) = Self::read_raw(&path, user_path).await {
            self.undo.push(path.clone(), content);
        }

        fs::remove_file(&path)
            .await
            .map_err(|e| VaultError::io(user_path, e))
    }

    /// Rename / move a file within the vault.
    /// Returns `AlreadyExists` if the destination already exists.
    pub async fn rename_file(
        &self,
        from_path: &str,
        to_path: &str,
    ) -> Result<(), VaultError> {
        let from = self.validate(from_path).await?;
        let to = self.validate(to_path).await?;

        let (from_exists, _, _) = Self::classify(&from, from_path).await?;
        if !from_exists {
            return Err(VaultError::NotFound(from_path.to_string()));
        }

        let (to_exists, _, _) = Self::classify(&to, to_path).await?;
        if to_exists {
            return Err(VaultError::AlreadyExists(to_path.to_string()));
        }

        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| VaultError::io(to_path, e))?;
        }

        fs::rename(&from, &to)
            .await
            .map_err(|e| VaultError::io(from_path, e))
    }

    /// List the immediate children of a directory.
    /// An empty or "/" path lists the vault root.
    ///
    /// Path validation (including the empty/"/" short-circuit) is always
    /// performed via `normalize_and_validate` — no bypass.
    pub async fn list_dir(&self, user_path: &str) -> Result<Vec<String>, VaultError> {
        let path = self.validate(user_path).await?;

        let (exists, is_dir, _) = Self::classify(&path, user_path).await?;
        if !exists {
            return Err(VaultError::NotFound(user_path.to_string()));
        }
        if !is_dir {
            return Err(VaultError::NotADirectory(user_path.to_string()));
        }

        let mut entries = fs::read_dir(&path)
            .await
            .map_err(|e| VaultError::io(user_path, e))?;

        let mut names = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| VaultError::io(user_path, e))?
        {
            names.push(entry.file_name().to_string_lossy().to_string());
        }

        names.sort();
        Ok(names)
    }

    /// Recursively list all regular files in the vault, returning vault-relative paths.
    ///
    /// Symlinks are not followed (directory symlinks are not recursed into and
    /// symlink entries are not listed as files).  Directories nested deeper
    /// than `MAX_DEPTH` are silently skipped.
    pub async fn list_all_files(&self) -> Result<Vec<String>, VaultError> {
        let root = self.root.clone();
        // Stack entries: (directory path, current depth)
        let mut stack: Vec<(PathBuf, usize)> = vec![(root.clone(), 0)];
        let mut results = Vec::new();

        while let Some((dir, depth)) = stack.pop() {
            let mut entries = fs::read_dir(&dir)
                .await
                .map_err(|e| VaultError::io(dir.display(), e))?;

            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|e| VaultError::io(dir.display(), e))?
            {
                let path = entry.path();
                // `DirEntry::metadata` does NOT follow symlinks (equivalent to
                // lstat), so symlinks appear as neither is_dir nor is_file.
                let meta = entry
                    .metadata()
                    .await
                    .map_err(|e| VaultError::io(path.display(), e))?;

                if meta.is_dir() {
                    if depth < MAX_DEPTH {
                        stack.push((path, depth + 1));
                    }
                } else if meta.is_file() {
                    if let Ok(relative) = path.strip_prefix(&root) {
                        results.push(relative.to_string_lossy().to_string());
                    }
                }
                // symlinks: skipped intentionally
            }
        }

        results.sort();
        Ok(results)
    }

    /// Restore the previous version of a file from the undo stack.
    ///
    /// Creates parent directories if they no longer exist (e.g. after a
    /// `delete_file` removed the last file in a directory and the directory
    /// was subsequently cleaned up externally).
    ///
    /// Returns the restored content.
    pub async fn undo_file(&self, user_path: &str) -> Result<String, VaultError> {
        let path = self.validate(user_path).await?;
        let content = self.undo.pop(&path)?;

        // Recreate parent directories in case they were removed after the file
        // was deleted (e.g. by delete_file + external directory cleanup).
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| VaultError::io(user_path, e))?;
        }

        fs::write(&path, content.as_bytes())
            .await
            .map_err(|e| VaultError::io(user_path, e))?;

        Ok(content)
    }

    /// Check whether `user_path` exists and is a regular file (async, non-blocking).
    pub async fn is_file(&self, user_path: &str) -> Result<bool, VaultError> {
        let path = self.validate(user_path).await?;
        match fs::metadata(&path).await {
            Ok(meta) => Ok(meta.is_file()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(VaultError::io(user_path, e)),
        }
    }

    /// Check whether `user_path` exists and is a directory (async, non-blocking).
    pub async fn is_dir(&self, user_path: &str) -> Result<bool, VaultError> {
        let path = self.validate(user_path).await?;
        match fs::metadata(&path).await {
            Ok(meta) => Ok(meta.is_dir()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(VaultError::io(user_path, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn make_vault(dir: &TempDir) -> VaultFs {
        let undo = Arc::new(UndoStack::new(10));
        // Canonicalize so the stored root is symlink-resolved, matching the
        // guarantee required by normalize_and_validate.
        let canonical = tokio::fs::canonicalize(dir.path())
            .await
            .expect("canonicalize temp dir");
        VaultFs::new(canonical, undo, 10 * 1024 * 1024)
    }

    #[tokio::test]
    async fn read_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("note.md"), "hello").unwrap();
        let vault = make_vault(&dir).await;
        assert_eq!(vault.read_file("note.md").await.unwrap(), "hello");
    }

    #[tokio::test]
    async fn read_missing_file_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let vault = make_vault(&dir).await;
        assert!(matches!(
            vault.read_file("missing.md").await.unwrap_err(),
            VaultError::NotFound(_)
        ));
    }

    #[tokio::test]
    async fn create_and_read_file() {
        let dir = tempfile::tempdir().unwrap();
        let vault = make_vault(&dir).await;
        vault.create_file("new.md", "content").await.unwrap();
        assert_eq!(vault.read_file("new.md").await.unwrap(), "content");
    }

    #[tokio::test]
    async fn create_existing_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("exists.md"), "x").unwrap();
        let vault = make_vault(&dir).await;
        assert!(matches!(
            vault.create_file("exists.md", "y").await.unwrap_err(),
            VaultError::AlreadyExists(_)
        ));
    }

    #[tokio::test]
    async fn write_saves_undo() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("note.md"), "v1").unwrap();
        let vault = make_vault(&dir).await;

        vault.write_file("note.md", "v2".to_string()).await.unwrap();
        assert_eq!(vault.read_file("note.md").await.unwrap(), "v2");

        // undo should restore v1
        let restored = vault.undo_file("note.md").await.unwrap();
        assert_eq!(restored, "v1");
        assert_eq!(vault.read_file("note.md").await.unwrap(), "v1");
    }

    #[tokio::test]
    async fn write_with_undo_snapshot_skips_second_read() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("note.md"), "v1").unwrap();
        let vault = make_vault(&dir).await;

        vault
            .write_file_with_undo("note.md", "v1".to_string(), "v2".to_string())
            .await
            .unwrap();
        assert_eq!(vault.read_file("note.md").await.unwrap(), "v2");

        let restored = vault.undo_file("note.md").await.unwrap();
        assert_eq!(restored, "v1");
    }

    #[tokio::test]
    async fn delete_removes_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("del.md"), "bye").unwrap();
        let vault = make_vault(&dir).await;
        vault.delete_file("del.md").await.unwrap();
        assert!(!dir.path().join("del.md").exists());
    }

    #[tokio::test]
    async fn rename_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("old.md"), "data").unwrap();
        let vault = make_vault(&dir).await;
        vault.rename_file("old.md", "new.md").await.unwrap();
        assert!(!dir.path().join("old.md").exists());
        assert_eq!(vault.read_file("new.md").await.unwrap(), "data");
    }

    #[tokio::test]
    async fn list_dir_returns_sorted_entries() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("b.md"), "").unwrap();
        std::fs::write(dir.path().join("a.md"), "").unwrap();
        let vault = make_vault(&dir).await;
        let entries = vault.list_dir("").await.unwrap();
        assert_eq!(entries, vec!["a.md", "b.md"]);
    }

    #[tokio::test]
    async fn list_dir_empty_path_is_vault_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("root.md"), "").unwrap();
        let vault = make_vault(&dir).await;
        // "" and "/" should both return the vault root listing
        let a = vault.list_dir("").await.unwrap();
        let b = vault.list_dir("/").await.unwrap();
        assert_eq!(a, b);
        assert!(a.contains(&"root.md".to_string()));
    }

    #[tokio::test]
    async fn list_all_files_recursive() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("root.md"), "").unwrap();
        std::fs::write(dir.path().join("sub/child.md"), "").unwrap();
        let vault = make_vault(&dir).await;
        let files = vault.list_all_files().await.unwrap();
        assert!(files.contains(&"root.md".to_string()));
        assert!(files.contains(&"sub/child.md".to_string()));
    }

    #[tokio::test]
    async fn create_with_nested_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let vault = make_vault(&dir).await;
        vault
            .create_file("a/b/c.md", "deep")
            .await
            .unwrap();
        assert_eq!(vault.read_file("a/b/c.md").await.unwrap(), "deep");
    }

    #[tokio::test]
    async fn rejects_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let vault = make_vault(&dir).await;
        assert!(matches!(
            vault.read_file("../../etc/passwd").await.unwrap_err(),
            VaultError::PathTraversal(_)
        ));
    }

    #[tokio::test]
    async fn is_file_and_is_dir_are_async() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.md"), "").unwrap();
        std::fs::create_dir(dir.path().join("d")).unwrap();
        let vault = make_vault(&dir).await;

        assert!(vault.is_file("f.md").await.unwrap());
        assert!(!vault.is_dir("f.md").await.unwrap());
        assert!(vault.is_dir("d").await.unwrap());
        assert!(!vault.is_file("d").await.unwrap());
        assert!(!vault.is_file("no.md").await.unwrap());
    }

    #[tokio::test]
    async fn undo_file_recreates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let vault = make_vault(&dir).await;

        // Create a file in a subdirectory, then delete it
        vault.create_file("sub/note.md", "original").await.unwrap();
        vault.delete_file("sub/note.md").await.unwrap();

        // Remove the now-empty subdirectory (simulating external cleanup)
        tokio::fs::remove_dir(dir.path().join("sub")).await.unwrap();

        // undo_file must recreate the parent directory
        let restored = vault.undo_file("sub/note.md").await.unwrap();
        assert_eq!(restored, "original");
        assert_eq!(vault.read_file("sub/note.md").await.unwrap(), "original");
    }
}
