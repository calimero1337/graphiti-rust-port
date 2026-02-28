use std::{io, path::{Path, PathBuf}};

use super::VaultError;

/// Validate and normalize a user-supplied vault-relative path.
///
/// # Parameters
/// - `canonical_root`: The vault root, already canonicalized (absolute, symlink-resolved).
///   Pass `VaultFs::root`, which is initialized from `Config::validate_vault()`. The root
///   is **not** re-canonicalized inside this function — the caller is responsible for
///   providing an already-canonical root, which avoids a redundant blocking syscall on
///   every tool invocation.
/// - `user_path`: Vault-relative path provided by the user.
///
/// # Security guarantees
/// 1. Rejects paths containing null bytes.
/// 2. Rejects paths whose components include `..` (pre-join check).
/// 3. For existing paths: uses `tokio::fs::canonicalize` (async, non-blocking) and
///    verifies the result starts with `canonical_root` (symlink-escape prevention).
/// 4. For non-existing paths: no `..` was present and the path was joined onto the
///    canonical root, so it is provably within the vault.
///
/// Returns the absolute path to use for filesystem operations.
pub async fn normalize_and_validate(
    canonical_root: &Path,
    user_path: &str,
) -> Result<PathBuf, VaultError> {
    // 1. Reject null bytes
    if user_path.bytes().any(|b| b == 0) {
        return Err(VaultError::PathTraversal(
            "Path contains null bytes".to_string(),
        ));
    }

    // 2. Reject `..` components — must check before joining
    for component in user_path.split('/') {
        if component == ".." {
            return Err(VaultError::PathTraversal(format!(
                "Path traversal detected: {}",
                user_path
            )));
        }
    }

    // Strip leading slashes so the path is always relative
    let cleaned = user_path.trim_start_matches('/');

    // Build the target path by joining onto the pre-canonicalized root.
    // No root canonicalization happens here (it was done once at startup).
    let target = if cleaned.is_empty() {
        canonical_root.to_path_buf()
    } else {
        canonical_root.join(cleaned)
    };

    // 3. For existing paths, do a full async symlink-aware canonicalization
    //    and verify the result is still under the vault root.
    match tokio::fs::canonicalize(&target).await {
        Ok(canonical_target) => {
            if !canonical_target.starts_with(canonical_root) {
                return Err(VaultError::PathTraversal(format!(
                    "Path escapes vault root: {}",
                    user_path
                )));
            }
            Ok(canonical_target)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            // 4. Non-existing path: no `..` was present, built on the canonical root,
            //    so the result is provably within the vault.
            Ok(target)
        }
        Err(e) => Err(VaultError::io(target.display(), e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn temp_vault() -> TempDir {
        tempfile::tempdir().expect("create temp dir")
    }

    async fn canonical(dir: &TempDir) -> PathBuf {
        tokio::fs::canonicalize(dir.path())
            .await
            .expect("canonicalize temp dir")
    }

    #[tokio::test]
    async fn valid_relative_path() {
        let vault = temp_vault();
        let root = canonical(&vault).await;
        let file = root.join("notes/test.md");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "content").unwrap();

        let result = normalize_and_validate(&root, "notes/test.md").await.unwrap();
        assert!(result.starts_with(&root));
    }

    #[tokio::test]
    async fn rejects_dotdot() {
        let vault = temp_vault();
        let root = canonical(&vault).await;
        let err = normalize_and_validate(&root, "notes/../../etc/passwd")
            .await
            .unwrap_err();
        assert!(matches!(err, VaultError::PathTraversal(_)));
    }

    #[tokio::test]
    async fn rejects_dotdot_at_root() {
        let vault = temp_vault();
        let root = canonical(&vault).await;
        let err = normalize_and_validate(&root, "../secret")
            .await
            .unwrap_err();
        assert!(matches!(err, VaultError::PathTraversal(_)));
    }

    #[tokio::test]
    async fn rejects_null_bytes() {
        let vault = temp_vault();
        let root = canonical(&vault).await;
        let err = normalize_and_validate(&root, "notes/te\0st.md")
            .await
            .unwrap_err();
        assert!(matches!(err, VaultError::PathTraversal(_)));
    }

    #[tokio::test]
    async fn strips_leading_slash() {
        let vault = temp_vault();
        let root = canonical(&vault).await;
        let file = root.join("test.md");
        fs::write(&file, "content").unwrap();

        let a = normalize_and_validate(&root, "test.md").await.unwrap();
        let b = normalize_and_validate(&root, "/test.md").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn empty_path_returns_vault_root() {
        let vault = temp_vault();
        let root = canonical(&vault).await;
        let result = normalize_and_validate(&root, "").await.unwrap();
        assert_eq!(result, root);
    }

    #[tokio::test]
    async fn nonexistent_file_is_allowed() {
        let vault = temp_vault();
        let root = canonical(&vault).await;
        // New file — doesn't exist yet, should not error
        let result = normalize_and_validate(&root, "new-note.md")
            .await
            .unwrap();
        assert!(result.starts_with(&root));
    }

    #[tokio::test]
    async fn rejects_symlink_escape() {
        let vault = temp_vault();
        let root = canonical(&vault).await;
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        fs::write(&secret, "secret").unwrap();

        // Create a symlink inside the vault that points outside
        let link_path = root.join("escape-link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&secret, &link_path).unwrap();

        #[cfg(unix)]
        {
            let err = normalize_and_validate(&root, "escape-link")
                .await
                .unwrap_err();
            assert!(matches!(err, VaultError::PathTraversal(_)));
        }
    }
}
