pub mod fs;
pub mod path;
pub mod undo;

pub use fs::VaultFs;
pub use undo::UndoStack;

/// All errors that can occur in vault operations.
#[derive(Debug, thiserror::Error)]
pub enum VaultError {
    #[error("Path traversal attempt: {0}")]
    PathTraversal(String),

    #[error("Path not found: {0}")]
    NotFound(String),

    #[error("File already exists: {0}")]
    AlreadyExists(String),

    #[error("Not a file: {0}")]
    NotAFile(String),

    #[error("Not a directory: {0}")]
    NotADirectory(String),

    #[error("File too large: {size} bytes (max: {max} bytes)")]
    FileTooLarge { size: usize, max: usize },

    #[error("Invalid line range: {0}")]
    InvalidRange(String),

    #[error("No match found for replacement text in '{path}'")]
    NoMatch { path: String },

    #[error("Found {count} matches in '{path}'. Provide more specific text.")]
    MultipleMatches { path: String, count: usize },

    #[error("No undo history for: {0}")]
    NoUndoHistory(String),

    #[error("Invalid regex: {0}")]
    InvalidRegex(String),

    #[error("Empty old_str is not allowed")]
    EmptyOldStr,

    #[error("File is not valid UTF-8: {0}")]
    NotUtf8(String),

    #[error("IO error on '{path}': {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

impl VaultError {
    /// Create an IO error with a path context.
    pub(crate) fn io(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        VaultError::Io {
            path: path.to_string(),
            source,
        }
    }
}
