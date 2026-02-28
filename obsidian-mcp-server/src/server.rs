use std::sync::Arc;

use rmcp::{
    ServerHandler,
    handler::server::tool::Parameters,
    model::{
        CallToolResult, Content, ServerCapabilities, ServerInfo,
        Implementation,
    },
    tool, tool_handler,
};
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::vault::{VaultError, VaultFs};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn ok(text: impl Into<String>) -> CallToolResult {
    CallToolResult::success(vec![Content::text(text.into())])
}

fn err(text: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![Content::text(text.into())])
}

fn vault_err(e: VaultError) -> CallToolResult {
    err(e.to_string())
}

/// Format file contents with 1-indexed line numbers.
fn with_line_numbers(content: &str) -> String {
    content
        .lines()
        .enumerate()
        .map(|(i, line)| format!("{}: {}", i + 1, line))
        .collect::<Vec<_>>()
        .join("\n")
}

// ─── Parameter types ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
struct ViewParams {
    /// Path to the file or directory, relative to vault root.
    path: String,
    /// Optional line range [start, end], both 1-indexed. Use -1 for end-of-file.
    view_range: Option<Vec<i64>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CreateParams {
    /// Vault-relative path for the new file.
    path: String,
    /// Initial content of the new file.
    file_text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct StrReplaceParams {
    /// Vault-relative path of the file to edit.
    path: String,
    /// The exact text to find. Must match exactly one location in the file.
    old_str: String,
    /// Replacement text.
    new_str: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct InsertParams {
    /// Vault-relative path of the file to edit.
    path: String,
    /// Line number after which to insert (0 = beginning of file, line count = append).
    insert_line: i64,
    /// Text to insert (may contain newlines).
    new_str: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct UndoEditParams {
    /// Vault-relative path of the file to undo.
    path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetWorkspaceFilesParams {
    /// Optional regex pattern to filter file paths.
    pattern: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchParams {
    /// Text or regex to search for.
    query: String,
    /// Scope search to this subdirectory (optional, vault-relative).
    path: Option<String>,
    /// If true, treat `query` as a regex. Default: false.
    regex: Option<bool>,
    /// Lines of context to show before and after each match. Default: 2.
    context_lines: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DeleteParams {
    /// Vault-relative path of the file to delete.
    path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RenameParams {
    /// Current vault-relative path of the file.
    old_path: String,
    /// New vault-relative path for the file.
    new_path: String,
}

// ─── Server ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ObsidianMcpServer {
    vault: Arc<VaultFs>,
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
}

impl ObsidianMcpServer {
    pub fn new(vault: Arc<VaultFs>) -> Self {
        Self {
            vault,
            tool_router: Self::tool_router(),
        }
    }

    // ── Tool implementations ──────────────────────────────────────────────────

    /// View: read a file with line numbers or list a directory.
    #[tool(description = "Read file contents (with line numbers) or list directory entries. Paths are relative to the vault root.")]
    async fn view(&self, Parameters(p): Parameters<ViewParams>) -> CallToolResult {
        debug!(path = %p.path, "view");

        let is_dir = match self.vault.is_dir(&p.path).await {
            Ok(v) => v,
            Err(e) => return vault_err(e),
        };

        if is_dir {
            // List directory
            match self.vault.list_dir(&p.path).await {
                Ok(entries) => ok(entries.join("\n")),
                Err(e) => vault_err(e),
            }
        } else {
            // Read file
            let content = match self.vault.read_file(&p.path).await {
                Ok(c) => c,
                Err(e) => return vault_err(e),
            };

            match p.view_range {
                None => ok(with_line_numbers(&content)),
                Some(range) => {
                    let lines: Vec<&str> = content.lines().collect();
                    let total = lines.len() as i64;

                    if range.len() != 2 {
                        return err("view_range must be [start, end] with exactly two elements");
                    }
                    let start_raw = range[0];
                    let end_raw = range[1];

                    let start = if start_raw < 1 {
                        return err(format!(
                            "view_range start must be >= 1, got {}",
                            start_raw
                        ));
                    } else {
                        (start_raw - 1) as usize
                    };

                    let end = if end_raw == -1 {
                        total as usize
                    } else if end_raw < start_raw {
                        return err(format!(
                            "view_range end ({}) must be >= start ({})",
                            end_raw, start_raw
                        ));
                    } else {
                        end_raw as usize
                    };

                    if start >= lines.len() {
                        return err(format!(
                            "view_range start {} exceeds file length ({} lines)",
                            start_raw,
                            lines.len()
                        ));
                    }

                    let slice = &lines[start..end.min(lines.len())];
                    let numbered = slice
                        .iter()
                        .enumerate()
                        .map(|(i, line)| format!("{}: {}", start + i + 1, line))
                        .collect::<Vec<_>>()
                        .join("\n");
                    ok(numbered)
                }
            }
        }
    }

    /// Create: write a new file (errors if it already exists).
    #[tool(description = "Create a new file in the vault. Fails if the file already exists. Use str_replace to modify existing files.")]
    async fn create(&self, Parameters(p): Parameters<CreateParams>) -> CallToolResult {
        debug!(path = %p.path, "create");
        match self.vault.create_file(&p.path, &p.file_text).await {
            Ok(()) => ok(format!("Created '{}'", p.path)),
            Err(e) => vault_err(e),
        }
    }

    /// str_replace: replace exactly one occurrence of `old_str` with `new_str`.
    #[tool(description = "Replace exact text in a file. old_str must match exactly one location. Saves the previous version for undo_edit.")]
    async fn str_replace(&self, Parameters(p): Parameters<StrReplaceParams>) -> CallToolResult {
        debug!(path = %p.path, "str_replace");

        if p.old_str.is_empty() {
            return err("old_str must not be empty");
        }

        let content = match self.vault.read_file(&p.path).await {
            Ok(c) => c,
            Err(e) => return vault_err(e),
        };

        let count = content.matches(&p.old_str[..]).count();

        match count {
            0 => vault_err(VaultError::NoMatch {
                path: p.path.clone(),
            }),
            1 => {
                let new_content = content.replacen(&p.old_str[..], &p.new_str, 1);
                // Pass the already-read content as the undo snapshot to avoid
                // a redundant re-read and its TOCTOU window inside write_file.
                match self.vault.write_file_with_undo(&p.path, content, new_content).await {
                    Ok(()) => ok(format!("Replaced text in '{}'", p.path)),
                    Err(e) => vault_err(e),
                }
            }
            n => vault_err(VaultError::MultipleMatches {
                path: p.path.clone(),
                count: n,
            }),
        }
    }

    /// insert: insert text at a given line number.
    #[tool(description = "Insert text at a specific line number. insert_line=0 inserts at the beginning; insert_line=<line count> appends at the end.")]
    async fn insert(&self, Parameters(p): Parameters<InsertParams>) -> CallToolResult {
        debug!(path = %p.path, insert_line = p.insert_line, "insert");

        let content = match self.vault.read_file(&p.path).await {
            Ok(c) => c,
            Err(e) => return vault_err(e),
        };

        let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        let line_count = lines.len() as i64;

        if p.insert_line < 0 || p.insert_line > line_count {
            return err(format!(
                "insert_line {} is out of range [0, {}]",
                p.insert_line, line_count
            ));
        }

        let insert_at = p.insert_line as usize;
        let new_lines: Vec<String> = p.new_str.lines().map(|l| l.to_string()).collect();

        for (i, line) in new_lines.into_iter().enumerate() {
            lines.insert(insert_at + i, line);
        }

        // Preserve trailing newline behaviour: if original ended with \n, keep it
        let mut new_content = lines.join("\n");
        if content.ends_with('\n') {
            new_content.push('\n');
        }

        // Pass the already-read content as the undo snapshot to avoid
        // a redundant re-read and its TOCTOU window inside write_file.
        match self.vault.write_file_with_undo(&p.path, content, new_content).await {
            Ok(()) => ok(format!("Inserted text at line {} in '{}'", p.insert_line, p.path)),
            Err(e) => vault_err(e),
        }
    }

    /// undo_edit: revert the last edit to a file.
    #[tool(description = "Undo the last edit to a file. Only one level of undo is guaranteed per edit operation.")]
    async fn undo_edit(&self, Parameters(p): Parameters<UndoEditParams>) -> CallToolResult {
        debug!(path = %p.path, "undo_edit");
        match self.vault.undo_file(&p.path).await {
            Ok(content) => ok(format!(
                "Reverted '{}' to previous version:\n{}",
                p.path,
                with_line_numbers(&content)
            )),
            Err(e) => vault_err(e),
        }
    }

    /// get_workspace_files: list all vault files, optionally filtered.
    #[tool(description = "List all files in the vault. Optionally filter by a regex pattern applied to the file path.")]
    async fn get_workspace_files(
        &self,
        Parameters(p): Parameters<GetWorkspaceFilesParams>,
    ) -> CallToolResult {
        debug!(pattern = ?p.pattern, "get_workspace_files");

        let files = match self.vault.list_all_files().await {
            Ok(f) => f,
            Err(e) => return vault_err(e),
        };

        let filtered = match &p.pattern {
            None => files,
            Some(pat) => {
                let re = match regex::Regex::new(pat) {
                    Ok(r) => r,
                    Err(e) => return err(format!("Invalid regex '{}': {}", pat, e)),
                };
                files.into_iter().filter(|f| re.is_match(f)).collect()
            }
        };

        if filtered.is_empty() {
            ok("(no files found)")
        } else {
            ok(filtered.join("\n"))
        }
    }

    /// search: grep-like search across vault files.
    #[tool(description = "Search for text across vault files. Returns matching lines with file path, line number, and optional context.")]
    async fn search(&self, Parameters(p): Parameters<SearchParams>) -> CallToolResult {
        debug!(query = %p.query, path = ?p.path, "search");

        let use_regex = p.regex.unwrap_or(false);
        let context_lines = p.context_lines.unwrap_or(2) as usize;

        let pattern: Box<dyn Fn(&str) -> bool + Send + Sync> = if use_regex {
            let re = match regex::Regex::new(&p.query) {
                Ok(r) => r,
                Err(e) => return err(format!("Invalid regex '{}': {}", p.query, e)),
            };
            Box::new(move |s: &str| re.is_match(s))
        } else {
            let q = p.query.clone();
            Box::new(move |s: &str| s.contains(q.as_str()))
        };

        // Get file list (optionally scoped to a subdirectory)
        let all_files = match self.vault.list_all_files().await {
            Ok(f) => f,
            Err(e) => return vault_err(e),
        };

        let files: Vec<String> = match &p.path {
            None => all_files,
            Some(scope) => {
                let scope_prefix = scope.trim_start_matches('/').trim_end_matches('/');
                all_files
                    .into_iter()
                    .filter(|f| {
                        f == scope_prefix
                            || f.starts_with(&format!("{}/", scope_prefix))
                    })
                    .collect()
            }
        };

        let mut output = Vec::new();

        for file_path in &files {
            let content = match self.vault.read_file(file_path).await {
                Ok(c) => c,
                Err(e) => {
                    warn!(file = %file_path, error = %e, "skipping unreadable file");
                    continue;
                }
            };

            let lines: Vec<&str> = content.lines().collect();
            // Use a BTreeSet for O(log n) membership tests instead of a Vec
            // with O(n) linear scans.
            let match_set: std::collections::BTreeSet<usize> = lines
                .iter()
                .enumerate()
                .filter(|(_, l)| pattern(l))
                .map(|(i, _)| i)
                .collect();

            if match_set.is_empty() {
                continue;
            }

            output.push(format!("=== {} ===", file_path));

            // Merge overlapping context windows
            let mut shown = std::collections::BTreeSet::new();
            for &mi in &match_set {
                let start = mi.saturating_sub(context_lines);
                let end = (mi + context_lines + 1).min(lines.len());
                for idx in start..end {
                    shown.insert(idx);
                }
            }

            let mut prev: Option<usize> = None;
            for idx in &shown {
                if let Some(p) = prev {
                    if idx.saturating_sub(p) > 1 {
                        output.push("---".to_string());
                    }
                }
                let marker = if match_set.contains(idx) { ">" } else { " " };
                output.push(format!("{} {:>4}: {}", marker, idx + 1, lines[*idx]));
                prev = Some(*idx);
            }
        }

        if output.is_empty() {
            ok(format!("No matches found for '{}'", p.query))
        } else {
            ok(output.join("\n"))
        }
    }

    /// delete: delete a file from the vault.
    #[tool(description = "Delete a file from the vault. The previous content is saved for undo_edit.")]
    async fn delete(&self, Parameters(p): Parameters<DeleteParams>) -> CallToolResult {
        debug!(path = %p.path, "delete");
        match self.vault.delete_file(&p.path).await {
            Ok(()) => ok(format!("Deleted '{}'", p.path)),
            Err(e) => vault_err(e),
        }
    }

    /// rename: move or rename a file within the vault.
    #[tool(description = "Rename or move a file within the vault. Fails if the destination already exists.")]
    async fn rename(&self, Parameters(p): Parameters<RenameParams>) -> CallToolResult {
        debug!(from = %p.old_path, to = %p.new_path, "rename");
        match self.vault.rename_file(&p.old_path, &p.new_path).await {
            Ok(()) => ok(format!("Renamed '{}' → '{}'", p.old_path, p.new_path)),
            Err(e) => vault_err(e),
        }
    }
}

// ─── ServerHandler ────────────────────────────────────────────────────────────

#[tool_handler]
impl ServerHandler for ObsidianMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "obsidian-mcp-server".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
            instructions: Some(
                "MCP server for Obsidian vault file operations. \
                All file paths are relative to the vault root. \
                Use get_workspace_files to discover files."
                    .into(),
            ),
        }
    }
}
