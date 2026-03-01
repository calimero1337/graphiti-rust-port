# Implementation Plan: TASK-GR-003 — Config Types & Utility Functions

## Overview

Implement configuration types and utility functions for graphiti-rs (PORTING_PLAN.md Group 2, Task 06 + config portion of types.rs). This covers 5 files across 2 modules with comprehensive test coverage.

---

## 1. `src/types.rs` — Configuration Types

### 1.1 `GraphitiConfig` struct

**Purpose**: Central configuration loaded from environment variables.

**Fields**:
```
neo4j_uri: String          — env: NEO4J_URI, default: "bolt://localhost:7687"
neo4j_user: String         — env: NEO4J_USER, default: "neo4j"
neo4j_password: String     — env: NEO4J_PASSWORD, required
openai_api_key: String     — env: OPENAI_API_KEY, required
embedding_dim: usize       — env: EMBEDDING_DIM, default: 1536
model_name: String         — env: MODEL_NAME, default: "gpt-4o"
small_model_name: String   — env: SMALL_MODEL_NAME, default: "gpt-4.1-nano"
group_id: Option<String>   — env: GROUP_ID, optional
```

**Derives**: `Debug, Clone, Serialize, Deserialize`

**Validation** (via `validator` derive):
- `neo4j_uri`: `#[validate(url)]`
- `neo4j_password`: `#[validate(length(min = 1))]`
- `openai_api_key`: `#[validate(length(min = 1))]`
- `embedding_dim`: Must be > 0 (custom validator or range)

**Methods**:
- `from_env() -> Result<Self>`: Load from environment using `dotenvy::dotenv().ok()` (non-fatal if .env missing), then `std::env::var()` for each field. Map missing required vars to `GraphitiError::Validation`. Call `validator::Validate::validate()` and map errors.
- `Default` impl with sensible defaults for optional fields.

**Edge cases**:
- `.env` file not present — must not panic, falls back to process env
- Empty string env vars — treated as missing for required fields
- Non-numeric `EMBEDDING_DIM` — map parse error to `GraphitiError::Validation`

### 1.2 Forward declarations (stubs only for this task)

These are defined fully in later tasks but need placeholder types so the crate compiles:

- `SearchConfig` — empty struct with `Debug, Clone, Serialize, Deserialize` derives (full implementation in search module, Task 15)
- `IngestionConfig` — empty struct with derives (full implementation in pipeline module)

**Note**: The `GraphitiClients` container is defined in Task 05 (traits), not in this task. Leave the existing TODO comment if it references GraphitiClients — it will be implemented when the driver/LLM traits are done.

---

## 2. `src/utils/mod.rs` — Module Re-exports

**Current state**: Module doc comment only (8 lines).

**Changes**:
- Declare submodules: `pub mod similarity;`, `pub mod text;`, `pub mod datetime;`
- Re-export key functions:
  ```
  pub use similarity::{cosine_similarity, normalize_l2};
  pub use text::{normalize_whitespace, truncate_with_ellipsis, extract_json_from_response, lucene_sanitize};
  pub use datetime::{parse_flexible_datetime, format_neo4j_datetime};
  ```

---

## 3. `src/utils/similarity.rs` — Vector Similarity

### 3.1 `cosine_similarity(a: &[f32], b: &[f32]) -> f32`

**Algorithm**:
1. If `a.len() != b.len()` or either is empty, return `0.0`
2. Convert to `ndarray::ArrayView1<f32>`
3. Compute `dot = a.dot(b)`
4. Compute `norm_a = a.dot(a).sqrt()`, `norm_b = b.dot(b).sqrt()`
5. If either norm is `0.0` (zero vector), return `0.0`
6. Return `dot / (norm_a * norm_b)`

**Edge cases**:
- Empty slices → 0.0
- Mismatched lengths → 0.0
- Zero vector (all zeros) → 0.0
- Identical vectors → 1.0
- Opposite vectors → -1.0
- NaN/Inf in input — ndarray handles naturally, result may be NaN (acceptable)

### 3.2 `normalize_l2(v: &[f32]) -> Vec<f32>`

**Algorithm**:
1. Convert to `ndarray::Array1<f32>`
2. Compute `norm = v.dot(v).sqrt()`
3. If `norm == 0.0`, return `vec![0.0; v.len()]`
4. Return `(v / norm).to_vec()`

**Edge cases**:
- Empty slice → empty vec
- Zero vector → zero vector (same length)
- Already normalized → unchanged (within floating point tolerance)

### Tests (in `#[cfg(test)] mod tests`):
- `test_cosine_identical_vectors` — expect 1.0
- `test_cosine_orthogonal_vectors` — expect 0.0 (e.g., [1,0] vs [0,1])
- `test_cosine_opposite_vectors` — expect -1.0
- `test_cosine_known_vectors` — hand-computed value, assert within epsilon
- `test_cosine_empty_vectors` — expect 0.0
- `test_cosine_mismatched_lengths` — expect 0.0
- `test_cosine_zero_vector` — expect 0.0
- `test_normalize_l2_unit` — result should have magnitude ~1.0
- `test_normalize_l2_zero` — zero vector stays zero
- `test_normalize_l2_empty` — empty stays empty

---

## 4. `src/utils/text.rs` — Text Processing

### 4.1 `normalize_whitespace(s: &str) -> String`

Replace consecutive whitespace (spaces, tabs, newlines) with a single space. Trim leading/trailing whitespace.

**Implementation**: Use `regex` with pattern `\s+` → replace with `" "`, then `.trim().to_string()`.

**Edge cases**: Empty string → empty string. Only whitespace → empty string. Unicode whitespace (e.g. non-breaking space U+00A0) — `\s` in Rust regex matches Unicode whitespace by default.

### 4.2 `truncate_with_ellipsis(s: &str, max_len: usize) -> String`

If `s.len() <= max_len`, return `s.to_string()`. Otherwise return `s[..max_len-3] + "..."`. Must respect UTF-8 char boundaries — use `s.char_indices()` to find the correct cut point.

**Implementation**:
1. If `max_len <= 3`, return `"..."[..max_len].to_string()`
2. Count chars. If char count <= `max_len`, return as-is.
3. Find byte offset of the `(max_len - 3)`th char boundary
4. Return `&s[..byte_offset]` + `"..."`

**Edge cases**: Empty string. `max_len = 0` → empty string. `max_len < 3` → just dots. Multi-byte UTF-8 chars (emoji, CJK).

### 4.3 `extract_json_from_response(s: &str) -> Option<&str>`

Extract the first JSON object or array from a potentially markdown-wrapped LLM response. LLMs often wrap JSON in ` ```json ... ``` ` blocks.

**Algorithm**:
1. Try to find ` ```json\n...\n``` ` — extract inner content
2. Try to find ` ```\n...\n``` ` — extract inner content
3. Find the first `{` or `[` and match to the last corresponding `}` or `]`
4. Return `None` if no JSON-like content found

**Implementation**: Use `str::find` and `str::rfind` — no regex needed for this. For the fenced code block, search for "```json" or "```" patterns.

**Edge cases**: No JSON → None. Nested braces. Multiple JSON objects (return first complete one). Empty fenced block.

### 4.4 `lucene_sanitize(s: &str) -> String`

Escape Lucene special characters for safe use in Neo4j full-text index queries.

**Lucene special chars**: `+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /`

**Implementation**: Iterate chars, prepend `\` before each special char. Handle `&&` and `||` as two-char sequences.

**Edge cases**: Empty string → empty string. String with no special chars → unchanged. All special chars.

### Tests:
- `test_normalize_whitespace_basic` — tabs, newlines collapsed
- `test_normalize_whitespace_empty` — empty string
- `test_normalize_whitespace_unicode` — non-breaking spaces
- `test_truncate_basic` — standard truncation
- `test_truncate_short_string` — no truncation needed
- `test_truncate_emoji` — multi-byte boundary handling
- `test_truncate_zero_length` — edge case
- `test_extract_json_fenced` — ` ```json {...} ``` `
- `test_extract_json_bare` — raw `{...}` in text
- `test_extract_json_none` — no JSON present
- `test_lucene_sanitize_special_chars`
- `test_lucene_sanitize_no_special`
- `test_lucene_sanitize_empty`

---

## 5. `src/utils/datetime.rs` — DateTime Helpers

### 5.1 `parse_flexible_datetime(s: &str) -> Option<DateTime<Utc>>`

Parse various datetime string formats into UTC `DateTime`.

**Supported formats** (in order of attempt):
1. RFC 3339 / ISO 8601: `"2024-01-15T10:30:00Z"`, `"2024-01-15T10:30:00+05:00"`
2. ISO 8601 without timezone (assume UTC): `"2024-01-15T10:30:00"`
3. Date only (midnight UTC): `"2024-01-15"`
4. Common US format: `"01/15/2024"`
5. Neo4j datetime string: `"2024-01-15T10:30:00.000000000Z"` (nanosecond precision)

**Implementation**: Use `chrono::DateTime::parse_from_rfc3339` first, then fall through to `NaiveDateTime::parse_from_str` with various format strings, converting to `DateTime<Utc>`.

**Edge cases**: Empty string → None. Invalid format → None. Timezone-aware → convert to UTC. Nanosecond precision.

### 5.2 `format_neo4j_datetime(dt: &DateTime<Utc>) -> String`

Format a `DateTime<Utc>` for Neo4j Cypher datetime literal.

**Output format**: `"2024-01-15T10:30:00.000000000Z"` (ISO 8601 with nanosecond precision and Z suffix).

**Implementation**: `dt.format("%Y-%m-%dT%H:%M:%S%.9fZ").to_string()`

**Edge cases**: Sub-second precision preserved. Midnight → explicit `T00:00:00.000000000Z`.

### Tests:
- `test_parse_rfc3339` — standard RFC 3339
- `test_parse_iso_no_tz` — bare ISO without timezone
- `test_parse_date_only` — date → midnight UTC
- `test_parse_neo4j_format` — nanosecond Neo4j string
- `test_parse_invalid` — garbage string → None
- `test_parse_empty` — empty string → None
- `test_format_neo4j` — roundtrip: format then parse
- `test_format_neo4j_midnight` — midnight edge case

---

## 6. Changes to `src/lib.rs`

No changes needed — `pub mod types;` and `pub mod utils;` are already declared.

---

## 7. Compilation Checks

After implementation:
1. `cargo check` — must pass with zero errors
2. `cargo clippy -- -D warnings` — must pass with zero warnings
3. `cargo test` — all new tests pass
4. Existing benchmark stubs must still compile

---

## 8. Files NOT Modified

- `src/errors.rs` — Already complete; no changes needed. `GraphitiError::Validation` variant already exists for config validation errors.
- `src/lib.rs` — Module declarations already present.
- `Cargo.toml` — All needed dependencies already declared.
- Benchmark files — Left as stubs (Task 28 fills them in).

---

## 9. Risk & Decision Notes

1. **`validator` crate URL validation**: `#[validate(url)]` from the `validator` crate requires the URL to be a full URL. Neo4j bolt URIs (`bolt://localhost:7687`) may not pass standard URL validation. If `bolt://` scheme isn't accepted, use `#[validate(length(min = 1))]` instead and add a custom validator that checks for valid scheme prefixes (`bolt://`, `neo4j://`, `neo4j+s://`).

2. **`extract_json_from_response` scope**: The task description says `extract_json_from_response` but the PORTING_PLAN mentions `lucene_sanitize`. Both are needed — include both.

3. **`truncate_with_ellipsis` operates on chars, not bytes**: This is critical for Unicode correctness. The max_len parameter counts Unicode scalar values (chars), not bytes.

4. **Thread safety**: All utility functions are pure — no `&mut self`, no global state. They are inherently `Send + Sync`.

---

## Subtask Plan

```yaml
max_parallel_subtasks: 3
subtasks:
  - id: "01"
    title: "Implement similarity.rs with cosine_similarity and normalize_l2 (tests first)"
    group: 1
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Create src/utils/similarity.rs with:
      - cosine_similarity(a: &[f32], b: &[f32]) -> f32 using ndarray dot product
      - normalize_l2(v: &[f32]) -> Vec<f32>

      Edge cases: empty slices (return 0.0), mismatched lengths (return 0.0),
      zero vectors (return 0.0), identical vectors (return 1.0).

      Write tests first (#[cfg(test)] mod tests), then implement.
      Tests: identical vectors→1.0, orthogonal→0.0, opposite→-1.0,
      known hand-computed value, empty, mismatched, zero vector,
      normalize_l2 unit magnitude, normalize_l2 zero, normalize_l2 empty.

      Do NOT modify utils/mod.rs yet (separate subtask).

  - id: "02"
    title: "Implement text.rs with text processing utilities (tests first)"
    group: 1
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Create src/utils/text.rs with:
      - normalize_whitespace(s: &str) -> String — collapse whitespace, trim
      - truncate_with_ellipsis(s: &str, max_len: usize) -> String — UTF-8 safe
      - extract_json_from_response(s: &str) -> Option<&str> — extract JSON from LLM response
      - lucene_sanitize(s: &str) -> String — escape Lucene special chars

      Use regex crate for normalize_whitespace.
      truncate_with_ellipsis must use char_indices() for UTF-8 boundary safety.
      extract_json_from_response: check ```json fenced blocks, then bare {/[ matching.
      lucene_sanitize: escape + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /

      Write tests first, then implement. Test edge cases:
      empty strings, unicode (emoji, CJK), only-whitespace,
      max_len=0, max_len<3, no JSON, nested braces, all special chars.

      Do NOT modify utils/mod.rs yet.

  - id: "03"
    title: "Implement datetime.rs with parsing and formatting helpers (tests first)"
    group: 1
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Create src/utils/datetime.rs with:
      - parse_flexible_datetime(s: &str) -> Option<DateTime<Utc>>
      - format_neo4j_datetime(dt: &DateTime<Utc>) -> String

      parse_flexible_datetime tries these formats in order:
      1. RFC 3339: "2024-01-15T10:30:00Z"
      2. ISO without tz (assume UTC): "2024-01-15T10:30:00"
      3. Date only (midnight UTC): "2024-01-15"
      4. Neo4j nanosecond: "2024-01-15T10:30:00.000000000Z"

      format_neo4j_datetime outputs: "%Y-%m-%dT%H:%M:%S%.9fZ"

      Use chrono::{DateTime, Utc, NaiveDate, NaiveDateTime}.
      No .unwrap() — use .ok() and Option chaining.

      Write tests first. Test: RFC3339, ISO no tz, date only, nanosecond,
      invalid string→None, empty→None, roundtrip format→parse.

      Do NOT modify utils/mod.rs yet.

  - id: "04"
    title: "Implement GraphitiConfig in types.rs with env loading and validation"
    group: 1
    agent: "rust-coding-agent"
    tags: [tdd-red, tdd-green]
    description: |
      Implement src/types.rs with:

      GraphitiConfig struct:
        neo4j_uri: String (default "bolt://localhost:7687")
        neo4j_user: String (default "neo4j")
        neo4j_password: String (required)
        openai_api_key: String (required)
        embedding_dim: usize (default 1536)
        model_name: String (default "gpt-4o")
        small_model_name: String (default "gpt-4.1-nano")
        group_id: Option<String>

      Derives: Debug, Clone, Serialize, Deserialize, Validate
      Validation: neo4j_password length(min=1), openai_api_key length(min=1),
        embedding_dim range(min=1). For neo4j_uri use length(min=1)
        (bolt:// URIs may not pass url validation).

      Methods:
      - from_env() -> crate::Result<Self>: calls dotenvy::dotenv().ok(),
        reads std::env::var for each field, applies defaults,
        calls self.validate() mapping ValidationErrors to GraphitiError::Validation.

      Also add placeholder stubs (empty structs with Debug, Clone, Serialize, Deserialize):
      - SearchConfig
      - IngestionConfig

      Keep the existing module doc comment.

      Tests: test_config_defaults (set required env vars, check defaults),
      test_config_missing_password (expect Validation error),
      test_config_custom_values.

      Use #[cfg(test)] with temp env var manipulation. Be careful about
      test isolation — env vars are process-global, so tests that modify
      env vars should restore them. Consider using a helper or serial test execution.

  - id: "05"
    title: "Wire up utils/mod.rs re-exports and verify cargo check/clippy/test"
    group: 2
    depends_on: ["01", "02", "03", "04"]
    agent: "rust-coding-agent"
    tags: [tdd-green, tdd-refactor]
    description: |
      Update src/utils/mod.rs to:
      1. Declare submodules: pub mod similarity; pub mod text; pub mod datetime;
      2. Re-export key functions:
         pub use similarity::{cosine_similarity, normalize_l2};
         pub use text::{normalize_whitespace, truncate_with_ellipsis, extract_json_from_response, lucene_sanitize};
         pub use datetime::{parse_flexible_datetime, format_neo4j_datetime};
      3. Keep the existing module doc comment.

      Then run:
      - cargo check (zero errors)
      - cargo clippy -- -D warnings (zero warnings)
      - cargo test (all tests pass)

      Fix any issues found by clippy or tests. Common issues:
      - Unused imports
      - Missing pub visibility
      - Clippy lints on float comparisons (use assert!((a-b).abs() < epsilon))

      Also verify the benchmark stubs still compile: cargo check --benches
```

