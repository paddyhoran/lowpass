use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// A character-offset slot span, matching the `generator` crate's JSONL
/// output. The two crates deliberately don't share Rust types — JSON is the
/// contract between them.
#[derive(Debug, Clone, Deserialize)]
pub struct SlotSpan {
    pub label: String,
    pub start: usize,
    pub end: usize,
}

/// One training example, matching one line of the `generator` crate's output.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingExample {
    pub text: String,
    pub intent: String,
    pub slots: Vec<SlotSpan>,
}

/// Reads a JSON Lines file produced by the `generator` crate.
pub fn read_jsonl(path: &Path) -> Result<Vec<TrainingExample>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).with_context(|| format!("parsing line: {line}")))
        .collect()
}
