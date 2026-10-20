//! Scaffolds a new model's folder structure under `models/<name>/`, with one
//! example tool so `just generate <name>` / `just train <name>` can run
//! immediately without the user having to hand-write a tool definition first.
//!
//! Usage:
//!   cargo run --bin init_model -- <name>

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};

use resources::name::normalize_model_name;

const EXAMPLE_TOOL: &str = "set_alarm";

const EXAMPLE_TEMPLATES: &str = r#"[
  "set alarm for {TIME}",
  "set an alarm for {TIME}",
  "wake me up at {TIME}",
  "alarm for {TIME}",
  "please set an alarm for {TIME}"
]
"#;

const EXAMPLE_PARAMS: &str = r#"{
  "TIME": ["7am", "7:00 AM", "6:30 pm", "midnight", "noon", "9:15am"]
}
"#;

fn main() -> Result<()> {
    let raw_name = std::env::args()
        .nth(1)
        .context("usage: init_model <name>")?;
    let name = normalize_model_name(&raw_name)?;
    if name != raw_name {
        println!(
            "note: using '{name}' instead of '{raw_name}' -- a model name becomes a Rust \
             crate and library name, which cannot contain hyphens"
        );
    }

    let tools_dir = PathBuf::from("models")
        .join(&name)
        .join("data")
        .join("tools");

    if tools_dir.is_dir() && fs::read_dir(&tools_dir)?.next().is_some() {
        println!(
            "{} already has tool definitions, leaving it as-is",
            tools_dir.display()
        );
        return Ok(());
    }

    let example_dir = tools_dir.join(EXAMPLE_TOOL);
    fs::create_dir_all(&example_dir)
        .with_context(|| format!("creating {}", example_dir.display()))?;
    fs::write(example_dir.join("templates.json"), EXAMPLE_TEMPLATES)
        .with_context(|| format!("writing {}/templates.json", example_dir.display()))?;
    fs::write(example_dir.join("params.json"), EXAMPLE_PARAMS)
        .with_context(|| format!("writing {}/params.json", example_dir.display()))?;

    println!(
        "scaffolded {} with example tool '{EXAMPLE_TOOL}'",
        tools_dir.display()
    );
    println!(
        "edit its templates.json/params.json (and add more tool directories), then run:\n  \
         just generate {name}\n  just train {name}"
    );
    Ok(())
}
