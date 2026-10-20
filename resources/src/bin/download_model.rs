//! Downloads the pretrained base encoder checkpoint that every trained
//! tool-calling model fine-tunes on top of.
//!
//! The fetch itself lives in `resources::download` so the training tests can
//! reuse it; this binary is just the CLI around it.
//!
//! Usage:
//!   cargo run --bin download_model -- [target_dir]
//!
//! Default target_dir: models/bert-tiny

use std::path::PathBuf;

use anyhow::Result;

use resources::download::{download_base_model, DEFAULT_TARGET_DIR};

fn main() -> Result<()> {
    let target_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_TARGET_DIR));

    download_base_model(&target_dir)?;

    println!("\nBase checkpoint ready at {}", target_dir.display());
    Ok(())
}
