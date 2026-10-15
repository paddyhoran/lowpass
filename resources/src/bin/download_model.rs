//! Downloads the pretrained base encoder checkpoint that every trained
//! tool-calling model fine-tunes on top of.
//!
//! Usage:
//!   cargo run --bin download_model -- [target_dir]
//!
//! Default target_dir: models/bert-tiny

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const BASE_URL: &str = "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main";
const FILES: &[&str] = &["config.json", "model.safetensors", "vocab.txt"];

fn main() -> Result<()> {
    let target_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/bert-tiny"));

    fs::create_dir_all(&target_dir)
        .with_context(|| format!("creating {}", target_dir.display()))?;

    for file in FILES {
        download(&format!("{BASE_URL}/{file}"), &target_dir.join(file))?;
    }

    println!("\nBase checkpoint ready at {}", target_dir.display());
    Ok(())
}

fn download(url: &str, dest: &Path) -> Result<()> {
    if dest.exists() {
        println!("skipping {} (already exists)", dest.display());
        return Ok(());
    }

    println!("fetching {url}...");
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("requesting {url}"))?;

    let mut reader = response.into_body().into_reader();
    let mut file = File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    let bytes =
        io::copy(&mut reader, &mut file).with_context(|| format!("writing {}", dest.display()))?;

    println!("  saved {} ({bytes} bytes)", dest.display());
    Ok(())
}
