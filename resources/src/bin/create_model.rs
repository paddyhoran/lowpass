//! Packages a trained model into a standalone Rust crate at
//! `models/<name>/<name>/`. The base encoder checkpoint and the trained
//! heads are embedded into the crate's source via `include_bytes!`/
//! `include_str!`, so the built library/binary needs no runtime assets —
//! and, since it depends on `lowpass-lib` only by path (no vendored source),
//! it stays a normal crate another project can depend on directly.
//!
//! Usage:
//!   cargo run --bin create_model -- <name>

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let name = std::env::args()
        .nth(1)
        .context("usage: create_model <name>")?;

    let model_dir = PathBuf::from("models").join(&name);
    let base_dir = PathBuf::from("models/bert-tiny");
    let artifacts_dir = model_dir.join("artifacts");
    let crate_dir = model_dir.join(&name);

    for required in [
        base_dir.join("config.json"),
        base_dir.join("model.safetensors"),
        base_dir.join("vocab.txt"),
    ] {
        anyhow::ensure!(
            required.exists(),
            "missing {} -- run `just start {name}` first",
            required.display()
        );
    }
    for required in [
        artifacts_dir.join("heads.safetensors"),
        artifacts_dir.join("vocab.json"),
    ] {
        anyhow::ensure!(
            required.exists(),
            "missing {} -- run `just train {name}` first",
            required.display()
        );
    }

    let assets_dir = crate_dir.join("assets");
    let src_dir = crate_dir.join("src");
    fs::create_dir_all(&assets_dir).with_context(|| format!("creating {}", assets_dir.display()))?;
    fs::create_dir_all(&src_dir).with_context(|| format!("creating {}", src_dir.display()))?;

    for (src, filename) in [
        (base_dir.join("config.json"), "config.json"),
        (base_dir.join("model.safetensors"), "model.safetensors"),
        (base_dir.join("vocab.txt"), "vocab.txt"),
        (artifacts_dir.join("heads.safetensors"), "heads.safetensors"),
        (artifacts_dir.join("vocab.json"), "vocab.json"),
    ] {
        let dest = assets_dir.join(filename);
        fs::copy(&src, &dest)
            .with_context(|| format!("copying {} to {}", src.display(), dest.display()))?;
    }

    fs::write(crate_dir.join("Cargo.toml"), cargo_toml(&name))
        .with_context(|| format!("writing {}/Cargo.toml", crate_dir.display()))?;
    fs::write(crate_dir.join(".gitignore"), "/target\n")
        .with_context(|| format!("writing {}/.gitignore", crate_dir.display()))?;
    fs::write(src_dir.join("lib.rs"), LIB_RS)
        .with_context(|| format!("writing {}/lib.rs", src_dir.display()))?;
    fs::write(src_dir.join("main.rs"), main_rs(&name))
        .with_context(|| format!("writing {}/main.rs", src_dir.display()))?;

    println!("packaged '{name}' at {}", crate_dir.display());
    println!(
        "try it:\n  cd {} && cargo run -- \"set alarm for 7am\"",
        crate_dir.display()
    );
    Ok(())
}

/// `[workspace]` here marks this crate as the root of its own workspace, so
/// Cargo doesn't try to fold it into the enclosing repo's workspace — it's
/// meant to stand alone, copyable to another project as-is.
fn cargo_toml(name: &str) -> String {
    format!(
        r#"[workspace]

[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[lib]
name = "{name}"
path = "src/lib.rs"

[[bin]]
name = "{name}"
path = "src/main.rs"

[dependencies]
lowpass-lib = {{ path = "../../../lowpass-lib" }}
serde_json = "1.0"
"#
    )
}

const LIB_RS: &str = r#"use std::sync::OnceLock;

use lowpass_lib::infer::LowPass;
use lowpass_lib::ToolCall;

static ENCODER_CONFIG: &str = include_str!("../assets/config.json");
static ENCODER_WEIGHTS: &[u8] = include_bytes!("../assets/model.safetensors");
static VOCAB_TXT: &[u8] = include_bytes!("../assets/vocab.txt");
static HEADS_WEIGHTS: &[u8] = include_bytes!("../assets/heads.safetensors");
static LABEL_VOCAB: &str = include_str!("../assets/vocab.json");

fn model() -> &'static LowPass {
    static MODEL: OnceLock<LowPass> = OnceLock::new();
    MODEL.get_or_init(|| {
        LowPass::load(ENCODER_CONFIG, ENCODER_WEIGHTS, VOCAB_TXT, HEADS_WEIGHTS, LABEL_VOCAB)
            .expect("embedded model assets failed to load")
    })
}

/// Runs `input` through the trained intent + slot model and returns the
/// resulting tool call.
pub fn lowpass(input: String) -> ToolCall {
    model().infer(&input).expect("inference failed")
}
"#;

fn main_rs(name: &str) -> String {
    format!(
        r#"use std::env;

fn main() {{
    let input = env::args().skip(1).collect::<Vec<_>>().join(" ");
    if input.is_empty() {{
        eprintln!("Usage: {name} \"<your command>\"");
        std::process::exit(1);
    }}

    let tool_call = {name}::lowpass(input);
    println!("{{}}", serde_json::to_string_pretty(&tool_call).unwrap());
}}
"#
    )
}
