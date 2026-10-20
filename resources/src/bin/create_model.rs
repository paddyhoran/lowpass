//! Packages a trained model into a standalone Rust crate at
//! `models/<name>/<name>/`. The base encoder checkpoint and the trained
//! heads are embedded into the crate's source via `include_bytes!`/
//! `include_str!`, so the built library and binary need no runtime assets.
//!
//! The generated crate depends on `lowpass-lib` by relative path, so it
//! builds as-is where it is written but is not self-contained: moving it
//! elsewhere means repointing that path at a checkout of this repo (or
//! vendoring `lowpass-lib` alongside it). Only the model weights travel with
//! the crate; the inference code does not.
//!
//! Usage:
//!   cargo run --bin create_model -- <name>

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};

use resources::name::normalize_model_name;

fn main() -> Result<()> {
    let raw_name = std::env::args()
        .nth(1)
        .context("usage: create_model <name>")?;
    let name = normalize_model_name(&raw_name)?;
    if name != raw_name {
        println!(
            "note: using '{name}' instead of '{raw_name}' -- a model name becomes a Rust \
             crate and library name, which cannot contain hyphens"
        );
    }

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
    fs::create_dir_all(&assets_dir)
        .with_context(|| format!("creating {}", assets_dir.display()))?;
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
/// Cargo doesn't try to fold it into the enclosing repo's workspace — it
/// builds independently of the rest of the repo, even though the
/// `lowpass-lib` path dependency still points back into it.
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
anyhow = "1.0"
serde_json = "1.0"
"#
    )
}

const LIB_RS: &str = r#"use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use lowpass_lib::infer::LowPass;
use lowpass_lib::ToolCall;

static ENCODER_CONFIG: &str = include_str!("../assets/config.json");
static ENCODER_WEIGHTS: &[u8] = include_bytes!("../assets/model.safetensors");
static VOCAB_TXT: &[u8] = include_bytes!("../assets/vocab.txt");
static HEADS_WEIGHTS: &[u8] = include_bytes!("../assets/heads.safetensors");
static LABEL_VOCAB: &str = include_str!("../assets/vocab.json");

/// Loads the embedded assets once, on first use.
///
/// `anyhow::Error` can't be cloned out of a `OnceLock`, so a load failure is
/// cached as its rendered message and re-wrapped on every later call.
fn model() -> Result<&'static LowPass> {
    static MODEL: OnceLock<std::result::Result<LowPass, String>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            LowPass::load(ENCODER_CONFIG, ENCODER_WEIGHTS, VOCAB_TXT, HEADS_WEIGHTS, LABEL_VOCAB)
                .map_err(|e| format!("{e:#}"))
        })
        .as_ref()
        .map_err(|e| anyhow!("loading embedded model assets: {e}"))
}

/// Runs `input` through the trained intent + slot model and returns the
/// resulting tool call.
pub fn lowpass(input: &str) -> Result<ToolCall> {
    model()?.infer(input)
}
"#;

fn main_rs(name: &str) -> String {
    format!(
        r#"use std::env;

use anyhow::{{bail, Result}};

fn main() -> Result<()> {{
    let input = env::args().skip(1).collect::<Vec<_>>().join(" ");
    if input.is_empty() {{
        bail!("usage: {name} \"<your command>\"");
    }}

    let tool_call = {name}::lowpass(&input)?;
    println!("{{}}", serde_json::to_string_pretty(&tool_call)?);
    Ok(())
}}
"#
    )
}
