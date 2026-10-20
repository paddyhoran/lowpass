//! Fetches the pretrained base encoder checkpoint that every trained
//! tool-calling model fine-tunes on top of.
//!
//! The download is pinned to an exact Hugging Face revision and every file is
//! verified against a known SHA-256 before it is accepted. A framework whose
//! selling point is deterministic local inference shouldn't silently pick up
//! different weights because an upstream branch moved.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

/// Default location of the shared base checkpoint, relative to the repo root.
pub const DEFAULT_TARGET_DIR: &str = "models/bert-tiny";

const REPO: &str = "google/bert_uncased_L-2_H-128_A-2";

/// Exact commit of [`REPO`] the checksums below were taken from. Pinned rather
/// than tracking `main`, so a re-run years from now fetches identical bytes.
const REVISION: &str = "30b0a37ccaaa32f332884b96992754e246e48c5f";

/// A file to fetch, with the SHA-256 its contents must hash to.
struct BaseFile {
    name: &'static str,
    sha256: &'static str,
}

const FILES: &[BaseFile] = &[
    BaseFile {
        name: "config.json",
        sha256: "508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39",
    },
    BaseFile {
        name: "model.safetensors",
        sha256: "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
    },
    BaseFile {
        name: "vocab.txt",
        sha256: "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    },
];

/// Downloads any missing base-checkpoint file into `target_dir`, verifying
/// each against its pinned SHA-256.
///
/// Safe to re-run: a file already present and matching its checksum is left
/// alone and costs nothing. A file present but *not* matching is an error
/// rather than a silent overwrite — see [`ensure_base_model`] for the quiet
/// variant used by tests.
pub fn download_base_model(target_dir: &Path) -> Result<()> {
    download_base_model_inner(target_dir, true)
}

/// Same as [`download_base_model`], but prints nothing when every file is
/// already present and valid.
///
/// Intended for test setup, where the checkpoint is a fixture rather than
/// something the user asked for.
pub fn ensure_base_model(target_dir: &Path) -> Result<()> {
    download_base_model_inner(target_dir, false)
}

fn download_base_model_inner(target_dir: &Path, verbose: bool) -> Result<()> {
    fs::create_dir_all(target_dir).with_context(|| format!("creating {}", target_dir.display()))?;

    for file in FILES {
        let dest = target_dir.join(file.name);

        if dest.exists() {
            let actual = sha256_file(&dest)?;
            if actual == file.sha256 {
                if verbose {
                    println!("{} already present and verified", dest.display());
                }
                continue;
            }
            bail!(
                "{} does not match the expected checksum for {REPO}@{REVISION}\n  \
                 expected {}\n  actual   {actual}\n\
                 Delete the file and re-run to fetch a clean copy.",
                dest.display(),
                file.sha256
            );
        }

        let url = format!(
            "https://huggingface.co/{REPO}/resolve/{REVISION}/{}",
            file.name
        );
        if verbose {
            println!("fetching {url}...");
        }
        download_verified(&url, &dest, file.sha256)?;
        if verbose {
            println!("  saved {} (sha256 verified)", dest.display());
        }
    }

    Ok(())
}

/// Downloads `url` to a sibling `.part` file, checks its hash, and only then
/// moves it into place — so an interrupted run can never leave a truncated
/// file that a later run would mistake for a complete one.
fn download_verified(url: &str, dest: &Path, expected_sha256: &str) -> Result<()> {
    let partial = partial_path(dest);

    let response = ureq::get(url)
        .call()
        .with_context(|| format!("requesting {url}"))?;
    let mut reader = response.into_body().into_reader();
    let mut file =
        File::create(&partial).with_context(|| format!("creating {}", partial.display()))?;
    io::copy(&mut reader, &mut file).with_context(|| format!("writing {}", partial.display()))?;
    drop(file);

    let actual = sha256_file(&partial)?;
    if actual != expected_sha256 {
        let _ = fs::remove_file(&partial);
        bail!(
            "checksum mismatch for {url}\n  expected {expected_sha256}\n  actual   {actual}\n\
             The upstream file may have changed, or the download was corrupted."
        );
    }

    fs::rename(&partial, dest)
        .with_context(|| format!("moving {} into place", partial.display()))?;
    Ok(())
}

fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher).with_context(|| format!("hashing {}", path.display()))?;
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
