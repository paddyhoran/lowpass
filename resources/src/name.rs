//! Model-name validation.
//!
//! A model's name ends up as a Cargo `[lib]` target name in the crate that
//! `create_model` packages, and as a path segment in `use <name>::lowpass`.
//! Cargo rejects library target names containing hyphens outright, so
//! `just create my-model` would otherwise produce a crate that cannot be
//! parsed, let alone built. Normalizing up front keeps that failure from
//! reaching the generated output.

use anyhow::{bail, Result};

/// Normalizes a user-supplied model name into a valid Rust identifier.
///
/// Hyphens become underscores; anything else that isn't alphanumeric or an
/// underscore, or a name starting with a digit, is rejected rather than
/// silently mangled.
///
/// Returns the normalized name, which the caller should use for every path
/// and identifier it derives — and should echo back if it differs from the
/// input, so the user knows what to pass to subsequent commands.
pub fn normalize_model_name(raw: &str) -> Result<String> {
    if raw.is_empty() {
        bail!("model name must not be empty");
    }

    let normalized: String = raw
        .chars()
        .map(|c| if c == '-' { '_' } else { c })
        .collect();

    if let Some(bad) = normalized
        .chars()
        .find(|c| !c.is_ascii_alphanumeric() && *c != '_')
    {
        bail!(
            "model name {raw:?} contains {bad:?}; \
             names must be ASCII letters, digits, underscores or hyphens"
        );
    }

    if normalized.starts_with(|c: char| c.is_ascii_digit()) {
        bail!("model name {raw:?} must not start with a digit");
    }

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_valid_names_alone() {
        assert_eq!(normalize_model_name("my_model").unwrap(), "my_model");
        assert_eq!(normalize_model_name("model2").unwrap(), "model2");
    }

    #[test]
    fn rewrites_hyphens_to_underscores() {
        assert_eq!(normalize_model_name("my-model").unwrap(), "my_model");
        assert_eq!(normalize_model_name("a-b-c").unwrap(), "a_b_c");
    }

    #[test]
    fn rejects_names_cargo_or_rust_would_choke_on() {
        for bad in ["", "my model", "my.model", "my/model", "2fast"] {
            assert!(
                normalize_model_name(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }
}
