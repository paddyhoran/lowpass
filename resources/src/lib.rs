//! Shared setup helpers behind the `resources` binaries.
//!
//! These live in a library rather than directly in the binaries so other
//! crates in the workspace can reuse them — in particular, `trainer`'s tests
//! call [`download::ensure_base_model`] to fetch the base checkpoint they
//! tokenize against.

pub mod download;
pub mod name;
