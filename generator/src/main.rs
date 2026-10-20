//! A very simple training data generator.
//!
//! Reads a directory of per-tool subdirectories, each containing:
//!   - `templates.json`: a list of sentence templates, e.g. `"set alarm for {TIME}"`
//!   - `params.json`: a map from placeholder name to every value it can take,
//!     e.g. `{"TIME": ["7am", "7:00 AM", ...]}`
//!
//! For every template, generates one example per combination of its
//! placeholders' values, and writes JSON Lines to stdout (or a file), one
//! example per line:
//!
//!   {"text": "set alarm for 7am", "intent": "set_alarm", "slots": [{"label": "TIME", "start": 14, "end": 17}]}
//!
//! Slot spans are byte offsets into `text`. Deliberately not tokenizer-specific
//! BIO tags — this output stays valid even if the base encoder is swapped
//! later, since converting spans to token-level tags depends on whichever
//! tokenizer is used.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Example {
    text: String,
    intent: String,
    slots: Vec<SlotSpan>,
}

#[derive(Debug, Serialize)]
struct SlotSpan {
    label: String,
    start: usize,
    end: usize,
}

#[derive(Debug)]
enum Segment {
    Literal(String),
    Placeholder(String),
}

/// Splits a template like `"set alarm for {TIME}"` into literal text and
/// placeholder segments, in order.
fn parse_template(template: &str) -> Result<Vec<Segment>> {
    let mut segments = Vec::new();
    let mut literal = String::new();
    let mut chars = template.chars();

    while let Some(c) = chars.next() {
        if c != '{' {
            literal.push(c);
            continue;
        }
        if !literal.is_empty() {
            segments.push(Segment::Literal(std::mem::take(&mut literal)));
        }
        let mut name = String::new();
        let mut closed = false;
        for c2 in chars.by_ref() {
            if c2 == '}' {
                closed = true;
                break;
            }
            name.push(c2);
        }
        if !closed {
            bail!("unterminated placeholder '{{{name}' in template {template:?}");
        }
        segments.push(Segment::Placeholder(name));
    }
    if !literal.is_empty() {
        segments.push(Segment::Literal(literal));
    }
    Ok(segments)
}

/// Placeholder names referenced by `segments`, in first-appearance order,
/// each listed once even if used more than once in the template.
fn unique_placeholders(segments: &[Segment]) -> Vec<String> {
    let mut names = Vec::new();
    for segment in segments {
        if let Segment::Placeholder(name) = segment {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    names
}

/// All combinations of one value per list, e.g. `[[a,b],[x,y]]` ->
/// `[[a,x],[a,y],[b,x],[b,y]]`.
fn cartesian_product(value_lists: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut combos: Vec<Vec<String>> = vec![Vec::new()];
    for values in value_lists {
        let mut next = Vec::with_capacity(combos.len() * values.len());
        for combo in &combos {
            for value in values {
                let mut extended = combo.clone();
                extended.push(value.clone());
                next.push(extended);
            }
        }
        combos = next;
    }
    combos
}

/// Substitutes `values` into `segments`, returning the resulting text plus
/// the (byte) span of each placeholder occurrence in that text.
fn substitute(segments: &[Segment], values: &HashMap<String, String>) -> (String, Vec<SlotSpan>) {
    let mut text = String::new();
    let mut slots = Vec::new();

    for segment in segments {
        match segment {
            Segment::Literal(literal) => text.push_str(literal),
            Segment::Placeholder(name) => {
                let value = &values[name];
                let start = text.len();
                text.push_str(value);
                let end = text.len();
                slots.push(SlotSpan {
                    label: name.clone(),
                    start,
                    end,
                });
            }
        }
    }

    (text, slots)
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("parsing {}", path.display()))
}

fn generate_for_tool(tool_name: &str, tool_dir: &Path) -> Result<Vec<Example>> {
    let templates: Vec<String> = read_json(&tool_dir.join("templates.json"))?;
    let params: HashMap<String, Vec<String>> = read_json(&tool_dir.join("params.json"))?;

    let mut examples = Vec::new();
    for template in &templates {
        let segments = parse_template(template)?;
        let placeholder_names = unique_placeholders(&segments);

        for name in &placeholder_names {
            if !params.contains_key(name) {
                bail!(
                    "tool '{tool_name}': template {template:?} references \
                     placeholder '{{{name}}}' but params.json has no '{name}' key"
                );
            }
        }

        let value_lists: Vec<Vec<String>> = placeholder_names
            .iter()
            .map(|name| params[name].clone())
            .collect();

        let combos = if value_lists.is_empty() {
            vec![Vec::new()]
        } else {
            cartesian_product(&value_lists)
        };

        for combo in combos {
            let values: HashMap<String, String> =
                placeholder_names.iter().cloned().zip(combo).collect();
            let (text, slots) = substitute(&segments, &values);
            examples.push(Example {
                text,
                intent: tool_name.to_string(),
                slots,
            });
        }
    }

    Ok(examples)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input_dir = args
        .next()
        .context("usage: generator <input_dir> [output_file]")?;
    let output_path = args.next();

    let input_dir = PathBuf::from(input_dir);
    let mut tool_dirs: Vec<PathBuf> = std::fs::read_dir(&input_dir)
        .with_context(|| format!("reading directory {}", input_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    tool_dirs.sort();

    let mut all_examples = Vec::new();
    for tool_dir in &tool_dirs {
        let tool_name = tool_dir
            .file_name()
            .expect("directory entry always has a file name")
            .to_string_lossy()
            .into_owned();

        let examples = generate_for_tool(&tool_name, tool_dir)?;
        eprintln!("{tool_name}: {} examples", examples.len());
        all_examples.extend(examples);
    }

    let mut writer: Box<dyn Write> = match &output_path {
        Some(path) => Box::new(BufWriter::new(
            File::create(path).with_context(|| format!("creating {path}"))?,
        )),
        None => Box::new(BufWriter::new(std::io::stdout())),
    };
    for example in &all_examples {
        writeln!(writer, "{}", serde_json::to_string(example)?)?;
    }

    eprintln!("total: {} examples", all_examples.len());
    Ok(())
}
