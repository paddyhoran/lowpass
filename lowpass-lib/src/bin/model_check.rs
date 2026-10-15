//! Architecture check (FRAMEWORK.md section 5, step 1): wires a pretrained
//! BERT-Tiny encoder up to two freshly-initialized heads (intent, slot) and
//! runs a forward pass over a handful of hand-written examples.
//!
//! The heads are untrained, so the predicted labels are meaningless — this
//! only proves the plumbing (weight loading, tokenization, shapes, schema
//! filtering) is wired correctly end to end before investing in a dataset.
//! Re-run this after any change to `model.rs`/`tokenizer.rs`, or when
//! swapping the base checkpoint, to confirm everything still loads and runs.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::Config as BertConfig;
use tokenizers::Tokenizer;

use lowpass_lib::model::IntentSlotModel;
use lowpass_lib::tokenizer::build_bert_tokenizer;

/// `google/bert_uncased_L-2_H-128_A-2` (BERT-Tiny), pre-downloaded into this
/// directory — see README for the fetch command. Fetching lives outside this
/// check; the production data/training pipeline will own real model
/// acquisition (versioning, checksums, swapping to MiniLM, etc).
const DEFAULT_MODEL_DIR: &str = "models/bert-tiny";

// Head 1 vocabulary: one entry per registered tool.
const INTENTS: &[&str] = &["set_alarm", "check_weather", "calculate"];

// Head 2 vocabulary: BIO tags across every tool's slots (see FRAMEWORK.md 2.2 -
// Schema Filtering is what keeps a tool from seeing another tool's tags).
const SLOT_LABELS: &[&str] = &[
    "O",
    "B-TIME",
    "I-TIME",
    "B-LOCATION",
    "I-LOCATION",
    "B-EXPRESSION",
    "I-EXPRESSION",
];

const EXAMPLES: &[&str] = &[
    "set alarm for 7am",
    "check weather in Seattle",
    "calculate 5 + 3",
];

fn model_files(model_dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");
    let vocab_path = model_dir.join("vocab.txt");

    for path in [&config_path, &weights_path, &vocab_path] {
        anyhow::ensure!(
            path.exists(),
            "missing {}\n\nFetch the BERT-Tiny checkpoint first, e.g.:\n\
             mkdir -p {model_dir:?} && cd {model_dir:?} && \\\n\
             curl -sLO https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/config.json && \\\n\
             curl -sLO https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/model.safetensors && \\\n\
             curl -sLO https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/vocab.txt",
            path.display()
        );
    }

    Ok((config_path, weights_path, vocab_path))
}

fn encode_batch(tokenizer: &Tokenizer, device: &Device) -> Result<(Tensor, Tensor)> {
    let encodings = tokenizer
        .encode_batch(EXAMPLES.to_vec(), true)
        .map_err(anyhow::Error::msg)
        .context("tokenizing examples")?;

    let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

    let mut input_ids = Vec::with_capacity(EXAMPLES.len());
    let mut token_type_ids = Vec::with_capacity(EXAMPLES.len());
    for encoding in &encodings {
        let mut ids = encoding.get_ids().to_vec();
        let mut type_ids = encoding.get_type_ids().to_vec();
        ids.resize(max_len, 0);
        type_ids.resize(max_len, 0);
        input_ids.push(ids);
        token_type_ids.push(type_ids);
    }

    let input_ids = Tensor::new(input_ids, device)?;
    let token_type_ids = Tensor::new(token_type_ids, device)?;
    Ok((input_ids, token_type_ids))
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let model_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));

    println!(
        "Loading BERT-Tiny checkpoint from {}...",
        model_dir.display()
    );
    let (config_path, weights_path, vocab_path) = model_files(&model_dir)?;

    let config_json = std::fs::read_to_string(&config_path).context("reading config.json")?;
    let bert_config: BertConfig =
        serde_json::from_str(&config_json).context("parsing config.json")?;

    println!("Building WordPiece tokenizer from vocab.txt...");
    let tokenizer = build_bert_tokenizer(&vocab_path)?;

    println!("Loading pretrained encoder weights + fresh classifier heads...");
    let vb_encoder =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    let vb_heads = VarBuilder::zeros(DType::F32, &device);

    let model = IntentSlotModel::load(
        vb_encoder,
        &bert_config,
        vb_heads,
        INTENTS.len(),
        SLOT_LABELS.len(),
    )?;

    let (input_ids, token_type_ids) = encode_batch(&tokenizer, &device)?;
    let attention_mask = input_ids.ne(0u32)?;

    let (intent_logits, slot_logits) =
        model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

    println!(
        "\nintent_logits shape: {:?} (expected [{}, {}])",
        intent_logits.dims(),
        EXAMPLES.len(),
        INTENTS.len()
    );
    println!(
        "slot_logits shape:   {:?} (expected [{}, seq_len, {}])\n",
        slot_logits.dims(),
        EXAMPLES.len(),
        SLOT_LABELS.len()
    );

    for (i, example) in EXAMPLES.iter().enumerate() {
        let intent_idx = intent_logits.i(i)?.argmax(0)?.to_scalar::<u32>()? as usize;
        println!(
            "\"{example}\" -> predicted intent: {} (untrained, expect noise)",
            INTENTS[intent_idx]
        );

        let per_token_slot = slot_logits.i(i)?.argmax(1)?.to_vec1::<u32>()?;
        let tags: Vec<&str> = per_token_slot
            .iter()
            .map(|&idx| SLOT_LABELS[idx as usize])
            .collect();
        println!("  per-token slot tags: {tags:?}");
    }

    println!("\nCheck complete: encoder + both heads loaded and forward pass succeeded.");
    Ok(())
}
