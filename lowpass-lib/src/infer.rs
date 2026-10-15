//! Inference over a trained intent + slot model: given raw text, run the
//! dual-head forward pass (see `model.rs`) and decode its output into a
//! [`ToolCall`].
//!
//! Everything here loads from in-memory bytes rather than file paths, so a
//! packaged model crate can embed its checkpoint via `include_bytes!` and
//! run with no runtime filesystem access.

use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::Config as BertConfig;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::model::IntentSlotModel;
use crate::tokenizer::build_bert_tokenizer_from_bytes;
use crate::vocab::Vocab;
use crate::ToolCall;

#[derive(Deserialize)]
struct LabelVocab {
    intents: Vec<String>,
    slot_tags: Vec<String>,
}

/// A loaded, ready-to-run model: the base encoder plus the trained
/// intent/slot heads, and the label vocabularies needed to turn their
/// predictions back into strings.
pub struct LowPass {
    tokenizer: Tokenizer,
    model: IntentSlotModel,
    intents: Vocab,
    slot_tags: Vocab,
    device: Device,
}

impl LowPass {
    /// Loads a model entirely from in-memory bytes:
    /// - `encoder_config_json`: the base encoder's `config.json`
    /// - `encoder_weights`: the base encoder's `model.safetensors`
    /// - `vocab_txt`: the base encoder's `vocab.txt`
    /// - `heads_weights`: the trained `heads.safetensors` (from `just train`)
    /// - `label_vocab_json`: the trained `vocab.json` (from `just train`)
    pub fn load(
        encoder_config_json: &str,
        encoder_weights: &[u8],
        vocab_txt: &[u8],
        heads_weights: &[u8],
        label_vocab_json: &str,
    ) -> Result<Self> {
        let device = Device::Cpu;

        let bert_config: BertConfig =
            serde_json::from_str(encoder_config_json).context("parsing encoder config.json")?;
        let tokenizer = build_bert_tokenizer_from_bytes(vocab_txt)?;

        let label_vocab: LabelVocab =
            serde_json::from_str(label_vocab_json).context("parsing label vocab.json")?;
        let intents = Vocab::new(label_vocab.intents);
        let slot_tags = Vocab::new(label_vocab.slot_tags);

        let vb_encoder =
            VarBuilder::from_buffered_safetensors(encoder_weights.to_vec(), DType::F32, &device)
                .context("loading encoder weights")?;
        let vb_heads =
            VarBuilder::from_buffered_safetensors(heads_weights.to_vec(), DType::F32, &device)
                .context("loading trained heads")?;
        let model = IntentSlotModel::load(
            vb_encoder,
            &bert_config,
            vb_heads,
            intents.len(),
            slot_tags.len(),
        )
        .context("wiring up encoder + heads")?;

        Ok(Self {
            tokenizer,
            model,
            intents,
            slot_tags,
            device,
        })
    }

    /// Runs `text` through the model and decodes the result into a `ToolCall`.
    pub fn infer(&self, text: &str) -> Result<ToolCall> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("tokenizing {text:?}"))?;

        let input_ids = Tensor::new(vec![encoding.get_ids().to_vec()], &self.device)?;
        let token_type_ids = Tensor::new(vec![encoding.get_type_ids().to_vec()], &self.device)?;
        let attention_mask = input_ids.ne(0u32)?;

        let (intent_logits, slot_logits) =
            self.model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let intent_id = intent_logits.i(0)?.argmax(0)?.to_scalar::<u32>()?;
        let tool = self.intents.label(intent_id).to_string();

        let tag_ids = slot_logits.i(0)?.argmax(1)?.to_vec1::<u32>()?;
        let params = decode_slots(
            encoding.get_offsets(),
            encoding.get_special_tokens_mask(),
            &tag_ids,
            &self.slot_tags,
            text,
        );

        Ok(ToolCall { tool, params })
    }
}

/// Reassembles per-token BIO tag predictions into slot values, using each
/// token's character offsets to slice the original text — the inverse of
/// what the training pipeline's `spans_to_bio_tags` does.
fn decode_slots(
    token_offsets: &[(usize, usize)],
    special_tokens_mask: &[u32],
    tag_ids: &[u32],
    slot_tags: &Vocab,
    text: &str,
) -> HashMap<String, String> {
    let mut params = HashMap::new();
    let mut current: Option<(String, usize, usize)> = None;

    let close = |current: &mut Option<(String, usize, usize)>,
                 params: &mut HashMap<String, String>| {
        if let Some((label, start, end)) = current.take() {
            params.insert(label, text[start..end].to_string());
        }
    };

    for (i, &tag_id) in tag_ids.iter().enumerate() {
        if special_tokens_mask[i] == 1 {
            continue;
        }
        let tag = slot_tags.label(tag_id);
        let (tok_start, tok_end) = token_offsets[i];

        if let Some(label) = tag.strip_prefix("B-") {
            close(&mut current, &mut params);
            current = Some((label.to_string(), tok_start, tok_end));
        } else if let Some(label) = tag.strip_prefix("I-") {
            match &mut current {
                Some((current_label, _, end)) if current_label == label => *end = tok_end,
                _ => {
                    // I- with no matching B- (a model mistake) — start a new span anyway.
                    close(&mut current, &mut params);
                    current = Some((label.to_string(), tok_start, tok_end));
                }
            }
        } else {
            close(&mut current, &mut params);
        }
    }
    close(&mut current, &mut params);

    params
}
