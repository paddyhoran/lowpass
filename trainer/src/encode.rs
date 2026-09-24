use anyhow::{Context, Result};
use tokenizers::Tokenizer;

use lowpass_lib::vocab::Vocab;

use crate::bio::spans_to_bio_tags;
use crate::training_data::TrainingExample;

/// One training example, tokenized and mapped to integer label ids — the
/// last step before batching into tensors.
#[derive(Debug, Clone)]
pub struct EncodedExample {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub intent_id: u32,
    /// One BIO tag id per token (same length as `input_ids`).
    pub slot_ids: Vec<u32>,
    /// 1.0 for tokens that should count toward the slot loss (i.e. not
    /// `[CLS]`/`[SEP]`/padding), 0.0 otherwise. Same length as `input_ids`.
    pub slot_mask: Vec<f32>,
}

pub fn encode_example(
    tokenizer: &Tokenizer,
    intents: &Vocab,
    slot_tags: &Vocab,
    example: &TrainingExample,
) -> Result<EncodedExample> {
    let encoding = tokenizer
        .encode(example.text.as_str(), true)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("tokenizing {:?}", example.text))?;

    let tags = spans_to_bio_tags(
        encoding.get_offsets(),
        encoding.get_special_tokens_mask(),
        &example.slots,
    );

    let intent_id = intents
        .id(&example.intent)
        .with_context(|| format!("intent {:?} not present in intent vocabulary", example.intent))?;

    let special_mask = encoding.get_special_tokens_mask();
    let mut slot_ids = Vec::with_capacity(tags.len());
    let mut slot_mask = Vec::with_capacity(tags.len());
    for (i, tag) in tags.iter().enumerate() {
        let id = slot_tags
            .id(tag)
            .with_context(|| format!("slot tag {tag:?} not present in slot-tag vocabulary"))?;
        slot_ids.push(id);
        slot_mask.push(if special_mask[i] == 1 { 0.0 } else { 1.0 });
    }

    Ok(EncodedExample {
        input_ids: encoding.get_ids().to_vec(),
        token_type_ids: encoding.get_type_ids().to_vec(),
        intent_id,
        slot_ids,
        slot_mask,
    })
}
