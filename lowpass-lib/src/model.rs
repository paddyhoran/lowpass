use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

/// Dual-head model per FRAMEWORK.md section 2: a shared BERT encoder feeding
/// two independent linear heads (intent classification, BIO slot tagging).
pub struct IntentSlotModel {
    encoder: BertModel,
    intent_head: Linear,
    slot_head: Linear,
}

impl IntentSlotModel {
    /// `vb_encoder` should point at the pretrained checkpoint's weights.
    /// `vb_heads` points at fresh (randomly-initialized) weights for the two heads,
    /// since no pretrained checkpoint contains an intent/slot classifier.
    pub fn load(
        vb_encoder: VarBuilder,
        bert_config: &BertConfig,
        vb_heads: VarBuilder,
        num_intents: usize,
        num_slot_labels: usize,
    ) -> Result<Self> {
        let encoder = BertModel::load(vb_encoder, bert_config)?;
        let hidden_size = bert_config.hidden_size;
        let intent_head = linear(hidden_size, num_intents, vb_heads.pp("intent_head"))?;
        let slot_head = linear(hidden_size, num_slot_labels, vb_heads.pp("slot_head"))?;
        Ok(Self {
            encoder,
            intent_head,
            slot_head,
        })
    }

    /// Runs the shared encoder once, then both heads in parallel over its output.
    ///
    /// Returns `(intent_logits, slot_logits)`:
    /// - `intent_logits`: `[batch, num_intents]`, from the `[CLS]` token embedding.
    /// - `slot_logits`: `[batch, seq_len, num_slot_labels]`, per-token BIO tag logits.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let sequence_output = self
            .encoder
            .forward(input_ids, token_type_ids, attention_mask)?;

        // [CLS] is always the first token; slice it out for sentence-level classification.
        let cls_embedding = sequence_output.i((.., 0, ..))?;

        let intent_logits = self.intent_head.forward(&cls_embedding)?;
        let slot_logits = self.slot_head.forward(&sequence_output)?;

        Ok((intent_logits, slot_logits))
    }
}
