//! Trains the intent + slot heads on generated data.
//!
//! Head-only fine-tuning: the pretrained BERT-Tiny encoder stays frozen
//! (loaded as plain, non-trainable tensors); only `intent_head` and
//! `slot_head` are updated.
//!
//! Usage:
//!   cargo run --bin train -- [data.jsonl] [base_model_dir] [artifacts_dir]
//!
//! Defaults: models/training_data.jsonl, models/bert-tiny, models/artifacts.
//! In practice `just train <name>` passes all three explicitly, pointing at
//! models/<name>/data/training_data.jsonl and models/<name>/artifacts.
//! Generate the training data first with `just generate <name>`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_transformers::models::bert::Config as BertConfig;
use rand::seq::SliceRandom;

use lowpass_lib::model::IntentSlotModel;
use lowpass_lib::tokenizer::build_bert_tokenizer;
use trainer::encode::{encode_example, EncodedExample};
use trainer::training_data::read_jsonl;
use trainer::vocab::{intent_vocab, slot_tag_vocab};

const BATCH_SIZE: usize = 16;
const EPOCHS: usize = 40;
const LEARNING_RATE: f64 = 1e-3;
const VAL_FRACTION: f64 = 0.15;

/// Pads a batch of encoded examples to the batch's longest sequence and
/// stacks them into tensors: `(input_ids, token_type_ids, intent_ids, slot_ids, slot_mask)`.
fn pad_batch(
    batch: &[&EncodedExample],
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let max_len = batch.iter().map(|e| e.input_ids.len()).max().unwrap_or(0);

    let mut input_ids = Vec::with_capacity(batch.len());
    let mut token_type_ids = Vec::with_capacity(batch.len());
    let mut slot_ids = Vec::with_capacity(batch.len());
    let mut slot_mask = Vec::with_capacity(batch.len());
    let mut intent_ids = Vec::with_capacity(batch.len());

    for example in batch {
        let mut ids = example.input_ids.clone();
        ids.resize(max_len, 0);

        let mut type_ids = example.token_type_ids.clone();
        type_ids.resize(max_len, 0);

        let mut s_ids = example.slot_ids.clone();
        s_ids.resize(max_len, 0);

        let mut s_mask = example.slot_mask.clone();
        s_mask.resize(max_len, 0.0);

        input_ids.push(ids);
        token_type_ids.push(type_ids);
        slot_ids.push(s_ids);
        slot_mask.push(s_mask);
        intent_ids.push(example.intent_id);
    }

    Ok((
        Tensor::new(input_ids, device)?,
        Tensor::new(token_type_ids, device)?,
        Tensor::new(intent_ids, device)?,
        Tensor::new(slot_ids, device)?,
        Tensor::new(slot_mask, device)?,
    ))
}

/// Loss function for the intent head.
fn intent_loss(intent_logits: &Tensor, intent_ids: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::loss::cross_entropy(intent_logits, intent_ids)?)
}

/// Per-token cross-entropy over slot tags, ignoring positions where
/// `slot_mask` is 0 (`[CLS]`/`[SEP]`/padding) — candle's built-in
/// `cross_entropy` has no notion of ignored positions, so this is a small
/// hand-rolled masked version of the same log-softmax + gather computation.
fn masked_slot_loss(slot_logits: &Tensor, slot_ids: &Tensor, slot_mask: &Tensor) -> Result<Tensor> {
    let (batch, seq_len, num_labels) = slot_logits.dims3()?;
    let logits_flat = slot_logits.reshape((batch * seq_len, num_labels))?;
    let targets_flat = slot_ids.reshape((batch * seq_len,))?;
    let mask_flat = slot_mask.reshape((batch * seq_len,))?;

    let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?;
    let picked = log_probs
        .gather(&targets_flat.unsqueeze(1)?, 1)?
        .squeeze(1)?;
    let masked = (picked * &mask_flat)?;
    let loss_sum = masked.sum_all()?.neg()?;
    let count = mask_flat.sum_all()?;
    Ok(loss_sum.broadcast_div(&count)?)
}

fn evaluate(model: &IntentSlotModel, val_set: &[EncodedExample], device: &Device) -> Result<()> {
    let mut correct_intents = 0usize;
    let mut total_slot_tokens = 0usize;
    let mut correct_slot_tokens = 0usize;

    for example in val_set {
        let (input_ids, token_type_ids, intent_ids, slot_ids, slot_mask) =
            pad_batch(&[example], device)?;
        let attention_mask = input_ids.ne(0u32)?;
        let (intent_logits, slot_logits) =
            model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let predicted_intent = intent_logits.i(0)?.argmax(0)?.to_scalar::<u32>()?;
        let true_intent = intent_ids.i(0)?.to_scalar::<u32>()?;
        if predicted_intent == true_intent {
            correct_intents += 1;
        }

        let predicted_tags = slot_logits.i(0)?.argmax(1)?.to_vec1::<u32>()?;
        let true_tags = slot_ids.i(0)?.to_vec1::<u32>()?;
        let mask = slot_mask.i(0)?.to_vec1::<f32>()?;

        for ((predicted, truth), m) in predicted_tags.iter().zip(true_tags.iter()).zip(mask.iter())
        {
            if *m > 0.0 {
                total_slot_tokens += 1;
                if predicted == truth {
                    correct_slot_tokens += 1;
                }
            }
        }
    }

    let intent_accuracy = 100.0 * correct_intents as f64 / val_set.len() as f64;
    println!(
        "intent accuracy: {intent_accuracy:.1}% ({correct_intents}/{})",
        val_set.len()
    );

    if total_slot_tokens > 0 {
        let slot_accuracy = 100.0 * correct_slot_tokens as f64 / total_slot_tokens as f64;
        println!(
            "slot token accuracy: {slot_accuracy:.1}% ({correct_slot_tokens}/{total_slot_tokens})"
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let data_path = args
        .next()
        .unwrap_or_else(|| "models/training_data.jsonl".to_string());
    let model_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "models/bert-tiny".to_string()),
    );
    let checkpoint_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "models/artifacts".to_string()),
    );

    let device = Device::Cpu;

    println!("Loading examples from {data_path}...");
    let examples = read_jsonl(Path::new(&data_path)).with_context(|| {
        format!("reading {data_path} -- did you run `just generate <name>` first?")
    })?;
    anyhow::ensure!(
        !examples.is_empty(),
        "no training examples found in {data_path}"
    );

    let intents = intent_vocab(&examples);
    let slot_tags = slot_tag_vocab(&examples);
    println!("intents ({}): {:?}", intents.len(), intents.labels());
    println!("slot tags ({}): {:?}", slot_tags.len(), slot_tags.labels());

    println!(
        "Building tokenizer and encoding {} examples...",
        examples.len()
    );
    let tokenizer = build_bert_tokenizer(&model_dir.join("vocab.txt"))?;
    let mut encoded: Vec<EncodedExample> = examples
        .iter()
        .map(|example| encode_example(&tokenizer, &intents, &slot_tags, example))
        .collect::<Result<_>>()?;

    let mut rng = rand::rng();
    encoded.shuffle(&mut rng);

    let val_size = ((encoded.len() as f64) * VAL_FRACTION).round() as usize;
    let val_size = val_size.clamp(1, encoded.len() - 1);
    let (val_set, train_set) = encoded.split_at(val_size);
    println!(
        "train: {} examples, validation: {} examples",
        train_set.len(),
        val_set.len()
    );

    let config_json = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    let bert_config: BertConfig = serde_json::from_str(&config_json)?;

    let vb_encoder = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_dir.join("model.safetensors")],
            DType::F32,
            &device,
        )?
    };
    let varmap = VarMap::new();
    let vb_heads = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = IntentSlotModel::load(
        vb_encoder,
        &bert_config,
        vb_heads,
        intents.len(),
        slot_tags.len(),
    )?;

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;

    println!("Training for {EPOCHS} epochs (batch size {BATCH_SIZE})...");
    let mut train_indices: Vec<usize> = (0..train_set.len()).collect();
    for epoch in 0..EPOCHS {
        train_indices.shuffle(&mut rng);
        let mut total_loss = 0f64;
        let mut num_batches = 0usize;

        for batch_indices in train_indices.chunks(BATCH_SIZE) {
            let batch: Vec<&EncodedExample> =
                batch_indices.iter().map(|&i| &train_set[i]).collect();
            let (input_ids, token_type_ids, intent_ids, slot_ids, slot_mask) =
                pad_batch(&batch, &device)?;
            let attention_mask = input_ids.ne(0u32)?;

            let (intent_logits, slot_logits) =
                model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

            let loss = (intent_loss(&intent_logits, &intent_ids)?
                + masked_slot_loss(&slot_logits, &slot_ids, &slot_mask)?)?;

            optimizer.backward_step(&loss)?;

            total_loss += loss.to_scalar::<f32>()? as f64;
            num_batches += 1;
        }

        println!(
            "epoch {:>3}: avg loss = {:.4}",
            epoch + 1,
            total_loss / num_batches as f64
        );
    }

    println!("\nEvaluating on {} held-out examples...", val_set.len());
    evaluate(&model, val_set, &device)?;

    std::fs::create_dir_all(&checkpoint_dir)?;
    varmap.save(checkpoint_dir.join("heads.safetensors"))?;

    let vocab_json = serde_json::json!({
        "intents": intents.labels(),
        "slot_tags": slot_tags.labels(),
    });
    std::fs::write(
        checkpoint_dir.join("vocab.json"),
        serde_json::to_string_pretty(&vocab_json)?,
    )?;

    println!(
        "\nSaved trained heads + label vocab to {}",
        checkpoint_dir.display()
    );
    Ok(())
}
