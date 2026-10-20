//! Builds the BERT WordPiece tokenizer by hand.
//!
//! `google/bert_uncased_L-2_H-128_A-2` (BERT-Tiny) ships only `vocab.txt`,
//! not a fast-tokenizer `tokenizer.json`, so there is nothing to load a
//! ready-made tokenizer from. We reconstruct the standard BERT pipeline from
//! that vocab file instead:
//!  - normalizer: lowercasing and accent stripping
//!  - pre-tokenizer: whitespace and punctuation splitting
//!  - post-processor: the `[CLS] $A [SEP]` template
use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::Tokenizer;

/// Builds the BERT `Tokenizer` from a path to the checkpoint's `vocab.txt`.
pub fn build_bert_tokenizer(vocab_path: &Path) -> Result<Tokenizer> {
    let vocab_txt =
        std::fs::read(vocab_path).with_context(|| format!("reading {}", vocab_path.display()))?;
    build_bert_tokenizer_from_bytes(&vocab_txt)
}

/// Same as [`build_bert_tokenizer`], but reads `vocab.txt`'s contents from
/// memory instead of a file path.
///
/// This is for callers that embed the vocab via `include_bytes!` and have no
/// filesystem access at runtime.
pub fn build_bert_tokenizer_from_bytes(vocab_txt: &[u8]) -> Result<Tokenizer> {
    let vocab = WordPiece::read_bytes(vocab_txt).map_err(anyhow::Error::msg)?;
    let wordpiece = WordPiece::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .map_err(anyhow::Error::msg)?;

    let mut tokenizer = Tokenizer::new(wordpiece);
    tokenizer
        .with_normalizer(Some(BertNormalizer::default()))
        .map_err(anyhow::Error::msg)?;
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));

    let cls_id = tokenizer
        .token_to_id("[CLS]")
        .expect("vocab.txt must contain [CLS]");
    let sep_id = tokenizer
        .token_to_id("[SEP]")
        .expect("vocab.txt must contain [SEP]");

    let post_processor = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .unwrap()
        .special_tokens(vec![("[CLS]", cls_id), ("[SEP]", sep_id)])
        .build()?;
    tokenizer.with_post_processor(Some(post_processor));

    Ok(tokenizer)
}
