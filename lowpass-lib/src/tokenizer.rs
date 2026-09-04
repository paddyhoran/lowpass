//! Implements our tokenizer implementation.
//!
//! BERT-Tiny has no `tokenizer.json` file that is can be loaded from so 
//! we need to manually add the following:
//!  - normalizer
//!  - pre-tokenizer
//!  - template format
use std::path::Path;

use anyhow::Result;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::Tokenizer;

/// `google/bert_uncased_L-2_H-128_A-2` (BERT-Tiny) ships only `vocab.txt`, not a
/// fast-tokenizer `tokenizer.json`. This reconstructs the standard BERT WordPiece
/// tokenizer (lowercase + accent-stripping normalizer, whitespace/punctuation
/// pre-tokenizer, `[CLS] $A [SEP]` template) directly from that vocab file.
pub fn build_bert_tokenizer(vocab_path: &Path) -> Result<Tokenizer> {
    let wordpiece = WordPiece::from_file(
        vocab_path
            .to_str()
            .expect("vocab path must be valid UTF-8"),
    )
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
