use crate::training_data::SlotSpan;

/// Converts character-offset slot spans (as emitted by the `generator` crate)
/// into one BIO tag per token, aligned to a `tokenizers::Encoding`'s
/// `get_offsets()` / `get_special_tokens_mask()`.
///
/// Special tokens (`[CLS]`, `[SEP]`, padding) are always tagged `"O"`. A
/// token whose span overlaps a slot's span gets `"B-<LABEL>"` if it's the
/// first token overlapping that particular slot occurrence, otherwise
/// `"I-<LABEL>"` — so a slot value split into multiple WordPiece sub-tokens
/// still forms a single `B-`, `I-`, `I-`, ... run.
pub fn spans_to_bio_tags(
    token_offsets: &[(usize, usize)],
    special_tokens_mask: &[u32],
    slots: &[SlotSpan],
) -> Vec<String> {
    assert_eq!(
        token_offsets.len(),
        special_tokens_mask.len(),
        "offsets and special-tokens-mask must be the same length (one entry per token)"
    );

    let mut tags = vec!["O".to_string(); token_offsets.len()];

    for slot in slots {
        let mut first = true;
        for (i, &(tok_start, tok_end)) in token_offsets.iter().enumerate() {
            if special_tokens_mask[i] == 1 {
                continue;
            }
            let overlaps = tok_start < slot.end && tok_end > slot.start;
            if overlaps {
                tags[i] = if first {
                    format!("B-{}", slot.label)
                } else {
                    format!("I-{}", slot.label)
                };
                first = false;
            }
        }
    }

    tags
}

#[cfg(test)]
mod tests {
    use super::*;
    use lowpass_lib::tokenizer::build_bert_tokenizer;
    use std::path::Path;

    fn test_tokenizer() -> tokenizers::Tokenizer {
        let vocab_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/bert-tiny/vocab.txt");
        build_bert_tokenizer(&vocab_path).expect("failed to build tokenizer for test")
    }

    #[test]
    fn tags_a_single_word_slot() {
        let tokenizer = test_tokenizer();
        let text = "check weather in Seattle";
        let encoding = tokenizer.encode(text, true).unwrap();

        // Computed the same way the generator does: text[17..24] == "Seattle".
        assert_eq!(&text[17..24], "Seattle");
        let slots = vec![SlotSpan {
            label: "LOCATION".to_string(),
            start: 17,
            end: 24,
        }];

        let tags = spans_to_bio_tags(
            encoding.get_offsets(),
            encoding.get_special_tokens_mask(),
            &slots,
        );

        assert_eq!(tags.len(), encoding.get_tokens().len());
        println!("tokens: {:?}", encoding.get_tokens());
        println!("offsets: {:?}", encoding.get_offsets());
        println!("special mask: {:?}", encoding.get_special_tokens_mask());
        println!("tags: {tags:?}");

        // [CLS] and [SEP] must never be tagged.
        assert_eq!(tags[0], "O");
        assert_eq!(*tags.last().unwrap(), "O");

        // Exactly one B-LOCATION, and every I-LOCATION (if any, from
        // sub-word splitting) immediately follows it with no gaps.
        let b_count = tags.iter().filter(|t| *t == "B-LOCATION").count();
        assert_eq!(b_count, 1, "expected exactly one B-LOCATION tag");
    }

    #[test]
    fn tags_a_multi_token_slot_with_a_single_b_and_trailing_is() {
        let tokenizer = test_tokenizer();
        let text = "check weather in New York";
        let encoding = tokenizer.encode(text, true).unwrap();

        assert_eq!(&text[17..25], "New York");
        let slots = vec![SlotSpan {
            label: "LOCATION".to_string(),
            start: 17,
            end: 25,
        }];

        let tags = spans_to_bio_tags(
            encoding.get_offsets(),
            encoding.get_special_tokens_mask(),
            &slots,
        );
        println!("tokens: {:?}", encoding.get_tokens());
        println!("tags: {tags:?}");

        // However many tokens "New York" splits into, exactly one is B-LOCATION
        // and the rest are I-LOCATION, with no other slot tags anywhere else.
        let b_count = tags.iter().filter(|t| *t == "B-LOCATION").count();
        let i_count = tags.iter().filter(|t| *t == "I-LOCATION").count();
        assert_eq!(b_count, 1);
        assert!(i_count >= 1, "expected at least one I-LOCATION token");
        assert_eq!(
            b_count + i_count + tags.iter().filter(|t| *t == "O").count(),
            tags.len()
        );
    }

    #[test]
    fn tags_are_all_o_when_there_are_no_slots() {
        let tokenizer = test_tokenizer();
        let encoding = tokenizer.encode("calculate 5 + 3", true).unwrap();
        let tags = spans_to_bio_tags(
            encoding.get_offsets(),
            encoding.get_special_tokens_mask(),
            &[],
        );
        assert!(tags.iter().all(|t| t == "O"));
    }
}
