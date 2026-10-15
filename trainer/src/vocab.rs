//! Builds label vocabularies from training data — the training-only
//! counterpart to `lowpass_lib::vocab::Vocab`.

use std::collections::BTreeSet;

use lowpass_lib::vocab::Vocab;

use crate::training_data::TrainingExample;

/// Takes a collection of training examples and builds a `Vocab` based on the
/// intents detected.
///
/// Items in a `Vocab` are unique, so duplicate intents are removed.
pub fn intent_vocab(examples: &[TrainingExample]) -> Vocab {
    let unique: BTreeSet<String> = examples.iter().map(|e| e.intent.clone()).collect();
    Vocab::new(unique.into_iter().collect())
}

/// `"O"` plus `"B-<LABEL>"`/`"I-<LABEL>"` for every distinct slot label seen
/// in the data.
pub fn slot_tag_vocab(examples: &[TrainingExample]) -> Vocab {
    let unique_labels: BTreeSet<String> = examples
        .iter()
        .flat_map(|e| e.slots.iter().map(|slot| slot.label.clone()))
        .collect();

    let mut tags = vec!["O".to_string()];
    for label in unique_labels {
        tags.push(format!("B-{label}"));
        tags.push(format!("I-{label}"));
    }
    Vocab::new(tags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_data::SlotSpan;

    fn example(intent: &str, labels: &[&str]) -> TrainingExample {
        TrainingExample {
            text: "irrelevant".to_string(),
            intent: intent.to_string(),
            slots: labels
                .iter()
                .map(|label| SlotSpan {
                    label: label.to_string(),
                    start: 0,
                    end: 1,
                })
                .collect(),
        }
    }

    #[test]
    fn builds_sorted_unique_intent_vocab() {
        let examples = vec![
            example("set_alarm", &[]),
            example("calculate", &[]),
            example("set_alarm", &[]),
        ];
        let vocab = intent_vocab(&examples);
        assert_eq!(
            vocab.labels(),
            &["calculate".to_string(), "set_alarm".to_string()]
        );
    }

    #[test]
    fn builds_bio_slot_tag_vocab_with_o_first() {
        let examples = vec![
            example("set_alarm", &["TIME"]),
            example("check_weather", &["LOCATION"]),
        ];
        let vocab = slot_tag_vocab(&examples);
        assert_eq!(
            vocab.labels(),
            &[
                "O".to_string(),
                "B-LOCATION".to_string(),
                "I-LOCATION".to_string(),
                "B-TIME".to_string(),
                "I-TIME".to_string(),
            ]
        );
    }
}
