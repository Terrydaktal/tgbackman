//! Small local fuzzy ranking adapter used by the search CLI.
//!
//! Keeping this implementation in the repository avoids a dependency on a
//! developer-owned remote repository and makes clean checkouts self-contained.

use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MessageField<'a> {
    pub priority: u8,
    pub value: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MessageCandidate<'a, 'b> {
    pub key: &'a str,
    pub fields: &'b [MessageField<'a>],
    pub score: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct MessageMatch<'a> {
    pub key: &'a str,
    score: f64,
    phrase: bool,
    matched_tokens: usize,
    field_quality: u8,
    span: usize,
    occurrences: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct MessageQuery {
    tokens: Vec<String>,
}

impl MessageQuery {
    pub fn new(query: &str) -> Option<Self> {
        let tokens = query
            .split_whitespace()
            .map(str::to_lowercase)
            .filter(|token| !token.is_empty())
            .fold(Vec::new(), |mut values, token| {
                if !values.contains(&token) {
                    values.push(token);
                }
                values
            });
        (!tokens.is_empty()).then_some(Self { tokens })
    }

    pub fn search_rank<'a, 'b>(
        &self,
        candidate: MessageCandidate<'a, 'b>,
    ) -> Option<MessageMatch<'a>> {
        let mut matched_tokens = 0;
        let mut occurrences = 0;
        let mut field_quality = u8::MAX;
        let mut first = usize::MAX;
        let mut last = 0;
        let mut offset = 0;
        let mut combined = String::new();
        for field in candidate.fields {
            let value = field.value.to_lowercase();
            if !combined.is_empty() {
                combined.push(' ');
            }
            combined.push_str(&value);
            let words = value.split_whitespace().collect::<Vec<_>>();
            for (index, word) in words.iter().enumerate() {
                for token in &self.tokens {
                    if word.contains(token) {
                        occurrences += 1;
                        if first == usize::MAX {
                            first = offset + index;
                        }
                        last = offset + index;
                    }
                }
            }
            let field_matches = self
                .tokens
                .iter()
                .filter(|token| words.iter().any(|word| word.contains(token.as_str())))
                .count();
            if field_matches > 0 {
                matched_tokens += field_matches;
                field_quality = field_quality.min(field.priority);
            }
            offset += words.len();
        }
        if matched_tokens == 0 {
            return None;
        }
        Some(MessageMatch {
            key: candidate.key,
            score: candidate.score,
            phrase: combined.contains(&self.tokens.join(" ")),
            matched_tokens,
            field_quality,
            span: last.saturating_sub(first),
            occurrences,
        })
    }
}

pub fn sort_matches(matches: &mut [MessageMatch<'_>]) {
    matches.sort_by(|left, right| {
        right
            .phrase
            .cmp(&left.phrase)
            .then_with(|| right.matched_tokens.cmp(&left.matched_tokens))
            .then_with(|| left.field_quality.cmp(&right.field_quality))
            .then_with(|| left.span.cmp(&right.span))
            .then_with(|| right.occurrences.cmp(&left.occurrences))
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.key.cmp(right.key))
            .then(Ordering::Equal)
    });
}
