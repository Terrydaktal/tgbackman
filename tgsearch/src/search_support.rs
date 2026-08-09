//! Small, repository-local FTS query support used by the standalone searcher.
//!
//! This keeps clean checkouts self-contained.  The previous implementation
//! depended on an untracked sibling `db-search` checkout, which made CI and
//! releases depend on the developer's filesystem layout.

pub(crate) mod query {
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct QueryToken {
        pub(crate) text: String,
        pub(crate) quoted: bool,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct SearchQuery {
        raw: String,
        tokens: Vec<QueryToken>,
    }

    impl SearchQuery {
        pub(crate) fn new(query: impl Into<String>) -> Self {
            let raw = query.into();
            let tokens = parse_query_tokens(&raw);
            Self { raw, tokens }
        }

        pub(crate) fn raw(&self) -> &str {
            &self.raw
        }

        pub(crate) fn tokens(&self) -> &[QueryToken] {
            &self.tokens
        }
    }

    fn push_split_tokens(tokens: &mut Vec<QueryToken>, current: &mut String) {
        let text = std::mem::take(current);
        for part in text.split_whitespace() {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                tokens.push(QueryToken {
                    text: trimmed.to_owned(),
                    quoted: false,
                });
            }
        }
    }

    fn push_token(tokens: &mut Vec<QueryToken>, current: &mut String, quoted: bool) {
        let text = std::mem::take(current);
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            tokens.push(QueryToken {
                text: trimmed.to_owned(),
                quoted,
            });
        }
    }

    pub(crate) fn parse_query_tokens(query: &str) -> Vec<QueryToken> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        for ch in query.chars() {
            if ch == '"' {
                if in_quotes {
                    push_token(&mut tokens, &mut current, true);
                    in_quotes = false;
                } else {
                    push_split_tokens(&mut tokens, &mut current);
                    in_quotes = true;
                }
            } else {
                current.push(ch);
            }
        }
        if in_quotes {
            push_token(&mut tokens, &mut current, true);
        } else {
            push_split_tokens(&mut tokens, &mut current);
        }
        tokens
    }
}

pub(crate) mod sqlite_fts {
    use super::query::{QueryToken, SearchQuery};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) enum SearchMode {
        Exact,
        Prefix,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct PreparedFtsQuery {
        pub(crate) match_query: String,
        pub(crate) highlight_terms: Vec<String>,
    }

    pub(crate) fn prepare_fts_query(query: &SearchQuery, mode: SearchMode) -> PreparedFtsQuery {
        let match_query = match mode {
            SearchMode::Exact => query.raw().to_owned(),
            SearchMode::Prefix => build_prefix_match_query(query.tokens()),
        };
        let highlight_terms = query.tokens().iter().filter_map(highlight_term).collect();
        PreparedFtsQuery {
            match_query,
            highlight_terms,
        }
    }

    pub(crate) fn rowid_match_subquery(fts_table: &str, parameter_index: usize) -> String {
        format!("SELECT rowid FROM {fts_table} WHERE {fts_table} MATCH ?{parameter_index}")
    }

    fn highlight_term(token: &QueryToken) -> Option<String> {
        if token.quoted {
            return (!token.text.is_empty()).then(|| token.text.to_lowercase());
        }
        if matches!(
            token.text.to_ascii_lowercase().as_str(),
            "or" | "and" | "not"
        ) {
            return None;
        }
        let terms = fts_token_fragments(&token.text);
        (!terms.is_empty()).then(|| terms.join(" ").trim_end_matches('*').to_lowercase())
    }

    fn build_prefix_match_query(tokens: &[QueryToken]) -> String {
        let mut parts = Vec::new();
        for token in tokens {
            if token.quoted {
                parts.push(format!("\"{}\"", token.text));
                continue;
            }
            if token.text.eq_ignore_ascii_case("or") {
                parts.push("OR".to_owned());
                continue;
            }
            for cleaned in fts_token_fragments(&token.text) {
                parts.push(if cleaned.ends_with('*') {
                    cleaned
                } else {
                    format!("{cleaned}*")
                });
            }
        }
        parts.join(" ")
    }

    fn fts_token_fragments(text: &str) -> Vec<String> {
        text.split(|ch: char| !ch.is_alphanumeric() && ch != '*')
            .filter(|part| !part.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::query::{parse_query_tokens, SearchQuery};
    use super::sqlite_fts::{prepare_fts_query, SearchMode};

    #[test]
    fn query_parser_preserves_phrases_and_prefixes() {
        let query = SearchQuery::new(r#"hello "two words" OR test"#);
        assert_eq!(query.tokens().len(), 4);
        assert_eq!(
            prepare_fts_query(&query, SearchMode::Prefix).match_query,
            "hello* \"two words\" OR test*"
        );
        assert_eq!(parse_query_tokens(r#""unclosed phrase"#).len(), 1);
    }
}
