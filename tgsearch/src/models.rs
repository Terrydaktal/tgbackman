#[derive(Clone)]
pub(crate) struct MatchRow {
    pub(crate) _id: i64,
    pub(crate) chat_name: String,
    pub(crate) chat_id: String,
    pub(crate) message_id: i64,
    pub(crate) backup_path: Option<String>,
    pub(crate) timestamp_unix: Option<i64>,
    pub(crate) sender: Option<String>,
    pub(crate) text: Option<String>,
    pub(crate) media_path: Option<String>,
}

pub(crate) struct ContextRow {
    pub(crate) _message_id: i64,
    pub(crate) timestamp: Option<String>,
    pub(crate) sender: String,
    pub(crate) text: String,
    pub(crate) media_type: Option<String>,
    pub(crate) media_path: Option<String>,
    pub(crate) is_match: bool,
}

pub(crate) struct MergedGroup {
    pub(crate) chat_id: String,
    pub(crate) chat_name: String,
    pub(crate) backup_path: Option<String>,
    pub(crate) match_ids: Vec<i64>,
    pub(crate) min_id: i64,
    pub(crate) max_id: i64,
}
