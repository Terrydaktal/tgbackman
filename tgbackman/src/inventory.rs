use std::collections::HashMap;

pub(crate) struct UnionFind {
    pub(crate) parent: HashMap<String, String>,
}
impl UnionFind {
    pub(crate) fn new() -> Self {
        Self {
            parent: HashMap::new(),
        }
    }
    pub(crate) fn find(&mut self, x: &str) -> String {
        if !self.parent.contains_key(x) {
            self.parent.insert(x.to_string(), x.to_string());
            return x.to_string();
        }
        let parent = self.parent.get(x).cloned().unwrap_or_else(|| x.to_string());
        if parent == x {
            return x.to_string();
        }
        let root = self.find(&parent);
        self.parent.insert(x.to_string(), root.clone());
        root
    }
    pub(crate) fn union(&mut self, x: &str, y: &str) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x != root_y {
            self.parent.insert(root_x, root_y);
        }
    }
}
