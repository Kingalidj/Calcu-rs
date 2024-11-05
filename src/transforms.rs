use std::fmt;

use serde::{Deserialize, Serialize};

use crate::atom::Expr;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Serialize, Deserialize)]
pub struct Step {
    pub(crate) to: Expr,
    pub(crate) from: Option<Expr>,
    pub(crate) explanation: String,
}

impl fmt::Debug for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(from) = &self.from {
            write!(f, "{}", from)?;
        }

        write!(f, "{}: {}", self.explanation, self.to)
    }
}
