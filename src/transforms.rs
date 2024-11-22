use std::fmt;

use serde::{Deserialize, Serialize};

use crate::atom::{Atom, Expr, PTR};

#[derive(Clone, Serialize, Deserialize, Hash)]
pub struct TransformStep {
    pub(crate) explanation: PTR<str>,
    //pub(crate) prev: Option<PTR<Expr>>,
    pub(crate) current: PTR<Atom>,
    pub(crate) refs: Vec<Expr>,
}

/*
impl fmt::Debug for TransformStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for prev in &self.refs {
            fmt::Debug::fmt(&(*prev).clone().steps(), f)?;
        }

        if self.explanation.is_empty() {
            return Ok(());
        }

        write!(f, "{}: [", self.explanation)?;

        let mut refs = self.refs.iter();
        if let Some(r) = refs.next() {
            write!(f, " {r}")?;
        }
        for r in refs {
            write!(f, ", {r}")?;
        }

        write!(f, " ] -> {}\n", self.current)?;

        Ok(())
    }
}
*/

//#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Serialize, Deserialize)]
//pub struct Step {
//    pub(crate) to: Expr,
//    pub(crate) from: Option<Expr>,
//    pub(crate) explanation: String,
//}
//
