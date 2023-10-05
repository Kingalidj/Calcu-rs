use derive_more::Display;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub struct Number {}

pub enum NumberKind {
    Integer,
    Rational,
    F64,
}
