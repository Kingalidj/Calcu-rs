#![allow(dead_code)]

use derive_more::Display;

use num::bigint::BigInt;
use num::rational::BigRational;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Display)]
pub struct Number {
    kind: NumberKind,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub enum NumberKind {
    Integer(BigInt),
    Rational(BigRational),

    Infinity(Infinity),
    NaN(NaN),
}

impl NumberKind {
    pub fn add(self, other: &Self) -> Self {
        use NumberKind as N;
        match self {
            N::Infinity(i) => i.add(other),
            N::NaN(n) => n.add(other),
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy)]
pub struct NaN;

impl NaN {
    pub fn add(self, _: &NumberKind) -> NumberKind {
        NumberKind::NaN(self)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy)]
pub enum Direction {
    #[display(fmt = "+")]
    Positive,
    #[display(fmt = "-")]
    Negitive,
    #[display(fmt = "")]
    UnSigned,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Display)]
#[display(fmt = "{dir}oo")]
pub struct Infinity {
    dir: Direction,
}

impl Default for Infinity {
    fn default() -> Self {
        Self {
            dir: Direction::UnSigned,
        }
    }
}

impl Infinity {
    pub fn pos() -> Self {
        Self {
            dir: Direction::Positive,
        }
    }

    pub fn neg() -> Self {
        Self {
            dir: Direction::Negitive,
        }
    }

    pub fn add(self, other: &NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Infinity(_) if self.dir == Direction::UnSigned => N::NaN(NaN),
            N::Infinity(Infinity { dir }) if self.dir != *dir => N::NaN(NaN),
            N::NaN(_) => N::NaN(NaN),
            _ => return N::Infinity(self),
        }
    }
}
