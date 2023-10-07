#![allow(dead_code)]

use calcurs_macros::Procagate;
use derive_more::Display;
use num::{BigInt, BigRational};

#[derive(Debug, Clone, Hash, Eq, PartialEq, Display)]
pub struct Number {
    kind: NumberKind,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Procagate)]
pub enum NumberKind {
    Integer(Integer),
    Rational(Rational),

    Infinity(Infinity),
    NaN(NaN),
}

impl NumberKind {
    pub fn add(self, other: &Self) -> Self {
        procagate_number_kind!(self, v => {v.add(other)})
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy)]
pub struct NaN;

impl NaN {
    pub fn add(self, _: &NumberKind) -> NumberKind {
        NumberKind::NaN(self)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy, Default)]
pub enum Direction {
    #[display(fmt = "+")]
    Positive,
    #[display(fmt = "-")]
    Negitive,
    #[display(fmt = "")]
    #[default]
    UnSigned,
}

impl Direction {
    pub fn from_num<I: num::Signed>(int: &I) -> Self {
        if int.is_negative() {
            Direction::Negitive
        } else if int.is_positive() {
            Direction::Positive
        } else {
            Direction::UnSigned
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Display, Default)]
#[display(fmt = "{dir}oo")]
pub struct Infinity {
    dir: Direction,
}

impl Infinity {
    pub fn new(dir: Direction) -> Self {
        Self { dir }
    }

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
            N::Infinity(inf) if self.dir != inf.dir => N::NaN(NaN),
            N::NaN(_) => N::NaN(NaN),
            _ => N::Infinity(self),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Default)]
pub struct Integer(BigInt);

impl From<Integer> for NumberKind {
    fn from(value: Integer) -> Self {
        NumberKind::Integer(value)
    }
}

impl Integer {
    pub fn add(self, other: &NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Infinity(_) | N::NaN(_) => other.clone(),
            N::Integer(i) => Integer(self.0 + &i.0).into(),
            N::Rational(r) => Rational(BigRational::from_integer(self.0) + &r.0).into(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Default)]
pub struct Rational(BigRational);

impl From<Rational> for NumberKind {
    fn from(value: Rational) -> Self {
        NumberKind::Rational(value)
    }
}

impl Rational {
    pub fn add(self, other: &NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Integer(i) => Rational(self.0 + BigRational::from_integer(i.0.clone())).into(),
            N::Rational(r) => Rational(self.0 + &r.0).into(),
            N::Infinity(_) | N::NaN(_) => other.clone(),
        }
    }
}
