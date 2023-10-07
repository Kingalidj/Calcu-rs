use std::ops;

use derive_more::Display;
use num::{BigInt, BigRational, Zero};

#[derive(Debug, Clone, Hash, Eq, PartialEq, Display)]
pub struct Number {
    kind: NumberKind,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub enum NumberKind {
    Integer(Integer),
    Rational(Rational),

    Infinity(Infinity),
    NaN(NaN),
}

impl NumberKind {
    pub fn add(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match (self, other) {
            (N::NaN(_), _) | (_, N::NaN(_)) => N::NaN(NaN),
            (N::Infinity(i), n) | (n, N::Infinity(i)) => i.add_inf(n),
            (N::Integer(i1), N::Integer(i2)) => i1.add_int(i2),
            (N::Integer(i), N::Rational(r)) | (N::Rational(r), N::Integer(i)) => r.add_int(i),
            (N::Rational(r1), N::Rational(r2)) => r1.add_rat(r2),
        }
    }

    pub fn sub(self, mut other: NumberKind) -> NumberKind {
        other = other.mul(integer(-1));
        self.add(other)
    }

    pub fn mul(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match (self, other) {
            (N::NaN(_), _) | (_, N::NaN(_)) => N::NaN(NaN),
            (N::Infinity(i), n) | (n, N::Infinity(i)) => i.mul_inf(n),
            (N::Integer(i1), N::Integer(i2)) => i1.mul_int(i2),
            (N::Integer(i), N::Rational(r)) | (N::Rational(r), N::Integer(i)) => r.mul_int(i),
            (N::Rational(r1), N::Rational(r2)) => r1.mul_rat(r2),
        }
    }

    pub fn div(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match (self, other) {
            (N::NaN(_), _) | (_, N::NaN(_)) => N::NaN(NaN),

            (N::Infinity(_), _) => todo!(),
            (n, N::Infinity(_)) => n.self_div_inf(),

            (N::Integer(i1), N::Integer(i2)) => Rational::div_rat(i1.into(), i2.into()),
            (N::Integer(i), N::Rational(r)) => Rational::div_rat(i.into(), r),
            (N::Rational(r), N::Integer(i)) => Rational::div_rat(r, i.into()),
            (N::Rational(r1), N::Rational(r2)) => Rational::div_rat(r1, r2),
        }
    }

    fn self_div_inf(&self) -> NumberKind {
        match self {
            NumberKind::Rational(_) | NumberKind::Integer(_) => integer(0),
            NumberKind::Infinity(_) => NaN.into(),
            NumberKind::NaN(_) => NaN.into(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy)]
pub struct NaN;

impl From<NaN> for NumberKind {
    fn from(value: NaN) -> Self {
        NumberKind::NaN(value)
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
    pub fn from_sign<I: num::Signed>(int: &I) -> Self {
        if int.is_negative() {
            Direction::Negitive
        } else if int.is_positive() {
            Direction::Positive
        } else {
            Direction::UnSigned
        }
    }

    pub fn neg(&self) -> Self {
        use Direction as D;
        match self {
            D::Positive => D::Negitive,
            D::Negitive => D::Positive,
            D::UnSigned => D::UnSigned,
        }
    }

    pub fn is_pos(&self) -> bool {
        matches!(self, Direction::Positive)
    }

    pub fn is_neg(&self) -> bool {
        matches!(self, Direction::Negitive)
    }

    pub fn is_unsign(&self) -> bool {
        matches!(self, Direction::UnSigned)
    }
}

impl ops::Mul for Direction {
    type Output = Direction;

    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.is_neg() {
            self.neg()
        } else {
            self
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Display, Default)]
#[display(fmt = "{dir}ꝏ ")]
pub struct Infinity {
    dir: Direction,
}

impl From<Infinity> for NumberKind {
    fn from(value: Infinity) -> Self {
        NumberKind::Infinity(value)
    }
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

    pub fn add_inf(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Infinity(_) if self.dir == Direction::UnSigned => N::NaN(NaN),
            N::Infinity(inf) if self.dir != inf.dir => N::NaN(NaN),
            N::NaN(_) => N::NaN(NaN),
            _ => N::Infinity(self),
        }
    }

    pub fn mul_inf(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Rational(r) => Infinity::new(self.dir * Direction::from_sign(&r.0)).into(),
            N::Integer(i) => Infinity::new(self.dir * Direction::from_sign(&i.0)).into(),
            N::Infinity(i) => Infinity::new(self.dir * i.dir).into(),
            N::NaN(_) => N::NaN(NaN),
        }
    }

    pub fn div_inf(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Integer(_) | N::Rational(_) => self.mul_inf(other),
            N::Infinity(_) | N::NaN(_) => NaN.into(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Default)]
pub struct Integer(BigInt);

pub fn integer(val: i32) -> NumberKind {
    Integer(BigInt::from(val)).into()
}

impl From<Integer> for Rational {
    fn from(value: Integer) -> Self {
        Rational(BigRational::from_integer(value.0))
    }
}

impl From<Integer> for NumberKind {
    fn from(value: Integer) -> Self {
        NumberKind::Integer(value)
    }
}

impl Integer {
    pub fn add_int(self, other: Integer) -> NumberKind {
        Integer(self.0 + other.0).into()
    }

    pub fn sub_int(self, other: Integer) -> NumberKind {
        Integer(self.0 - other.0).into()
    }

    pub fn mul_int(self, other: Integer) -> NumberKind {
        Integer(self.0 * other.0).into()
    }

    pub fn div_int(self, other: Integer) -> NumberKind {
        if other.0.is_zero() {
            Infinity::new(Direction::from_sign(&self.0)).into()
        } else {
            Integer(self.0 / other.0).into()
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
    pub fn simplify(self) -> NumberKind {
        if self.0.is_integer() {
            Integer(self.0.to_integer()).into()
        } else {
            self.into()
        }
    }

    pub fn add_int(self, i: Integer) -> NumberKind {
        self.add_rat(i.into())
    }

    pub fn sub_int(self, i: Integer) -> NumberKind {
        self.sub_rat(i.into())
    }

    pub fn mul_int(self, i: Integer) -> NumberKind {
        self.mul_rat(i.into())
    }

    pub fn div_int(self, i: Integer) -> NumberKind {
        self.div_rat(i.into())
    }

    pub fn add_rat(self, r: Rational) -> NumberKind {
        Rational(self.0 + r.0).simplify().into()
    }

    pub fn sub_rat(self, r: Rational) -> NumberKind {
        Rational(self.0 - r.0).simplify().into()
    }

    pub fn mul_rat(self, r: Rational) -> NumberKind {
        Rational(self.0 * r.0).simplify().into()
    }

    pub fn div_rat(self, r: Rational) -> NumberKind {
        if r.0.is_zero() {
            Infinity::new(Direction::from_sign(&self.0)).into()
        } else {
            Rational(self.0 / r.0).simplify().into()
        }
    }
}
