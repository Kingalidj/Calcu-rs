use std::ops;

use calcurs_macros::Procagate;
use derive_more::Display;
use num::{rational::Ratio, Zero};
use num_traits::One;

use crate::{
    base::Base,
    traits::{CalcursType, Num},
};

// #[derive(Debug, Clone, Hash, Eq, PartialEq, Display)]
// pub struct Number {
//     pub kind: NumberKind,
// }

// impl Num for Number {
//     fn is_zero(&self) -> bool {
//         self.kind.is_zero()
//     }

//     fn is_one(&self) -> bool {
//         self.kind.is_one()
//     }

//     fn sign(&self) -> Sign {
//         self.kind.sign()
//     }
// }

// impl CalcursType for Number {
//     #[inline]
//     fn base(self) -> Base {
//         BaseKind::Number(self).base()
//     }
// }

// todo maybe try into + precedence

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Procagate)]
pub enum Number {
    Integer(Integer),
    Rational(Rational),
    Infinity(Infinity),
    NaN(NaN),
}

impl Num for Number {
    fn is_zero(&self) -> bool {
        procagate_number!(self, v => { v.is_zero() })
    }

    fn is_one(&self) -> bool {
        procagate_number!(self, v => { v.is_one() })
    }

    fn sign(&self) -> Sign {
        procagate_number!(self, v => { v.sign() })
    }
}

// impl From<NumberKind> for Number {
//     fn from(value: Number) -> Self {
//         Number { kind: value }
//     }
// }

impl Number {
    //TODO take &mut
    pub fn add_kind<T: Into<Number>>(self, other: T) -> Number {
        let other = other.into();
        use Number as N;
        match (self, other) {
            (N::NaN(_), _) | (_, N::NaN(_)) => N::NaN(NaN),

            (N::Infinity(i), n) | (n, N::Infinity(i)) => i.add_inf(n),

            (N::Integer(i1), N::Integer(i2)) => i1.add_int(i2),

            (N::Integer(i), N::Rational(r)) | (N::Rational(r), N::Integer(i)) => r.add_int(i),
            (N::Rational(r1), N::Rational(r2)) => r1.add_rat(r2),
        }
    }

    pub fn sub_kind(self, mut other: Number) -> Number {
        other = other.mul_kind(Integer::num(-1));
        self.add_kind(other)
    }

    pub fn mul_kind(self, other: Number) -> Number {
        use Number as N;
        match (self, other) {
            (N::NaN(_), _) | (_, N::NaN(_)) => N::NaN(NaN),
            (N::Infinity(i), n) | (n, N::Infinity(i)) => i.mul_inf(n),
            (N::Integer(i1), N::Integer(i2)) => i1.mul_int(i2),
            (N::Integer(i), N::Rational(r)) | (N::Rational(r), N::Integer(i)) => r.mul_int(i),
            (N::Rational(r1), N::Rational(r2)) => r1.mul_rat(r2),
        }
    }

    pub fn div_kind(self, other: Number) -> Number {
        use Number as N;
        match (self, other) {
            (N::NaN(_), _) | (_, N::NaN(_)) => N::NaN(NaN),

            (N::Infinity(inf), n) => inf.div_inf(n),
            (n, N::Infinity(_)) => n.self_div_inf(),

            (N::Integer(i1), N::Integer(i2)) => Rational::div_rat(i1.into(), i2.into()),
            (N::Integer(i), N::Rational(r)) => Rational::div_rat(i.into(), r),
            (N::Rational(r), N::Integer(i)) => Rational::div_rat(r, i.into()),
            (N::Rational(r1), N::Rational(r2)) => Rational::div_rat(r1, r2),
        }
    }

    pub fn is_zero(&self) -> bool {
        procagate_number!(self, v => { v.is_zero() })
    }

    fn self_div_inf(&self) -> Number {
        match self {
            Number::Rational(_) | Number::Integer(_) => Integer::num(0),
            Number::Infinity(_) => NaN.into(),
            Number::NaN(_) => NaN.into(),
        }
    }

    #[inline]
    pub fn base(self) -> Base {
        Base::Number(self.into()).base()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy)]
pub struct NaN;

impl Num for NaN {
    fn is_zero(&self) -> bool {
        false
    }

    fn is_one(&self) -> bool {
        false
    }

    fn sign(&self) -> Sign {
        Sign::UnSigned
    }
}

impl CalcursType for NaN {
    #[inline]
    fn base(self) -> Base {
        Number::from(Number::NaN(self)).base()
    }
}

impl NaN {
    pub fn is_zero(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy, Default)]
pub enum Sign {
    #[display(fmt = "+")]
    Positive,
    #[display(fmt = "-")]
    Negitive,
    #[display(fmt = "")]
    #[default]
    UnSigned,
}

impl Sign {
    pub fn from_sign<I: num::Signed>(int: &I) -> Self {
        if int.is_negative() {
            Sign::Negitive
        } else if int.is_positive() {
            Sign::Positive
        } else {
            Sign::UnSigned
        }
    }

    pub fn neg(&self) -> Self {
        use Sign as D;
        match self {
            D::Positive => D::Negitive,
            D::Negitive => D::Positive,
            D::UnSigned => D::UnSigned,
        }
    }

    pub fn is_pos(&self) -> bool {
        matches!(self, Sign::Positive)
    }

    pub fn is_neg(&self) -> bool {
        matches!(self, Sign::Negitive)
    }

    pub fn is_unsign(&self) -> bool {
        matches!(self, Sign::UnSigned)
    }
}

impl ops::Mul for Sign {
    type Output = Sign;

    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.is_neg() {
            self.neg()
        } else {
            self
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Display, Default)]
#[display(fmt = "{dir}oo")]
pub struct Infinity {
    dir: Sign,
}

impl Num for Infinity {
    fn is_zero(&self) -> bool {
        false
    }

    fn is_one(&self) -> bool {
        false
    }

    fn sign(&self) -> Sign {
        self.dir
    }
}

impl CalcursType for Infinity {
    #[inline]
    fn base(self) -> Base {
        Number::from(Number::Infinity(self)).base()
    }
}

impl Infinity {
    pub fn new(dir: Sign) -> Self {
        Self { dir }
    }

    pub fn is_zero(&self) -> bool {
        false
    }

    pub fn pos() -> Self {
        Self {
            dir: Sign::Positive,
        }
    }

    pub fn neg() -> Self {
        Self {
            dir: Sign::Negitive,
        }
    }

    pub fn add_inf(self, other: Number) -> Number {
        use Number as N;
        match other {
            N::Infinity(_) if self.dir == Sign::UnSigned => N::NaN(NaN),
            N::Infinity(inf) if self.dir != inf.dir => N::NaN(NaN),
            N::NaN(_) => N::NaN(NaN),
            _ => N::Infinity(self),
        }
    }

    pub fn mul_inf(self, other: Number) -> Number {
        use Number as N;
        match other {
            N::Rational(r) => Infinity::new(self.dir * r.sign()).into(),
            N::Integer(i) => Infinity::new(self.dir * i.sign()).into(),
            N::Infinity(i) => Infinity::new(self.dir * i.dir).into(),
            N::NaN(_) => N::NaN(NaN),
        }
    }

    pub fn div_inf(self, other: Number) -> Number {
        use Number as N;
        match other {
            N::Infinity(_) | N::Integer(_) | N::Rational(_) => self.mul_inf(other),
            N::NaN(_) => NaN.into(),
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Display, Default)]
pub struct Integer(i64);

impl Num for Integer {
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }

    fn sign(&self) -> Sign {
        match self.0.signum() {
            1 => Sign::Positive,
            -1 => Sign::Negitive,
            _ => Sign::UnSigned,
        }
    }
}

impl CalcursType for Integer {
    #[inline]
    fn base(self) -> Base {
        Number::from(Number::Integer(self)).base()
    }
}

impl From<Integer> for Rational {
    fn from(value: Integer) -> Self {
        Rational(Ratio::from_integer(value.0))
    }
}

impl Integer {
    pub fn new(val: i64) -> Self {
        Self(val)
    }

    pub fn num(val: i64) -> Number {
        Self(val).into()
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn add_int(self, other: Integer) -> Number {
        Integer(self.0 + other.0).into()
    }

    pub fn sub_int(self, other: Integer) -> Number {
        Integer(self.0 - other.0).into()
    }

    pub fn mul_int(self, other: Integer) -> Number {
        Integer(self.0 * other.0).into()
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Display, Default)]
pub struct Rational(Ratio<i64>);

impl Num for Rational {
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }

    fn sign(&self) -> Sign {
        match self.0.numer().signum() {
            1 => Sign::Positive,
            -1 => Sign::Negitive,
            _ => Sign::UnSigned,
        }
    }
}

impl CalcursType for Rational {
    #[inline]
    fn base(self) -> Base {
        Number::from(Number::Rational(self)).base()
    }
}

impl Rational {
    pub fn new(mut val: i64, mut denom: i64) -> Self {
        if denom.is_negative() {
            val *= -1;
            denom *= -1;
        }
        Self(Ratio::new(val, denom))
    }

    pub fn num(val: i64, denom: i64) -> Number {
        Self::new(val, denom).cleanup()
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn cleanup(self) -> Number {
        if self.0.is_integer() {
            Integer(self.0.to_integer()).into()
        } else {
            self.into()
        }
    }

    pub fn add_int(self, i: Integer) -> Number {
        self.add_rat(i.into())
    }

    pub fn sub_int(self, i: Integer) -> Number {
        self.sub_rat(i.into())
    }

    pub fn mul_int(self, i: Integer) -> Number {
        self.mul_rat(i.into())
    }

    pub fn div_int(self, i: Integer) -> Number {
        self.div_rat(i.into())
    }

    pub fn add_rat(self, r: Rational) -> Number {
        Rational(self.0 + r.0).cleanup()
    }

    pub fn sub_rat(self, r: Rational) -> Number {
        Rational(self.0 - r.0).cleanup()
    }

    pub fn mul_rat(self, r: Rational) -> Number {
        Rational(self.0 * r.0).cleanup()
    }

    pub fn div_rat(self, r: Rational) -> Number {
        if r.0.is_zero() {
            // todo: complex inf
            Infinity::new(self.sign()).into()
        } else {
            Rational(self.0 / r.0).cleanup()
        }
    }

    pub fn pow_int(self, i: Integer) -> Number {
        //TODO: is max correct?
        let exp = i.0.try_into().unwrap_or(i32::MAX);
        if exp == 0 {
            return Integer::num(1);
        }

        self.0.pow(exp);
        self.into()
    }
}

#[cfg(test)]
mod num_test {
    use crate::prelude::*;

    #[test]
    fn rational() {
        assert!(Rational::new(1, 2).is_pos());
        assert!(Rational::new(1, -2).is_neg());
        assert!(Rational::new(-1, -2).is_pos());
        assert!(Rational::new(-1, 2).is_neg());
    }
}
