use std::ops;

use calcurs_macros::Procagate;
use derive_more::Display;
use lazy_static::lazy_static;
use num::{bigint, BigInt, BigRational, Zero};
use num_traits::One;

use crate::{
    base::Base,
    base::BasicKind,
    binop::{Add, Mul},
    traits::{CalcursType, Numeric},
};

lazy_static! {
    pub static ref ZERO: Number = Integer::num(0).into();
    pub static ref ONE: Number = Integer::num(1).into();
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Display)]
pub struct Number {
    pub kind: NumberKind,
}

impl Numeric for Number {
    fn is_zero(&self) -> bool {
        self.kind.is_zero()
    }

    fn is_one(&self) -> bool {
        self.kind.is_one()
    }

    fn sign(&self) -> Sign {
        self.kind.sign()
    }
}

impl CalcursType for Number {
    fn base(self) -> Base {
        BasicKind::Number(self).into()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Procagate)]
pub enum NumberKind {
    Integer(Integer),
    Rational(Rational),

    Infinity(Infinity),
    NaN(NaN),
}

impl Numeric for NumberKind {
    fn is_zero(&self) -> bool {
        procagate_number_kind!(self, v => { v.is_zero() })
    }

    fn is_one(&self) -> bool {
        procagate_number_kind!(self, v => { v.is_zero() })
    }

    fn sign(&self) -> Sign {
        procagate_number_kind!(self, v => { v.sign() })
    }
}

impl From<NumberKind> for Number {
    fn from(value: NumberKind) -> Self {
        Number { kind: value }
    }
}

impl NumberKind {
    pub fn add_kind<T: Into<NumberKind>>(self, other: T) -> NumberKind {
        let other = other.into();
        use NumberKind as N;
        match (self, other) {
            (N::NaN(_), _) | (_, N::NaN(_)) => N::NaN(NaN),
            (N::Infinity(i), n) | (n, N::Infinity(i)) => i.add_inf(n),
            (N::Integer(i1), N::Integer(i2)) => i1.add_int(i2),
            (N::Integer(i), N::Rational(r)) | (N::Rational(r), N::Integer(i)) => r.add_int(i),
            (N::Rational(r1), N::Rational(r2)) => r1.add_rat(r2),
        }
    }

    pub fn sub_kind(self, mut other: NumberKind) -> NumberKind {
        other = other.mul_kind(Integer::num(-1));
        self.add_kind(other)
    }

    pub fn mul_kind(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match (self, other) {
            (N::NaN(_), _) | (_, N::NaN(_)) => N::NaN(NaN),
            (N::Infinity(i), n) | (n, N::Infinity(i)) => i.mul_inf(n),
            (N::Integer(i1), N::Integer(i2)) => i1.mul_int(i2),
            (N::Integer(i), N::Rational(r)) | (N::Rational(r), N::Integer(i)) => r.mul_int(i),
            (N::Rational(r1), N::Rational(r2)) => r1.mul_rat(r2),
        }
    }

    pub fn div_kind(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
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
        procagate_number_kind!(self, v => { v.is_zero() })
    }

    fn self_div_inf(&self) -> NumberKind {
        match self {
            NumberKind::Rational(_) | NumberKind::Integer(_) => Integer::num(0),
            NumberKind::Infinity(_) => NaN.into(),
            NumberKind::NaN(_) => NaN.into(),
        }
    }

    pub fn base(self) -> Base {
        BasicKind::Number(self.into()).into()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy)]
pub struct NaN;

impl Numeric for NaN {
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
    fn base(self) -> Base {
        Number::from(NumberKind::NaN(self)).base()
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

impl Numeric for Infinity {
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
    fn base(self) -> Base {
        Number::from(NumberKind::Infinity(self)).base()
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

    pub fn add_inf(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Infinity(_) if self.dir == Sign::UnSigned => N::NaN(NaN),
            N::Infinity(inf) if self.dir != inf.dir => N::NaN(NaN),
            N::NaN(_) => N::NaN(NaN),
            _ => N::Infinity(self),
        }
    }

    pub fn mul_inf(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Rational(r) => Infinity::new(self.dir * r.sign()).into(),
            N::Integer(i) => Infinity::new(self.dir * i.sign()).into(),
            N::Infinity(i) => Infinity::new(self.dir * i.dir).into(),
            N::NaN(_) => N::NaN(NaN),
        }
    }

    pub fn div_inf(self, other: NumberKind) -> NumberKind {
        use NumberKind as N;
        match other {
            N::Infinity(_) | N::Integer(_) | N::Rational(_) => self.mul_inf(other),
            N::NaN(_) => NaN.into(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Default)]
pub struct Integer(BigInt);

impl Numeric for Integer {
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }

    fn sign(&self) -> Sign {
        match self.0.sign() {
            bigint::Sign::Minus => Sign::Negitive,
            bigint::Sign::NoSign => Sign::UnSigned,
            bigint::Sign::Plus => Sign::Positive,
        }
    }
}

impl CalcursType for Integer {
    fn base(self) -> Base {
        Number::from(NumberKind::Integer(self)).base()
    }
}

impl From<Integer> for Rational {
    fn from(value: Integer) -> Self {
        Rational(BigRational::from_integer(value.0))
    }
}

impl Integer {
    pub fn new(val: i32) -> Self {
        Self(val.into())
    }

    pub fn num(val: i32) -> NumberKind {
        Self(val.into()).into()
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn add_int(self, other: Integer) -> NumberKind {
        Integer(self.0 + other.0).into()
    }

    pub fn sub_int(self, other: Integer) -> NumberKind {
        Integer(self.0 - other.0).into()
    }

    pub fn mul_int(self, other: Integer) -> NumberKind {
        Integer(self.0 * other.0).into()
    }

    // pub fn div_int(self, other: Integer) -> NumberKind {
    //     if other.0.is_zero() {
    //         Infinity::new(Sign::from_sign(&self.0)).into()
    //     } else {
    //         Integer(self.0 / other.0).into()
    //     }
    // }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Default)]
pub struct Rational(BigRational);

impl Numeric for Rational {
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }

    fn sign(&self) -> Sign {
        match self.0.numer().sign() {
            bigint::Sign::Minus => Sign::Negitive,
            bigint::Sign::NoSign => Sign::UnSigned,
            bigint::Sign::Plus => Sign::Positive,
        }
    }
}

impl CalcursType for Rational {
    fn base(self) -> Base {
        Number::from(NumberKind::Rational(self)).base()
    }
}

impl Rational {
    pub fn new(mut val: i32, mut denom: i32) -> Self {
        if denom.is_negative() {
            val *= -1;
            denom *= -1;
        }
        Self(BigRational::new(val.into(), denom.into()))
    }

    pub fn num(val: i32, denom: i32) -> NumberKind {
        Self::new(val, denom).cleanup()
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn cleanup(self) -> NumberKind {
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
        Rational(self.0 + r.0).cleanup()
    }

    pub fn sub_rat(self, r: Rational) -> NumberKind {
        Rational(self.0 - r.0).cleanup()
    }

    pub fn mul_rat(self, r: Rational) -> NumberKind {
        Rational(self.0 * r.0).cleanup()
    }

    pub fn div_rat(self, r: Rational) -> NumberKind {
        if r.0.is_zero() {
            Infinity::new(self.sign()).into()
        } else {
            Rational(self.0 / r.0).cleanup()
        }
    }
}

impl ops::Add for Base {
    type Output = Base;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(self, rhs)
    }
}

impl ops::Mul for Base {
    type Output = Base;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(self, rhs)
    }
}

impl ops::Sub for Base {
    type Output = Base;

    fn sub(self, rhs: Self) -> Self::Output {
        Add::add(self, Mul::mul(Integer::new(-1), rhs))
    }
}

#[cfg(test)]
mod test_numbers {

    use crate::prelude::*;

    use pretty_assertions::assert_eq;

    macro_rules! nk {
        (+inf) => {
            NumberKind::Infinity(Infinity::pos())
        };

        (-inf) => {
            NumberKind::Infinity(Infinity::neg())
        };

        (inf) => {
            NumberKind::Infinity(Infinity::default())
        };

        (nan) => {
            NumberKind::NaN(NaN)
        };

        ($int: literal) => {
            Integer::num($int)
        };

        ($val: literal / $denom: literal) => {
            Rational::num($val, $denom)
        };
    }

    macro_rules! c_impl {
        (+inf) => {
            Infinity::pos()
        };

        (-inf) => {
            Infinity::neg()
        };

        (inf) => {
            Infinity::default()
        };

        (nan) => {
            NaN
        };

        ($int: literal) => {
            Integer::num($int)
        };

        ($val: literal / $denom: literal) => {
            Rational::num($val, $denom)
        };

        (v($var: tt)) => {
            Variable::new(stringify!($var))
        };
    }

    macro_rules! c {
        ($($tt: tt)+) => {
            c_impl!($($tt)+).base()
        };
    }

    #[test]
    fn rational() {
        assert!(Rational::new(1, 2).is_pos());
        assert!(Rational::new(1, -2).is_neg());
        assert!(Rational::new(-1, -2).is_pos());
        assert!(Rational::new(-1, 2).is_neg());
    }

    #[test]
    fn add_binop() {
        assert_eq!(c!(2) + c!(3), c!(5));
        assert_eq!(c!(v(x)) + c!(v(x)) + c!(3), c!(3) + c!(v(x)) + c!(v(x)));
        assert_eq!(c!(-1) + c!(3), c!(2));
        assert_eq!(c!(-3) + c!(1 / 2), c!(-5 / 2));
        assert_eq!(c!(1 / 2) + c!(1 / 2), c!(1));
        assert_eq!(c!(inf) + c!(4), c!(inf));
        assert_eq!(c!(-inf) + c!(4), c!(-inf));
        assert_eq!(c!(+inf) + c!(+inf), c!(+inf));
        assert_eq!(c!(-inf) + c!(+inf), c!(nan));
        assert_eq!(c!(nan) + c!(inf), c!(nan));
        assert_eq!(c!(4 / 2), c!(2));
    }

    #[test]
    fn mul_binop() {
        assert_eq!(c!(-1) * c!(3), c!(-3));
        assert_eq!(c!(-1) * c!(0), c!(0));
        assert_eq!(c!(-3) * c!(1 / 2), c!(-3 / 2));
        assert_eq!(c!(1 / 2) * c!(1 / 2), c!(1 / 4));
        assert_eq!(c!(inf) * c!(4), c!(inf));
        assert_eq!(c!(-inf) * c!(4 / 2), c!(-inf));
        assert_eq!(c!(+inf) * c!(4), c!(+inf));
        assert_eq!(c!(+inf) * c!(-1), c!(-inf));
        assert_eq!(c!(+inf) * c!(+inf), c!(+inf));
        assert_eq!(c!(-inf) * c!(+inf), c!(-inf));
        assert_eq!(c!(nan) * c!(inf), c!(nan));
    }

    #[test]
    fn sub_binop() {
        assert_eq!(c!(-1) - c!(3), c!(-4));
        assert_eq!(c!(-3) - c!(1 / 2), c!(-7 / 2));
        assert_eq!(c!(1 / 2) - c!(1 / 2), c!(0));
        assert_eq!(c!(inf) - c!(4), c!(inf));
        assert_eq!(c!(-inf) - c!(4 / 2), c!(-inf));
        assert_eq!(c!(+inf) - c!(4), c!(+inf));
        assert_eq!(c!(+inf) - c!(+inf), c!(nan));
        assert_eq!(c!(-inf) - c!(+inf), c!(-inf));
        assert_eq!(c!(nan) - c!(inf), c!(nan));
    }

    #[test]
    fn div_num() {
        assert_eq!(nk!(-1).div_kind(nk!(3)), nk!(-1 / 3));
        assert_eq!(nk!(-1).div_kind(nk!(0)), nk!(-inf));
        assert_eq!(nk!(-3).div_kind(nk!(1 / 2)), nk!(-6));
        assert_eq!(nk!(1 / 2).div_kind(nk!(1 / 2)), nk!(1));
        assert_eq!(nk!(inf).div_kind(nk!(4)), nk!(inf));
        assert_eq!(nk!(-inf).div_kind(nk!(4 / 2)), nk!(-inf));
        assert_eq!(nk!(+inf).div_kind(nk!(4)), nk!(+inf));
        assert_eq!(nk!(+inf).div_kind(nk!(-1)), nk!(-inf));
        assert_eq!(nk!(+inf).div_kind(nk!(+inf)), nk!(+inf));
        assert_eq!(nk!(-inf).div_kind(nk!(+inf)), nk!(-inf));
        assert_eq!(nk!(nan).div_kind(nk!(inf)), nk!(nan));
    }
}
