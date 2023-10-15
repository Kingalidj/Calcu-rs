use std::{collections::HashMap, fmt, ops};

use calcurs_macros::Procagate;
use derive_more::Display;
use num::{bigint, BigInt, BigRational, Zero};
use num_traits::One;

use crate::{Basic, BasicKind, CalcursType};

#[derive(Debug, Clone, Hash, Eq, PartialEq, Display)]
pub struct Number {
    kind: NumberKind,
}

impl Numberic for Number {
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
    fn to_basic(self) -> Basic {
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

impl Numberic for NumberKind {
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

pub trait Numberic {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn sign(&self) -> Sign;

    fn is_pos(&self) -> bool {
        self.sign().is_pos()
    }

    fn is_neg(&self) -> bool {
        self.sign().is_neg()
    }
}

impl NumberKind {
    pub fn add<T: Into<NumberKind>>(self, other: T) -> NumberKind {
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

    pub fn sub(self, mut other: NumberKind) -> NumberKind {
        other = other.mul(Integer::num(-1));
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
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display, Copy)]
pub struct NaN;

impl Numberic for NaN {
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
    fn to_basic(self) -> Basic {
        Number::from(NumberKind::NaN(self)).to_basic()
    }
}

impl NaN {
    pub fn is_zero(&self) -> bool {
        false
    }
}

impl From<NaN> for NumberKind {
    fn from(value: NaN) -> Self {
        NumberKind::NaN(value)
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

impl Numberic for Infinity {
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
    fn to_basic(self) -> Basic {
        Number::from(NumberKind::Infinity(self)).to_basic()
    }
}

impl From<Infinity> for NumberKind {
    fn from(value: Infinity) -> Self {
        NumberKind::Infinity(value)
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

impl Numberic for Integer {
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
    fn to_basic(self) -> Basic {
        Number::from(NumberKind::Integer(self)).to_basic()
    }
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
    pub fn new(val: i32) -> Number {
        NumberKind::Integer(Self(val.into())).into()
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

impl Numberic for Rational {
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
    fn to_basic(self) -> Basic {
        Number::from(NumberKind::Rational(self)).to_basic()
    }
}

impl From<Rational> for NumberKind {
    fn from(value: Rational) -> Self {
        NumberKind::Rational(value)
    }
}

impl Rational {
    pub fn new(val: i32, denom: i32) -> Number {
        Self::num(val, denom).into()
    }

    pub fn num(mut val: i32, mut denom: i32) -> NumberKind {
        if denom.is_negative() {
            val *= -1;
            denom *= -1;
        }
        Self(BigRational::new(val.into(), denom.into())).simplify()
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

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
            Infinity::new(self.sign()).into()
        } else {
            Rational(self.0 / r.0).simplify().into()
        }
    }
}

/// Represents addition in symbolic expressions
///
/// Implemented with a coefficient and a hashmap: \
/// coeff + key1*value1 + key2*value2 + ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Add {
    coeff: NumberKind,
    arg_map: HashMap<BasicKind, NumberKind>,
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.arg_map.iter();

        if let Some((k, v)) = iter.next() {
            write!(f, "{v}{k}")?;
        }

        for (k, v) in iter {
            write!(f, " + {v}{k}")?;
        }

        if !self.coeff.is_zero() {
            write!(f, " + {}", self.coeff)?;
        }

        Ok(())
    }
}

impl std::hash::Hash for Add {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "Add".hash(state);
        self.coeff.hash(state);
        for a in &self.arg_map {
            a.hash(state);
        }
    }
}

impl CalcursType for Add {
    fn to_basic(self) -> Basic {
        BasicKind::Add(self).into()
    }
}

impl Add {
    pub fn add<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Basic {
        let b1 = b1.to_basic();
        let b2 = b2.to_basic();
        Self::add_kind(b1.kind, b2.kind)
    }

    pub fn add_kind(b1: BasicKind, b2: BasicKind) -> Basic {
        let add = Self {
            coeff: Integer::num(0),
            arg_map: Default::default(),
        };
        add.append_basic(b1).append_basic(b2).simplify()
    }

    fn simplify(self) -> Basic {
        //TODO remvoe zero
        if self.arg_map.is_empty() {
            Number::from(self.coeff).to_basic()
        } else {
            self.to_basic()
        }
    }

    fn append_basic(mut self, b: BasicKind) -> Self {
        use BasicKind as B;

        match b {
            B::Boolean(_) => todo!(),

            B::Number(num) => {
                self.coeff = self.coeff.add(num.kind);
            }
            B::Mul(mut mul) => {
                let mut coeff = Integer::num(1);
                (mul.coeff, coeff) = (coeff, mul.coeff);
                self.append_term(coeff, BasicKind::Mul(mul));
            }

            B::Add(add) => {
                add.arg_map.into_iter().for_each(|(term, coeff)| {
                    self.append_term(coeff, term);
                });

                self.coeff = self.coeff.add(add.coeff);
            }
            B::Var(_) | B::Dummy => {
                let term = Integer::num(1);
                self.append_term(term, b);
            }
        }

        self
    }

    /// adds the term: coeff*b
    fn append_term(&mut self, coeff: NumberKind, b: BasicKind) {
        if let Some(mut key) = self.arg_map.remove(&b) {
            key = key.add(coeff);

            if !key.is_zero() {
                self.arg_map.insert(b, key);
            }
        } else {
            if !coeff.is_zero() {
                self.arg_map.insert(b, coeff);
            }
        }
    }
}

impl ops::Add for Basic {
    type Output = Basic;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(self, rhs)
    }
}

/// Represents multiplication in symbolic expressions
///
/// Implemented with a coefficient and a hashmap: \
/// coeff * key1^value1 * key2^value2 * ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mul {
    coeff: NumberKind,
    arg_map: HashMap<BasicKind, BasicKind>,
}

impl Mul {
    pub fn mul<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Basic {
        let b1 = b1.to_basic();
        let b2 = b2.to_basic();

        Self {
            coeff: Integer::num(1),
            arg_map: Default::default(),
        }
        .append_basic(b1.kind)
        .append_basic(b2.kind)
        .simplify()
    }

    fn append_basic(mut self, b: BasicKind) -> Self {
        use BasicKind as B;

        match b {
            B::Number(n) => {
                todo!("is_zero")
            }
            B::Mul(mul) => {
                self.coeff = self.coeff.mul(mul.coeff);
                mul.arg_map
                    .into_iter()
                    .for_each(|(key, val)| self.append_term(key, val))
            }
            _ => {
                let exp = BasicKind::Number(Integer::num(1).into());
                self.append_term(b, exp);
            }
        }
        self
    }

    /// adds the term: b^exp
    fn append_term(&mut self, b: BasicKind, exp: BasicKind) {
        if let Some(key) = self.arg_map.remove(&b) {
            let exp = Add::add_kind(key, exp);
            match exp.kind {
                BasicKind::Number(ref num) if num.is_zero() => (),
                _ => {
                    self.arg_map.insert(b, exp.kind);
                }
            }
        } else {
            self.arg_map.insert(b, exp);
        }
    }

    fn simplify(self) -> Basic {
        if self.coeff.is_zero() || self.arg_map.is_empty() {
            Number::from(self.coeff).to_basic()
        } else {
            self.to_basic()
        }
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.arg_map.iter();

        if let Some((k, v)) = iter.next() {
            write!(f, "{v}^{k}")?;
        }

        for (k, v) in iter {
            write!(f, " * {v}^{k}")?;
        }

        if !self.coeff.is_one() {
            write!(f, " * {}", self.coeff)?;
        }

        Ok(())
    }
}

impl std::hash::Hash for Mul {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "Mul".hash(state);
        self.coeff.hash(state);
        for a in &self.arg_map {
            a.hash(state);
        }
    }
}

impl CalcursType for Mul {
    fn to_basic(self) -> Basic {
        BasicKind::Mul(self).into()
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
            NaN.to_basic()
        };

        ($int: literal) => {
            Integer::new($int)
        };

        ($val: literal / $denom: literal) => {
            Rational::new($val, $denom)
        };

        (v($var: tt)) => {
            Variable::new(stringify!($var))
        };
    }

    macro_rules! c {
        ($($tt: tt)+) => {
            c_impl!($($tt)+).to_basic()
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
    fn add_num() {
        assert_eq!(nk!(-1).add(nk!(3)), nk!(2));
        assert_eq!(nk!(-3).add(nk!(1 / 2)), nk!(-5 / 2));
        assert_eq!(nk!(1 / 2).add(nk!(1 / 2)), nk!(1));
        assert_eq!(nk!(inf).add(nk!(4)), nk!(inf));
        assert_eq!(nk!(-inf).add(nk!(4 / 2)), nk!(-inf));
        assert_eq!(nk!(+inf).add(nk!(4)), nk!(+inf));
        assert_eq!(nk!(+inf).add(nk!(+inf)), nk!(+inf));
        assert_eq!(nk!(-inf).add(nk!(+inf)), nk!(nan));
        assert_eq!(nk!(nan).add(nk!(inf)), nk!(nan));
        assert_eq!(nk!(4 / 2), nk!(2));
    }

    #[test]
    fn sub_num() {
        assert_eq!(nk!(-1).sub(nk!(3)), nk!(-4));
        assert_eq!(nk!(-3).sub(nk!(1 / 2)), nk!(-7 / 2));
        assert_eq!(nk!(1 / 2).sub(nk!(1 / 2)), nk!(0));
        assert_eq!(nk!(inf).sub(nk!(4)), nk!(inf));
        assert_eq!(nk!(-inf).sub(nk!(4 / 2)), nk!(-inf));
        assert_eq!(nk!(+inf).sub(nk!(4)), nk!(+inf));
        assert_eq!(nk!(+inf).sub(nk!(+inf)), nk!(nan));
        assert_eq!(nk!(-inf).sub(nk!(+inf)), nk!(-inf));
        assert_eq!(nk!(nan).sub(nk!(inf)), nk!(nan));
    }

    #[test]
    fn mul_num() {
        assert_eq!(nk!(-1).mul(nk!(3)), nk!(-3));
        assert_eq!(nk!(-1).mul(nk!(0)), nk!(0));
        assert_eq!(nk!(-3).mul(nk!(1 / 2)), nk!(-3 / 2));
        assert_eq!(nk!(1 / 2).mul(nk!(1 / 2)), nk!(1 / 4));
        assert_eq!(nk!(inf).mul(nk!(4)), nk!(inf));
        assert_eq!(nk!(-inf).mul(nk!(4 / 2)), nk!(-inf));
        assert_eq!(nk!(+inf).mul(nk!(4)), nk!(+inf));
        assert_eq!(nk!(+inf).mul(nk!(-1)), nk!(-inf));
        assert_eq!(nk!(+inf).mul(nk!(+inf)), nk!(+inf));
        assert_eq!(nk!(-inf).mul(nk!(+inf)), nk!(-inf));
        assert_eq!(nk!(nan).mul(nk!(inf)), nk!(nan));
    }

    #[test]
    fn div_num() {
        assert_eq!(nk!(-1).div(nk!(3)), nk!(-1 / 3));
        assert_eq!(nk!(-1).div(nk!(0)), nk!(-inf));
        assert_eq!(nk!(-3).div(nk!(1 / 2)), nk!(-6));
        assert_eq!(nk!(1 / 2).div(nk!(1 / 2)), nk!(1));
        assert_eq!(nk!(inf).div(nk!(4)), nk!(inf));
        assert_eq!(nk!(-inf).div(nk!(4 / 2)), nk!(-inf));
        assert_eq!(nk!(+inf).div(nk!(4)), nk!(+inf));
        assert_eq!(nk!(+inf).div(nk!(-1)), nk!(-inf));
        assert_eq!(nk!(+inf).div(nk!(+inf)), nk!(+inf));
        assert_eq!(nk!(-inf).div(nk!(+inf)), nk!(-inf));
        assert_eq!(nk!(nan).div(nk!(inf)), nk!(nan));
    }
}
