use std::fmt;

pub use crate::rational::Rational;
use crate::{
    base::{Base, CalcursType, Num},
    numeric::constants::UNDEF,
    rational::NonZeroUInt,
};

pub mod constants {
    pub use super::*;

    /// + (1 / 1)
    pub const ONE: Number = Number::Rational(Rational {
        is_neg: false,
        numer: 1,
        denom: NonZeroUInt::new(1),
    });

    /// - (1 / 1)
    pub const MINUS_ONE: Number = Number::Rational(Rational {
        is_neg: true,
        numer: 1,
        denom: NonZeroUInt::new(1),
    });

    /// + (0 / 1)
    pub const ZERO: Number = Number::Rational(Rational {
        is_neg: false,
        numer: 0,
        denom: NonZeroUInt::new(1),
    });

    /// undefined
    pub const UNDEF: Number = Number::Undefined(Undefined);
}

#[derive(Debug, Clone, Hash, PartialEq, PartialOrd, Ord, Eq, Copy)]
pub enum Sign {
    Pos,
    Neg,
}

impl fmt::Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Sign as S;
        let s = match self {
            S::Pos => "",
            S::Neg => "-",
        };
        write!(f, "{s}")
    }
}

impl Sign {
    pub fn neg(&self) -> Self {
        use Sign as D;
        match self {
            D::Pos => D::Neg,
            D::Neg => D::Pos,
        }
    }

    pub fn is_pos(&self) -> bool {
        matches!(self, Sign::Pos)
    }

    pub fn is_neg(&self) -> bool {
        matches!(self, Sign::Neg)
    }

    pub fn mul_opt(mut self, other: Option<Self>) -> Self {
        if let Some(other) = other {
            self *= other
        }
        self
    }
}

impl std::ops::Mul for Sign {
    type Output = Sign;

    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.is_neg() {
            self.neg()
        } else {
            self
        }
    }
}

impl std::ops::MulAssign for Sign {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Infinity {
    sign: Sign,
}

impl fmt::Display for Infinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}oo", self.sign)
    }
}

impl Num for Infinity {
    #[inline]
    fn is_zero(&self) -> bool {
        false
    }

    #[inline]
    fn is_one(&self) -> bool {
        false
    }

    #[inline]
    fn is_neg_one(&self) -> bool {
        false
    }

    #[inline]
    fn sign(&self) -> Option<Sign> {
        Some(self.sign)
    }
}

impl CalcursType for Infinity {
    #[inline]
    fn base(self) -> Base {
        Number::Infinity(self).base()
    }
}

impl Infinity {
    #[inline]
    pub fn new(dir: Sign) -> Self {
        Self { sign: dir }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        false
    }

    #[inline]
    pub fn pos() -> Self {
        Self { sign: Sign::Pos }
    }

    #[inline]
    pub fn neg() -> Self {
        Self { sign: Sign::Neg }
    }

    pub fn add_num(self, n: Number) -> Number {
        use Number as N;
        match n {
            N::Rational(_) => self.into(),
            N::Infinity(inf) => match self.sign == inf.sign {
                true => self.into(),
                false => UNDEF,
            },
            UNDEF => n,
        }
    }

    pub fn sub_num(self, n: Number) -> Number {
        use Number as N;
        match n {
            N::Rational(_) | N::Infinity(_) => self.into(),
            UNDEF => n,
        }
    }

    pub fn mul_num(self, n: Number) -> Number {
        use Number as N;
        match n {
            N::Rational(r) => Infinity::new(self.sign.mul_opt(r.sign())).into(),
            N::Infinity(inf) => Infinity::new(self.sign * inf.sign).into(),
            UNDEF => n,
        }
    }

    pub fn div_num(self, n: Number) -> Number {
        self.mul_num(n)
    }
}

#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Copy)]
pub struct Undefined;

impl fmt::Display for Undefined {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Undefined")
    }
}

impl Num for Undefined {
    #[inline]
    fn is_zero(&self) -> bool {
        false
    }

    #[inline]
    fn is_one(&self) -> bool {
        false
    }

    #[inline]
    fn is_neg_one(&self) -> bool {
        false
    }

    #[inline]
    fn sign(&self) -> Option<Sign> {
        None
    }
}

impl CalcursType for Undefined {
    #[inline]
    fn base(self) -> Base {
        Number::Undefined(self).base()
    }
}

impl Undefined {
    pub fn is_zero(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum Number {
    Rational(Rational),

    Infinity(Infinity),
    Undefined(Undefined),
}

macro_rules! for_each_number {
    ($self: ident, $v:ident => $bod: tt) => {
        match $self {
            Number::Rational($v) => $bod,
            Number::Infinity($v) => $bod,
            Number::Undefined($v) => $bod,
        }
    };
}

impl From<Rational> for Number {
    fn from(value: Rational) -> Self {
        Number::Rational(value)
    }
}

impl From<Infinity> for Number {
    fn from(value: Infinity) -> Self {
        Number::Infinity(value)
    }
}

impl From<Undefined> for Number {
    fn from(value: Undefined) -> Self {
        Number::Undefined(value)
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for_each_number!(self, v => { write!(f, "{v}")})
    }
}

impl CalcursType for Number {
    #[inline]
    fn base(self) -> Base {
        Base::Number(self)
    }
}

impl Num for Number {
    #[inline]
    fn is_zero(&self) -> bool {
        for_each_number!(self, v => { v.is_zero() })
    }

    #[inline]
    fn is_one(&self) -> bool {
        for_each_number!(self, v => { v.is_one() })
    }

    #[inline]
    fn is_neg_one(&self) -> bool {
        for_each_number!(self, v => { v.is_neg_one() })
    }

    #[inline]
    fn sign(&self) -> Option<Sign> {
        for_each_number!(self, v => { v.sign() })
    }
}

impl Number {
    pub fn add_num(self, n: Number) -> Number {
        use Number as N;
        match (self, n) {
            (UNDEF, _) | (_, UNDEF) => UNDEF,
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.add_num(n),
            (N::Rational(r1), N::Rational(r2)) => r1.add_ratio(r2).into(),
        }
    }

    pub fn sub_num(self, n: Number) -> Number {
        use Number as N;
        match (self, n) {
            (UNDEF, _) | (_, UNDEF) => UNDEF,
            (N::Infinity(inf), n) => inf.sub_num(n),
            (n, N::Infinity(inf)) => n.sub_inf(inf),
            (N::Rational(r1), N::Rational(r2)) => r1.sub_ratio(r2).into(),
        }
    }

    pub fn mul_num(self, n: Number) -> Number {
        use Number as N;
        match (self, n) {
            (UNDEF, _) | (_, UNDEF) => UNDEF,
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.mul_num(n),
            (N::Rational(r1), N::Rational(r2)) => r1.mul_ratio(r2).into(),
        }
    }

    pub fn div_num(self, n: Number) -> Number {
        use Number as N;
        match (self, n) {
            (UNDEF, _) | (_, UNDEF) => UNDEF,
            (N::Infinity(inf), n) => inf.div_num(n),
            (n, N::Infinity(inf)) => n.div_inf(inf),

            (N::Rational(r1), N::Rational(r2)) => r1.div_ratio(r2).into(),
        }
    }

    fn sub_inf(self, inf: Infinity) -> Number {
        use Number as N;
        match self {
            N::Rational(_) => inf.into(),
            N::Infinity(i) => i.sub_num(inf.into()),
            UNDEF => self,
        }
    }

    fn div_inf(self, mut inf: Infinity) -> Number {
        use Number as N;
        match self {
            N::Rational(r) => {
                let sign = r.sign();
                inf.sign = inf.sign.mul_opt(sign);
                inf.into()
            }
            N::Infinity(i) => i.div_num(inf.into()),
            UNDEF => self,
        }
    }
}

impl std::ops::Add for Number {
    type Output = Number;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_num(rhs)
    }
}

impl std::ops::AddAssign for Number {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_num(rhs);
    }
}

impl std::ops::Sub for Number {
    type Output = Number;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_num(rhs)
    }
}

impl std::ops::SubAssign for Number {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub_num(rhs);
    }
}

impl std::ops::Mul for Number {
    type Output = Number;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_num(rhs)
    }
}

impl std::ops::MulAssign for Number {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_num(rhs);
    }
}

impl std::ops::Div for Number {
    type Output = Number;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_num(rhs)
    }
}

impl std::ops::DivAssign for Number {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.div_num(rhs);
    }
}
