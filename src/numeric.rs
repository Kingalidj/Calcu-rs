use std::{
    cmp::Ordering,
    fmt::{self, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{
    base::{Base, CalcursType, SubsDict},
    numeric::constants::{ONE, UNDEF, ZERO},
    pattern::pat,
    rational::{NonZeroUInt, Rational},
};

pub mod constants {
    pub use super::*;

    /// + (1 / 1)
    pub const ONE: Numeric = Numeric::Rational(Rational {
        sign: Sign::Positive,
        numer: 1,
        denom: NonZeroUInt::new(1),
    });

    /// - (1 / 1)
    pub const MINUS_ONE: Numeric = Numeric::Rational(Rational {
        sign: Sign::Negative,
        numer: 1,
        denom: NonZeroUInt::new(1),
    });

    /// + (0 / 1)
    pub const ZERO: Numeric = Numeric::Rational(Rational {
        sign: Sign::Positive,
        numer: 0,
        denom: NonZeroUInt::new(1),
    });

    /// undefined
    pub const UNDEF: Numeric = Numeric::Undefined(Undefined);
}

#[derive(Debug, Clone, Hash, PartialEq, PartialOrd, Eq, Copy)]
pub enum Sign {
    Positive,
    Negative,
}

impl Ord for Sign {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Sign::Positive, Sign::Negative) => Ordering::Greater,
            (Sign::Negative, Sign::Positive) => Ordering::Less,
            (Sign::Negative, Sign::Negative) | (Sign::Positive, Sign::Positive) => Ordering::Equal,
        }
    }
}

impl Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Sign as S;
        let s = match self {
            S::Positive => "",
            S::Negative => "-",
        };
        write!(f, "{s}")
    }
}

impl Sign {
    pub fn neg(&self) -> Self {
        use Sign as S;
        match self {
            S::Positive => S::Negative,
            S::Negative => S::Positive,
        }
    }

    #[inline]
    pub const fn is_pos(&self) -> bool {
        matches!(self, Sign::Positive)
    }

    #[inline]
    pub const fn is_neg(&self) -> bool {
        matches!(self, Sign::Negative)
    }

    pub fn mul_opt(mut self, other: Option<Self>) -> Self {
        if let Some(other) = other {
            self *= other
        }
        self
    }
}

impl Mul for Sign {
    type Output = Sign;

    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.is_neg() {
            self.neg()
        } else {
            self
        }
    }
}

impl MulAssign for Sign {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Infinity {
    pub(crate) sign: Sign,
}

impl Display for Infinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}oo", self.sign)
    }
}

impl CalcursType for Infinity {
    #[inline(always)]
    fn base(self) -> Base {
        Numeric::Infinity(self).base()
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
        Self {
            sign: Sign::Positive,
        }
    }

    #[inline]
    pub fn neg() -> Self {
        Self {
            sign: Sign::Negative,
        }
    }

    pub fn add_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match n {
            N::Rational(_) => self.into(),
            N::Infinity(inf) => match self.sign == inf.sign {
                true => self.into(),
                false => UNDEF,
            },
            UNDEF => n,
        }
    }

    pub fn sub_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match n {
            N::Rational(_) | N::Infinity(_) => self.into(),
            UNDEF => n,
        }
    }

    pub fn mul_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match n {
            N::Rational(r) => Infinity::new(self.sign * r.sign).into(),
            N::Infinity(inf) => Infinity::new(self.sign * inf.sign).into(),
            N::Undefined(_) => n,
        }
    }

    pub fn div_num(self, n: Numeric) -> Numeric {
        self.mul_num(n)
    }
}

#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Copy)]
pub struct Undefined;

impl Display for Undefined {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Undefined")
    }
}

impl CalcursType for Undefined {
    #[inline(always)]
    fn base(self) -> Base {
        Numeric::Undefined(self).base()
    }
}

/// defines all numeric types
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Numeric {
    Rational(Rational),

    Infinity(Infinity),

    /// only used if the calculation itself is undefined, e.g 0 / 0
    /// not the same as [f64::NAN]
    Undefined(Undefined),
}

impl Ord for Numeric {
    fn cmp(&self, other: &Self) -> Ordering {
        pat!(use);
        match (self, other) {
            (num_pat!(Rational: r1), num_pat!(Rational: r2)) => r1.cmp(r2),
            (num_pat!(Infinity: i1), num_pat!(Infinity: i2)) => i1.cmp(i2),
            (num_pat!(undef), num_pat!(undef)) => Ordering::Equal,

            (_, num_pat!(undef)) => Ordering::Greater,
            (num_pat!(undef), _) => Ordering::Less,

            (num_pat!(Rational: _), num_pat!(+oo)) => Ordering::Less,
            (num_pat!(+oo), num_pat!(Rational: _)) => Ordering::Greater,
            (num_pat!(Rational: _), num_pat!(-oo)) => Ordering::Greater,
            (num_pat!(-oo), num_pat!(Rational: _)) => Ordering::Less,
        }
    }
}

impl PartialOrd for Numeric {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

macro_rules! for_each_number {
    ($self: ident, $v:ident => $bod: tt) => {
        match $self {
            Numeric::Rational($v) => $bod,
            Numeric::Infinity($v) => $bod,
            Numeric::Undefined($v) => $bod,
        }
    };
}

impl Numeric {
    pub const fn is_zero(&self) -> bool {
        pat!(use);
        matches!(self, pat!(Numeric: 0))
    }

    pub const fn is_one(&self) -> bool {
        pat!(use);
        matches!(self, pat!(Numeric: 1))
    }

    pub const fn is_minus_one(&self) -> bool {
        pat!(use);
        matches!(self, pat!(Numeric: -1))
    }

    pub const fn is_negative(&self) -> bool {
        pat!(use);
        matches!(self, pat!(Numeric: -))
    }

    pub const fn is_positive(&self) -> bool {
        pat!(use);
        matches!(self, pat!(Numeric: +))
    }

    pub fn subs(self, _dict: &SubsDict) -> Self {
        self
    }
}

impl From<Rational> for Numeric {
    fn from(value: Rational) -> Self {
        Numeric::Rational(value)
    }
}

impl From<Infinity> for Numeric {
    fn from(value: Infinity) -> Self {
        Numeric::Infinity(value)
    }
}

impl From<Undefined> for Numeric {
    fn from(value: Undefined) -> Self {
        Numeric::Undefined(value)
    }
}

impl Display for Numeric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for_each_number!(self, v => { write!(f, "{v}")})
    }
}

impl CalcursType for Numeric {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Numeric(self)
    }
}

impl Numeric {
    pub fn add_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (UNDEF, _) | (_, UNDEF) => UNDEF,
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.add_num(n),
            (N::Rational(r1), N::Rational(r2)) => r1.add_ratio(r2).into(),
        }
    }

    pub fn sub_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (UNDEF, _) | (_, UNDEF) => UNDEF,
            (N::Infinity(inf), n) => inf.sub_num(n),
            (n, N::Infinity(inf)) => n.sub_inf(inf),
            (N::Rational(r1), N::Rational(r2)) => r1.sub_ratio(r2).into(),
        }
    }

    pub fn mul_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (UNDEF, _) | (_, UNDEF) => UNDEF,
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.mul_num(n),
            (N::Rational(r1), N::Rational(r2)) => r1.mul_ratio(r2).into(),
        }
    }

    pub fn div_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (UNDEF, _) | (_, UNDEF) => UNDEF,
            (N::Infinity(inf), n) => inf.div_num(n),
            (n, N::Infinity(inf)) => n.div_inf(inf),

            (N::Rational(r1), N::Rational(r2)) => r1.div_ratio(r2).into(),
        }
    }

    pub fn checked_pow_num(self, n: Numeric) -> Option<Numeric> {
        pat!(use);

        Some(match (self, n) {
            (_, num_pat!(undef)) | (num_pat!(undef), _) => UNDEF,

            // 0^0 = undefined
            (num_pat!(0), num_pat!(0)) => UNDEF,

            // x^(-oo) = 0
            (num_pat!(Rational: _), num_pat!(-oo)) => ZERO,

            // x^(+oo) = +oo
            (num_pat!(Rational: _), num_pat!(+oo)) => Infinity::pos().into(),

            // 1^x = 1
            (num_pat!(1), _) => ONE,

            // x^0 = 1 | x != 0
            (n, num_pat!(0)) if !n.is_zero() => ONE,

            // 0^x = undefined | x < 0
            (num_pat!(0), num_pat!(-)) => UNDEF,

            // 0^x = 0 | x > 0
            (num_pat!(0), num_pat!(+)) => ZERO,

            // n^-1 = 1/n | n != 0
            (num_pat!(Rational: r), num_pat!(-1)) => (Rational::one() / r).into(),

            (num_pat!(Rational: r1), num_pat!(Rational: r2)) => {
                let exp = r2.numer;

                todo!()
            }

            _ => todo!(),
        })
    }

    fn sub_inf(self, inf: Infinity) -> Numeric {
        use Numeric as N;
        match self {
            N::Rational(_) => inf.into(),
            N::Infinity(i) => i.sub_num(inf.into()),
            UNDEF => self,
        }
    }

    fn div_inf(self, mut inf: Infinity) -> Numeric {
        use Numeric as N;
        match self {
            N::Rational(r) => {
                inf.sign *= r.sign;
                inf.into()
            }
            N::Infinity(i) => i.div_num(inf.into()),
            UNDEF => self,
        }
    }
}

impl Add for Numeric {
    type Output = Numeric;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_num(rhs)
    }
}

impl AddAssign for Numeric {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_num(rhs);
    }
}

impl Sub for Numeric {
    type Output = Numeric;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_num(rhs)
    }
}

impl SubAssign for Numeric {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub_num(rhs);
    }
}

impl Mul for Numeric {
    type Output = Numeric;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_num(rhs)
    }
}

impl MulAssign for Numeric {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_num(rhs);
    }
}

impl Div for Numeric {
    type Output = Numeric;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_num(rhs)
    }
}

impl DivAssign for Numeric {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.div_num(rhs);
    }
}
