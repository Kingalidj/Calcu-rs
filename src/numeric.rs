use num::integer::Roots;
use std::{
    cmp::Ordering,
    fmt::{self, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{
    base::{Base, CalcursType, SubsDict},
    pattern::itm,
    rational::{NonZero, Rational},
};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub enum Sign {
    Positive,
    Negative,
}

impl PartialOrd for Sign {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

impl<I: num::Integer> From<I> for Sign {
    // Positive for value >= 0
    fn from(value: I) -> Self {
        if value >= I::zero() {
            Sign::Positive
        } else {
            Sign::Negative
        }
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
    pub fn num(self) -> Numeric {
        self.into()
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
            N::Rational(_) => self.num(),
            N::Infinity(inf) => match self.sign == inf.sign {
                true => self.num(),
                false => Undefined.num(),
            },
            N::Undefined(_) => n,
        }
    }

    pub fn sub_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match n {
            N::Rational(_) | N::Infinity(_) => self.into(),
            N::Undefined(_) => n,
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

impl Undefined {
    pub fn num(self) -> Numeric {
        self.into()
    }
}

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
        match (self, other) {
            (itm!(num: Rational: r1), itm!(num: Rational: r2)) => r1.cmp(r2),
            (itm!(num: Infinity: i1), itm!(num: Infinity: i2)) => i1.cmp(i2),
            (itm!(num: undef), itm!(num: undef)) => Ordering::Equal,

            (_, itm!(num: undef)) => Ordering::Greater,
            (itm!(num: undef), _) => Ordering::Less,

            (itm!(num: Rational: _), itm!(num: +oo)) => Ordering::Less,
            (itm!(num: +oo), itm!(num: Rational: _)) => Ordering::Greater,
            (itm!(num: Rational: _), itm!(num: -oo)) => Ordering::Greater,
            (itm!(num: -oo), itm!(num: Rational: _)) => Ordering::Less,
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
        matches!(self, itm!(num: 0))
    }

    pub const fn is_one(&self) -> bool {
        matches!(self, itm!(num: 1))
    }

    pub const fn is_minus_one(&self) -> bool {
        matches!(self, itm!(num: -1))
    }

    pub const fn is_negative(&self) -> bool {
        matches!(self, itm!(num: -))
    }

    pub const fn is_positive(&self) -> bool {
        matches!(self, itm!(num: +))
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
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.add_num(n),
            (N::Rational(r1), N::Rational(r2)) => r1.add(r2).into(),
        }
    }

    pub fn sub_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) => inf.sub_num(n),
            (n, N::Infinity(inf)) => n.sub_inf(inf),
            (N::Rational(r1), N::Rational(r2)) => r1.sub(r2).into(),
        }
    }

    pub fn mul_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.mul_num(n),
            (N::Rational(r1), N::Rational(r2)) => r1.mul(r2).into(),
        }
    }

    pub fn div_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) => inf.div_num(n),
            (n, N::Infinity(inf)) => n.div_inf(inf),

            (N::Rational(r1), N::Rational(r2)) => r1.div(r2).into(),
        }
    }

    pub fn checked_pow_num(self, n: Numeric) -> Option<Numeric> {
        //let test: itm!(num: Undefined) = Undefined;
        Some(match (self, n) {
            (_, itm!(num: undef)) | (itm!(num: undef), _) => Undefined.num(),

            // 0^0 = undefined
            (itm!(num: 0), itm!(num: 0)) => Undefined.num(),

            // x^(-oo) = 0
            (itm!(num: Rational: _), itm!(num: -oo)) => Rational::zero().num(),

            // x^(+oo) = +oo
            (itm!(num: Rational: _), itm!(num: +oo)) => Infinity::pos().num(),

            // 1^x = 1
            (itm!(num: 1), _) => Rational::one().num(),

            // x^0 = 1 | x != 0
            (n, itm!(num: 0)) if !n.is_zero() => Rational::one().num(),

            // 0^x = undefined | x < 0
            (itm!(num: 0), itm!(num: -)) => Undefined.num(),

            // 0^x = 0 | x > 0
            (itm!(num: 0), itm!(num: +)) => Rational::zero().num(),

            // n^-1 = 1/n | n != 0
            (itm!(num: Rational: r), itm!(num: -1)) => (Rational::one() / r).num(),

            (itm!(num: Rational: r1), itm!(num: Rational: r2)) => {
                let mut exp = r2.try_apply_expon()?;
                let mut base = r1.try_apply_expon()?;
                exp.reduce_frac();
                base.reduce_frac();

                let mut root = base.numer;
                if exp.denom() != 1 {
                    root = base.numer.nth_root(exp.denom().try_into().ok()?);
                    if root * root != base.numer {
                        return None;
                    };
                }
                let numer = root.pow(exp.numer.try_into().ok()?);

                let mut root = base.denom();
                if exp.denom() != 1 {
                    root = base.denom().nth_root(exp.denom().try_into().ok()?);
                    if root * root != base.denom() {
                        return None;
                    };
                }
                let denom = root.pow(exp.numer.try_into().ok()?);

                let sign = if exp.numer() % 2 == 0 {
                    Sign::Positive
                } else {
                    base.sign
                };

                Rational::reduced(sign, numer, NonZero::new(denom), 0).num()
            }

            _ => return None,
        })
    }

    fn sub_inf(self, inf: Infinity) -> Numeric {
        use Numeric as N;
        match self {
            N::Rational(_) => inf.into(),
            N::Infinity(i) => i.sub_num(inf.into()),
            N::Undefined(_) => self,
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
            N::Undefined(_) => self,
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
