use num::integer::Roots;
use std::{cmp::Ordering, fmt, ops};

use crate::{
    base::{Base, CalcursType, SubsDict},
    pattern::{self, Item},
    rational::{NonZero, Rational},
};

/// defines all numeric types
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Numeric {
    // F64(f64),
    Rational(Rational),
    Infinity(Infinity),
    Undefined(Undefined),
}

/// only used if the result is provenly undefined, e.g 0 / 0
///
/// not the same as [f64::NAN]
#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, Copy)]
pub struct Undefined;

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Infinity {
    pub(crate) sign: Sign,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub enum Sign {
    Positive,
    Negative,
}

impl Numeric {
    pub fn subs(self, _dict: &SubsDict) -> Self {
        self
    }

    pub const fn desc(&self) -> pattern::Pattern {
        pattern::Pattern::Itm(match self {
            Numeric::Rational(r) => r.desc().to_item(),
            Numeric::Infinity(Infinity { sign }) => sign.desc().union(Item::Inf),
            Numeric::Undefined(_) => Item::Undef,
        })
    }
    pub fn add_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.add_num(n),
            (N::Rational(r1), N::Rational(r2)) => (r1 + r2).into(),
        }
    }

    pub fn sub_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) => inf.sub_num(n),
            (n, N::Infinity(inf)) => n.sub_inf(inf),
            (N::Rational(r1), N::Rational(r2)) => (r1 - r2).into(),
        }
    }

    pub fn mul_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.mul_num(n),
            (N::Rational(r1), N::Rational(r2)) => (r1 * r2).into(),
        }
    }

    pub fn div_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) => inf.div_num(n),
            (n, N::Infinity(inf)) => n.div_inf(inf),

            (N::Rational(r1), N::Rational(r2)) => (r1 / r2).into(),
        }
    }

    pub fn checked_pow_num(self, n: Numeric) -> Option<Numeric> {
        match (self, n) {
            (Numeric::Rational(r1), Numeric::Rational(r2)) => {
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

                Some(Rational::reduced(sign, numer, NonZero::new(denom), 0).num())
            }
            _ => None,
        }
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

    pub const fn desc(&self) -> pattern::Item {
        match self {
            Sign::Positive => Item::Pos,
            Sign::Negative => Item::Neg,
        }
    }

    pub fn mul_opt(mut self, other: Option<Self>) -> Self {
        if let Some(other) = other {
            self *= other
        }
        self
    }
}

impl Undefined {
    pub fn num(self) -> Numeric {
        self.into()
    }
}

impl ops::Add for Numeric {
    type Output = Numeric;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_num(rhs)
    }
}

impl ops::AddAssign for Numeric {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_num(rhs);
    }
}

impl ops::Sub for Numeric {
    type Output = Numeric;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_num(rhs)
    }
}

impl ops::SubAssign for Numeric {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub_num(rhs);
    }
}

impl ops::Mul for Numeric {
    type Output = Numeric;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_num(rhs)
    }
}

impl ops::MulAssign for Numeric {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_num(rhs);
    }
}

impl ops::Div for Numeric {
    type Output = Numeric;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_num(rhs)
    }
}

impl ops::DivAssign for Numeric {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.div_num(rhs);
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

impl ops::MulAssign for Sign {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Ord for Numeric {
    fn cmp(&self, other: &Self) -> Ordering {
        use Numeric as N;
        match (self, other) {
            (N::Rational(r1), N::Rational(r2)) => r1.cmp(r2),
            (N::Infinity(i1), N::Infinity(i2)) => i1.cmp(i2),
            (N::Undefined(_), N::Undefined(_)) => Ordering::Equal,

            (N::Rational(_), N::Infinity(inf)) => match inf.sign {
                Sign::Positive => Ordering::Less,
                Sign::Negative => Ordering::Greater,
            },
            (N::Infinity(inf), N::Rational(_)) => match inf.sign {
                Sign::Positive => Ordering::Greater,
                Sign::Negative => Ordering::Less,
            },

            (_, N::Undefined(_)) => Ordering::Greater,
            (N::Undefined(_), _) => Ordering::Less,
        }
    }
}

impl PartialOrd for Numeric {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

impl CalcursType for Numeric {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Numeric(self)
    }
}

impl CalcursType for Infinity {
    #[inline(always)]
    fn base(self) -> Base {
        Numeric::Infinity(self).base()
    }
}

impl CalcursType for Undefined {
    #[inline(always)]
    fn base(self) -> Base {
        Numeric::Undefined(self).base()
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

impl fmt::Display for Numeric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Numeric::Rational(v) => write!(f, "{v}"),
            Numeric::Infinity(v) => write!(f, "{v}"),
            Numeric::Undefined(v) => write!(f, "{v}"),
        }
    }
}

impl fmt::Display for Undefined {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Undefined")
    }
}

impl fmt::Display for Infinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}oo", self.sign)
    }
}

impl fmt::Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Sign as S;
        let s = match self {
            S::Positive => "",
            S::Negative => "-",
        };
        write!(f, "{s}")
    }
}
