use num::integer::Roots;
use std::{cmp::Ordering, fmt, ops};

use crate::{
    base::{Base, CalcursType, Described},
    pattern::{self, Item},
    rational::{NonZero, Rational},
};

/// defines all numeric types
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum Numeric {
    Float(Float),
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

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Float(pub(crate) f64);

impl Numeric {
    pub fn add_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.add_num(n),
            (N::Float(f1), N::Float(f2)) => (f1 + f2).into(),
            (N::Float(f), N::Rational(r)) | (N::Rational(r), N::Float(f)) => {
                (r.as_float() + f).into()
            }
            (N::Rational(r1), N::Rational(r2)) => r1.convert_add(r2),
        }
    }

    pub fn sub_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) => inf.sub_num(n),
            (n, N::Infinity(inf)) => n.sub_inf(inf),
            (N::Float(f1), N::Float(f2)) => (f1 - f2).into(),
            (N::Float(f), N::Rational(r)) | (N::Rational(r), N::Float(f)) => {
                (r.as_float() - f).into()
            }
            (N::Rational(r1), N::Rational(r2)) => r1.convert_sub(r2),
        }
    }

    pub fn mul_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.mul_num(n),
            (N::Float(f1), N::Float(f2)) => (f1 * f2).into(),
            (N::Float(f), N::Rational(r)) | (N::Rational(r), N::Float(f)) => {
                (r.as_float() * f).into()
            }
            (N::Rational(r1), N::Rational(r2)) => r1.convert_mul(r2),
        }
    }

    pub fn div_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) => inf.div_num(n),
            (n, N::Infinity(inf)) => n.div_inf(inf),
            (N::Float(f1), N::Float(f2)) => f1 / f2,
            (N::Float(f), N::Rational(r)) | (N::Rational(r), N::Float(f)) => r.as_float() / f,
            (N::Rational(r1), N::Rational(r2)) => r1.convert_div(r2),
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
            N::Float(_) => inf.into(),
            N::Infinity(i) => i.sub_num(inf.into()),
            N::Undefined(_) => self,
        }
    }

    fn div_inf(self, inf: Infinity) -> Numeric {
        use Numeric as N;
        match self {
            N::Rational(_) => Rational::zero().into(),
            N::Float(_) => Float(0f64).into(),
            N::Infinity(i) => i.div_num(inf.into()),
            N::Undefined(_) => self,
        }
    }
}

impl Described for Numeric {
    fn desc(&self) -> pattern::Pattern {
        pattern::Pattern::Itm(match self {
            Numeric::Float(_) => Item::Float,
            Numeric::Rational(r) => r.desc().to_item(),
            Numeric::Infinity(i) => i.desc().to_item(),
            Numeric::Undefined(_) => Item::Undef,
        })
    }
}

impl Float {
    pub fn new<F: Into<f64>>(f: F) -> Self {
        let f: f64 = f.into();
        debug_assert!(!f.is_nan());
        Float(f)
    }

    pub fn sign(&self) -> Sign {
        if self.0.is_sign_negative() {
            Sign::Negative
        } else {
            Sign::Positive
        }
    }

    pub const fn desc(&self) -> pattern::Pattern {
        unimplemented!()
    }
}

impl Described for Float {
    fn desc(&self) -> pattern::Pattern {
        Item::Float.into()
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
            N::Rational(_) | N::Float(_) => self.num(),
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
            N::Rational(_) | N::Float(_) | N::Infinity(_) => self.into(),
            N::Undefined(_) => n,
        }
    }

    pub fn mul_num(self, n: Numeric) -> Numeric {
        use Numeric as N;
        match n {
            N::Rational(r) => Infinity::new(self.sign * r.sign).into(),
            N::Float(f) => Infinity::new(self.sign * f.sign()).into(),
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

impl Described for Undefined {
    fn desc(&self) -> pattern::Pattern {
        Item::Undef.into()
    }
}

impl std::hash::Hash for Float {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let bits = (self.0 + 0.0f64).to_bits();
        state.write_u64(bits)
    }
}

impl ops::Add for Float {
    type Output = Float;

    fn add(self, rhs: Self) -> Self::Output {
        Float(self.0 + rhs.0)
    }
}

impl ops::Sub for Float {
    type Output = Float;

    fn sub(self, rhs: Self) -> Self::Output {
        Float(self.0 - rhs.0)
    }
}

impl ops::Mul for Float {
    type Output = Float;

    fn mul(self, rhs: Self) -> Self::Output {
        Float(self.0 * rhs.0)
    }
}

impl ops::Div for Float {
    type Output = Numeric;

    fn div(self, rhs: Self) -> Self::Output {
        let f = self.0 / rhs.0;

        if f.is_nan() {
            Undefined.into()
        } else {
            Float(f).into()
        }
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
            (N::Undefined(_), N::Undefined(_)) => Ordering::Equal,
            // undefined last
            (_, N::Undefined(_)) => Ordering::Greater,
            (N::Undefined(_), _) => Ordering::Less,

            (N::Infinity(i1), N::Infinity(i2)) => i1.cmp(i2),
            (_, N::Infinity(inf)) => match inf.sign {
                Sign::Positive => Ordering::Less,
                Sign::Negative => Ordering::Greater,
            },
            (N::Infinity(inf), _) => match inf.sign {
                Sign::Positive => Ordering::Greater,
                Sign::Negative => Ordering::Less,
            },

            (N::Rational(r1), N::Rational(r2)) => r1.cmp(r2),
            (N::Float(f1), N::Float(f2)) => f1.cmp(f2),
            (N::Rational(r), N::Float(f)) => r.as_float().cmp(f),
            (N::Float(f), N::Rational(r)) => f.cmp(&r.as_float()),
        }
    }
}

impl Eq for Float {}
impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Float {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0 < other.0 {
            Ordering::Less
        } else if self.0 > other.0 {
            Ordering::Greater
        } else {
            Ordering::Equal
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

impl Described for Infinity {
    fn desc(&self) -> pattern::Pattern {
        self.sign.desc().union(Item::Inf).into()
    }
}

impl CalcursType for Undefined {
    #[inline(always)]
    fn base(self) -> Base {
        Numeric::Undefined(self).base()
    }
}

impl CalcursType for Float {
    fn base(self) -> Base {
        Numeric::Float(self).base()
    }
}

impl From<Rational> for Numeric {
    fn from(value: Rational) -> Self {
        Numeric::Rational(value)
    }
}

impl From<Float> for Numeric {
    fn from(value: Float) -> Self {
        Numeric::Float(value)
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

impl From<f64> for Numeric {
    fn from(value: f64) -> Self {
        if value.is_nan() {
            Undefined.into()
        } else if value.is_infinite() {
            if value.is_sign_negative() {
                Infinity::neg().into()
            } else {
                Infinity::pos().into()
            }
        } else {
            Float(value).into()
        }
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
            Numeric::Float(v) => write!(f, "{:?}", v.0),
            Numeric::Infinity(v) => write!(f, "{v}"),
            Numeric::Undefined(v) => write!(f, "{v}"),
        }
    }
}

impl fmt::Debug for Numeric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Numeric::Rational(v) => write!(f, "{:?}", v),
            Numeric::Float(v) => write!(f, "F({:?})", v.0),
            Numeric::Infinity(v) => write!(f, "{}", v),
            Numeric::Undefined(v) => write!(f, "{}", v),
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
