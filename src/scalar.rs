use fmt::Display;
use ordered_float::Float as OrdFloat;
use std::{cmp::Ordering, fmt, ops};

use calcu_rs::{
    expression::{CalcursType, Expr},
    pattern::Item,
    rational::Rational,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scalar {
    Rational(Rational),
    Undefined,
}

impl From<Scalar> for Expr {
    fn from(value: Scalar) -> Self {
        match value {
            Scalar::Rational(r) => Expr::Rational(r),
            Scalar::Undefined => Expr::Undefined,
        }
    }
}

impl Scalar {
    pub fn desc(&self) -> Item {
        match self {
            Scalar::Rational(r) => r.desc(),
            Scalar::Undefined => Item::Undef,
        }
    }

    pub fn pow(self, exp: Self) -> Option<Self> {
        use Scalar as S;
        match (self, exp) {
            (S::Undefined, _) | (_, S::Undefined) => Some(S::Undefined),
            (S::Rational(r1), S::Rational(r2)) => {
                if r1.is_zero() && r2.is_zero() || r1.is_zero() && r2.is_neg() {
                    return Some(S::Undefined);
                }
                if r1.is_zero() {
                    return Some(S::Rational(Rational::zero()));
                }
                // TODO: return even if pow not fully applied
                let (pow, rem) = r1.pow(r2);
                if rem.is_one() {
                    Some(S::Rational(pow))
                } else {
                    None
                }
            }
        }
    }
}

macro_rules! impl_from_for_scalar {
    ($typ:ident) => {
        impl From<$typ> for Scalar {
            #[inline(always)]
            fn from(value: $typ) -> Scalar {
                Scalar::$typ(value.into())
            }
        }
    };
}

impl_from_for_scalar!(Rational);

impl ops::Add for Scalar {
    type Output = Scalar;

    fn add(self, rhs: Self) -> Self::Output {
        use Scalar as S;
        match (self, rhs) {
            (S::Undefined, _) | (_, S::Undefined) => S::Undefined,
            (S::Rational(r1), S::Rational(r2)) => S::Rational(r1 + r2),
        }
    }
}

impl ops::Sub for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Self::Output {
        use Scalar as S;
        match (self, rhs) {
            (S::Undefined, _) | (_, S::Undefined) => S::Undefined,
            (S::Rational(r1), S::Rational(r2)) => S::Rational(r1 - r2),
        }
    }
}
impl ops::Mul for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Self) -> Self::Output {
        use Scalar as S;
        match (self, rhs) {
            (S::Undefined, _) | (_, S::Undefined) => S::Undefined,
            (S::Rational(r1), S::Rational(r2)) => S::Rational(r1 * r2),
        }
    }
}
impl ops::Div for Scalar {
    type Output = Scalar;

    fn div(self, rhs: Self) -> Self::Output {
        use Scalar as S;
        match (self, rhs) {
            (S::Undefined, _) | (_, S::Undefined) => S::Undefined,
            (S::Rational(r1), S::Rational(r2)) => match r1 / r2 {
                None => S::Undefined,
                Some(quot) => S::Rational(quot),
            },
        }
    }
}

//TODO: turn to rational if integer?
pub type FloatType = ordered_float::OrderedFloat<f64>;
#[derive(Debug, Clone, PartialOrd, Ord, Eq, PartialEq, Copy, Hash)]
pub struct Float(pub(crate) FloatType);

impl Float {
    pub fn new<F: Into<f64>>(f: F) -> Self {
        Float(FloatType::from(f.into()))
    }
    //pub fn new<F: <f64>>(f: F) -> Self {
    //    let f: f64 = f.into();
    //    debug_assert!(!f.is_nan());
    //    Float(f)
    //}

    pub fn sign(&self) -> Sign {
        if self.0.is_sign_negative() {
            Sign::Negative
        } else {
            Sign::Positive
        }
    }

    pub fn pow(self, rhs: Float) -> Float {
        Float(self.0.powf(rhs.0))
    }
}

//impl CalcursType for Float {
//    fn desc(&self) -> Item {
//        let mut desc = Item::Float;
//
//        if self.0 == 0f64 {
//            desc |= Item::Zero;
//        }
//
//        desc
//    }
//}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Infinity {
    pub(crate) sign: Sign,
}

impl Infinity {
    #[inline]
    pub fn new(dir: Sign) -> Self {
        Self { sign: dir }
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
}

//impl CalcursType for Infinity {
//    fn desc(&self) -> Item {
//        self.sign.desc().union(Item::Inf)
//    }
//}

/// sign of a value. zero is regarded as positive
#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub enum Sign {
    Positive,
    Negative,
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

    pub const fn desc(&self) -> Item {
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
    type Output = Expr;

    fn div(self, rhs: Self) -> Self::Output {
        let f = self.0 / rhs.0;

        if f.is_nan() {
            Expr::Undefined
        } else {
            //Float(f).into()
            todo!()
        }
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

pub trait Signed {
    fn sign(&self) -> Sign;
}

impl<I: num::Signed> Signed for I {
    fn sign(&self) -> Sign {
        match num::Signed::is_negative(self) {
            true => Sign::Negative,
            false => Sign::Positive,
        }
    }
}

impl<F: Into<f64>> From<F> for Float {
    fn from(value: F) -> Self {
        let val = value.into();
        Self::new(val)
    }
}

impl Display for Infinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}oo", self.sign)
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

impl Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Scalar as S;
        match self {
            S::Rational(r) => write!(f, "{}", r),
            S::Undefined => write!(f, "{}", Expr::Undefined),
        }
    }
}
