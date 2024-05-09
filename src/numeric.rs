
use std::{cmp::Ordering, fmt, ops};
//use std::os::watchos;

use num::Zero;

use crate::{
    expression::{CalcursType, Expr},
    pattern::Item,
};

/// defines all numeric types
//#[derive(Clone, Copy, Hash, PartialEq, Eq)]
//pub enum Numeric {
//    Float(Float),
//    Rational(Rational),
//    Infinity(Infinity),
//
//    /// only used if the result is provenly undefined, e.g 0 / 0
//    ///
//    /// not the same as [f64::NAN]
//    Undefined,
//}

// TODO: move to [Expr]
//impl Numeric {
//    pub fn checked_pow_num(self, n: Numeric) -> Option<Numeric> {
//        match (self, n) {
//            (Numeric::Rational(r1), Numeric::Rational(r2)) => {
//                let mut exp = r2.try_apply_expon()?;
//                let mut base = r1.try_apply_expon()?;
//                panic!();
//                //exp.reduce_frac();
//                //base.reduce_frac();
//
//                let mut root = base.numer;
//                if exp.denom() != 1 {
//                    root = base.numer.nth_root(exp.denom().try_into().ok()?);
//                    if root * root != base.numer {
//                        return None;
//                    };
//                }
//                let numer = root.pow(exp.numer.try_into().ok()?);
//
//                let mut root = base.denom();
//                if exp.denom() != 1 {
//                    root = base.denom().nth_root(exp.denom().try_into().ok()?);
//                    if root * root != base.denom() {
//                        return None;
//                    };
//                }
//                let denom = root.pow(exp.numer.try_into().ok()?);
//
//                let sign = if exp.numer() % 2 == 0 {
//                    Sign::Positive
//                } else {
//                    base.sign
//                };
//
//                Some(Rational::reduced(sign, numer, UNonZero::new(denom).unwrap(), 0).num())
//            }
//            _ => None,
//        }
//    }
//
//    fn sub_inf(self, inf: Infinity) -> Numeric {
//        use Numeric as N;
//        match self {
//            N::Rational(_) => inf.into(),
//            N::Float(_) => inf.into(),
//            N::Infinity(i) => i.sub_num(inf.into()),
//            N::Undefined => self,
//        }
//    }
//
//    fn div_inf(self, inf: Infinity) -> Numeric {
//        use Numeric as N;
//        match self {
//            N::Rational(_) => Rational::zero().into(),
//            N::Float(_) => Float(0f64).into(),
//            N::Infinity(i) => i.div_num(inf.into()),
//            N::Undefined => self,
//        }
//    }
//}

//impl Ord for Numeric {
//    fn cmp(&self, other: &Self) -> Ordering {
//        use Numeric as N;
//        match (self, other) {
//            (N::Undefined, N::Undefined) => Ordering::Equal,
//            // undefined last
//            (_, N::Undefined) => Ordering::Greater,
//            (N::Undefined, _) => Ordering::Less,
//
//            (N::Infinity(i1), N::Infinity(i2)) => i1.cmp(i2),
//            (_, N::Infinity(inf)) => match inf.sign {
//                Sign::Positive => Ordering::Less,
//                Sign::Negative => Ordering::Greater,
//            },
//            (N::Infinity(inf), _) => match inf.sign {
//                Sign::Positive => Ordering::Greater,
//                Sign::Negative => Ordering::Less,
//            },
//
//            (N::Rational(r1), N::Rational(r2)) => r1.cmp(r2),
//            (N::Float(f1), N::Float(f2)) => f1.cmp(f2),
//            (N::Rational(r), N::Float(f)) => r.as_float().cmp(f),
//            (N::Float(f), N::Rational(r)) => f.cmp(&r.as_float()),
//        }
//    }
//}

//impl CalcursType for Numeric {
//    fn desc(&self) -> Item {
//        use Numeric as N;
//        match self {
//            N::Float(_) => Item::Float,
//            N::Rational(r) => r.desc(),
//            N::Infinity(i) => i.desc(),
//            N::Undefined => Item::Undef,
//        }
//    }
//
//    //#[inline(always)]
//    //fn base(self) -> Expr {
//    //    use Expr as B;
//    //    use Numeric as N;
//    //    match self {
//    //        N::Float(f) => B::Float(f),
//    //        N::Rational(r) => B::Rational(r),
//    //        N::Infinity(i) => B::Infinity(i),
//    //        N::Undefined => B::Undefined,
//    //    }
//    //}
//
//    //fn free_of(&self, other: &Base) -> bool {
//    //    match self {
//    //        Numeric::Float(x) => x.free_of(other),
//    //        Numeric::Rational(x) => x.free_of(other),
//    //        Numeric::Infinity(x) => x.free_of(other),
//    //        Numeric::Undefined => &Base::Undefined != other,
//    //    }
//    //}
//
//    //fn operands(&mut self) -> Vec<&mut Base> {
//    //    panic!("should have been called by parent")
//    //}
//}

//impl From<Numeric> for Expr {
//    fn from(value: Numeric) -> Self {
//        use Expr as B;
//        use Numeric as N;
//        match value {
//            N::Float(f) => B::Float(f),
//            N::Rational(r) => B::Rational(r),
//            N::Infinity(i) => B::Infinity(i),
//            N::Undefined => B::Undefined,
//        }
//    }
//}
//impl TryFrom<Expr> for Numeric {
//    type Error = Expr;
//
//    fn try_from(value: Expr) -> Result<Self, Self::Error> {
//        use Expr as B;
//        use Numeric as N;
//        match value {
//            B::Float(f) => Ok(N::Float(f)),
//            B::Rational(r) => Ok(N::Rational(r)),
//            B::Infinity(i) => Ok(N::Infinity(i)),
//            B::Undefined => Ok(N::Undefined),
//
//            _ => Err(value),
//        }
//    }
//}
///// avoids copy of potential large expression
//impl TryFrom<&Expr> for Numeric {
//    type Error = ();
//
//    fn try_from(value: &Expr) -> Result<Self, Self::Error> {
//        use Expr as B;
//        use Numeric as N;
//        match value {
//            B::Float(f) => Ok(N::Float(*f)),
//            B::Rational(r) => Ok(N::Rational(*r)),
//            B::Infinity(i) => Ok(N::Infinity(*i)),
//            B::Undefined => Ok(N::Undefined),
//
//            _ => Err(()),
//        }
//    }
//}

//TODO: turn to rational if integer?
#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Float(pub(crate) f64);

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

    pub fn pow(self, rhs: Float) -> Float {
        Float(self.0.powf(rhs.0))
    }
}

impl std::hash::Hash for Float {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let bits = (self.0 + 0.0f64).to_bits();
        state.write_u64(bits)
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

impl CalcursType for Float {
    fn desc(&self) -> Item {
        let mut desc = Item::Float;

        if self.0.is_zero() {
            desc |= Item::Zero;
        }

        desc
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Infinity {
    pub(crate) sign: Sign,
}

impl Infinity {
    #[inline]
    pub fn new(dir: Sign) -> Self {
        Self { sign: dir }
    }

    //#[inline]
    //pub fn num(self) -> Numeric {
    //    self.into()
    //}

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

    //pub fn add_num(self, n: Numeric) -> Numeric {
    //    use Numeric as N;
    //    match n {
    //        N::Rational(_) | N::Float(_) => self.num(),
    //        N::Infinity(inf) => match self.sign == inf.sign {
    //            true => self.num(),
    //            false => Numeric::Undefined,
    //        },
    //        N::Undefined => n,
    //    }
    //}

    //pub fn sub_num(self, n: Numeric) -> Numeric {
    //    use Numeric as N;
    //    match n {
    //        N::Rational(_) | N::Float(_) | N::Infinity(_) => self.into(),
    //        N::Undefined => n,
    //    }
    //}

    //pub fn mul_num(self, n: Numeric) -> Numeric {
    //    use Numeric as N;
    //    match n {
    //        N::Rational(r) => Infinity::new(self.sign * r.sign).into(),
    //        N::Float(f) => Infinity::new(self.sign * f.sign()).into(),
    //        N::Infinity(inf) => Infinity::new(self.sign * inf.sign).into(),
    //        N::Undefined => n,
    //    }
    //}

    //pub fn div_num(self, n: Numeric) -> Numeric {
    //    self.mul_num(n)
    //}
}

impl CalcursType for Infinity {
    fn desc(&self) -> Item {
        self.sign.desc().union(Item::Inf)
    }
}

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

// OPERATOR IMPLS

//impl ops::Add for Numeric {
//    type Output = Numeric;
//
//    fn add(self, rhs: Self) -> Self::Output {
//        use Numeric as N;
//        match (self, rhs) {
//            (N::Undefined, _) | (_, N::Undefined) => Numeric::Undefined,
//            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.add_num(n),
//            (N::Float(f1), N::Float(f2)) => (f1 + f2).into(),
//            (N::Float(f), N::Rational(r)) | (N::Rational(r), N::Float(f)) => {
//                (r.as_float() + f).into()
//            }
//            (N::Rational(r1), N::Rational(r2)) => r1.convert_add(r2),
//        }
//    }
//}
//impl ops::AddAssign for Numeric {
//    fn add_assign(&mut self, rhs: Self) {
//        *self = *self + rhs;
//    }
//}
//impl ops::Sub for Numeric {
//    type Output = Numeric;
//
//    fn sub(self, rhs: Self) -> Self::Output {
//        use Numeric as N;
//        match (self, rhs) {
//            (N::Undefined, _) | (_, N::Undefined) => Numeric::Undefined,
//            (N::Infinity(inf), n) => inf.sub_num(n),
//            (n, N::Infinity(inf)) => n.sub_inf(inf),
//            (N::Float(f1), N::Float(f2)) => (f1 - f2).into(),
//            (N::Float(f), N::Rational(r)) | (N::Rational(r), N::Float(f)) => {
//                (r.as_float() - f).into()
//            }
//            (N::Rational(r1), N::Rational(r2)) => r1.convert_sub(r2),
//        }
//    }
//}
//impl ops::SubAssign for Numeric {
//    fn sub_assign(&mut self, rhs: Self) {
//        *self = *self - rhs;
//    }
//}
//impl ops::Mul for Numeric {
//    type Output = Numeric;
//
//    fn mul(self, rhs: Self) -> Self::Output {
//        use Numeric as N;
//        match (self, rhs) {
//            (N::Undefined, _) | (_, N::Undefined) => Numeric::Undefined,
//            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.mul_num(n),
//            (N::Float(f1), N::Float(f2)) => (f1 * f2).into(),
//            (N::Float(f), N::Rational(r)) | (N::Rational(r), N::Float(f)) => {
//                (r.as_float() * f).into()
//            }
//            (N::Rational(r1), N::Rational(r2)) => r1.convert_mul(r2),
//        }
//    }
//}
//impl ops::MulAssign for Numeric {
//    fn mul_assign(&mut self, rhs: Self) {
//        *self = *self * rhs;
//    }
//}
//impl ops::Div for Numeric {
//    type Output = Numeric;
//
//    fn div(self, rhs: Self) -> Self::Output {
//        use Numeric as N;
//        match (self, rhs) {
//            (N::Undefined, _) | (_, N::Undefined) => Numeric::Undefined,
//            (N::Infinity(inf), n) => inf.div_num(n),
//            (n, N::Infinity(inf)) => n.div_inf(inf),
//            (N::Float(f1), N::Float(f2)) => f1 / f2,
//            (N::Float(f), N::Rational(r)) | (N::Rational(r), N::Float(f)) => r.as_float() / f,
//            (N::Rational(r1), N::Rational(r2)) => r1.convert_div(r2),
//        }
//    }
//}
//impl ops::DivAssign for Numeric {
//    fn div_assign(&mut self, rhs: Self) {
//        *self = *self / rhs;
//    }
//}

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
            Float(f).into()
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

//impl PartialOrd for Numeric {
//    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//        Some(self.cmp(other))
//    }
//}

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

// FROM IMPLS

//impl From<Rational> for Numeric {
//    fn from(value: Rational) -> Self {
//        Numeric::Rational(value)
//    }
//}
//impl From<Float> for Numeric {
//    fn from(value: Float) -> Self {
//        Numeric::Float(value)
//    }
//}
//impl From<Infinity> for Numeric {
//    fn from(value: Infinity) -> Self {
//        Numeric::Infinity(value)
//    }
//}
//impl From<f64> for Numeric {
//    fn from(value: f64) -> Self {
//        if value.is_nan() {
//            Numeric::Undefined
//        } else if value.is_infinite() {
//            if value.is_sign_negative() {
//                Infinity::neg().into()
//            } else {
//                Infinity::pos().into()
//            }
//        } else {
//            Float(value).into()
//        }
//    }
//}

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

// DISPLAY IMPLS

//impl fmt::Display for Numeric {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        match self {
//            Numeric::Rational(v) => write!(f, "{v}"),
//            Numeric::Float(v) => write!(f, "{:?}", v.0),
//            Numeric::Infinity(v) => write!(f, "{v}"),
//            Numeric::Undefined => write!(f, "undefined"),
//        }
//    }
//}
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
//impl fmt::Debug for Numeric {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        match self {
//            Numeric::Rational(v) => write!(f, "{:?}", v),
//            Numeric::Float(v) => write!(f, "F({:?})", v.0),
//            Numeric::Infinity(v) => write!(f, "{}", v),
//            Numeric::Undefined => write!(f, "{}", Numeric::Undefined),
//        }
//    }
//}
