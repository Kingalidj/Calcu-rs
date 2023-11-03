use calcurs_macros::Procagate;

pub use crate::rational::Rational;
use crate::{
    base::Base,
    traits::{CalcursType, Num},
};

#[derive(
    Debug, Clone, Hash, PartialEq, PartialOrd, Ord, Eq, derive_more::Display, Copy, Default,
)]
pub enum Sign {
    #[display(fmt = "+")]
    Positive,
    #[display(fmt = "-")]
    Negative,
    #[display(fmt = "")]
    #[default]
    UnSigned,
    //TODO: remove Unsigned
}

impl Sign {
    pub fn from_sign<I: num::Signed + std::fmt::Debug>(v: I) -> Self {
        if v.is_negative() {
            Sign::Negative
        } else if v.is_positive() {
            Sign::Positive
        } else {
            Sign::UnSigned
        }
    }

    pub fn neg(&self) -> Self {
        use Sign as D;
        match self {
            D::Positive => D::Negative,
            D::Negative => D::Positive,
            D::UnSigned => D::UnSigned,
        }
    }

    pub fn is_pos(&self) -> bool {
        matches!(self, Sign::Positive)
    }

    pub fn is_neg(&self) -> bool {
        matches!(self, Sign::Negative)
    }

    pub fn is_unsign(&self) -> bool {
        matches!(self, Sign::UnSigned)
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

#[derive(
    Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, derive_more::Display, Default,
)]
#[display(fmt = "{sign}oo")]
pub struct Infinity {
    sign: Sign,
}

impl Num for Infinity {
    fn is_zero(&self) -> bool {
        false
    }

    fn is_one(&self) -> bool {
        false
    }

    fn is_neg_one(&self) -> bool {
        false
    }

    fn sign(&self) -> Sign {
        self.sign
    }
}

impl CalcursType for Infinity {
    #[inline]
    fn base(self) -> Base {
        Number::Infinity(self).base()
    }
}

impl Infinity {
    pub fn new(dir: Sign) -> Self {
        Self { sign: dir }
    }

    pub fn is_zero(&self) -> bool {
        false
    }

    pub fn pos() -> Self {
        Self {
            sign: Sign::Positive,
        }
    }

    pub fn neg() -> Self {
        Self {
            sign: Sign::Negative,
        }
    }

    pub fn add_num(self, n: Number) -> Number {
        use Number as N;
        match n {
            N::Rational(_) => self.into(),
            N::Infinity(inf) => match self.sign == inf.sign {
                true => self.into(),
                false => Undefined.into(),
            },
            N::Undefined(_) => n,
        }
    }

    pub fn sub_num(self, n: Number) -> Number {
        use Number as N;
        match n {
            N::Rational(_) | N::Infinity(_) => self.into(),
            N::Undefined(_) => n,
        }
    }

    pub fn mul_num(self, n: Number) -> Number {
        use Number as N;
        match n {
            N::Rational(r) => Infinity::new(self.sign * r.sign()).into(),
            N::Infinity(inf) => Infinity::new(self.sign * inf.sign).into(),
            N::Undefined(_) => n,
        }
    }

    pub fn div_num(self, n: Number) -> Number {
        self.mul_num(n)
    }
}

#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, derive_more::Display, Copy)]
pub struct Undefined;

impl Num for Undefined {
    fn is_zero(&self) -> bool {
        false
    }

    fn is_one(&self) -> bool {
        false
    }

    fn is_neg_one(&self) -> bool {
        false
    }

    fn sign(&self) -> Sign {
        Sign::UnSigned
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

#[derive(
    Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, derive_more::Display, Procagate,
)]
pub enum Number {
    Rational(Rational),

    Infinity(Infinity),
    Undefined(Undefined),
}

impl CalcursType for Number {
    #[inline]
    fn base(self) -> Base {
        Base::Number(self)
    }
}

impl Num for Number {
    fn is_zero(&self) -> bool {
        procagate_number!(self, v => { v.is_zero() })
    }

    fn is_one(&self) -> bool {
        procagate_number!(self, v => { v.is_one() })
    }

    fn is_neg_one(&self) -> bool {
        procagate_number!(self, v => { v.is_neg_one() })
    }

    fn sign(&self) -> Sign {
        procagate_number!(self, v => { v.sign() })
    }
}

impl Number {
    pub fn add_num(self, n: Number) -> Number {
        use Number as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.add_num(n),
            (N::Rational(r1), N::Rational(r2)) => r1.add_ratio(r2),
        }
    }

    pub fn sub_num(self, n: Number) -> Number {
        use Number as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) => inf.sub_num(n),
            (n, N::Infinity(inf)) => n.sub_inf(inf),
            (N::Rational(r1), N::Rational(r2)) => r1.sub_ratio(r2),
        }
    }

    pub fn mul_num(self, n: Number) -> Number {
        use Number as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) | (n, N::Infinity(inf)) => inf.mul_num(n),
            (N::Rational(r1), N::Rational(r2)) => r1.mul_ratio(r2),
        }
    }

    pub fn div_num(self, n: Number) -> Number {
        use Number as N;
        match (self, n) {
            (N::Undefined(_), _) | (_, N::Undefined(_)) => Undefined.into(),
            (N::Infinity(inf), n) => inf.div_num(n),
            (n, N::Infinity(inf)) => n.div_inf(inf),

            (N::Rational(r1), N::Rational(r2)) => r1.div_ratio(r2),
        }
    }

    fn sub_inf(self, inf: Infinity) -> Number {
        use Number as N;
        match self {
            N::Rational(_) => inf.into(),
            N::Infinity(i) => i.sub_num(inf.into()),
            N::Undefined(_) => self,
        }
    }

    fn div_inf(self, mut inf: Infinity) -> Number {
        use Number as N;
        match self {
            N::Rational(r) => {
                let sign = r.sign();
                inf.sign *= sign;
                inf.into()
            }
            N::Infinity(i) => i.div_num(inf.into()),
            N::Undefined(_) => self,
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
