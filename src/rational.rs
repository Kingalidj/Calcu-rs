use fmt::{Debug, Display};
use malachite::{
    self as mal,
    num::{
        arithmetic::traits::{Abs, DivRem, PowAssign, Sign as MalSign},
        conversion::traits::IsInteger,
    },
};
use std::cmp::Ordering;
use std::fmt::Formatter;
use std::{fmt, ops};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rational(pub(crate) mal::Rational);

impl Rational {
    pub const MINUS_ONE: Self = Rational(mal::Rational::const_from_signed(-1));
    pub const ZERO: Self = Rational(mal::Rational::const_from_signed(0));
    pub const ONE: Self = Rational(mal::Rational::const_from_signed(1));
    pub const TWO: Self = Rational(mal::Rational::const_from_signed(2));

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        matches!(self.0.sign(), Ordering::Equal)
    }
    pub fn is_one(&self) -> bool {
        self == &Rational::ONE
    }
    #[inline(always)]
    pub fn is_pos(&self) -> bool {
        matches!(self.0.sign(), Ordering::Greater)
    }
    #[inline(always)]
    pub fn is_neg(&self) -> bool {
        matches!(self.0.sign(), Ordering::Less)
    }

    #[inline(always)]
    pub fn is_int(&self) -> bool {
        self.0.is_integer()
    }

    #[inline(always)]
    pub fn try_into_int(&self) -> Option<i64> {
        if self.is_int() {
            let n = self.0.numerator_ref().clone();
            i64::try_from(&n).ok()
        } else {
            None
        }
    }

    /// none if [self] is zero
    #[inline(always)]
    pub fn inverse(self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            let is_neg = self.is_neg();
            let (num, denom) = self.0.to_numerator_and_denominator();
            let mut r = mal::Rational::from_naturals(denom, num);
            // num and denom are unsigned
            if is_neg {
                r *= Rational::MINUS_ONE.0;
            }
            Some(Self(r))
        }
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// will calculate [self] to the power of an integer number.
    ///
    /// if the exponent is (a/b) non-int: we calculate the power to the int quotient of a/b
    /// and return the remainder: (self^quot, rem).
    ///
    /// n^(a/b) = n^(quot + rest) = n^(quot) * n^(rest)
    ///
    /// quot: Int, rest: Fraction, n^(quot): Rational
    ///
    /// returns the input if calculation was not possible
    pub fn pow(mut self, mut rhs: Self) -> (Self, Self) {
        if self.is_zero() && rhs.is_zero() {
            panic!("0^0");
        }

        if rhs.is_zero() {
            return (Rational::ONE, Rational::ZERO);
        }

        // inverse if exponent is negative
        if rhs.is_neg() {
            self = self.inverse().unwrap();
            rhs = rhs.abs();
        }

        debug_assert!(rhs.is_pos());

        if rhs.is_int() {
            let exp = rhs.0.numerator_ref();
            if let Ok(exp) = u64::try_from(exp) {
                self.0.pow_assign(exp);
                return (self, Rational::ZERO);
            } else {
                return (self, rhs);
            }
        }

        // ensure that the exponent is < 1
        // a^(b/c) -> ( b/c -> quot + rem ) -> a^quot * a^rem  // apply the quotient
        if rhs.0.numerator_ref() > rhs.0.denominator_ref() {
            let (num, den) = rhs.0.to_numerator_and_denominator();
            let (quot, rem) = num.div_rem(den);
            let rem_exp = Self(mal::Rational::from(rem));

            if let Ok(apply_exp) = u64::try_from(&quot) {
                self.0.pow_assign(apply_exp);
                return (self, rem_exp);
            }
        }

        // no change
        (self, rhs)
    }
}

const NAT_ZERO: mal::Natural = mal::Natural::const_from(0);
const NAT_ONE: mal::Natural = mal::Natural::const_from(1);

impl ops::Add for Rational {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        self.0 += rhs.0;
        self
    }
}
impl ops::Add<&Rational> for Rational {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.0 += &rhs.0;
        self
    }
}
impl ops::AddAssign for Rational {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl ops::AddAssign<&Rational> for Rational {
    fn add_assign(&mut self, rhs: &Self) {
        self.0 += &rhs.0;
    }
}
impl ops::Sub for Rational {
    type Output = Self;

    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self.0 -= rhs.0;
        self
    }
}
impl ops::Sub<&Rational> for Rational {
    type Output = Self;

    #[inline(always)]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.0 -= &rhs.0;
        self
    }
}
impl ops::SubAssign for Rational {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl ops::SubAssign<&Rational> for Rational {
    fn sub_assign(&mut self, rhs: &Self) {
        self.0 -= &rhs.0;
    }
}
impl ops::Mul for Rational {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self.0 *= rhs.0;
        self
    }
}
impl ops::Mul<&Rational> for Rational {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.0 *= &rhs.0;
        self
    }
}
impl ops::MulAssign for Rational {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}
impl ops::MulAssign<&Rational> for Rational {
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 *= &rhs.0;
    }
}
impl ops::Div for Rational {
    type Output = Option<Self>;

    #[inline(always)]
    fn div(mut self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            None
        } else {
            self.0 = self.0 / rhs.0;
            Some(self)
        }
    }
}
impl ops::Div<&Rational> for Rational {
    type Output = Option<Self>;

    #[inline(always)]
    fn div(mut self, rhs: &Self) -> Self::Output {
        if rhs.is_zero() {
            None
        } else {
            self.0 = self.0 / &rhs.0;
            Some(self)
        }
    }
}

impl From<i64> for Rational {
    fn from(value: i64) -> Self {
        Self(mal::Rational::from(value))
    }
}
impl From<i32> for Rational {
    fn from(value: i32) -> Self {
        Self(mal::Rational::from(value))
    }
}
impl From<(u64, u64)> for Rational {
    fn from(value: (u64, u64)) -> Self {
        let n = mal::Natural::from(value.0);
        let d = mal::Natural::from(value.1);
        Self(mal::Rational::from_naturals(n, d))
    }
}
impl From<(i64, i64)> for Rational {
    fn from(value: (i64, i64)) -> Self {
        let is_neg = (value.0 * value.1) < 0;
        let n = mal::Natural::from(value.0.unsigned_abs());
        let d = mal::Natural::from(value.1.unsigned_abs());
        let mut r = mal::Rational::from_naturals(n, d);

        if is_neg {
            r *= Rational::MINUS_ONE.0;
        }
        Self(r)
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Debug for Rational {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
