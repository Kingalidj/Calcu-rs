use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter},
    ops,
};

use malachite::{
    self as mal,
    num::{
        arithmetic::traits::{Abs, Gcd, DivRem, PowAssign, Sign as MalSign},
        conversion::traits::IsInteger,
    },
};
use ref_cast::RefCast;

#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, RefCast)]
#[repr(transparent)]
pub struct Int(mal::Integer);

impl Int {
    pub const ZERO: Int = Int(mal::Integer::const_from_signed(0));
    pub const ONE: Int = Int(mal::Integer::const_from_signed(1));
    pub const TWO: Int = Int(mal::Integer::const_from_signed(2));

    pub fn binomial_coeff(n: &Int, k: &Int) -> Int {
        Self(mal::num::arithmetic::traits::BinomialCoefficient::binomial_coefficient(&n.0, &k.0))
    }

    pub fn range_inclusive(start: Self, stop: Self) -> num::iter::RangeInclusive<Int> {
        num::iter::range_inclusive(start, stop)
    }

    pub fn is_one(&self) -> bool {
        self == &Int::ONE
    }

    pub fn is_zero(&self) -> bool {
        self == &Int::ZERO
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let n = self.0.unsigned_abs_ref().gcd(other.0.unsigned_abs_ref());
        Self(n.into())
    }
}

impl fmt::Debug for Int {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Display for Int {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl num::One for Int {
    fn one() -> Self {
        Int::ONE
    }
}

impl num::ToPrimitive for Int {
    fn to_i64(&self) -> Option<i64> {
        i64::try_from(&self.0).ok()
    }

    fn to_u64(&self) -> Option<u64> {
        u64::try_from(&self.0).ok()
    }

    fn to_u128(&self) -> Option<u128> {
        u128::try_from(&self.0).ok()
    }

    fn to_i128(&self) -> Option<i128> {
        i128::try_from(&self.0).ok()
    }
}

impl TryFrom<Rational> for Int {
    type Error = ();
    fn try_from(value: Rational) -> Result<Self, Self::Error> {
        match value.to_int() {
            Some(int) => Ok(int),
            None => Err(()),
        }
    }
}
impl From<Int> for Rational {
    fn from(value: Int) -> Self {
        Rational::from(value.0)
    }
}
impl From<u64> for Int {
    fn from(value: u64) -> Self {
        Self(mal::Integer::from(value))
    }
}

impl ops::Add for Int {
    type Output = Int;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl ops::AddAssign for Int {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() + rhs.0;
    }
}
impl ops::AddAssign<&Int> for Int {
    fn add_assign(&mut self, rhs: &Self) {
        self.0 = self.0.clone() + &rhs.0;
    }
}
impl ops::Add<&Int> for Int {
    type Output = Int;
    fn add(self, rhs: &Self) -> Self::Output {
        Self(self.0 + &rhs.0)
    }
}
impl ops::Sub for Int {
    type Output = Int;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}
impl ops::SubAssign for Int {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() - rhs.0;
    }
}
impl ops::SubAssign<&Int> for Int {
    fn sub_assign(&mut self, rhs: &Self) {
        self.0 = self.0.clone() - &rhs.0;
    }
}
impl ops::Sub<&Int> for Int {
    type Output = Int;
    fn sub(self, rhs: &Self) -> Self::Output {
        Self(self.0 - &rhs.0)
    }
}
impl ops::Mul for Int {
    type Output = Int;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}
impl ops::MulAssign for Int {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() * rhs.0;
    }
}
impl ops::MulAssign<&Int> for Int {
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 = self.0.clone() * &rhs.0;
    }
}
impl ops::Mul<&Int> for Int {
    type Output = Int;
    fn mul(self, rhs: &Self) -> Self::Output {
        Self(self.0 * &rhs.0)
    }
}
impl ops::Div for Int {
    type Output = Int;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}
impl ops::Div<&Int> for Int {
    type Output = Int;
    fn div(self, rhs: &Self) -> Self::Output {
        Self(self.0 / &rhs.0)
    }
}
impl ops::DivAssign for Int {
    fn div_assign(&mut self, rhs: Self) {
        self.0 = self.0.clone() / rhs.0;
    }
}
impl ops::DivAssign<&Int> for Int {
    fn div_assign(&mut self, rhs: &Self) {
        self.0 = self.0.clone() / &rhs.0;
    }
}

#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rational(pub(crate) mal::Rational);

impl Rational {
    pub const MINUS_TWO: Self = Rational(mal::Rational::const_from_signed(-2));
    pub const MINUS_ONE: Self = Rational(mal::Rational::const_from_signed(-1));
    pub const ZERO: Self = Rational(mal::Rational::const_from_signed(0));
    pub const ONE: Self = Rational(mal::Rational::const_from_signed(1));
    pub const TWO: Self = Rational(mal::Rational::const_from_signed(2));

    pub const fn new_int(n: i64) -> Self {
        Rational(mal::Rational::const_from_signed(n))
    }

    pub fn numer(&self) -> Int {
        Int(mal::Integer::from(self.0.numerator_ref().clone()))
        //UInt::ref_cast(self.0.numerator_ref())
    }

    pub fn to_int(&self) -> Option<Int> {
        Some(Int(mal::Integer::try_from(self.0.clone()).ok()?))
    }

    pub fn denom(&self) -> Int {
        Int(mal::Integer::from(self.0.denominator_ref().clone()))
    }

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

    pub fn div_rem(&self) -> (Self, Self) {
        let denom = self.denom();
        let (num, den) = self.0.to_numerator_and_denominator();
        let (quot, rem) = num.div_rem(den);
        (Self(mal::Rational::from(quot)), (Self(mal::Rational::from(rem)) / Self::from(denom)).unwrap())

    }

    /// will calculate [self] to the power of an integer number.
    ///
    /// if the exponent is (a/b) non-int: we calculate the power to the int quotient of a/b
    /// and return the remainder: (self^quot, rem).
    ///
    /// n^(a/b) = n^(quot + rem) = n^(quot) * n^(rem) -> (n^quot, rem)
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

    pub fn pow_basic(self, rhs: Self) -> Option<Self> {
        let (pow, rest) = self.pow(rhs);
        if rest != Rational::ZERO {
            None
        } else {
            Some(pow)
        }
    }
}

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
            self.0 /= rhs.0;
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
            self.0 /= &rhs.0;
            Some(self)
        }
    }
}

impl From<u128> for Rational {
    fn from(value: u128) -> Self {
        Self(mal::Rational::from(value))
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
impl From<mal::Integer> for Rational {
    fn from(value: mal::Integer) -> Self {
        Self(mal::Rational::from(value))
    }
}
impl From<(u64, u64)> for Rational {
    fn from(value: (u64, u64)) -> Self {
        let n = mal::Integer::from(value.0);
        let d = mal::Integer::from(value.1);
        Self(mal::Rational::from_integers(n, d))
    }
}
impl From<(i64, i64)> for Rational {
    fn from(value: (i64, i64)) -> Self {
        let is_neg = (value.0 * value.1) < 0;
        let n = mal::Integer::from(value.0.unsigned_abs());
        let d = mal::Integer::from(value.1.unsigned_abs());
        let mut r = mal::Rational::from_integers(n, d);

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
