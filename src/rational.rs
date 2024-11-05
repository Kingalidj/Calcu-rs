use std::{cmp::Ordering, ops, str::FromStr};

use calcurs_macros::arith_ops;
use derive_more::{
    Add, AddAssign, Debug, Display, Div, DivAssign, From, Into, Mul, MulAssign, Sub, SubAssign,
};
use serde::{Deserialize, Serialize};

use malachite::{
    self as mal,
    num::{arithmetic::traits as marith, conversion::traits as mconv},
};

#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display, From, Into)]
#[arith_ops(ref, self.0)]
#[from(i32, u32, i64, u64)]
#[into(mal::Rational)]
#[debug("{}", self.0)]
#[repr(transparent)]
pub struct Int(mal::Integer);

impl num::Zero for Int {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

impl ops::Rem<&Int> for Int {
    type Output = Self;

    fn rem(self, rhs: &Self) -> Self::Output {
        Int(self.0.rem(&rhs.0))
    }
}

impl ops::Rem for Int {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Int(self.0.rem(rhs.0))
    }
}

impl num::Num for Int {
    type FromStrRadixErr = ();

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let int = mconv::FromStringBase::from_string_base(radix as u8, str).ok_or(())?;
        Ok(Int(int))
    }
}

impl num::Integer for Int {
    fn div_floor(&self, other: &Self) -> Self {
        Int(marith::DivMod::div_mod(&self.0, &other.0).0)
    }

    fn mod_floor(&self, other: &Self) -> Self {
        Int(marith::Mod::mod_op(&self.0, &other.0))
    }

    fn gcd(&self, other: &Self) -> Self {
        Int(marith::Gcd::gcd(self.0.unsigned_abs_ref(), other.0.unsigned_abs_ref()).into())
    }

    fn lcm(&self, other: &Self) -> Self {
        Int(marith::Lcm::lcm(self.0.unsigned_abs_ref(), other.0.unsigned_abs_ref()).into())
    }

    fn is_multiple_of(&self, other: &Self) -> bool {
        self.mod_floor(other).is_zero()
    }

    fn is_even(&self) -> bool {
        marith::Parity::even(&self.0)
    }

    fn is_odd(&self) -> bool {
        marith::Parity::odd(&self.0)
    }

    fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (quot, rem) = marith::DivRem::div_rem(&self.0, &other.0);
        (Int(quot), Int(rem))
    }
}

impl num::FromPrimitive for Int {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Int(n.into()))
    }

    fn from_i128(n: i128) -> Option<Self> {
        Some(Int(n.into()))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Int(n.into()))
    }

    fn from_u128(n: u128) -> Option<Self> {
        Some(Int(n.into()))
    }
}

impl num_integer::Roots for Int {
    fn nth_root(&self, n: u32) -> Self {
        if self.is_pos() {
            Int(marith::FloorRoot::floor_root(&self.0, n.into()))
        } else {
            Int(marith::CeilingRoot::ceiling_root(&self.0, n.into()))
        }
    }
}

impl Int {
    pub const MINUS_TWO: Int = Int(mal::Integer::const_from_signed(-2));
    pub const MINUS_ONE: Int = Int(mal::Integer::const_from_signed(-1));
    pub const ZERO: Int = Int(mal::Integer::const_from_signed(0));
    pub const ONE: Int = Int(mal::Integer::const_from_signed(1));
    pub const TWO: Int = Int(mal::Integer::const_from_signed(2));

    pub fn binomial_coeff(n: &Int, k: &Int) -> Int {
        Self(marith::BinomialCoefficient::binomial_coefficient(
            &n.0, &k.0,
        ))
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
    pub fn is_pos(&self) -> bool {
        self > &Int::ZERO
    }
    pub fn is_neg(&self) -> bool {
        self < &Int::ZERO
    }
    pub fn is_even(&self) -> bool {
        marith::Parity::even(&self.0)
    }
    pub fn is_odd(&self) -> bool {
        marith::Parity::odd(&self.0)
    }
    pub fn abs(&self) -> Self {
        Self(self.0.unsigned_abs_ref().into())
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let n = marith::Gcd::gcd(self.0.unsigned_abs_ref(), other.0.unsigned_abs_ref());
        Self(n.into())
    }

    pub fn prime_factorize(&self) {
        //num_prime::nt_funcs::factorize(self.0);
        todo!()
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

#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display, From)]
#[arith_ops(ref, self.0)]
#[from(i64, i32, u64, u32, Int)]
#[debug("{}", self.0)]
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
        let sign = match self.is_neg() {
            true => Int::MINUS_ONE,
            false => Int::ONE,
        };
        sign * Int(mal::Integer::from(self.0.numerator_ref().clone()))
    }

    pub fn to_int(&self) -> Option<Int> {
        Some(Int(mal::Integer::try_from(self.0.clone()).ok()?))
    }

    pub fn denom(&self) -> Int {
        Int(mal::Integer::from(self.0.denominator_ref().clone()))
    }

    pub fn is_min_two(&self) -> bool {
        self == &Self::MINUS_TWO
    }
    pub fn is_min_one(&self) -> bool {
        self == &Self::MINUS_ONE
    }
    pub fn is_zero(&self) -> bool {
        self == &Self::ZERO
    }
    pub fn is_one(&self) -> bool {
        self == &Self::ONE
    }
    pub fn is_two(&self) -> bool {
        self == &Self::TWO
    }
    pub fn is_pos(&self) -> bool {
        matches!(marith::Sign::sign(&self.0), Ordering::Greater)
    }
    pub fn is_neg(&self) -> bool {
        matches!(marith::Sign::sign(&self.0), Ordering::Less)
    }
    pub fn is_int(&self) -> bool {
        mconv::IsInteger::is_integer(&self.0)
    }
    pub fn is_fraction(&self) -> bool {
        !self.is_int()
    }
    pub fn is_even(&self) -> bool {
        if self.is_int() {
            marith::Parity::even(self.0.numerator_ref())
        } else {
            false
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
        Self(marith::Abs::abs(self.0))
    }

    pub fn div_rem(&self) -> (Self, Self) {
        let denom = self.denom();
        let (num, den) = self.0.to_numerator_and_denominator();
        let (quot, rem) = marith::DivRem::div_rem(num, den);
        (
            Self(mal::Rational::from(quot)),
            (Self(mal::Rational::from(rem)) / Self::from(denom)),
        )
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
                marith::PowAssign::pow_assign(&mut self.0, exp);
                return (self, Rational::ZERO);
            } else {
                return (self, rhs);
            }
        }

        // ensure that the exponent is < 1
        // a^(b/c) -> ( b/c -> quot + rem ) -> a^quot * a^rem  // apply the quotient
        if rhs.0.numerator_ref() > rhs.0.denominator_ref() {
            let (num, den) = rhs.0.to_numerator_and_denominator();
            let (quot, rem) = marith::DivRem::div_rem(num, den);
            let rem_exp = Self(mal::Rational::from(rem));

            if let Ok(apply_exp) = u64::try_from(&quot) {
                marith::PowAssign::pow_assign(&mut self.0, apply_exp);
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

impl Serialize for Int {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}
impl<'de> Deserialize<'de> for Int {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Self(mal::Integer::from_str(&s).unwrap()))
    }
}
impl Serialize for Rational {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}
impl<'de> Deserialize<'de> for Rational {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Self(mal::Rational::from_str(&s).unwrap()))
    }
}

impl<I: Into<i64>> From<(I, I)> for Rational {
    fn from(value: (I, I)) -> Self {
        let value = (value.0.into(), value.1.into());
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
