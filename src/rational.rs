use std::{
    cmp::Ordering,
    fmt::{self, Display},
    hash::Hash,
    ops,
};

use num::Integer;

use crate::{
    base::{Base, CalcursType},
    numeric::{Numeric, Sign},
};

/// Nonzero integer value
///
/// will panic if set to 0
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct NonZero<T> {
    pub(crate) non_zero_val: T,
}

impl<T: Integer + Copy> NonZero<T> {
    /// panics if arg is 0
    #[inline(always)]
    pub fn new(n: T) -> Self {
        debug_assert!(n != T::zero());
        NonZero { non_zero_val: n }
    }

    /// panics if arg is 0
    #[inline(always)]
    pub fn set(&mut self, n: T) {
        debug_assert!(n != T::zero());
        self.non_zero_val = n;
    }

    #[inline(always)]
    pub fn val(&self) -> T {
        self.non_zero_val
    }
}

impl<T: Integer + Copy> From<T> for NonZero<T> {
    fn from(value: T) -> Self {
        NonZero::new(value)
    }
}

impl<T: Integer + Copy> ops::Mul for NonZero<T> {
    type Output = NonZero<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        (self.val().mul(rhs.val())).into()
    }
}

impl<T: Integer + Copy + ops::MulAssign> ops::MulAssign for NonZero<T> {
    fn mul_assign(&mut self, rhs: Self) {
        self.non_zero_val *= rhs.val();
    }
}

impl<T: Integer + Copy> ops::Div for NonZero<T> {
    type Output = NonZero<T>;

    fn div(self, rhs: Self) -> Self::Output {
        (self.val().div(rhs.val())).into()
    }
}

impl<T: Integer + Copy + ops::DivAssign> ops::DivAssign for NonZero<T> {
    fn div_assign(&mut self, rhs: Self) {
        self.non_zero_val /= rhs.val();
    }
}

impl<T: Integer + Copy> ops::Mul<T> for NonZero<T> {
    type Output = NonZero<T>;

    fn mul(self, rhs: T) -> Self::Output {
        (self.val().mul(rhs)).into()
    }
}

impl<T: Integer + Copy + ops::MulAssign> ops::MulAssign<T> for NonZero<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.non_zero_val *= rhs;
    }
}

impl<T: Integer + Copy> ops::Div<T> for NonZero<T> {
    type Output = NonZero<T>;

    fn div(self, rhs: T) -> Self::Output {
        (self.val().div(rhs)).into()
    }
}

impl<T: Integer + Copy + ops::DivAssign> ops::DivAssign<T> for NonZero<T> {
    fn div_assign(&mut self, rhs: T) {
        self.non_zero_val /= rhs;
    }
}

impl<T: Display + Integer + Copy> Display for NonZero<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.val())
    }
}

type UInt = u64;

/// Represents a rational number
///
/// implemented with two [UInt]: numer and a denom, where denom is of type [NonZeroUInt] \
/// the sign is given by [Sign]
/// and the exponent is defined by a [i32]
#[derive(Debug, Clone, Eq, Copy)]
pub struct Rational {
    pub(crate) sign: Sign,
    pub(crate) numer: UInt,
    pub(crate) denom: NonZero<UInt>,
    pub(crate) expon: i32, // * 10^e
}

impl From<UInt> for Rational {
    fn from(numer: UInt) -> Self {
        if numer == 0 {
            return Self::zero();
        }
        Rational::reduced(Sign::Positive, numer, NonZero::new(1), 0)
    }
}

impl From<i32> for Rational {
    fn from(numer: i32) -> Self {
        if numer == 0 {
            return Self::zero();
        }
        Rational::reduced(
            Sign::from(numer),
            numer.unsigned_abs() as UInt,
            NonZero::new(1),
            0,
        )
    }
}

impl From<(i32, i32)> for Rational {
    fn from(value: (i32, i32)) -> Self {
        {
            let num = value.0;
            let den = value.1;
            if den == 0 {
                panic!("Rational::from: found 0 denominator")
            }

            let sign = match num.is_negative() || den.is_negative() {
                false => Sign::Positive,
                true => Sign::Negative,
            };

            let numer = num.unsigned_abs() as UInt;
            let denom = NonZero::from(den.unsigned_abs() as UInt);
            let expon = 0;

            Rational {
                sign,
                numer,
                denom,
                expon,
            }
            .reduce()
        }
    }
}

impl Hash for Rational {
    //TODO: look at ratio hash
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.numer != 0 {
            self.sign.hash(state);
        }
        self.numer.hash(state);
        self.denom.hash(state);
    }
}

impl PartialEq for Rational {
    fn eq(&self, other: &Self) -> bool {
        if self.numer == 0 && other.numer == 0 {
            true
        } else {
            self.sign == other.sign
                && self.numer == other.numer
                && self.denom == other.denom
                && self.expon == other.expon
        }
    }
}

impl CalcursType for Rational {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Numeric(self.into())
    }
}

impl Rational {
    pub const fn one() -> Self {
        Self {
            numer: 1,
            denom: NonZero { non_zero_val: 1 },
            sign: Sign::Positive,
            expon: 0,
        }
    }

    pub const fn zero() -> Self {
        Self {
            numer: 0,
            denom: NonZero { non_zero_val: 1 },
            sign: Sign::Positive,
            expon: 0,
        }
    }

    pub const fn minus_one() -> Self {
        Self {
            numer: 1,
            denom: NonZero { non_zero_val: 1 },
            sign: Sign::Negative,
            expon: 0,
        }
    }

    pub fn new(num: i32, den: i32) -> Self {
        if den == 0 {
            panic!("Rational::new: found 0 denominator")
        }

        let sign = match num.is_negative() || den.is_negative() {
            false => Sign::Positive,
            true => Sign::Negative,
        };

        let numer = num.unsigned_abs() as UInt;
        let denom = NonZero::new(den.unsigned_abs() as UInt);
        let expon = 0;

        Self {
            sign,
            numer,
            denom,
            expon,
        }
        .reduce()
    }

    pub(crate) fn reduced(sign: Sign, numer: UInt, denom: NonZero<UInt>, expon: i32) -> Self {
        Self {
            sign,
            numer,
            denom,
            expon,
        }
        .reduce()
    }

    pub fn num(self) -> Numeric {
        self.into()
    }

    #[inline]
    pub const fn numer(&self) -> UInt {
        self.numer
    }

    #[inline]
    pub const fn denom(&self) -> UInt {
        self.denom.non_zero_val
    }

    pub const fn is_zero(&self) -> bool {
        self.numer == 0
    }

    pub const fn is_one(&self) -> bool {
        self.numer == 1 && self.denom() == 1 && self.sign.is_pos() && self.expon == 0
    }

    // reduces only the fraction part
    #[inline]
    pub(crate) fn reduce_frac(mut self) -> Self {
        match (self.numer, self.denom()) {
            (_, 0) => unreachable!(),

            // 0 / x => 0 / 1
            (0, _) => {
                self.denom.set(1);
                self.sign = Sign::Positive;
            }

            // x / x => 1
            (n, d) if n == d => {
                self.numer = 1;
                self.denom.set(1);
            }
            _ => {
                let g = self.numer.gcd(&self.denom());
                if g != 1 {
                    self.numer /= g;
                    self.denom /= g;
                }
            }
        }
        self
    }

    #[inline]
    pub(crate) fn reduce(mut self) -> Self {
        if self.numer > self.denom() {
            let e = self.numer.ilog10() - self.denom().ilog10() + 1;
            self.denom *= 10u64.pow(e);
            self.expon += e as i32;
        }

        self = self.reduce_frac();

        // reduce denom
        let mut den = self.denom();
        // reminder: den != 0
        while den % 10 == 0 && den / 10 >= self.numer {
            den /= 10;
            self.expon -= 1;
        }
        self.denom.set(den);
        self
    }

    /// tuple acts as a fraction, e.g 1 / 3 => (1, 3)
    ///
    /// assume f1.1 != 0 and f2.1 != 0
    #[inline]
    fn unsigned_add(f1: (UInt, UInt), f2: (UInt, UInt)) -> Rational {
        let lcm = f1.1.lcm(&f2.1);
        let lhs_numer = f1.0 * lcm / f1.1;
        let rhs_numer = f2.0 * lcm / f2.1;
        Rational::reduced(Sign::Positive, lhs_numer + rhs_numer, NonZero::new(lcm), 0)
    }

    /// tuple acts as a fraction, e.g 1 / 3 => (1, 3)
    ///
    /// assume f1.1 != 0 and f2.1 != 0 and f1 >= f2
    #[inline]
    fn unsigned_sub(f1: (UInt, UInt), f2: (UInt, UInt)) -> Rational {
        let lcm = f1.1.lcm(&f2.1);
        let lhs_numer = f1.0 * lcm / f1.1;
        let rhs_numer = f2.0 * lcm / f2.1;
        Rational::reduced(Sign::Positive, lhs_numer - rhs_numer, NonZero::new(lcm), 0)
    }

    pub fn abs(&self) -> Self {
        let mut res = *self;
        res.sign = Sign::Positive;
        res
    }

    pub fn div_ratio(self, other: Self) -> Self {
        let (mut lhs, rhs) = (self, other);
        lhs.sign *= rhs.sign;
        let gcd_ac = lhs.numer.gcd(&rhs.numer);
        let gcd_bd = lhs.denom().gcd(&rhs.denom());
        lhs.numer /= gcd_ac;
        lhs.numer = lhs.numer * rhs.denom() / gcd_bd;
        lhs.denom /= gcd_bd;
        lhs.denom *= rhs.numer / gcd_ac;
        lhs.expon -= rhs.expon;
        lhs.reduce()
    }

    pub fn mul_ratio(self, other: Self) -> Self {
        let (mut lhs, rhs) = (self, other);
        lhs.sign *= rhs.sign;
        let gcd_ad = lhs.numer.gcd(&rhs.denom());
        let gcd_bc = lhs.denom().gcd(&rhs.numer);
        lhs.numer /= gcd_ad;
        lhs.numer = lhs.numer * rhs.numer / gcd_bc;
        lhs.denom /= gcd_bc;
        lhs.denom *= rhs.denom() / gcd_ad;
        lhs.expon += rhs.expon;
        lhs.reduce()
    }

    pub fn sub_ratio(self, other: Self) -> Self {
        let mut rhs = other;
        rhs.sign *= Sign::Negative;
        self.add_ratio(rhs)
    }

    /// helper function, will not reduce the resulting fraction
    #[inline]
    fn apply_exp(&mut self) {
        if self.expon > 0 {
            self.numer *= 10u32.pow(self.expon.unsigned_abs()) as UInt;
        } else {
            self.denom *= 10u32.pow(self.expon.unsigned_abs()) as UInt;
        }
        self.expon = 0;
    }

    #[inline]
    pub(crate) fn try_apply_exp(mut self) -> Option<Self> {
        if self.expon > 0 {
            self.numer *= 10u32.checked_pow(self.expon.try_into().ok()?)? as u64;
        } else {
            self.denom *= 10u32.checked_pow(self.expon.abs().try_into().ok()?)? as u64;
        }
        self.expon = 0;
        Some(self)
    }

    pub(crate) fn factor_and_apply_expon(mut lhs: Self, mut rhs: Self) -> (Self, Self, i32) {
        let factor = (lhs.expon + rhs.expon) / 2;
        lhs.expon -= factor;
        rhs.expon -= factor;
        lhs.apply_exp();
        rhs.apply_exp();
        (lhs, rhs, factor)
    }

    pub fn add_ratio(self, other: Self) -> Self {
        let (mut lhs, rhs) = (self, other);

        if lhs.denom == rhs.denom && lhs.sign == rhs.sign && lhs.expon == rhs.expon {
            lhs.numer = lhs.numer + rhs.numer;
            return lhs.reduce();
        }

        let (lhs, rhs, factor) = Self::factor_and_apply_expon(lhs, rhs);

        use Sign as S;
        let mut res;
        match (lhs.sign, rhs.sign) {
            (S::Positive, S::Positive) | (S::Negative, S::Negative) => {
                res = Self::unsigned_add((lhs.numer, lhs.denom()), (rhs.numer, rhs.denom()));
                res.sign = lhs.sign;
            }

            // -lhs + rhs
            (S::Negative, S::Positive) => {
                let lhs_abs = lhs.abs();
                let rhs_abs = rhs.abs();

                if lhs_abs >= rhs_abs {
                    // lhs >= rhs => -(lhs - rhs)
                    res = Self::unsigned_sub((lhs.numer, lhs.denom()), (rhs.numer, rhs.denom()));
                    res.sign = Sign::Negative;
                } else {
                    // rhs > lhs => rhs - lhs
                    res = Self::unsigned_sub((rhs.numer, rhs.denom()), (lhs.numer, lhs.denom()));
                }
            }

            // lhs - rhs
            (S::Positive, S::Negative) => {
                let lhs_abs = lhs.abs();
                let rhs_abs = rhs.abs();

                if lhs_abs >= rhs_abs {
                    // lhs >= rhs => lhs - rhs
                    res = Self::unsigned_sub((lhs.numer, lhs.denom()), (rhs.numer, rhs.denom()));
                } else {
                    // rhs > lhs => -(lhs + rhs)
                    res = Self::unsigned_sub((rhs.numer, rhs.denom()), (lhs.numer, lhs.denom()));
                    res.sign = Sign::Negative;
                }
            }
        }
        res.expon += factor;
        res.reduce()
    }
}

impl ops::Neg for Rational {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.sign *= Sign::Negative;
        self
    }
}

//TODO: checked add
impl ops::Add for Rational {
    type Output = Rational;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_ratio(rhs)
    }
}

impl ops::Sub for Rational {
    type Output = Rational;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_ratio(rhs)
    }
}

impl ops::Mul for Rational {
    type Output = Rational;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_ratio(rhs)
    }
}

impl ops::Div for Rational {
    type Output = Rational;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_ratio(rhs)
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.sign != other.sign {
            return match self.sign {
                Sign::Positive => Ordering::Greater,
                Sign::Negative => Ordering::Less,
            };
        }

        if self.expon != other.expon {
            // TODO: factor out e
            let lhs = self.try_apply_exp();
            let rhs = other.try_apply_exp();
            if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                return lhs.cmp(&rhs);
            } else {
                todo!()
            }
        }

        if self.denom() == other.denom() {
            return self.numer.cmp(&other.numer);
        }

        if self.numer == other.numer {
            return self.denom().cmp(&other.denom());
        }

        let (self_int, self_rem) = self.numer.div_mod_floor(&self.denom());
        let (other_int, other_rem) = other.numer.div_mod_floor(&other.denom());

        match self_int.cmp(&other_int) {
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
            Ordering::Equal => match (self_rem == 0, other_rem == 0) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                (false, false) => {
                    let self_recip =
                        Rational::reduced(self.sign, self.denom(), NonZero::new(self_rem), 0);
                    let other_recip =
                        Rational::reduced(other.sign, other.denom(), NonZero::new(other_rem), 0);
                    self_recip.cmp(&other_recip).reverse()
                }
            },
        }
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign.is_neg() {
            write!(f, "-")?;
        }

        if self.denom() == 1 {
            write!(f, "{}", self.numer)?;
        } else {
            write!(f, "({} / {})", self.numer, self.denom)?;
        }

        if self.expon != 0 {
            write!(f, "e{}", self.expon)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod rational_test {

    use crate::prelude::*;
    use crate::rational::NonZero;
    use pretty_assertions::assert_eq;

    macro_rules! r {
        ($v: literal) => {
            Rational::new($v, 1)
        };

        ($numer: literal / $denom: literal) => {
            Rational::new($numer, $denom)
        };
    }

    #[test]
    fn exprs() {
        assert_eq!(r!(1) + r!(1), r!(2));
        assert_eq!(r!(1 / 3) + r!(2 / 3), r!(1));
        assert_eq!(r!(1 / 3) - r!(2 / 3), r!(-1 / 3));
        assert_eq!(r!(1 / -3) * r!(3), r!(-1));
        assert!(r!(2) > r!(1));
        assert!(r!(2) >= r!(2));
        assert!(r!(2 / 4) <= r!(4 / 8));
        assert!(r!(5 / 128) > r!(11 / 2516));
    }

    #[test]
    #[should_panic]
    fn non_zero() {
        NonZero::new(0);
    }
}
