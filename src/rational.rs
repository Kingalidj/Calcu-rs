use core::{cmp, fmt, hash::Hash, ops};

use num::Integer;

use crate::{
    base::{Base, CalcursType},
    numeric::{Float, Numeric, Sign},
    pattern::{Item, Pattern},
};

pub type RatioTyp = u64;

/// Represents a rational number
///
/// DISCLAIMER: a rational numer should have a unique representation
///
/// sign * numer / denom * 10^(expon): \
/// (numer / denom) should always be between 1 and 0.1
#[derive(Clone, PartialEq, Eq, Copy, Hash)]
pub struct Rational {
    pub(crate) sign: Sign,
    pub(crate) numer: RatioTyp,
    pub(crate) denom: NonZero,
    pub(crate) expon: i32, // * 10^e
}

/// Nonzero integer value
///
/// will panic if set to 0
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NonZero {
    pub(crate) non_zero_val: RatioTyp,
}

/// helper struct for [Rational], simple fraction without any reformatting
#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
struct Fraction {
    numer: RatioTyp,
    denom: RatioTyp,
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

        let numer = num.unsigned_abs() as RatioTyp;
        let denom = NonZero::new(den.unsigned_abs() as RatioTyp);
        Self::reduced(sign, numer, denom, 0)
    }

    pub(crate) fn reduced(sign: Sign, numer: RatioTyp, denom: NonZero, expon: i32) -> Self {
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
    pub const fn numer(&self) -> RatioTyp {
        self.numer
    }

    #[inline]
    pub const fn denom(&self) -> RatioTyp {
        self.denom.non_zero_val
    }

    pub fn as_float(&self) -> Float {
        let n = self.numer as f64;
        let d = self.denom() as f64;
        let e = self.expon as f64;
        let sign = if self.sign.is_neg() { -1f64 } else { 1f64 };

        let f = sign * (n * 10f64.powf(e) / d);
        assert!(!f.is_nan(), "not possible if denom is not zero?");
        Float(f)
    }

    /// reduces only the fraction part
    #[inline]
    pub(crate) fn reduce_frac(&mut self) {
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
    }

    #[inline]
    pub(crate) fn reduce(mut self) -> Self {
        if self.numer == 0 {
            return Self::zero();
        }

        if self.numer > self.denom() {
            let e = self.numer.ilog10() - self.denom().ilog10();
            self.denom *= 10u64.pow(e);
            self.expon += e as i32;
        }

        self.reduce_frac();

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

    pub fn abs(&self) -> Self {
        let mut res = *self;
        res.sign = Sign::Positive;
        res
    }

    /// helper function, will not reduce the resulting fraction
    #[inline]
    fn apply_expon(&mut self) {
        if self.expon > 0 {
            self.numer *= 10u32.pow(self.expon.unsigned_abs()) as RatioTyp;
        } else {
            self.denom *= 10u32.pow(self.expon.unsigned_abs()) as RatioTyp;
        }
        self.expon = 0;
    }

    #[inline]
    pub(crate) fn try_apply_expon(mut self) -> Option<Self> {
        if self.expon > 0 {
            self.numer *= 10u32.checked_pow(self.expon.try_into().ok()?)? as RatioTyp;
        } else {
            self.denom *= 10u32.checked_pow(self.expon.abs().try_into().ok()?)? as RatioTyp;
        }
        self.expon = 0;
        Some(self)
    }

    #[inline]
    fn factor_expon(mut lhs: Self, mut rhs: Self) -> (Self, Self, i32) {
        if lhs.expon == rhs.expon {
            (lhs, rhs, 0)
        } else if lhs.desc().is(Item::Zero) {
            let factor = rhs.expon;
            lhs.expon = factor;
            (lhs, rhs, factor)
        } else if rhs.desc().is(Item::Zero) {
            let factor = lhs.expon;
            rhs.expon = factor;
            (lhs, rhs, factor)
        } else {
            let factor = (lhs.expon + rhs.expon) / 2;
            lhs.expon -= factor;
            rhs.expon -= factor;
            lhs.apply_expon();
            rhs.apply_expon();
            (lhs, rhs, factor)
        }
    }

    //#[inline]
    //fn factor_and_apply_expon(mut lhs: Self, mut rhs: Self) -> (Self, Self, i32) {
    //    let factor = (lhs.expon + rhs.expon) / 2;
    //    lhs.expon -= factor;
    //    rhs.expon -= factor;
    //    lhs.apply_expon();
    //    rhs.apply_expon();
    //    (lhs, rhs, factor)
    //}

    #[inline]
    pub const fn desc(&self) -> Pattern {
        let sign = self.sign.desc();

        let flag = if self.numer == 0 {
            Item::Zero
        } else if self.numer == 1 && self.denom() == 1 && self.expon == 0 {
            Item::UOne.union(sign)
        } else if self.denom() == 1 && self.expon >= 0 {
            Item::Int.union(sign)
        } else {
            sign
        };

        Pattern::Itm(flag.union(Item::Rational))
    }

    fn format_for_print(&self) -> Self {
        let mut r = *self;

        while r.denom() % 10 == 0 {
            r.expon -= 1;
            r.denom /= 10;
        }

        let max_len = 3;
        let max_num = 999;

        if r.expon.abs() <= max_len && r.expon.abs() > 1 && r.numer < max_num {
            r.apply_expon();
            r.reduce_frac();
        }

        r
    }
}

impl NonZero {
    /// panics if arg is 0
    #[inline(always)]
    pub const fn new(n: RatioTyp) -> Self {
        debug_assert!(n != 0);
        NonZero { non_zero_val: n }
    }

    /// panics if arg is 0
    #[inline(always)]
    pub fn set(&mut self, n: RatioTyp) {
        debug_assert!(n != 0);
        self.non_zero_val = n;
    }

    #[inline(always)]
    pub const fn val(&self) -> RatioTyp {
        self.non_zero_val
    }
}

impl From<RatioTyp> for Rational {
    #[inline]
    fn from(numer: RatioTyp) -> Self {
        if numer == 0 {
            return Self::zero();
        }
        Rational::reduced(Sign::Positive, numer, NonZero::new(1), 0)
    }
}

impl From<i32> for Rational {
    #[inline]
    fn from(numer: i32) -> Self {
        if numer == 0 {
            return Self::zero();
        }
        Rational::reduced(
            Sign::from(numer),
            numer.unsigned_abs() as RatioTyp,
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

            let numer = num.unsigned_abs() as RatioTyp;
            let denom = NonZero::from(den.unsigned_abs() as RatioTyp);
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

impl From<(RatioTyp, RatioTyp)> for Rational {
    #[inline]
    fn from(value: (RatioTyp, RatioTyp)) -> Self {
        Rational::reduced(Sign::Positive, value.0, value.1.into(), 0)
    }
}

impl CalcursType for Rational {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Numeric(self.into())
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if self.sign != other.sign {
            return self.sign.cmp(&other.sign);
        } else if self.expon != other.expon {
            return self.expon.cmp(&other.expon);
        } else if self.denom == other.denom {
            return self.numer.cmp(&other.numer);
        } else if self.numer == other.numer {
            return self.denom.cmp(&other.denom);
        }

        let (self_int, self_rem) = self.numer.div_mod_floor(&self.denom());
        let (other_int, other_rem) = other.numer.div_mod_floor(&other.denom());

        use cmp::Ordering as Ord;
        match self_int.cmp(&other_int) {
            Ord::Greater => Ord::Greater,
            Ord::Less => Ord::Less,
            Ord::Equal => match (self_rem == 0, other_rem == 0) {
                (true, true) => Ord::Equal,
                (true, false) => Ord::Less,
                (false, true) => Ord::Greater,
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

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
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
        let (lhs, rhs, factor) = Rational::factor_expon(self, rhs);

        let lhs_f = Fraction::from((lhs.numer(), lhs.denom()));
        let rhs_f = Fraction::from((rhs.numer(), rhs.denom()));

        let mut res;
        if lhs.sign == rhs.sign {
            res = Self::from(lhs_f + rhs_f);
            res.sign = lhs.sign;
            res.expon += factor;
            return res;
        }

        // lhs.sign != rhs.sign
        // -(lhs - rhs) or rhs - lhs
        if lhs.abs() >= rhs.abs() {
            res = Self::from(lhs_f - rhs_f);
            res.sign = lhs.sign;
        } else {
            // -(lhs + rhs) or rhs - lhs
            res = Self::from(rhs_f - lhs_f);
            res.sign = rhs.sign;
        }
        res.expon += factor;
        res.reduce()
    }
}

impl ops::Sub for Rational {
    type Output = Rational;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut rhs = rhs;
        rhs.sign *= Sign::Negative;
        self + rhs
    }
}

impl ops::Mul for Rational {
    type Output = Rational;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.sign *= rhs.sign;
        let gcd_ad = self.numer.gcd(&rhs.denom());
        let gcd_bc = self.denom().gcd(&rhs.numer);
        self.numer /= gcd_ad;
        self.numer = self.numer * rhs.numer / gcd_bc;
        self.denom /= gcd_bc;
        self.denom *= rhs.denom() / gcd_ad;
        self.expon += rhs.expon;
        self.reduce()
    }
}

impl ops::Div for Rational {
    type Output = Rational;

    fn div(mut self, rhs: Self) -> Self::Output {
        self.sign *= rhs.sign;
        let gcd_ac = self.numer.gcd(&rhs.numer);
        let gcd_bd = self.denom().gcd(&rhs.denom());
        self.numer /= gcd_ac;
        self.numer = self.numer * rhs.denom() / gcd_bd;
        self.denom /= gcd_bd;
        self.denom *= rhs.numer / gcd_ac;
        self.expon -= rhs.expon;
        self.reduce()
    }
}

impl ops::Mul for NonZero {
    type Output = NonZero;

    fn mul(self, rhs: Self) -> Self::Output {
        (self.val().mul(rhs.val())).into()
    }
}

impl ops::MulAssign for NonZero {
    fn mul_assign(&mut self, rhs: Self) {
        self.non_zero_val *= rhs.val();
    }
}

impl ops::Div for NonZero {
    type Output = NonZero;

    fn div(self, rhs: Self) -> Self::Output {
        (self.val().div(rhs.val())).into()
    }
}

impl ops::DivAssign for NonZero {
    fn div_assign(&mut self, rhs: Self) {
        self.non_zero_val /= rhs.val();
    }
}

impl ops::Mul<RatioTyp> for NonZero {
    type Output = NonZero;

    fn mul(self, rhs: RatioTyp) -> Self::Output {
        (self.val().mul(rhs)).into()
    }
}

impl ops::MulAssign<RatioTyp> for NonZero {
    fn mul_assign(&mut self, rhs: RatioTyp) {
        self.non_zero_val *= rhs;
    }
}

impl ops::Div<RatioTyp> for NonZero {
    type Output = NonZero;

    fn div(self, rhs: RatioTyp) -> Self::Output {
        (self.val().div(rhs)).into()
    }
}

impl ops::DivAssign<RatioTyp> for NonZero {
    fn div_assign(&mut self, rhs: RatioTyp) {
        self.non_zero_val /= rhs;
    }
}

impl ops::Add for Fraction {
    type Output = Fraction;

    /// assume self.denom != 0 and rhs.denom != 0
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.denom == rhs.denom {
            let denom = self.denom;
            let numer = self.numer + rhs.numer;
            return Self { numer, denom };
        }

        let lcm = self.denom.lcm(&rhs.denom);
        let lhs_numer = self.numer * (lcm / self.denom);
        let rhs_numer = rhs.numer * (lcm / rhs.denom);

        Self {
            numer: lhs_numer + rhs_numer,
            denom: lcm,
        }
    }
}

impl ops::Sub for Fraction {
    type Output = Fraction;

    /// assume self.denom != 0 and rhs.denom != 0 and self >= rhs
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.denom == rhs.denom {
            let denom = self.denom;
            let numer = self.numer - rhs.numer;
            return Self { numer, denom };
        }

        let lcm = self.denom.lcm(&rhs.denom);
        let lhs_numer = self.numer * (lcm / self.denom);
        let rhs_numer = rhs.numer * (lcm / rhs.denom);

        Self {
            numer: lhs_numer - rhs_numer,
            denom: lcm,
        }
    }
}

impl From<Fraction> for Rational {
    #[inline]
    fn from(value: Fraction) -> Self {
        (value.numer, value.denom).into()
    }
}

impl From<(RatioTyp, RatioTyp)> for Fraction {
    #[inline]
    fn from(value: (RatioTyp, RatioTyp)) -> Self {
        Fraction {
            numer: value.0,
            denom: value.1,
        }
    }
}

impl From<RatioTyp> for NonZero {
    #[inline]
    fn from(value: RatioTyp) -> Self {
        NonZero::new(value)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign.is_neg() {
            write!(f, "-")?;
        }

        let r = self.format_for_print();

        if r.denom() == 1 {
            write!(f, "{}", r.numer)?;
        } else {
            write!(f, "{}/{}", r.numer, r.denom)?;
        }

        if r.expon != 0 {
            write!(f, " e{}", r.expon)?;
        }

        Ok(())
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "R(")?;
        if self.sign.is_neg() {
            write!(f, "-")?;
        }

        write!(f, "{}/{}", self.numer, self.denom)?;
        write!(f, ")")
    }
}

impl fmt::Display for NonZero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.val())
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
