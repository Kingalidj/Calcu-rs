use std::{cmp, fmt, hash::Hash, ops};
use std::fmt::Formatter;

use calcu_rs::numeric::{Float, Sign};
use calcu_rs::operator2::Pow;
use num::{integer::Roots, Integer};
use std::num::NonZeroU64;

use crate::{
    expression::{CalcursType, Expr},
    pattern::Item,
};

const NNZ_ONE: UNonZero = UNonZero::new_unchecked(1);

/// Represents a rational number
///
/// DISCLAIMER: a rational numer should have a unique representation
///
/// sign * numer / denom * 10^(expon): \
/// (numer / denom) should always be between 1 and 0.1
#[derive(Clone, PartialEq, Eq, Copy, Hash)]
pub struct Rational {
    pub(crate) sign: Sign,
    pub(crate) numer: u64,
    pub(crate) denom: UNonZero,
    pub(crate) exponent: i32, // * 10^e
}

impl Rational {
    pub const ONE: Expr = Expr::Rational(Rational {
        numer: 1,
        denom: NNZ_ONE,
        sign: Sign::Positive,
        exponent: 0,
    });

    pub const MINUS_ONE: Expr = Expr::Rational(Rational {
        numer: 1,
        denom: NNZ_ONE,
        sign: Sign::Negative,
        exponent: 0,
    });

    pub const ZERO: Expr = Expr::Rational(Rational {
        numer: 0,
        denom: NNZ_ONE,
        sign: Sign::Positive,
        exponent: 0,
    });

    pub const fn one() -> Self {
        Self {
            numer: 1,
            denom: NNZ_ONE,
            sign: Sign::Positive,
            exponent: 0,
        }
    }

    pub const fn zero() -> Self {
        Self {
            numer: 0,
            denom: NNZ_ONE,
            sign: Sign::Positive,
            exponent: 0,
        }
    }

    pub const fn minus_one() -> Self {
        Self {
            numer: 1,
            denom: NNZ_ONE,
            sign: Sign::Negative,
            exponent: 0,
        }
    }

    pub fn new(num: i64, den: i64) -> Self {
        if den == 0 {
            panic!("Rational::new: found 0 denominator")
        }

        let sign = match num.is_negative() || den.is_negative() {
            false => Sign::Positive,
            true => Sign::Negative,
        };

        let numer = num.unsigned_abs();
        let denom = UNonZero::new(den.unsigned_abs()).unwrap();
        Self::reduced(sign, numer, denom, 0)
    }

    pub(crate) fn reduced(sign: Sign, numer: u64, denom: UNonZero, exponent: i32) -> Self {
        Self {
            sign,
            numer,
            denom,
            exponent,
        }
        .reduce()
    }

    #[inline]
    pub const fn numer(&self) -> u64 {
        self.numer
    }

    #[inline]
    pub const fn denom(&self) -> u64 {
        self.denom.get()
    }

    pub fn as_float(&self) -> Float {
        let n = self.numer as f64;
        let d = self.denom() as f64;
        let e = self.exponent as f64;
        let sign = if self.sign.is_neg() { -1f64 } else { 1f64 };

        let f = sign * (n * 10f64.powf(e) / d);
        assert!(!f.is_nan(), "not possible if denom is not zero?");
        Float(f)
    }

    /// will either multiply the two rationals or turn them to floats and then multiply them.
    /// Multiplication will always result in a [Expr]
    pub(crate) fn base_mul(self, rhs: Self) -> Expr {
        if let Some(prod) = self * rhs {
            Expr::Rational(prod)
        } else {
            let (f1, f2) = (self.to_float(), rhs.to_float());
            Expr::Float(f1 * f2)
        }
    }

    /// reduces only the fraction part, ignores exponent
    #[inline]
    fn reduce_frac(&mut self) {
        match (self.numer, self.denom()) {
            (_, 0) => unreachable!(),

            // 0 / x => 0 / 1
            (0, _) => {
                self.denom = NNZ_ONE;
                self.sign = Sign::Positive;
            }

            // x / x => 1
            (n, d) if n == d => {
                self.numer = 1;
                self.denom = NNZ_ONE;
            }
            _ => {
                let g = self.numer.gcd(&self.denom());
                if g != 1 {
                    self.numer /= g;
                    self.denom /= UNonZero::new(g).unwrap();
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
            self.denom *= UNonZero::new_unchecked(10u64.pow(e));
            self.exponent += e as i32;
        }

        self.reduce_frac();

        // reduce denom
        let mut den = self.denom();
        // reminder: den != 0
        while den % 10 == 0 && den / 10 >= self.numer {
            den /= 10;
            self.exponent -= 1;
        }
        self.denom = UNonZero::new(den).unwrap();
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
        if self.exponent > 0 {
            self.numer *= 10u64.pow(self.exponent.unsigned_abs());
        } else {
            self.denom *= UNonZero::new(10u64.pow(self.exponent.unsigned_abs())).unwrap();
        }
        self.exponent = 0;
    }

    #[inline]
    pub(crate) fn try_apply_expon(mut self) -> Option<Self> {
        if self.exponent > 0 {
            let rhs = 10u64.checked_pow(self.exponent.try_into().ok()?)?;
            self.numer = self.numer.checked_mul(rhs)?;
        } else {
            let rhs = 10u64.checked_pow(self.exponent.abs().try_into().ok()?)?;
            let lhs = self.denom();
            let denom = lhs.checked_mul(rhs)?;
            self.denom = UNonZero::new(denom)?;
        }
        self.exponent = 0;
        Some(self)
    }

    /// when adding [Rational] we need to make sure that both exponents are equal
    #[inline]
    fn factor_expon(mut lhs: Self, mut rhs: Self) -> (Self, Self, i32) {
        if lhs.exponent == rhs.exponent {
            (lhs, rhs, 0)
        } else if lhs.desc().is(Item::Zero) {
            let factor = rhs.exponent;
            lhs.exponent = factor;
            (lhs, rhs, factor)
        } else if rhs.desc().is(Item::Zero) {
            let factor = lhs.exponent;
            rhs.exponent = factor;
            (lhs, rhs, factor)
        } else {
            let factor = (lhs.exponent + rhs.exponent) / 2;
            lhs.exponent -= factor;
            rhs.exponent -= factor;
            // TODO: try_apply?
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

    pub fn factorial(self) -> Self {
        let d = self.desc();

        if d.is(Item::Zero) {
            Self::one()
        } else if d.is(Item::PosInt) {
            let mut res = 1;

            for i in 1..self.numer {
                res *= i;
            }

            Rational::from(res)
        } else {
            panic!("factorial of {self} not defined");
        }
    }

    fn format_for_print(&self) -> Self {
        let mut r = *self;

        while r.denom() % 10 == 0 {
            r.exponent -= 1;
            r.denom /= UNonZero::new_unchecked(10);
        }

        let max_len = 3;
        let max_num = 999;

        if r.exponent.abs() <= max_len && r.exponent.abs() > 1 && r.numer < max_num {
            r.apply_expon();
            r.reduce_frac();
        }

        if r.exponent == 1 {
            r.apply_expon();
            r.reduce_frac();
        }

        r
    }
    pub(crate) fn convert_add(self, rhs: Rational) -> Expr {
        if let Some(sum) = self + rhs {
            sum.into()
        } else {
            let f_lhs: f64 = self.into();
            let f_rhs: f64 = rhs.into();
            let sum = f_lhs + f_rhs;
            Expr::Float(Float(sum))
        }
    }

    pub(crate) fn convert_sub(self, rhs: Rational) -> Expr {
        if let Some(diff) = self - rhs {
            diff.into()
        } else {
            let f_lhs: f64 = self.into();
            let f_rhs: f64 = rhs.into();
            let diff = f_lhs - f_rhs;
            Expr::Float(Float(diff))
        }
    }

    pub(crate) fn convert_mul(self, rhs: Rational) -> Expr {
        if let Some(prod) = self * rhs {
            prod.into()
        } else {
            let f_lhs: f64 = self.into();
            let f_rhs: f64 = rhs.into();
            let mul = f_lhs * f_rhs;
            Expr::Float(Float(mul))
        }
    }

    pub(crate) fn convert_div(self, rhs: Rational) -> Expr {
        if let Some(prod) = self / rhs {
            prod.into()
        } else {
            let f_lhs: f64 = self.into();
            let f_rhs: f64 = rhs.into();
            let mul = f_lhs / f_rhs;
            Expr::Float(Float(mul))
        }
    }

    /// helper function for [checked_pow]
    #[inline]
    fn checked_int_pow(&self, exponent: u64) -> Option<Self> {
        let numer = self.numer.checked_pow(exponent.try_into().ok()?)?;
        let denom = self.denom().checked_pow(exponent.try_into().ok()?)?;
        Some(Rational::from((numer, denom)))
    }

    /// tries to calculate the power. It is possible to apply just part of the
    /// exponent, so we return the changed (base, exponent). If the power
    /// was fully calculated (.., exponent) will be one
    /// eg:
    /// a, b, c: Integers, (q, r) = (quotient, reminder) of b / c
    /// a^(b / c) -> a.apply_pow(b / c) -> out: (a^q, a^r) -> a^q * a^r
    pub(crate) fn apply_pow(self, exponent: Rational) -> (Rational, Rational) {
        let (mut base, mut exp) =
            if let (Some(b), Some(e)) = (self.try_apply_expon(), exponent.try_apply_expon()) {
                (b, e)
            } else {
                return (self, exponent);
            };

        exp.reduce_frac();
        base.reduce_frac();

        if exp.sign.is_neg() {
            if let Some(inv) = base.inverse() {
                base = inv;
                base.exponent *= -1;
                exp.sign = Sign::Positive;
            } else {
                // 0^-1
                panic!("0^-1 should have been handled");
            }
        }

        // integer exponent
        if exp.desc().is(Item::Int) {
            // (a / b) ^ (c / 1) => (a / b) ^ c => a^c / b^c
            let exponent = exp.numer;
            if let Some(res) = self.checked_int_pow(exponent) {
                return (res, Rational::one());
            }
        }

        // exponent > 1 -> try to apply the quotient
        if self.numer > self.denom() {
            //base ^ ( a / b) == base ^ (quot + rem / b) == base^quot * base^(rem / b)
            let (quot, rem) = exp.numer.div_rem(&exp.denom());

            if let Some(new_base) = base.checked_int_pow(quot) {
                let new_exp = Rational::from((rem, exp.denom()));
                // check if we can apply new_exp
                return new_base.apply_pow(new_exp);
            }
        }

        // just check if root exists, e.g a^(1 / c)
        let root = (
            base.numer.nth_root(exp.denom() as u32),
            base.denom().nth_root(exp.denom() as u32),
        );

        if (root.0 * root.0, root.1 * root.1) == (base.numer, base.denom()) {
            base.numer = root.0;
            base.denom = UNonZero::new(root.1).unwrap();
            exp.denom = NNZ_ONE;
            return (base, exp);
        }

        (base, exp)
    }

    pub(crate) fn inverse(&mut self) -> Option<Self> {
        if self.numer == 0 {
            None
        } else {
            let tmp = self.numer;
            self.numer = self.denom();
            self.denom = UNonZero::new(tmp).unwrap();
            Some(*self)
        }
    }

    pub(crate) fn to_float(self) -> Float {
        let mut f = self.numer as f64 / self.denom() as f64;
        f *= 10f64.powi(self.exponent);
        Float(f)
    }

    #[inline(always)]
    pub(crate) const fn is_int(&self) -> bool {
        self.denom() == 1 && self.exponent >= 0
    }

    #[inline(always)]
    pub(crate) fn to_int(self) -> i64 {
        let sign = match self.sign {
            Sign::Positive => 1i64,
            Sign::Negative => -1i64,
        };
        let numer = self.numer as i64;
        let exponent = self.exponent as u32;
        sign * numer * 10i64.pow(exponent)
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if self.sign != other.sign {
            return self.sign.cmp(&other.sign);
        } else if self.exponent != other.exponent {
            return self.exponent.cmp(&other.exponent);
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
                    let self_recip = Rational::reduced(
                        self.sign,
                        self.denom(),
                        UNonZero::new(self_rem).unwrap(),
                        0,
                    );
                    let other_recip = Rational::reduced(
                        other.sign,
                        other.denom(),
                        UNonZero::new(other_rem).unwrap(),
                        0,
                    );
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
impl ops::Add for Rational {
    type Output = Option<Rational>;

    fn add(self, rhs: Self) -> Self::Output {
        let (lhs, rhs, factor) = Rational::factor_expon(self, rhs);

        let lhs_f = Fraction::from((lhs.numer(), lhs.denom()));
        let rhs_f = Fraction::from((rhs.numer(), rhs.denom()));

        let mut res;
        if lhs.sign == rhs.sign {
            res = Self::from((lhs_f + rhs_f)?);
            res.sign = lhs.sign;
            res.exponent = res.exponent.checked_add(factor)?;
            return Some(res);
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
        res.exponent += factor;
        Some(res.reduce())
    }
}
impl ops::Sub for Rational {
    type Output = Option<Rational>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut rhs = rhs;
        rhs.sign *= Sign::Negative;
        self + rhs
    }
}
impl ops::Mul for Rational {
    type Output = Option<Rational>;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.sign *= rhs.sign;
        let gcd_ad = self.numer.gcd(&rhs.denom());
        let gcd_bc = self.denom().gcd(&rhs.numer);
        // divisions should be safe
        self.numer /= gcd_ad;
        self.numer = self.numer.checked_mul(rhs.numer)? / gcd_bc;
        self.denom /= UNonZero::new(gcd_bc).unwrap();
        self.denom *= UNonZero::new(rhs.denom() / gcd_ad).unwrap();
        self.exponent += rhs.exponent;
        Some(self.reduce())
    }
}
impl ops::Div for Rational {
    type Output = Option<Rational>;

    fn div(mut self, rhs: Self) -> Self::Output {
        self.sign *= rhs.sign;
        let gcd_ac = self.numer.gcd(&rhs.numer);
        let gcd_bd = self.denom().gcd(&rhs.denom());
        self.numer /= gcd_ac;
        self.numer = self.numer.checked_mul(rhs.denom())? / gcd_bd;
        self.denom /= UNonZero::new(gcd_bd).unwrap();
        self.denom = UNonZero::new(
            self.denom
                .get()
                .checked_mul(rhs.numer.checked_div(gcd_ac)?)?,
        )
        .unwrap();
        self.exponent -= rhs.exponent;
        Some(self.reduce())
    }
}
impl ops::AddAssign for Rational {
    fn add_assign(&mut self, rhs: Self) {
        *self = (*self + rhs).unwrap();
    }
}
impl ops::MulAssign for Rational {
    fn mul_assign(&mut self, rhs: Self) {
        *self = (*self * rhs).unwrap();
    }
}

impl CalcursType for Rational {
    #[inline]
    fn desc(&self) -> Item {
        let sign = self.sign.desc();

        let flag = if self.numer == 0 {
            Item::Zero
        } else if self.numer == 1 && self.denom() == 1 && self.exponent == 0 {
            Item::UOne.union(sign)
        } else if self.denom() == 1 && self.exponent >= 0 {
            Item::Int.union(sign)
        } else {
            sign
        };

        flag.union(Item::Rational)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash, Ord, PartialOrd)]
pub struct UNonZero(NonZeroU64);

impl UNonZero {
    pub const fn new(val: u64) -> Option<Self> {
        if let Some(nnz) = NonZeroU64::new(val) {
            Some(Self(nnz))
        } else {
            None
        }
        //Some(Self(NonZeroU64::new(val)?))
    }

    pub const fn new_unchecked(val: u64) -> Self {
        debug_assert!(val != 0);
        unsafe { Self(NonZeroU64::new_unchecked(val)) }
    }

    pub const fn get(&self) -> u64 {
        self.0.get()
    }
}

/// helper struct for [Rational], simple fraction without any reformatting
#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
struct Fraction {
    numer: u64,
    denom: u64,
}

impl ops::Add for Fraction {
    type Output = Option<Fraction>;

    /// assume self.denom != 0 and rhs.denom != 0
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.denom == rhs.denom {
            let denom = self.denom;
            let numer = self.numer.checked_add(rhs.numer)?;
            return Some(Self { numer, denom });
        }

        let lcm = self.denom.lcm(&rhs.denom);
        let lhs_numer = self.numer.checked_mul(lcm.checked_div(self.denom)?)?;
        let rhs_numer = rhs.numer.checked_mul(lcm.checked_div(rhs.denom)?)?;

        Some(Self {
            numer: lhs_numer + rhs_numer,
            denom: lcm,
        })
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

impl ops::Add for UNonZero {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0.get() + rhs.0.get()).expect("add of two nnz should be nnz")
    }
}
impl ops::AddAssign for UNonZero {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl ops::Sub for UNonZero {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.0 > rhs.0, "underflow or zero");
        Self::new(self.0.get() - rhs.0.get()).unwrap()
    }
}
impl ops::SubAssign for UNonZero {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl ops::Mul for UNonZero {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.0.get() * rhs.0.get()).expect("mul must be nnz")
    }
}
impl ops::MulAssign for UNonZero {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl ops::Div for UNonZero {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.0.get() / rhs.0.get()).expect("div must be nnz")
    }
}
impl ops::DivAssign for UNonZero {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl From<(u64, u64)> for Fraction {
    #[inline]
    fn from(value: (u64, u64)) -> Self {
        Fraction {
            numer: value.0,
            denom: value.1,
        }
    }
}

impl From<u64> for Rational {
    #[inline]
    fn from(numer: u64) -> Self {
        if numer == 0 {
            return Self::zero();
        }
        Rational::reduced(Sign::Positive, numer, UNonZero::new(1).unwrap(), 0)
    }
}
impl From<i32> for Rational {
    fn from(val: i32) -> Self {
        (val as i64).into()
    }
}
impl From<i64> for Rational {
    #[inline]
    fn from(numer: i64) -> Self {
        if numer == 0 {
            return Self::zero();
        }
        Rational::reduced(
            Sign::from(numer),
            numer.unsigned_abs(),
            UNonZero::new(1).unwrap(),
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

            let numer = num.unsigned_abs() as u64;
            let denom = UNonZero::new(den.unsigned_abs() as u64).unwrap();
            let expon = 0;

            Rational {
                sign,
                numer,
                denom,
                exponent: expon,
            }
            .reduce()
        }
    }
}
impl From<(u64, u64)> for Rational {
    #[inline]
    fn from(value: (u64, u64)) -> Self {
        Rational::reduced(
            Sign::Positive,
            value.0,
            UNonZero::new(value.1).expect("nonzero denom"),
            0,
        )
    }
}
impl From<Rational> for f64 {
    fn from(value: Rational) -> Self {
        let mut val: f64 = value.numer() as f64 * 10f64.powf(value.exponent as f64);
        val /= value.denom() as f64;

        val
    }
}
impl From<Fraction> for Rational {
    #[inline]
    fn from(value: Fraction) -> Self {
        (value.numer, value.denom).into()
    }
}

impl fmt::Display for UNonZero {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
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

        if r.exponent != 0 {
            write!(f, " e{}", r.exponent)?;
        }

        Ok(())
    }
}
