use std::{
    fmt::{self, Display},
    hash::Hash,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
};

use crate::{
    base::{Base, CalcursType},
    numeric::{Number, Sign, Undefined},
};
use num::Integer;

type UInt = u64;

/// Nonzero integer value
///
/// will panic otherwise
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct NonZeroUInt(pub(crate) UInt);

impl NonZeroUInt {
    #[inline]
    /// panics if arg is 0
    pub const fn new(n: UInt) -> Self {
        if n == 0 {
            panic!("NonZeroUInt::new: found 0");
        } else {
            NonZeroUInt(n)
        }
    }

    /// panics if arg is 0
    pub fn set(&mut self, n: UInt) {
        if n == 0 {
            panic!("NonZeroUInt::set: found 0");
        }

        self.0 = n;
    }

    /// panics if arg is 0
    fn div(&mut self, n: UInt) {
        if n == 0 {
            panic!("NonZeroUInt::div: found 0");
        }

        self.0 /= n;
    }

    /// panics if arg is 0
    fn mul(&mut self, n: UInt) {
        if n == 0 {
            panic!("NonZeroUInt::div: found 0");
        }

        self.0 *= n;
    }
}

impl Deref for NonZeroUInt {
    type Target = UInt;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Represents a rational number
///
/// implemented with two [UInt]: numer and a denom, where denom is of type [NonZeroUInt] \
/// Sign defined with a boolean field
#[derive(Debug, Clone, Eq, Copy)]
pub struct Rational {
    pub(crate) sign: Sign,
    pub(crate) numer: UInt,
    pub(crate) denom: NonZeroUInt,
}

impl Hash for Rational {
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
            self.sign == other.sign && self.numer == other.numer && self.denom == other.denom
        }
    }
}

impl CalcursType for Rational {
    fn base(self) -> Base {
        Base::Number(self.into())
    }
}

impl Rational {
    pub const fn one() -> Self {
        Self {
            numer: 1,
            denom: NonZeroUInt::new(1),
            sign: Sign::Positive,
        }
    }

    pub fn new(num: i32, den: i32) -> Self {
        if let (_, 0) = (num, den) {
            panic!("Rational::new: found 0 denominator")
        }

        let sign = match (num * den).is_positive() {
            true => Sign::Positive,
            false => Sign::Negative,
        };

        let numer = num.unsigned_abs() as UInt;
        let denom = NonZeroUInt::new(den.unsigned_abs() as UInt);

        Self { sign, numer, denom }.reduce()
    }

    pub fn frac_num(num: i32, den: i32) -> Number {
        match (num, den) {
            (_, 0) => Undefined.into(),
            _ => Self::new(num, den).into(),
        }
    }

    pub fn int_num(n: i32) -> Number {
        let sign = match n.is_positive() {
            true => Sign::Positive,
            false => Sign::Negative,
        };
        let numer = n.unsigned_abs() as UInt;
        let denom = NonZeroUInt::new(1);
        Self { sign, numer, denom }.into()
    }

    #[inline]
    pub fn numer(&self) -> UInt {
        self.numer
    }

    #[inline]
    pub const fn denom(&self) -> UInt {
        self.denom.0
    }

    fn reduce(mut self) -> Self {
        match (self.numer, *self.denom) {
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
                let g = self.numer.gcd(&self.denom);
                self.numer /= g;
                self.denom.div(g);
            }
        }

        self
    }

    /// tuple acts as a fraction, e.g 1 / 3 => (1, 3)
    ///
    /// assume f.1 != 0 and f1.1 != f2.1
    #[inline]
    fn unsigned_add(f1: (UInt, UInt), f2: (UInt, UInt)) -> Rational {
        let lcm = f1.1.lcm(&f2.1);
        let lhs_numer = f1.0 * (lcm / f1.1);
        let rhs_numer = f2.0 * (lcm / f2.1);
        Rational {
            sign: Sign::Positive,
            numer: lhs_numer + rhs_numer,
            denom: NonZeroUInt::new(lcm),
        }
        .reduce()
    }

    /// tuple acts as a fraction, e.g 1 / 3 => (1, 3)
    ///
    /// assume f.1 != 0 and f1.1 != f2.1 and f1 >= f2
    #[inline]
    fn unsigned_sub(f1: (UInt, UInt), f2: (UInt, UInt)) -> Rational {
        let lcm = f1.1.lcm(&f2.1);
        let lhs_numer = f1.0 * (lcm / f1.1);
        let rhs_numer = f2.0 * (lcm / f2.1);
        Rational {
            sign: Sign::Positive,
            numer: lhs_numer - rhs_numer,
            denom: NonZeroUInt::new(lcm),
        }
        .reduce()
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
        let gcd_bd = lhs.denom.gcd(&rhs.denom);
        lhs.numer /= gcd_ac;
        lhs.numer *= *rhs.denom / gcd_bd;
        lhs.denom.div(gcd_bd);
        lhs.denom.mul(rhs.numer / gcd_ac);
        lhs
    }

    pub fn mul_ratio(self, other: Self) -> Self {
        let (mut lhs, rhs) = (self, other);
        lhs.sign *= rhs.sign;
        let gcd_ad = lhs.numer.gcd(&rhs.denom);
        let gcd_bc = lhs.denom.gcd(&rhs.numer);
        lhs.numer /= gcd_ad;
        lhs.numer *= rhs.numer / gcd_bc;
        lhs.denom.div(gcd_bc);
        lhs.denom.mul(*rhs.denom / gcd_ad);
        lhs
    }

    pub fn sub_ratio(self, other: Self) -> Self {
        let mut rhs = other;
        rhs.sign = rhs.sign * Sign::Negative;
        self.add_ratio(rhs)
    }

    pub fn add_ratio(self, other: Self) -> Self {
        let (mut lhs, rhs) = (self, other);

        if lhs.denom == rhs.denom && lhs.sign == rhs.sign {
            lhs.numer += rhs.numer;
            return lhs.reduce();
        }

        match (lhs.sign.is_pos(), rhs.sign.is_pos()) {
            (true, true) | (false, false) => {
                let mut res = Self::unsigned_add((lhs.numer, *lhs.denom), (rhs.numer, *rhs.denom));
                res.sign = lhs.sign;
                res
            }

            // -lhs + rhs
            (false, true) => {
                let lhs_abs = lhs.abs();
                let rhs_abs = rhs.abs();

                if lhs_abs >= rhs_abs {
                    // lhs >= rhs => -(lhs - rhs)
                    let mut res =
                        Self::unsigned_sub((lhs.numer, *lhs.denom), (rhs.numer, *rhs.denom));
                    res.sign = Sign::Negative;
                    res
                } else {
                    // rhs > lhs => rhs - lhs
                    Self::unsigned_sub((rhs.numer, *rhs.denom), (lhs.numer, *lhs.denom))
                }
            }

            // lhs - rhs
            (true, false) => {
                let lhs_abs = lhs.abs();
                let rhs_abs = rhs.abs();

                if lhs_abs >= rhs_abs {
                    // lhs >= rhs => lhs - rhs
                    Self::unsigned_sub((lhs.numer, *lhs.denom), (rhs.numer, *rhs.denom))
                } else {
                    // rhs > lhs => -(lhs + rhs)
                    let mut res =
                        Self::unsigned_sub((rhs.numer, *rhs.denom), (lhs.numer, *lhs.denom));
                    res.sign = Sign::Negative;
                    res
                }
            }
        }
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.sign *= Sign::Negative;
        self
    }
}

//TODO: checked add
impl Add for Rational {
    type Output = Rational;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_ratio(rhs)
    }
}

impl Sub for Rational {
    type Output = Rational;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_ratio(rhs)
    }
}

impl Mul for Rational {
    type Output = Rational;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_ratio(rhs)
    }
}

impl Div for Rational {
    type Output = Rational;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_ratio(rhs)
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let v1 = self.numer * *other.denom;
        let v2 = other.numer * *self.denom;
        v1.cmp(&v2)
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign.is_neg() {
            write!(f, "-")?;
        }

        if *self.denom == 1 {
            write!(f, "{}", self.numer)
        } else {
            write!(f, "({} / {})", self.numer, self.denom)
        }
    }
}

impl Display for NonZeroUInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod rational_test {

    use crate::prelude::*;
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
}
