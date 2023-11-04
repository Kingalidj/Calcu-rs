use crate::{
    base::{Base, CalcursType, Num},
    numeric::{Number, Sign, Undefined},
};
use num::Integer;

type UInt = u64;

/// Nonzero integer value
///
/// will panic otherwise
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct NonZeroUInt(UInt);

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

impl std::ops::Deref for NonZeroUInt {
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
#[derive(Debug, Clone, Eq, Copy, Hash)]
pub struct Rational {
    pub(crate) is_neg: bool,
    pub(crate) numer: UInt,
    pub(crate) denom: NonZeroUInt,
}

impl PartialEq for Rational {
    fn eq(&self, other: &Self) -> bool {
        if self.numer == 0 && other.numer == 0 {
            true
        } else {
            self.is_neg == other.is_neg && self.numer == other.numer && self.denom == other.denom
        }
    }
}

impl CalcursType for Rational {
    fn base(self) -> Base {
        Base::Number(self.into())
    }
}

impl Num for Rational {
    #[inline]
    fn is_zero(&self) -> bool {
        self.numer == 0
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.numer == 1 && *self.denom == 1 && !self.is_neg
    }

    #[inline]
    fn is_neg_one(&self) -> bool {
        self.numer == 1 && *self.denom == 1 && self.is_neg
    }

    #[inline]
    fn sign(&self) -> Option<Sign> {
        Some(match self.is_neg {
            true => Sign::Neg,
            false => Sign::Pos,
        })
    }
}

impl Rational {
    pub fn new(num: i32, den: i32) -> Self {
        match (num, den) {
            (_, 0) => panic!("Rational::new: found 0 denominator"),
            _ => (),
        }

        let is_neg = (num * den).is_negative();
        let numer = num.unsigned_abs() as UInt;
        let denom = NonZeroUInt::new(den.unsigned_abs() as UInt);

        Self {
            is_neg,
            numer,
            denom,
        }
        .reduce()
        .into()
    }

    pub fn frac_num(num: i32, den: i32) -> Number {
        match (num, den) {
            (_, 0) => return Undefined.into(),
            _ => Self::new(num, den).into(),
        }
    }

    pub fn int_num(n: i32) -> Number {
        let is_neg = n.is_negative();
        let numer = n.unsigned_abs() as UInt;
        let denom = NonZeroUInt::new(1);
        Self {
            is_neg,
            numer,
            denom,
        }
        .into()
    }

    #[inline]
    pub fn numer(&self) -> UInt {
        self.numer
    }

    #[inline]
    pub fn denom(&self) -> UInt {
        self.denom.0
    }

    fn reduce(mut self) -> Self {
        match (self.numer, *self.denom) {
            (_, 0) => unreachable!(),

            // 0 / x => 0 / 1
            (0, _) => {
                self.denom.set(1);
                self.is_neg = false;
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
            is_neg: false,
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
            is_neg: false,
            numer: lhs_numer - rhs_numer,
            denom: NonZeroUInt::new(lcm),
        }
        .reduce()
    }

    pub fn abs(&self) -> Self {
        let mut res = *self;
        res.is_neg = false;
        res
    }

    pub fn div_ratio(self, other: Self) -> Self {
        let (mut lhs, rhs) = (self, other);
        lhs.is_neg = lhs.is_neg || rhs.is_neg;
        let gcd_ac = lhs.numer.gcd(&rhs.numer);
        let gcd_bd = lhs.denom.gcd(&rhs.denom);
        lhs.numer /= gcd_ac;
        lhs.numer *= *rhs.denom / gcd_bd;
        lhs.denom.div(gcd_bd);
        lhs.denom.mul(rhs.numer / gcd_ac);
        lhs.is_neg = lhs.is_neg && (lhs.numer != 0);
        lhs
    }

    pub fn mul_ratio(self, other: Self) -> Self {
        let (mut lhs, rhs) = (self, other);
        lhs.is_neg = lhs.is_neg || rhs.is_neg;
        let gcd_ad = lhs.numer.gcd(&rhs.denom);
        let gcd_bc = lhs.denom.gcd(&rhs.numer);
        lhs.numer /= gcd_ad;
        lhs.numer *= rhs.numer / gcd_bc;
        lhs.denom.div(gcd_bc);
        lhs.denom.mul(*rhs.denom / gcd_ad);
        lhs.is_neg = lhs.is_neg && (lhs.numer != 0);
        lhs
    }

    pub fn sub_ratio(self, other: Self) -> Self {
        let mut rhs = other;
        rhs.is_neg = !rhs.is_neg;
        self.add_ratio(rhs).into()
    }

    pub fn add_ratio(self, other: Self) -> Self {
        let (mut lhs, rhs) = (self, other);

        if lhs.denom == rhs.denom && lhs.is_neg == rhs.is_neg {
            lhs.numer += rhs.numer;
            return lhs.reduce();
        }

        match (lhs.is_pos(), rhs.is_pos()) {
            (true, true) | (false, false) => {
                let mut res = Self::unsigned_add((lhs.numer, *lhs.denom), (rhs.numer, *rhs.denom));
                res.is_neg = lhs.is_neg;
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
                    res.is_neg = true;
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
                    res.is_neg = true;
                    res
                }
            }
        }
    }
}

impl std::ops::Neg for Rational {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.is_neg = !self.is_neg;
        self
    }
}

//TODO: checked add
impl std::ops::Add for Rational {
    type Output = Rational;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_ratio(rhs)
    }
}

impl std::ops::Sub for Rational {
    type Output = Rational;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_ratio(rhs)
    }
}

impl std::ops::Mul for Rational {
    type Output = Rational;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_ratio(rhs)
    }
}

impl std::ops::Div for Rational {
    type Output = Rational;

    fn div(self, rhs: Self) -> Self::Output {
        self.div_ratio(rhs)
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let v1 = self.numer * *other.denom;
        let v2 = other.numer * *self.denom;
        v1.partial_cmp(&v2)
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_neg {
            write!(f, "-")?;
        }

        if *self.denom == 1 {
            write!(f, "{}", self.numer)
        } else {
            write!(f, "({} / {})", self.numer, self.denom)
        }
    }
}

impl std::fmt::Display for NonZeroUInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
