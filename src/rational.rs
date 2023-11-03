use crate::{
    base::Base,
    numeric::{Infinity, Number, Sign, Undefined},
    traits::{CalcursType, Num},
};
use num::Integer;

type UInt = u64;

#[repr(transparent)]
#[derive(Debug, derive_more::Display, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NonZeroUInt(UInt);

impl NonZeroUInt {
    #[inline]
    pub const fn new(n: UInt) -> Self {
        if n == 0 {
            panic!("NonZeroUInt::new: found 0");
        } else {
            NonZeroUInt(n)
        }
    }

    /// panics if n is not 0
    pub const fn set(&mut self, n: UInt) {
        if n == 0 {
            panic!("NonZeroUInt::set: found 0");
        }

        self.0 = n;
    }

    /// panics if n is not 0
    fn div(&mut self, n: UInt) {
        if n == 0 {
            panic!("NonZeroUInt::div: found 0");
        }

        self.0 /= n;
    }

    /// panics if n is not 0
    fn mul(&mut self, n: UInt) {
        if n == 0 {
            panic!("NonZeroUInt::div: found 0");
        }

        self.0 *= n;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    is_neg: bool,
    numer: UInt,
    denom: NonZeroUInt,
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_neg {
            write!(f, "-")?;
        }

        if self.denom() == 1 {
            write!(f, "{}", self.numer)
        } else {
            write!(f, "{} / {}", self.numer, self.denom)
        }
    }
}

impl CalcursType for Rational {
    fn base(self) -> Base {
        Base::Number(self.into())
    }
}

impl Num for Rational {
    fn is_zero(&self) -> bool {
        self.numer == 0
    }

    fn is_one(&self) -> bool {
        self.numer == 1 && self.denom() == 1 && !self.is_neg
    }

    fn is_neg_one(&self) -> bool {
        self.numer == 1 && self.denom() == 1 && self.is_neg
    }

    fn sign(&self) -> Sign {
        match self.is_neg {
            true => Sign::Negative,
            false => Sign::Positive,
        }
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
            (0, 0) => return Undefined.into(),
            (_, 0) => return Infinity::new(Sign::UnSigned).into(), // TODO
            _ => (),
        }

        Self::new(num, den).into()
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
        match (self.numer(), self.denom()) {
            (_, 0) => unreachable!(),

            (0, _) => {
                self.denom.set(1);
                self.is_neg = false;
            }

            (n, d) if n == d => {
                self.numer = 1;
                self.denom.set(1);
            }
            _ => {
                let g = self.numer.gcd(&self.denom());
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

    // TODO: user std::ops
    pub fn div_ratio(self, other: Self) -> Number {
        (self / other).into()
    }

    pub fn mul_ratio(self, other: Self) -> Number {
        (self * other).into()
    }

    pub fn sub_ratio(self, other: Self) -> Number {
        (self - other).into()
    }

    pub fn add_ratio(self, other: Self) -> Number {
        (self + other).into()
    }

    pub fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            let mut inv = *self;
            inv.denom.set(self.numer);
            inv.numer = self.denom();
            Some(inv)
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
        let mut lhs = self;

        if lhs.denom == rhs.denom && lhs.is_neg == rhs.is_neg {
            lhs.numer += rhs.numer;
            return lhs.reduce();
        }

        match (lhs.is_pos(), rhs.is_pos()) {
            (true, true) | (false, false) => {
                let mut res =
                    Self::unsigned_add((lhs.numer, lhs.denom()), (rhs.numer, rhs.denom()));
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
                        Self::unsigned_sub((lhs.numer, lhs.denom()), (rhs.numer, rhs.denom()));
                    res.is_neg = true;
                    res
                } else {
                    // rhs > lhs => rhs - lhs
                    Self::unsigned_sub((rhs.numer, rhs.denom()), (lhs.numer, lhs.denom()))
                }
            }

            // lhs - rhs
            (true, false) => {
                let lhs_abs = lhs.abs();
                let rhs_abs = rhs.abs();

                if lhs_abs >= rhs_abs {
                    // lhs >= rhs => lhs - rhs
                    Self::unsigned_sub((lhs.numer, lhs.denom()), (rhs.numer, rhs.denom()))
                } else {
                    // rhs > lhs => -(lhs + rhs)
                    let mut res =
                        Self::unsigned_sub((rhs.numer, rhs.denom()), (lhs.numer, lhs.denom()));
                    res.is_neg = true;
                    res
                }
            }
        }
    }
}

impl std::ops::Sub for Rational {
    type Output = Rational;

    fn sub(self, mut rhs: Self) -> Self::Output {
        rhs.is_neg = !rhs.is_neg;
        self + rhs
    }
}

impl std::ops::Mul for Rational {
    type Output = Rational;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.is_neg = self.is_neg || rhs.is_neg;
        let gcd_ad = self.numer.gcd(&rhs.denom());
        let gcd_bc = self.denom().gcd(&rhs.numer);
        self.numer /= gcd_ad;
        self.numer *= rhs.numer / gcd_bc;
        self.denom.div(gcd_bc);
        self.denom.mul(rhs.denom() / gcd_ad);
        self
    }
}

impl std::ops::Div for Rational {
    type Output = Rational;

    fn div(mut self, rhs: Self) -> Self::Output {
        self.is_neg = self.is_neg || rhs.is_neg;
        let gcd_ac = self.numer.gcd(&rhs.numer);
        let gcd_bd = self.denom().gcd(&rhs.denom());
        self.numer /= gcd_ac;
        self.numer *= rhs.denom() / gcd_bd;
        self.denom.div(gcd_bd);
        self.denom.mul(rhs.numer / gcd_ac);
        self
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let v1 = self.numer * other.denom();
        let v2 = other.numer * self.denom();
        v1.partial_cmp(&v2)
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
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
    }
}
