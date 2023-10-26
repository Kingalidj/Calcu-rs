use crate::{
    base::Base,
    numeric::{Infinity, Number, Sign, Undefined},
    traits::{CalcursType, Num},
};
use num::Integer;

type UInt = u64;

#[repr(transparent)]
#[derive(Debug, derive_more::Display, Copy, Clone, PartialEq, Eq, Hash)]
struct NonZeroUInt(UInt);

impl NonZeroUInt {
    #[inline]
    fn new(n: UInt) -> Option<Self> {
        if n == 0 {
            None
        } else {
            Some(NonZeroUInt(n))
        }
    }

    /// panics if n is not 0
    fn set(&mut self, n: UInt) {
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
        self.numer == 1 && self.denom() == 1 && self.is_neg == false
    }

    fn sign(&self) -> Sign {
        match self.is_neg {
            true => Sign::Negitive,
            false => Sign::Positive,
        }
    }
}

impl Rational {
    pub fn frac(num: i32, den: i32) -> Number {
        match (num, den) {
            (0, 0) => return Undefined.into(),
            (_, 0) => return Infinity::new(Sign::UnSigned).into(), // TODO
            _ => (),
        }

        let is_neg = (num * den).is_negative();
        let numer = num.abs() as UInt;
        let denom = NonZeroUInt::new(den.abs() as UInt).unwrap();

        Self {
            is_neg,
            numer,
            denom,
        }
        .reduce()
        .into()
    }

    pub fn int(n: i32) -> Number {
        let is_neg = n.is_negative();
        let numer = n.abs() as UInt;
        let denom = NonZeroUInt::new(1).unwrap();
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

    /// assume denom != 0 and f1.denom != f2.denom
    #[inline]
    fn unsigned_add(f1: (UInt, UInt), f2: (UInt, UInt)) -> Rational {
        let lcm = f1.1.lcm(&f2.1);
        let lhs_numer = f1.0 * (lcm / f1.1);
        let rhs_numer = f2.0 * (lcm / f2.1);
        Rational {
            is_neg: false,
            numer: lhs_numer + rhs_numer,
            denom: NonZeroUInt::new(lcm).unwrap(),
        }
        .reduce()
    }

    /// assume denom != 0 and f1.denom != f2.denom and f1 >= f2
    #[inline]
    fn unsigned_sub(f1: (UInt, UInt), f2: (UInt, UInt)) -> Rational {
        let lcm = f1.1.lcm(&f2.1);
        let lhs_numer = f1.0 * (lcm / f1.1);
        let rhs_numer = f2.0 * (lcm / f2.1);
        Rational {
            is_neg: false,
            numer: lhs_numer - rhs_numer,
            denom: NonZeroUInt::new(lcm).unwrap(),
        }
        .reduce()
    }

    pub fn abs(&self) -> Self {
        let mut res = *self;
        res.is_neg = false;
        res
    }

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
}

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
        self.numer /= gcd_ad.clone();
        self.numer *= rhs.numer / gcd_bc.clone();
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
        self.numer /= gcd_ac.clone();
        self.numer *= rhs.denom() / gcd_bd.clone();
        self.denom.div(gcd_bd);
        self.denom.mul(rhs.numer / gcd_ac);
        self.into()
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        let self_val = self.numer * other.denom();
        let other_val = other.numer * self.denom();

        if self_val < other_val {
            Some(Ordering::Less)
        } else if self_val > other_val {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
