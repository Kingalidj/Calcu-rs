use std::{
    collections::BTreeMap,
    fmt::{self, Display},
};

use crate::{
    base::{Base, CalcursType},
    numeric::{Numeric, Undefined},
    pattern::itm,
    rational::Rational,
};

/// helper container for [Add]
///
///  n1 * mul_1 + n2 * mul_2 + n3 * mul_3 + ... \
/// <=> n1 * {pow_1_1 * pow_1_2 * ... } + n2 * { pow_2_1 * pow_2_2 * ...} + ...
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub(crate) struct AddArgs {
    // TODO: maybe Vec<(_, _)>?
    __args: BTreeMap<MulArgs, Numeric>,
}

impl AddArgs {
    pub fn insert_mul(&mut self, mul: Mul) {
        if mul.coeff.is_zero() {
            return;
        }

        if let Some(coeff) = self.__args.get_mut(&mul.args) {
            // 2x + 3x => 5x
            *coeff += mul.coeff;

            if coeff.is_zero() {
                self.__args.remove(&mul.args);
            }
        } else {
            self.__args.insert(mul.args, mul.coeff);
        }
    }

    #[inline]
    pub fn into_mul_iter(self) -> impl Iterator<Item = Mul> {
        self.__args
            .into_iter()
            .map(|(args, coeff)| Mul { coeff, args })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.__args.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.__args.is_empty()
    }

    #[inline]
    pub fn is_mul(&self) -> bool {
        self.len() == 1
    }

    #[inline]
    pub fn into_mul(mut self) -> Option<Mul> {
        if self.is_mul() {
            let (args, coeff) = self.__args.pop_first().unwrap();
            Some(Mul { args, coeff })
        } else {
            None
        }
    }
}

impl Display for AddArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.__args.iter().rev();

        if let Some((args, coeff)) = iter.next() {
            Mul::fmt_parts(args, coeff, f)?;
        }

        for (args, coeff) in iter {
            write!(f, " + ")?;
            Mul::fmt_parts(args, coeff, f)?;
        }

        Ok(())
    }
}

/// Represents addition in symbolic expressions
///
/// Implemented with a coefficient and an [AddArgs]: \
/// coeff + mul1 + mul2 + mul3...
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Add {
    pub(crate) coeff: Numeric,
    pub(crate) args: AddArgs,
}

pub type Sub = Add;

impl Add {
    /// n1 + n2
    pub fn add(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            coeff: Rational::zero().num(),
            args: Default::default(),
        }
        .arg(n1.base())
        .arg(n2.base())
        .reduce()
    }

    /// n1 + (-1 * n2)
    #[inline]
    pub fn sub(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Add::add(n1, Mul::mul(Rational::minus_one(), n2))
    }

    pub fn subs(self, dict: &crate::base::SubsDict) -> Base {
        let mut sum = self.coeff.base();
        for mul in self.args.into_mul_iter() {
            sum += mul.subs(dict);
        }
        sum
    }

    fn arg(mut self, b: Base) -> Self {
        use Base as B;
        match b {
            B::Numeric(num) => self.coeff += num,
            B::Mul(mul) => self.args.insert_mul(mul),
            B::Add(mut add) => {
                if add.args.len() > self.args.len() {
                    (add, self) = (self, add);
                }

                self.coeff += add.coeff;
                add.args
                    .into_mul_iter()
                    .for_each(|mul| self.args.insert_mul(mul));
            }

            base @ (B::Symbol(_) | B::Pow(_) | B::Derivative(_)) => {
                self.args.insert_mul(Mul::from_base(base))
            }
        };

        self
    }

    fn reduce(self) -> Base {
        match (self.coeff, self.args) {
            // x + {} => x
            (x, args) if args.is_empty() => x.base(),
            // 0 + x * y => x * y
            (itm!(num: 0), x) if x.is_mul() => x.into_mul().unwrap().base(),

            (coeff, args) => Self { coeff, args }.base(),
        }
    }
}

impl CalcursType for Add {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Add(self)
    }
}

impl Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.args)?;

        if !self.coeff.is_zero() {
            write!(f, " + {}", self.coeff)?;
        }

        Ok(())
    }
}

/// helper container for [Mul]
///
/// k1 ^ v1 * k2 ^ v2 * k3 ^ v3 * ...
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MulArgs {
    __args: BTreeMap<Base, Base>,
}

impl MulArgs {
    pub fn insert_pow(&mut self, mut pow: Pow) {
        if let Some(exp) = self.__args.remove(&pow.base) {
            pow.exp = Add::add(exp, pow.exp);
            self.__args.insert(pow.base, pow.exp);
        } else {
            self.__args.insert(pow.base, pow.exp);
        }
    }

    #[inline]
    pub fn into_pow_iter(self) -> impl Iterator<Item = Pow> {
        self.__args.into_iter().map(|(base, exp)| Pow { base, exp })
    }

    #[inline]
    pub fn is_pow(&self) -> bool {
        self.__args.len() == 1
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.__args.len()
    }

    #[inline]
    pub fn try_into_pow(mut self) -> Option<Pow> {
        if self.is_pow() {
            let (base, exp) = self.__args.pop_first().unwrap();
            Some(Pow { base, exp })
        } else {
            None
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.__args.is_empty()
    }

    #[inline]
    pub fn from_base(b: Base) -> Self {
        let pow = Pow::from_base(b);
        let __args = BTreeMap::from([(pow.base, pow.exp)]);
        Self { __args }
    }
}

impl Display for MulArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.__args.iter();

        if let Some((base, exp)) = iter.next() {
            Pow::fmt_parts(base, exp, f)?;
        }

        for (base, exp) in iter {
            write!(f, " * ")?;
            Pow::fmt_parts(base, exp, f)?;
        }

        Ok(())
    }
}

/// Represents multiplication in symbolic expressions
///
/// Implemented with a coefficient and a hashmap: \
/// coeff * pow1 * pow2 * pow3...
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Mul {
    pub(crate) coeff: Numeric,
    pub(crate) args: MulArgs,
}

pub type Div = Mul;

impl Mul {
    /// n1 * n2
    pub fn mul(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            coeff: Rational::one().num(),
            args: MulArgs::default(),
        }
        .arg(n1.base())
        .arg(n2.base())
        .reduce()
    }

    /// n1 * (1 / n2)
    #[inline]
    pub fn div(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Mul::mul(n1, Pow::pow(n2, Rational::minus_one()))
    }

    pub fn subs(self, dict: &crate::base::SubsDict) -> Base {
        let mut prod = self.coeff.base();
        for pow in self.args.into_pow_iter() {
            prod *= pow.subs(dict);
        }
        prod
    }

    fn arg(mut self, b: Base) -> Self {
        use Base as B;

        if Undefined.base() == b {
            self.coeff = Undefined.into();
            return self;
        } else if self.coeff == Rational::zero().num() || Rational::one().base() == b {
            return self;
        }

        match b {
            B::Numeric(num) => self.coeff *= num,
            B::Pow(pow) => self.args.insert_pow(*pow),
            B::Mul(mut mul) => {
                if mul.args.len() > self.args.len() {
                    (self, mul) = (mul, self);
                }

                self.coeff *= mul.coeff;
                mul.args
                    .into_pow_iter()
                    .for_each(|pow| self.args.insert_pow(pow))
            }
            base @ (B::Symbol(_) | B::Add(_) | B::Derivative(_)) => {
                self.args.insert_pow(Pow::from_base(base))
            }
        }

        self
    }

    fn reduce(self) -> Base {
        match (self.coeff, self.args) {
            (itm!(num: undef), _) => Undefined.base(),

            // 0 * x => 0
            (itm!(num: 0), _) => Rational::zero().base(),

            // 1 * x => x
            (itm!(num: 1), args) if args.is_pow() => args.try_into_pow().unwrap().base(),

            // x * {} => x
            (coeff, args) if args.is_empty() => coeff.base(),

            (coeff, args) => Self { coeff, args }.base(),
        }
    }

    fn fmt_parts(args: &MulArgs, coeff: &Numeric, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if args.len() == 1 {
            if coeff.is_one() {
                write!(f, "{args}")?;
            } else {
                write!(f, "{coeff}{args}")?;
            }
        } else {
            write!(f, "{coeff} * {args}")?;
        }
        Ok(())
    }

    #[inline]
    fn from_base(b: Base) -> Self {
        Self {
            coeff: Rational::one().num(),
            args: MulArgs::from_base(b),
        }
    }
}

impl CalcursType for Mul {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Mul(self)
    }
}

impl Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Mul::fmt_parts(&self.args, &self.coeff, f)
    }
}

// TODO: pow of number
/// base^exp
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pow {
    pub(crate) base: Base,
    pub(crate) exp: Base,
}

impl Pow {
    pub fn pow(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            base: n1.base(),
            exp: n2.base(),
        }
        .reduce()
    }

    pub fn subs(self, dict: &crate::base::SubsDict) -> Base {
        let base = self.base.subs(dict);
        let exp = self.exp.subs(dict);
        Pow::pow(base, exp)
    }

    fn reduce(self) -> Base {
        match (self.base, self.exp) {
            (itm!(undef), _) | (_, itm!(undef)) => Undefined.base(),

            // 0^0 = undefined
            (itm!(0), itm!(0)) => Undefined.base(),

            // 1^x = 1
            (itm!(1), _) => Rational::one().base(),

            // x^1 = x
            (x, itm!(1)) => x,

            // x^0 = 1 | x != 0
            (itm!(Numeric: n), itm!(0)) if !n.is_zero() => Rational::one().base(),

            // 0^x = undefined | x < 0
            (itm!(0), itm!(-)) => Undefined.base(),

            // 0^x = 0 | x > 0
            (itm!(0), itm!(+)) => Rational::zero().base(),

            // n^-1 = 1/n | n != 0
            (itm!(Rational: r), itm!(-1)) => (Rational::one() / r).base(),

            (itm!(Numeric: n1), itm!(Numeric: n2)) => {
                n1.checked_pow_num(n2).map(|n| n.base()).unwrap_or(
                    Self {
                        base: n1.base(),
                        exp: n2.base(),
                    }
                    .base(),
                )
            }

            // (x^y)^z = x^(y*z)
            (base, exp) => Self { base, exp }.base(),
        }
    }

    #[inline]
    fn from_base(b: Base) -> Self {
        if let Base::Pow(pow) = b {
            *pow
        } else {
            Pow {
                base: b,
                exp: Rational::one().base(),
            }
        }
    }

    fn fmt_parts(base: &Base, exp: &Base, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let itm!(1) = exp {
            write!(f, "{base}")
        } else {
            write!(f, "{base}^{exp}")
        }
    }
}

impl CalcursType for Pow {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Pow(self.into())
    }
}

impl Display for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Pow::fmt_parts(&self.base, &self.exp, f)
    }
}

#[cfg(test)]
mod op_test {
    use crate::base;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    #[test_case(1, base!(2), base!(3), base!(5))]
    #[test_case(2, base!(1 / 2), base!(1 / 2), base!(1))]
    #[test_case(3, base!(v: x), base!(v: x), base!(v: x) * base!(2))]
    #[test_case(4, base!(-3), base!(1 / 2), base!(-5 / 2))]
    #[test_case(5, base!(inf), base!(4), base!(inf))]
    #[test_case(6, base!(neg_inf), base!(4), base!(neg_inf))]
    #[test_case(7, base!(pos_inf), base!(pos_inf), base!(pos_inf))]
    #[test_case(8, base!(neg_inf), base!(pos_inf), base!(nan))]
    #[test_case(9, base!(nan), base!(pos_inf), base!(nan))]
    #[test_case(10, base!(4 / 2), base!(0), base!(2))]
    fn add(_case: u32, x: Base, y: Base, z: Base) {
        let expr = x + y;
        assert_eq!(expr, z);
    }

    #[test_case(1, base!(-1), base!(3), base!(-4))]
    #[test_case(2, base!(-3), base!(1 / 2), base!(-7 / 2))]
    #[test_case(3, base!(1 / 2), base!(1 / 2), base!(0))]
    #[test_case(4, base!(inf), base!(4), base!(inf))]
    #[test_case(5, base!(neg_inf), base!(4 / 2), base!(neg_inf))]
    #[test_case(6, base!(pos_inf), base!(4), base!(pos_inf))]
    #[test_case(7, base!(pos_inf), base!(pos_inf), base!(nan))]
    #[test_case(8, base!(neg_inf), base!(pos_inf), base!(neg_inf))]
    #[test_case(9, base!(nan), base!(inf), base!(nan))]
    fn sub(_case: u32, x: Base, y: Base, z: Base) {
        let expr = x - y;
        assert_eq!(expr, z)
    }

    #[test_case(1, base!(-1), base!(3), base!(-3))]
    #[test_case(2, base!(-1), base!(0), base!(0))]
    #[test_case(3, base!(-1), base!(3) * base!(0), base!(0))]
    #[test_case(4, base!(-3), base!(1 / 2), base!(-3 / 2))]
    #[test_case(5, base!(1 / 2), base!(1 / 2), base!(1 / 4))]
    #[test_case(6, base!(inf), base!(4), base!(inf))]
    #[test_case(7, base!(neg_inf), base!(4 / 2), base!(neg_inf))]
    #[test_case(8, base!(pos_inf), base!(4), base!(pos_inf))]
    #[test_case(9, base!(pos_inf), base!(-1), base!(neg_inf))]
    #[test_case(10, base!(pos_inf), base!(pos_inf), base!(pos_inf))]
    #[test_case(11, base!(neg_inf), base!(pos_inf), base!(neg_inf))]
    #[test_case(12, base!(nan), base!(inf), base!(nan))]
    fn mul(_case: u32, x: Base, y: Base, z: Base) {
        let expr = x * y;
        assert_eq!(expr, z);
    }

    #[test_case(1, base!(0), base!(0), base!(nan))]
    #[test_case(2, base!(0), base!(5), base!(0))]
    #[test_case(3, base!(5), base!(0), base!(nan))]
    #[test_case(4, base!(5), base!(5), base!(1))]
    #[test_case(5, base!(1), base!(3), base!(1 / 3))]
    fn div(_case: u32, x: Base, y: Base, z: Base) {
        let div = x / y;
        assert_eq!(div, z);
    }

    #[test_case(1, base!(1), base!(1 / 100), base!(1))]
    #[test_case(2, base!(4), base!(1), base!(4))]
    #[test_case(3, base!(0), base!(0), base!(nan))]
    #[test_case(4, base!(0), base!(-3 / 4), base!(nan))]
    #[test_case(5, base!(0), base!(3 / 4), base!(0))]
    #[test_case(6, base!(1 / 2), base!(-1), base!(4 / 2))]
    fn pow(_case: u32, x: Base, y: Base, z: Base) {
        let expr = x.pow(y);
        assert_eq!(expr, z);
    }

    #[test]
    fn polynom() {
        assert_eq!(
            base!(v: x) * base!(v: x) * base!(2) + base!(3) * base!(v: x) + base!(4 / 3),
            base!(4 / 3) + (base!(v: x).pow(base!(2))) * base!(2) + base!(3) * base!(v: x)
        );
    }
}
