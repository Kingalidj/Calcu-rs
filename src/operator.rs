use std::{
    collections::BTreeMap,
    fmt::{self, Display},
};

use crate::{
    base::CalcursType,
    numeric::constants::{MINUS_ONE, ONE, UNDEF, ZERO},
    pattern::pat,
};

pat!(use);

use Base as B;

/// helper container for [Add]
///
///  n1 * mul_1 + n2 * mul_2 + n3 * mul_3 + ... \
/// <=> n1 * {pow_1_1 * pow_1_2 * ... } + n2 * { pow_2_1 * pow_2_2 * ...} + ...
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub(crate) struct AddArgs {
    pub(crate) args: BTreeMap<MulArgs, Number>,
}

impl AddArgs {
    pub fn insert_mul(&mut self, mul: Mul) {
        if let pat!(Number: 0) = mul.coeff {
            return;
        }

        if let Some(coeff) = self.args.get_mut(&mul.args) {
            // 2x + 3x => 5x
            *coeff += mul.coeff;

            if let pat!(Number: 0) = coeff {
                self.args.remove(&mul.args);
            }
        } else {
            self.args.insert(mul.args, mul.coeff);
        }
    }

    #[inline]
    pub fn into_mul_iter(self) -> impl Iterator<Item = Mul> {
        self.args
            .into_iter()
            .map(|(args, coeff)| Mul { coeff, args })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.args.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    #[inline]
    pub fn is_mul(&self) -> bool {
        self.len() == 1
    }

    #[inline]
    pub fn into_mul(mut self) -> Option<Mul> {
        if self.is_mul() {
            let (args, coeff) = self.args.pop_first().unwrap();
            Some(Mul { args, coeff })
        } else {
            None
        }
    }
}

impl Display for AddArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.args.iter().rev();

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
    pub(crate) coeff: Number,
    pub(crate) args: AddArgs,
}

pub type Sub = Add;

impl Add {
    /// n1 + n2
    pub fn add(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            coeff: ZERO,
            args: Default::default(),
        }
        .arg(n1.base())
        .arg(n2.base())
        .reduce()
    }

    /// n1 + (-1 * n2)
    #[inline]
    pub fn sub(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Add::add(n1, Mul::mul(MINUS_ONE, n2))
    }

    fn arg(mut self, b: Base) -> Self {
        match b {
            B::Number(num) => self.coeff += num,
            B::Mul(mul) => self.args.insert_mul(mul),
            B::Add(add) => {
                self.coeff += add.coeff;
                add.args
                    .into_mul_iter()
                    .for_each(|mul| self.args.insert_mul(mul));
            }

            base @ (B::Symbol(_) | B::Pow(_)) => self.args.insert_mul(Mul::from_base(base)),
        };

        self
    }

    fn reduce(self) -> Base {
        match (self.coeff, self.args) {
            // x + {} => x
            (x, args) if args.is_empty() => x.base(),
            // 0 + x * y => x * y
            (pat!(Number: 0), x) if x.is_mul() => x.into_mul().unwrap().base(),

            (coeff, args) => Self { coeff, args }.base(),
        }
    }
}

impl CalcursType for Add {
    #[inline]
    fn base(self) -> Base {
        B::Add(self)
    }
}

impl Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.args)?;

        if let pat!(Number: 0) = self.coeff {
        } else {
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
    pub(crate) args: BTreeMap<Base, Base>,
}

impl MulArgs {
    pub fn insert_pow(&mut self, mut pow: Pow) {
        if let Some(exp) = self.args.remove(&pow.base) {
            pow.exp = Add::add(exp, pow.exp);
            self.args.insert(pow.base, pow.exp);
        } else {
            self.args.insert(pow.base, pow.exp);
        }
    }

    #[inline]
    pub fn into_pow_iter(self) -> impl Iterator<Item = Pow> {
        self.args.into_iter().map(|(base, exp)| Pow { base, exp })
    }

    #[inline]
    pub fn is_pow(&self) -> bool {
        self.args.len() == 1
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.args.len()
    }

    #[inline]
    pub fn into_pow(mut self) -> Option<Pow> {
        if self.is_pow() {
            let (base, exp) = self.args.pop_first().unwrap();
            Some(Pow { base, exp })
        } else {
            None
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    #[inline]
    pub fn from_base(b: Base) -> Self {
        let pow = Pow::from_base(b);
        let args = BTreeMap::from([(pow.base, pow.exp)]);
        Self { args }
    }
}

impl Display for MulArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.args.iter();

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
    pub(crate) coeff: Number,
    pub(crate) args: MulArgs,
}

pub type Div = Mul;

impl Mul {
    /// n1 * n2
    pub fn mul(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            coeff: ONE,
            args: MulArgs::default(),
        }
        .arg(n1.base())
        .arg(n2.base())
        .reduce()
    }

    /// n1 * (1 / n2)
    #[inline]
    pub fn div(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Mul::mul(n1, Pow::pow(n2, MINUS_ONE))
    }

    fn arg(mut self, b: Base) -> Self {
        if B::Number(UNDEF) == b {
            self.coeff = Undefined.into();
            return self;
        } else if self.coeff == ZERO || B::Number(ONE) == b {
            return self;
        }

        match b {
            B::Number(num) => self.coeff *= num,
            B::Pow(pow) => self.args.insert_pow(*pow),
            B::Mul(mul) => {
                self.coeff *= mul.coeff;
                mul.args
                    .into_pow_iter()
                    .for_each(|pow| self.args.insert_pow(pow))
            }
            base @ (B::Symbol(_) | B::Add(_)) => self.args.insert_pow(Pow::from_base(base)),
        }

        self
    }

    fn reduce(self) -> Base {
        match (self.coeff, self.args) {
            (pat!(Number: undef), _) => Undefined.base(),

            // 0 * x => 0
            (pat!(Number: 0), _) => ZERO.base(),

            // 1 * x => x
            (pat!(Number: 1), args) if args.is_pow() => args.into_pow().unwrap().base(),

            // x * {} => x
            (coeff, args) if args.is_empty() => coeff.base(),

            (coeff, args) => Self { coeff, args }.base(),
        }
    }

    fn fmt_parts(args: &MulArgs, coeff: &Number, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if args.len() == 1 {
            if let pat!(Number: 1) = coeff {
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
            coeff: ONE,
            args: MulArgs::from_base(b),
        }
    }
}

impl CalcursType for Mul {
    fn base(self) -> Base {
        B::Mul(self)
    }
}

impl Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Mul::fmt_parts(&self.args, &self.coeff, f)
    }
}

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

    fn reduce(self) -> Base {
        match (self.base, self.exp) {
            (pat!(undef), _) | (_, pat!(undef)) => Undefined.base(),

            // 0^0 = undefined
            (pat!(0), pat!(0)) => Undefined.base(),

            // 1^x = 1
            (pat!(1), _) => ONE.base(),

            // x^1 = x
            (x, pat!(1)) => x,

            // x^0 = 1 | x != 0
            (pat!(Number: n), pat!(0)) if !n.is_zero() => ONE.base(),

            // 0^x = undefined | x < 0
            (pat!(0), pat!(-)) => Undefined.base(),

            // 0^x = 0 | x > 0
            (pat!(0), pat!(+)) => ZERO.base(),

            // n^-1 = 1/n | n != 0
            (pat!(Rational: r), pat!(-1)) => (Rational::one() / r).base(),

            // (x^y)^z = x^(y*z)
            (base, exp) => Self { base, exp }.base(),
        }
    }

    #[inline]
    fn from_base(b: Base) -> Self {
        if let B::Pow(pow) = b {
            *pow
        } else {
            Pow {
                base: b,
                exp: B::Number(ONE),
            }
        }
    }

    fn fmt_parts(base: &Base, exp: &Base, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if &B::Number(ONE) == exp {
            write!(f, "{base}")
        } else {
            write!(f, "{base}^{exp}")
        }
    }
}

impl CalcursType for Pow {
    fn base(self) -> Base {
        B::Pow(self.into())
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
    fn add(case: u32, x: Base, y: Base, z: Base) {
        let expr = x + y;
        eprintln!("case {case}: {expr} = {z}");
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
    fn sub(case: u32, x: Base, y: Base, z: Base) {
        let expr = x - y;
        eprintln!("case {case}: {expr} = {z}");
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
    fn mul(case: u32, x: Base, y: Base, z: Base) {
        let expr = x * y;
        eprintln!("case {case}: {expr} = {z}");
        assert_eq!(expr, z);
    }

    #[test_case(1, base!(0), base!(0), base!(nan))]
    #[test_case(2, base!(0), base!(5), base!(0))]
    #[test_case(3, base!(5), base!(0), base!(nan))]
    #[test_case(4, base!(5), base!(5), base!(1))]
    #[test_case(5, base!(1), base!(3), base!(1 / 3))]
    fn div(case: u32, x: Base, y: Base, z: Base) {
        let div = x / y;
        eprintln!("case {case}: {div} = {z}");
        assert_eq!(div, z);
    }

    #[test_case(1, base!(1), base!(1 / 100), base!(1))]
    #[test_case(2, base!(4), base!(1), base!(4))]
    #[test_case(3, base!(0), base!(0), base!(nan))]
    #[test_case(4, base!(0), base!(-3 / 4), base!(nan))]
    #[test_case(5, base!(0), base!(3 / 4), base!(0))]
    #[test_case(6, base!(1 / 2), base!(-1), base!(4 / 2))]
    fn pow(case: u32, x: Base, y: Base, z: Base) {
        let expr = x ^ y;
        eprintln!("case {case}: {expr} = {z}");
        assert_eq!(expr, z);
    }

    #[test]
    fn polynom() {
        assert_eq!(
            base!(v: x) * base!(v: x) * base!(2) + base!(3) * base!(v: x) + base!(4 / 3),
            base!(4 / 3) + (base!(v: x) ^ base!(2)) * base!(2) + base!(3) * base!(v: x)
        );
    }
}
