use std::collections::BTreeMap;

use crate::{
    base::{Base, CalcursType, Num},
    numeric::{
        constants::{MINUS_ONE, ONE, UNDEF, ZERO},
        Number, Undefined,
    },
};

use Base as B;

/// helper container for [Add]
///
///  n1 * mul_1 + n2 * mul_2 + n3 * mul_3 + ...
/// <=> n1 * {pow_1_1 * pow_1_2 * ... } + n2 * { pow_2_1 * pow_2_2 * ...} + ...
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub(crate) struct AddArgs {
    pub(crate) args: BTreeMap<MulArgs, Number>,
}

impl AddArgs {
    pub fn insert_mul(&mut self, mul: Mul) {
        if mul.coeff.is_zero() {
            return;
        }

        if let Some(coeff) = self.args.get_mut(&mul.args) {
            // 2x + 3x => 5x
            *coeff += mul.coeff;
            if coeff.is_zero() {
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
    pub fn to_mul(mut self) -> Option<Mul> {
        if self.is_mul() {
            let (args, coeff) = self.args.pop_first().unwrap();
            Some(Mul { args, coeff })
        } else {
            None
        }
    }
}

impl std::fmt::Display for AddArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.args.iter().rev();

        if let Some((args, coeff)) = iter.next() {
            Mul::fmt_parts(args, coeff, f)?;
        }

        while let Some((args, coeff)) = iter.next() {
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

            base @ (B::Var(_) | B::Dummy | B::Pow(_)) => self.args.insert_mul(Mul::from_base(base)),
        };

        self
    }

    fn reduce(self) -> Base {
        match (self.coeff, self.args) {
            // x + {} => x
            (x, args) if args.is_empty() => x.base(),
            // 0 + x * y => x * y
            (n, x) if n.is_zero() && x.is_mul() => x.to_mul().unwrap().base(),

            (coeff, args) => Self { coeff, args }.base(),
        }
    }
}

impl CalcursType for Add {
    #[inline]
    fn base(self) -> Base {
        B::Add(self.into())
    }
}

impl std::fmt::Display for Add {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.args)?;

        if !self.coeff.is_zero() {
            write!(f, " + {}", self.coeff)?;
        };

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
    pub fn to_pow(mut self) -> Option<Pow> {
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

impl std::fmt::Display for MulArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.args.iter();

        if let Some((base, exp)) = iter.next() {
            Pow::fmt_parts(base, exp, f)?;
        }

        while let Some((base, exp)) = iter.next() {
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
            base @ (B::Var(_) | B::Dummy | B::Add(_)) => self.args.insert_pow(Pow::from_base(base)),
        }

        self
    }

    fn reduce(self) -> Base {
        match (self.coeff, self.args) {
            (UNDEF, _) => UNDEF.base(),
            // 0 * x => 0
            (n, _) if n.is_zero() => n.base(),
            // 1 * x => x
            (n, args) if n.is_one() && args.is_pow() => args.to_pow().unwrap().base(),
            // x * {} => x
            (n, args) if args.is_empty() => n.base(),

            (coeff, args) => Self { coeff, args }.base(),
        }
    }

    fn fmt_parts(
        args: &MulArgs,
        coeff: &Number,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
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
            coeff: ONE,
            args: MulArgs::from_base(b),
        }
    }
}

impl CalcursType for Mul {
    fn base(self) -> Base {
        B::Mul(self.into())
    }
}

impl std::fmt::Display for Mul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
        use B::Number as N;

        match (self.base, self.exp) {
            (N(UNDEF), N(UNDEF)) => Undefined.base(),

            // 1^x = 1
            (N(n1), _) if n1 == ONE => N(n1),
            // x^1 = x
            (n1, N(n2)) if n2 == ONE => n1,

            // 0^0 = undefined
            (N(n1), N(n2)) if n1 == ZERO && n2 == ZERO => N(UNDEF),

            // x^0 = 1 | x != 0 [0^0 already handled]
            (_, N(n2)) if n2 == ZERO => N(ONE),

            // 0^x = undefined | x < 0
            (n1, N(n2)) if n1 == N(ZERO) && n2.is_neg() => N(UNDEF),

            // 0^x = 0 | x > 0 [0^x | x = 0 & x < 0 already handled]
            (N(n1), _) if n1 == ZERO => N(n1),

            // n^-1 = 1/n | n != 0 [0^x already handled]
            (N(n1), N(n2)) if n1 != ZERO && n2.is_neg_one() => N(ONE / n1),

            // (x^y)^z = x^(y*z)
            //(B::Pow(pow), z) => Pow::pow(pow.base, Mul::mul(pow.exp, z)),
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

    fn fmt_parts(base: &Base, exp: &Base, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::fmt::Display for Pow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Pow::fmt_parts(&self.base, &self.exp, f)
    }
}

#[cfg(test)]
mod op_test {
    use crate::base;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    #[test_case(base!(2), base!(3), base!(5))]
    #[test_case(base!(1 / 2), base!(1 / 2), base!(1))]
    #[test_case(base!(v: x), base!(v: x), base!(v: x) * base!(2))]
    #[test_case(base!(-3), base!(1 / 2), base!(-5 / 2))]
    #[test_case(base!(inf), base!(4), base!(inf))]
    #[test_case(base!(neg_inf), base!(4), base!(neg_inf))]
    #[test_case(base!(pos_inf), base!(pos_inf), base!(pos_inf))]
    #[test_case(base!(neg_inf), base!(pos_inf), base!(nan))]
    #[test_case(base!(nan), base!(pos_inf), base!(nan))]
    #[test_case(base!(4 / 2), base!(0), base!(2))]
    fn add(x: Base, y: Base, z: Base) {
        assert_eq!(x + y, z);
    }

    #[test_case(base!(-1), base!(3), base!(-4))]
    #[test_case(base!(-3), base!(1 / 2), base!(-7 / 2))]
    #[test_case(base!(1 / 2), base!(1 / 2), base!(0))]
    #[test_case(base!(inf), base!(4), base!(inf))]
    #[test_case(base!(neg_inf), base!(4 / 2), base!(neg_inf))]
    #[test_case(base!(pos_inf), base!(4), base!(pos_inf))]
    #[test_case(base!(pos_inf), base!(pos_inf), base!(nan))]
    #[test_case(base!(neg_inf), base!(pos_inf), base!(neg_inf))]
    #[test_case(base!(nan), base!(inf), base!(nan))]
    fn sub(x: Base, y: Base, z: Base) {
        assert_eq!(x - y, z)
    }

    #[test_case(base!(-1), base!(3), base!(-3))]
    #[test_case(base!(-1), base!(0), base!(0))]
    #[test_case(base!(-1), base!(3) * base!(0), base!(0))]
    #[test_case(base!(-3), base!(1 / 2), base!(-3 / 2))]
    #[test_case(base!(1 / 2), base!(1 / 2), base!(1 / 4))]
    #[test_case(base!(inf), base!(4), base!(inf))]
    #[test_case(base!(neg_inf), base!(4 / 2), base!(neg_inf))]
    #[test_case(base!(pos_inf), base!(4), base!(pos_inf))]
    #[test_case(base!(pos_inf), base!(-1), base!(neg_inf))]
    #[test_case(base!(pos_inf), base!(pos_inf), base!(pos_inf))]
    #[test_case(base!(neg_inf), base!(pos_inf), base!(neg_inf))]
    #[test_case(base!(nan), base!(inf), base!(nan))]
    fn mul(x: Base, y: Base, z: Base) {
        assert_eq!(x * y, z);
    }

    #[test_case(base!(0), base!(0), base!(nan))]
    #[test_case(base!(0), base!(5), base!(0))]
    #[test_case(base!(5), base!(0), base!(nan))]
    #[test_case(base!(5), base!(5), base!(1))]
    #[test_case(base!(1), base!(3), base!(1 / 3))]
    fn div(x: Base, y: Base, z: Base) {
        assert_eq!(x / y, z);
    }

    #[test_case(base!(1), base!(1 / 100), base!(1))]
    #[test_case(base!(4), base!(1), base!(4))]
    #[test_case(base!(0), base!(0), base!(nan))]
    #[test_case(base!(0), base!(-3 / 4), base!(nan))]
    #[test_case(base!(0), base!(3 / 4), base!(0))]
    #[test_case(base!(1 / 2), base!(-1), base!(4 / 2))]
    fn pow(x: Base, y: Base, z: Base) {
        assert_eq!(x ^ y, z);
    }

    #[test]
    fn polynom() {
        assert_eq!(
            base!(v: x) * base!(v: x) * base!(2) + base!(3) * base!(v: x) + base!(4 / 3),
            base!(4 / 3) + (base!(v: x) ^ base!(2)) * base!(2) + base!(3) * base!(v: x)
        );
    }
}
