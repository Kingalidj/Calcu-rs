use std::{collections::BTreeMap, fmt};

use crate::{
    base::{Base, CalcursType},
    numeric::{Infinity, Numeric, Undefined},
    pattern::{get_itm, Item, Pattern},
    rational::Rational,
};

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

// TODO: pow of number
/// base^exp
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pow {
    pub(crate) base: Base,
    pub(crate) exp: Base,
}

/// helper container for [Add]
///
///  n1 * mul_1 + n2 * mul_2 + n3 * mul_3 + ... \
/// <=> n1 * {pow_1_1 * pow_1_2 * ... } + n2 * { pow_2_1 * pow_2_2 * ...} + ...
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub(crate) struct AddArgs {
    // TODO: maybe Vec<(_, _)>?
    __args: BTreeMap<MulArgs, Numeric>,
}

/// helper container for [Mul]
///
/// k1 ^ v1 * k2 ^ v2 * k3 ^ v3 * ...
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct MulArgs {
    __args: BTreeMap<Base, Base>,
}

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

    pub fn desc(&self) -> Pattern {
        let op = Item::Add;
        let lhs = self.coeff.desc().to_item();
        let rhs = self.args.desc().to_item();
        Pattern::Binary { lhs, op, rhs }
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
        let coeff = self.coeff.desc();

        if self.args.is_empty() {
            self.coeff.base()
        } else if coeff.is(Item::Zero) && self.args.is_mul() {
            self.args.into_mul().base()
        } else {
            self.base()
        }
    }
}

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
        let coeff = self.coeff.desc();

        if self.args.is_empty() {
            self.coeff.base()

        } else if coeff.is(Item::Undef) {
            Undefined.base()

        } else if coeff.is(Item::Zero) {
            Rational::zero().base()

        } else if coeff.is(Item::One) && self.args.is_pow() {
            self.args.into_pow().base()
        } else {
            self.base()
        }
    }

    fn fmt_parts(args: &MulArgs, coeff: &Numeric, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if args.len() == 1 {
            if coeff.desc().is(Item::One) {
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
    pub fn desc(&self) -> Pattern {
        let op = Item::Mul;
        let lhs = self.coeff.desc().to_item();
        let rhs = self.args.desc().to_item();
        Pattern::Binary { lhs, op, rhs }
    }

    #[inline]
    fn from_base(b: Base) -> Self {
        Self {
            coeff: Rational::one().num(),
            args: MulArgs::from_base(b),
        }
    }
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
        let b = self.base.desc();
        let e = self.exp.desc();

        if (b.is(Item::Undef) || e.is(Item::Undef))
            || (b.is(Item::Zero) && e.is(Item::Zero))
            || (b.is(Item::Zero) && e.is(Item::Neg))
        {
            // 0^0 / 0^-n => undef
            Undefined.base()

        } else if b.is(Item::One) {
            // 1^x => x
            self.base

        } else if b.is(Item::Numeric) && !b.is(Item::Zero) && e.is(Item::Zero) {
            // x^0 if x != 0 => 1
            Rational::one().base()

        } else if b.is(Item::Zero) && e.is(Item::Pos) {
            // 0^x if x > 0 => 0
            Rational::zero().base()

        } else if b.is(Item::Numeric) && e.is(Item::PosInf) {
            // x^(+oo) = +oo
            Infinity::pos().base()

        } else if b.is(Item::Numeric) && e.is(Item::NegInf) {
            // x^(-oo) = 0
            Rational::zero().base()

        } else if b.is(Item::Rational) && e.is(Item::MinusOne) {
            // n^-1 => 1 / n
            let r = get_itm!(Rational: self.base);
            (Rational::one() / r).base()

        } else if b.is(Item::Numeric) && e.is(Item::Numeric) {
            let n1 = get_itm!(Numeric: self.base);
            let n2 = get_itm!(Numeric: self.exp);
            n1.checked_pow_num(n2).map(|n| n.base()).unwrap_or(
                Pow {
                    base: n1.base(),
                    exp: n2.base(),
                }
                .base(),
            )
        } else {
            self.base()
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

    #[inline]
    pub fn desc(&self) -> Pattern {
        let op = Item::Pow;
        let lhs = self.base.desc().to_item();
        let rhs = self.exp.desc().to_item();
        Pattern::Binary { lhs, op, rhs }
    }

    fn fmt_parts(base: &Base, exp: &Base, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if exp.desc().is(Item::One) {
            return write!(f, "{base}");
        }

        let use_paren = match base {
            Base::Numeric(n) => {
                if n.desc().is(Item::Int) {
                    false
                } else {
                    true
                }
            },
            Base::Symbol(_) => false,
            _ => true,
        };

        if use_paren {
            write!(f, "({base})")
        } else {
            write!(f, "{base}")
        }?;




        if exp.desc().is(Item::Int) {
            write!(f, "^{exp}")
        }
        else {
            write!(f, "^({exp})")
        }
    }
}

impl AddArgs {
    pub fn insert_mul(&mut self, mul: Mul) {
        if mul.coeff.desc().is(Item::Zero) {
            return;
        }

        if let Some(coeff) = self.__args.get_mut(&mul.args) {
            // 2x + 3x => 5x
            *coeff += mul.coeff;

            if coeff.desc().is(Item::Zero) {
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
    pub fn into_mul(mut self) -> Mul {
        if self.is_mul() {
            let (args, coeff) = self.__args.pop_first().unwrap();
            Mul { args, coeff }
        } else {
            panic!("AddArgs::into_mul: not possible")
        }
    }

    pub fn desc(&self) -> Pattern {
        if self.__args.len() == 1 {
            let (args, coeff) = self.__args.first_key_value().unwrap();
            let op = Item::Mul;
            let lhs = coeff.desc().to_item();
            let rhs = args.desc().to_item();
            Pattern::Binary { lhs, op, rhs }
        } else {
            Pattern::Itm(Item::Add)
        }
    }
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
    pub fn into_pow(mut self) -> Pow {
        if self.is_pow() {
            let (base, exp) = self.__args.pop_first().unwrap();
            Pow { base, exp }
        } else {
            panic!("MulArgs::into_pow: not possible")
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

    #[inline]
    pub fn desc(&self) -> Pattern {
        Pattern::Itm(Item::Mul)
    }
}

impl CalcursType for Add {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Add(self)
    }
}

impl CalcursType for Mul {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Mul(self)
    }
}

impl CalcursType for Pow {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Pow(self.into())
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.args)?;

        if !self.coeff.desc().is(Item::Zero) {
            write!(f, " + {}", self.coeff)?;
        }

        Ok(())
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Mul::fmt_parts(&self.args, &self.coeff, f)
    }
}

impl fmt::Display for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Pow::fmt_parts(&self.base, &self.exp, f)
    }
}

impl fmt::Display for AddArgs {
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

impl fmt::Display for MulArgs {
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
