use std::collections::BTreeMap;

use crate::{
    base::Base,
    numeric::{Number, Undefined},
    traits::{Bool, CalcursType, Num},
};

use crate::constants as C;
use Base as B;

//  n1 * mul_1 + n2 * mul_2 + n3 * mul_3 + ...
// <=> n1 * {pow_1_1 * pow_1_2 * ... } + n2 * { pow_2_1 * pow_2_2 * ...} + ...
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
struct AddArgs {
    args: BTreeMap<MulArgs, Number>,
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
    coeff: Number,
    args: AddArgs,
}

pub type Sub = Add;

impl Add {
    /// n1 + n2
    pub fn add(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            coeff: C::ZERO,
            args: Default::default(),
        }
        .arg(n1.base())
        .arg(n2.base())
        .reduce()
    }

    /// n1 + (-1 * n2)
    #[inline]
    pub fn sub(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Add::add(n1, Mul::mul(C::MINUS_ONE, n2))
    }

    fn arg(mut self, b: Base) -> Self {
        match b {
            B::BooleanAtom(atom) => self.coeff += atom.to_num(),
            B::Number(num) => self.coeff += num,

            B::Mul(mul) => self.args.insert_mul(*mul),
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
            // 0 + x => x
            (C::ZERO, x) if x.is_mul() => x.to_mul().unwrap().base(),

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

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct MulArgs {
    args: BTreeMap<Base, Base>,
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
    coeff: Number,
    args: MulArgs,
}

pub type Div = Mul;

impl Mul {
    /// n1 * n2
    pub fn mul(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            coeff: C::ONE,
            args: MulArgs::default(),
        }
        .arg(n1.base())
        .arg(n2.base())
        .reduce()
    }

    /// n1 * (1 / n2)
    #[inline]
    pub fn div(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Mul::mul(n1, Pow::pow(n2, C::MINUS_ONE))
    }

    fn arg(mut self, b: Base) -> Self {
        if self.coeff == C::ZERO || B::Number(C::ONE) == b {
            return self;
        }

        match b {
            B::BooleanAtom(atom) => self.coeff *= atom.to_num(),
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
            // 0 * x => 0
            (C::ZERO, _) => C::ZERO.base(),
            // 1 * x => x
            (C::ONE, x) if x.is_pow() => x.to_pow().unwrap().base(),
            // x * {} => x
            (x, args) if args.is_empty() => x.base(),

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
            coeff: C::ONE,
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
    base: Base,
    exp: Base,
}

impl Pow {
    pub fn pow(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            base: n1.base(),
            exp: n2.base(),
        }
        .reduce()
    }

    fn reduce(mut self) -> Base {
        if let B::BooleanAtom(atom) = self.base {
            self.base = atom.to_num().base();
        }

        if let B::BooleanAtom(atom) = self.exp {
            self.exp = atom.to_num().base();
        }

        match (self.base, self.exp) {
            // 1^x = 1
            (one @ B::Number(C::ONE), _) => one,
            // x^1 = x
            (x, B::Number(C::ONE)) => x,

            // 0^0 = undefined
            (B::Number(C::ZERO), B::Number(C::ZERO)) => Undefined.base(),

            // x^0 = 1 | x != 0 [0^0 already handled]
            (_, B::Number(C::ZERO)) => C::ONE.base(),

            // 0^x = undefined | x < 0
            (B::Number(C::ZERO), B::Number(x)) if x.is_neg() => Undefined.base(),

            // 0^x = 0 | x > 0 [0^x | x = 0 & x < 0 already handled]
            (zero @ B::Number(C::ZERO), _) => zero,

            // n^-1 = 1/n | n != 0 [0^x already handled]
            (B::Number(n), B::Number(C::MINUS_ONE)) if !n.is_zero() => C::ONE.div_num(n).base(),

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
                exp: B::Number(C::ONE),
            }
        }
    }

    fn fmt_parts(base: &Base, exp: &Base, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let B::Number(C::ONE) = exp {
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

    #[test_case(base!(1), base!(3), base!(1 / 3))]
    fn div(x: Base, y: Base, z: Base) {
        assert_eq!(x / y, z);
    }

    //#[test]
    //fn and() {
    //    assert_eq!(c!(false) & c!(true), c!(false));
    //    assert_eq!(c!(true) & c!(true), c!(true));
    //    assert_eq!(c!(false) & c!(v: x), c!(false));
    //    assert_eq!(c!(true) & c!(v: x), c!(v: x));
    //    assert_eq!(c!(v: y) & c!(v: x), c!(v: x) & c!(v: y));

    //    assert_eq!(c!(false) & c!(3), c!(false));
    //    assert_eq!(c!(true) & c!(0), c!(false));
    //    assert_eq!(c!(true) & c!(10), c!(true));
    //}

    //#[test]
    //fn or() {
    //    assert_eq!(c!(false) | c!(true), c!(true));
    //    assert_eq!(c!(true) | c!(v: x), c!(true));
    //    assert_eq!(c!(false) | c!(v: x), c!(v: x));
    //    assert_eq!(c!(v: y) | c!(v: x), c!(v: x) | c!(v: y));

    //    assert_eq!(c!(false) | c!(3), c!(true));
    //    assert_eq!(c!(true) | c!(0), c!(true));
    //    assert_eq!(c!(false) | c!(0), c!(false));
    //}

    //#[test]
    //fn not() {
    //    assert_eq!(!c!(false), c!(true));
    //    assert_eq!(!c!(true), c!(false));
    //    assert_eq!(!c!(v: x), !c!(v: x));
    //    assert!(!c!(v: x) != c!(v: x));
    //}

    //#[test]
    //fn bool_expr() {
    //    assert_eq!(c!(false) & c!(false) | c!(true), c!(true));
    //    assert_eq!(c!(false) | c!(false) & c!(true), c!(false));
    //    assert_eq!(c!(false) & c!(true) | c!(false), c!(false));
    //    assert_eq!(c!(true) & (c!(false) | c!(true)), c!(true));
    //}
}

#[cfg(hide)]
mod __ {
    use std::{collections::HashSet, fmt, hash::Hash};

    use crate::{
        base::{Base, PTR},
        boolean::{BoolValue, BooleanAtom},
        constants::{FALSE, TRUE},
        traits::{Bool, CalcursType},
    };

    /// Represents or in symbolic expressions
    ///
    /// Implemented with a HashSet: \
    /// v1 ∨ v2 ∨ v3...
    ///
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Or {
        arg_set: HashSet<Base>,
    }

    impl fmt::Display for Or {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut iter = self.arg_set.iter();

            if let Some(a) = iter.next() {
                write!(f, "{}", a)?;
            }

            for arg in iter {
                write!(f, " ∨ {}", arg)?;
            }

            Ok(())
        }
    }

    impl Hash for Or {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            "Or".hash(state);
            for arg in &self.arg_set {
                arg.hash(state)
            }
        }
    }

    impl CalcursType for Or {
        #[inline]
        fn base(self) -> Base {
            Base::Or(self).base()
        }
    }

    impl Bool for Or {
        fn bool_val(&self) -> BoolValue {
            if self.arg_set.contains(&TRUE) {
                BoolValue::True
            } else if self.arg_set.is_empty()
                || (self.arg_set.len() == 1 && self.arg_set.contains(&FALSE))
            {
                BoolValue::False
            } else {
                BoolValue::Unknown
            }
        }
    }

    impl Or {
        pub fn or(b1: impl CalcursType, b2: impl CalcursType) -> Base {
            Self {
                arg_set: Default::default(),
            }
            .or_arg(b1.base())
            .or_arg(b2.base())
            .reduce()
        }

        fn or_arg(mut self, b: Base) -> Self {
            use Base as B;

            if self.is_true() {
                return self;
            }

            match b {
                B::Or(or) => {
                    or.arg_set.into_iter().for_each(|b| {
                        self.or_term(b);
                    });
                }

                B::Number(n) => {
                    if let Some(bool) = n.to_bool() {
                        self.or_term(B::BooleanAtom(bool))
                    } else {
                        self.or_term(B::Number(n));
                    }
                }

                // neutral element
                B::BooleanAtom(BooleanAtom::False) => (),

                B::Not(_)
                | B::BooleanAtom(_)
                | B::And(_)
                | B::Add(_)
                | B::Mul(_)
                | B::Pow(_)
                | B::Var(_)
                | B::Dummy => self.or_term(b),
            }

            self
        }

        /// adds the term b to [Or]
        fn or_term(&mut self, b: Base) {
            self.arg_set.insert(b);
        }

        fn reduce(self) -> Base {
            if self.arg_set.len() == 1 {
                return self.arg_set.into_iter().next().unwrap().base();
            }

            match self.to_bool() {
                Some(b) => b.base(),
                None => self.base(),
            }
        }
    }

    /// Represents or in symbolic expressions
    ///
    /// Implemented with a HashSet: \
    /// v1 ∧ v2 ∧ v3...
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct And {
        arg_set: HashSet<Base>,
    }

    impl Bool for And {
        fn bool_val(&self) -> BoolValue {
            if self.arg_set.contains(&FALSE) {
                BoolValue::False
            } else if self.arg_set.is_empty()
                || (self.arg_set.len() == 1 && self.arg_set.contains(&TRUE))
            {
                BoolValue::True
            } else {
                BoolValue::Unknown
            }
        }
    }

    impl CalcursType for And {
        #[inline]
        fn base(self) -> Base {
            Base::And(self).base()
        }
    }

    impl fmt::Display for And {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut iter = self.arg_set.iter();

            if let Some(a) = iter.next() {
                write!(f, "{}", a)?;
            }

            for arg in iter {
                write!(f, " ∧ {}", arg)?;
            }

            Ok(())
        }
    }

    impl Hash for And {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            "And".hash(state);
            for arg in &self.arg_set {
                arg.hash(state)
            }
        }
    }

    impl And {
        pub fn and(b1: impl CalcursType, b2: impl CalcursType) -> Base {
            Self {
                arg_set: Default::default(),
            }
            .and_arg(b1.base())
            .and_arg(b2.base())
            .reduce()
        }

        fn and_arg(mut self, b: Base) -> Self {
            use Base as B;

            if self.is_false() {
                return self;
            }

            match b {
                B::And(and) => {
                    and.arg_set.into_iter().for_each(|b| {
                        self.and_term(b);
                    });
                }

                B::Number(n) => {
                    if let Some(bool) = n.to_bool() {
                        self.and_term(B::BooleanAtom(bool));
                    } else {
                        self.and_term(B::Number(n));
                    }
                }

                // neutral element
                B::BooleanAtom(BooleanAtom::True) => (),

                B::BooleanAtom(_)
                | B::Not(_)
                | B::Or(_)
                | B::Var(_)
                | B::Mul(_)
                | B::Pow(_)
                | B::Add(_)
                | B::Dummy => self.and_term(b),
            }
            self
        }

        fn and_term(&mut self, b: Base) {
            self.arg_set.insert(b);
        }

        fn reduce(self) -> Base {
            if self.arg_set.len() == 1 {
                return self.arg_set.into_iter().next().unwrap().base();
            }
            match self.to_bool() {
                Some(b) => b.base(),
                None => self.base(),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Not {
        val: PTR<Base>,
    }

    impl fmt::Display for Not {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "¬({})", self.val)
        }
    }

    impl CalcursType for Not {
        #[inline]
        fn base(self) -> Base {
            Base::Not(self).base()
        }
    }

    impl Not {
        pub fn new(b: Base) -> Self {
            Self { val: b.into() }
        }

        pub fn not(b: impl CalcursType) -> Base {
            use Base as B;
            let b = b.base();

            match b {
                B::Not(not) => not.val.base(),

                B::BooleanAtom(a) => match a {
                    BooleanAtom::True => FALSE.clone(),
                    BooleanAtom::False => TRUE.clone(),
                },

                B::Number(n) => {
                    if let Some(bool) = n.to_bool() {
                        bool.base()
                    } else {
                        Self::new(B::Number(n)).base()
                    }
                }

                B::Pow(_) | B::Var(_) | B::Add(_) | B::Mul(_) | B::Or(_) | B::And(_) | B::Dummy => {
                    Self::new(b).base()
                }
            }
        }
    }
}
