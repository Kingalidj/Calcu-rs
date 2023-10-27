use std::{
    collections::{HashMap, HashSet},
    fmt,
    hash::Hash,
};

use crate::{
    base::{Base, PTR},
    boolean::{BoolValue, BooleanAtom},
    constants::{FALSE, ONE, TRUE},
    numeric::{Number, Rational, Undefined},
    traits::{Bool, CalcursType, Num},
};

/// Represents addition in symbolic expressions
///
/// Implemented with a coefficient and a hashmap: \
/// coeff + key1 * value1 + key2 * value2 + ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Add {
    coeff: Number,
    arg_map: HashMap<Base, Number>,
}
pub type Sub = Add;

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.arg_map.is_empty() {
            return write!(f, "{}", self.coeff);
        }

        let mut iter = self.arg_map.iter();

        write!(f, "(")?;

        if let Some((k, v)) = iter.next() {
            write!(f, "{v}{k}")?;
        }

        for (k, v) in iter {
            write!(f, " + {v}{k}")?;
        }

        if !self.coeff.is_zero() {
            write!(f, " + {}", self.coeff)?;
        }

        write!(f, ")")?;

        Ok(())
    }
}

impl Hash for Add {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "Add".hash(state);
        self.coeff.hash(state);
        for a in self.arg_map.iter() {
            a.hash(state);
        }
    }
}

impl CalcursType for Add {
    #[inline]
    fn base(self) -> Base {
        Base::Add(self.into()).base()
    }
}

impl Add {
    /// a + b
    pub fn add(b1: impl CalcursType, b2: impl CalcursType) -> Base {
        Self {
            // TODO: default element
            coeff: Rational::int_num(0),
            arg_map: Default::default(),
        }
        .add_arg(b1.base())
        .add_arg(b2.base())
        .reduce()
    }

    /// a - b
    pub fn sub(b1: impl CalcursType, b2: impl CalcursType) -> Base {
        Self {
            coeff: Rational::int_num(0),
            arg_map: Default::default(),
        }
        .add_arg(b1.base())
        .add_arg(Mul::mul(b2.base(), Rational::int_num(-1)))
        .reduce()
    }

    fn reduce(self) -> Base {
        if self.arg_map.is_empty() {
            self.coeff.base()
        } else {
            self.base()
        }
    }

    fn add_arg(mut self, b: Base) -> Self {
        use Base as B;

        match b {
            B::BooleanAtom(bool) => self = self.add_num(bool.to_num()),
            B::Number(num) => self = self.add_num(num),

            B::Mul(mut mul) => {
                let mut coeff = Rational::int_num(1);
                (mul.coeff, coeff) = (coeff, mul.coeff);
                self.add_term(coeff, Base::Mul(mul));
            }

            B::Add(add) => {
                add.arg_map.into_iter().for_each(|(term, coeff)| {
                    self.add_term(coeff, term);
                });

                self.coeff = self.coeff.add_num(add.coeff);
            }
            B::Pow(_) | B::Not(_) | B::And(_) | B::Or(_) | B::Var(_) | B::Dummy => {
                let coeff = ONE.clone();
                self.add_term(coeff, b);
            }
        }

        self
    }

    fn add_num(mut self, n: Number) -> Self {
        if !n.is_zero() {
            self.coeff = self.coeff.add_num(n);
        }
        self
    }

    /// adds the term: coeff * b
    fn add_term(&mut self, coeff: Number, b: Base) {
        if let Some(mut key) = self.arg_map.remove(&b) {
            key = key.add_num(coeff);

            if !key.is_zero() {
                self.arg_map.insert(b, key);
            }
        } else if !coeff.is_zero() {
            self.arg_map.insert(b, coeff);
        }
    }
}

/// Represents multiplication in symbolic expressions
///
/// Implemented with a coefficient and a hashmap: \
/// coeff * key1^value1 * key2^value2 * ...
///
/// coeff is used to extract all [Number]s
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mul {
    coeff: Number,
    arg_map: HashMap<Base, Base>,
}
pub type Div = Mul;

impl Mul {
    /// a * b
    pub fn mul(b1: impl CalcursType, b2: impl CalcursType) -> Base {
        let b1 = b1.base();
        let b2 = b2.base();

        Self {
            coeff: Rational::int_num(1),
            arg_map: Default::default(),
        }
        .mul_arg(b1)
        .mul_arg(b2)
        .reduce()
    }

    /// a * b^-1 <=> a * (1 / b)
    pub fn div(b1: impl CalcursType, b2: impl CalcursType) -> Base {
        let b1 = b1.base();
        let b2 = b2.base();

        Self {
            coeff: Rational::int_num(1),
            arg_map: Default::default(),
        }
        .mul_arg(b1)
        .mul_arg(Pow::pow(b2, Rational::int_num(-1)))
        .reduce()
    }

    fn mul_arg(mut self, b: Base) -> Self {
        use Base as B;

        if self.coeff.is_zero() {
            return self;
        }

        match b {
            B::BooleanAtom(b) => self = self.mul_num(b.to_num()),
            B::Number(n) => self = self.mul_num(n),

            B::Mul(mul) => {
                self.coeff = self.coeff.mul_num(mul.coeff);
                mul.arg_map
                    .into_iter()
                    .for_each(|(key, val)| self.mul_term(key, val))
            }

            B::Pow(pow) => {
                self.mul_term(pow.base, pow.exp);
            }

            B::Not(_) | B::And(_) | B::Or(_) | B::Var(_) | B::Add(_) | B::Dummy => {
                let exp = ONE.clone().base();
                self.mul_term(b, exp);
            }
        }

        self
    }

    /// adds the term: b^exp
    fn mul_term(&mut self, b: Base, exp: Base) {
        if let Some(key) = self.arg_map.remove(&b) {
            let exp = Add::add(key, exp);
            self.arg_map.insert(b, exp);
        } else {
            self.arg_map.insert(b, exp);
        }
    }

    fn mul_num(mut self, n: Number) -> Self {
        if n.is_zero() {
            self.arg_map.clear();
            self.coeff = n;
        } else if !n.is_one() {
            self.coeff = self.coeff.mul_num(n);
        }
        self
    }

    fn reduce(self) -> Base {
        if self.coeff.is_zero() || self.arg_map.is_empty() {
            self.coeff.base()
        } else {
            self.base()
        }
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.arg_map.is_empty() {
            return write!(f, "{}", self.coeff);
        }

        let mut iter = self.arg_map.iter();

        write!(f, "(")?;

        if let Some((k, v)) = iter.next() {
            write!(f, "{k}^{v}")?;
        }

        for (k, v) in iter {
            write!(f, " * {k}^{v}")?;
        }

        if !self.coeff.is_one() {
            write!(f, " * {}", self.coeff)?;
        }

        write!(f, ")")?;

        Ok(())
    }
}

impl Hash for Mul {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "Mul".hash(state);
        self.coeff.hash(state);
        for a in self.arg_map.iter() {
            a.hash(state);
        }
    }
}

impl CalcursType for Mul {
    #[inline]
    fn base(self) -> Base {
        Base::Mul(self.into()).base()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pow {
    base: Base,
    exp: Base,
}

impl CalcursType for Pow {
    fn base(self) -> Base {
        Base::Pow(self.into())
    }
}

impl fmt::Display for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}^{})", self.base, self.exp)
    }
}

impl Pow {
    pub fn pow(b: impl CalcursType, e: impl CalcursType) -> Base {
        use Base as B;

        let base = b.base();
        let exp = e.base();

        match (base, exp) {
            (B::Number(n1), B::Number(n2)) if n1.is_zero() && n2.is_zero() => Undefined.base(),

            (_, B::Number(n)) if n.is_zero() => Rational::int_num(1).base(),
            (base, B::Number(n)) if n.is_one() => base,

            (B::Number(num), B::Number(n)) if n.is_neg_one() => {
                Rational::int_num(1).div_num(num).base()
            }

            (Base::Number(n1), B::Number(n2)) => {
                todo!()
            }

            // (a^e)^n => a^(e * n)
            (B::Pow(mut pow), B::Number(num)) => {
                pow.exp = Add::add(pow.exp, num.base());
                pow.base()
            }

            (base, exp) => Self { base, exp }.base(),
        }
    }
}

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

#[cfg(test)]
mod op_test {
    use crate::prelude::*;
    use pretty_assertions::assert_eq;

    macro_rules! c {
        (+inf) => {
            Infinity::pos().base()
        };

        (-inf) => {
            Infinity::neg().base()
        };

        (inf) => {
            Infinity::default().base()
        };

        (nan) => {
            Undefined.base()
        };

        (false) => {
            FALSE.clone()
        };

        (true) => {
            TRUE.clone()
        };

        ($int: literal) => {
            Rational::int_num($int).base()
        };

        ($val: literal / $denom: literal) => {
            Rational::frac_num($val, $denom).base()
        };

        (v: $var: tt) => {
            Variable::new(stringify!($var)).base()
        };
    }

    #[test]
    fn add() {
        assert_eq!(c!(2) + c!(3), c!(5));
        assert_eq!(c!(v:x) + c!(v:x) + c!(3), c!(3) + c!(v:x) + c!(v:x));
        assert_eq!(c!(-1) + c!(3), c!(2));
        assert_eq!(c!(-3) + c!(1 / 2), c!(-5 / 2));
        assert_eq!(c!(1 / 2) + c!(1 / 2), c!(1));
        assert_eq!(c!(inf) + c!(4), c!(inf));
        assert_eq!(c!(-inf) + c!(4), c!(-inf));
        assert_eq!(c!(+inf) + c!(+inf), c!(+inf));
        assert_eq!(c!(-inf) + c!(+inf), c!(nan));
        assert_eq!(c!(nan) + c!(inf), c!(nan));
        assert_eq!(c!(4 / 2), c!(2));
    }

    #[test]
    fn mul() {
        assert_eq!(c!(-1) * c!(3), c!(-3));
        assert_eq!(c!(-1) * c!(0), c!(0));
        assert_eq!(c!(-1) * c!(3) * c!(0), c!(0));
        assert_eq!(c!(-3) * c!(1 / 2), c!(-3 / 2));
        assert_eq!(c!(1 / 2) * c!(1 / 2), c!(1 / 4));
        assert_eq!(c!(inf) * c!(4), c!(inf));
        assert_eq!(c!(-inf) * c!(4 / 2), c!(-inf));
        assert_eq!(c!(+inf) * c!(4), c!(+inf));
        assert_eq!(c!(+inf) * c!(-1), c!(-inf));
        assert_eq!(c!(+inf) * c!(+inf), c!(+inf));
        assert_eq!(c!(-inf) * c!(+inf), c!(-inf));
        assert_eq!(c!(nan) * c!(inf), c!(nan));
    }

    #[test]
    fn div() {
        assert_eq!(c!(1) / c!(3), c!(1 / 3));
    }

    #[test]
    fn sub() {
        assert_eq!(c!(-1) - c!(3), c!(-4));
        assert_eq!(c!(-3) - c!(1 / 2), c!(-7 / 2));
        assert_eq!(c!(1 / 2) - c!(1 / 2), c!(0));
        assert_eq!(c!(inf) - c!(4), c!(inf));
        assert_eq!(c!(-inf) - c!(4 / 2), c!(-inf));
        assert_eq!(c!(+inf) - c!(4), c!(+inf));
        assert_eq!(c!(+inf) - c!(+inf), c!(nan));
        assert_eq!(c!(-inf) - c!(+inf), c!(-inf));
        assert_eq!(c!(nan) - c!(inf), c!(nan));
    }

    #[test]
    fn num_expr() {}

    #[test]
    fn and() {
        assert_eq!(c!(false) & c!(true), c!(false));
        assert_eq!(c!(true) & c!(true), c!(true));
        assert_eq!(c!(false) & c!(v:x), c!(false));
        assert_eq!(c!(true) & c!(v:x), c!(v:x));
        assert_eq!(c!(v:y) & c!(v:x), c!(v:x) & c!(v:y));

        assert_eq!(c!(false) & c!(3), c!(false));
        assert_eq!(c!(true) & c!(0), c!(false));
        assert_eq!(c!(true) & c!(10), c!(true));
    }

    #[test]
    fn or() {
        assert_eq!(c!(false) | c!(true), c!(true));
        assert_eq!(c!(true) | c!(v:x), c!(true));
        assert_eq!(c!(false) | c!(v:x), c!(v:x));
        assert_eq!(c!(v:y) | c!(v:x), c!(v:x) | c!(v:y));

        assert_eq!(c!(false) | c!(3), c!(true));
        assert_eq!(c!(true) | c!(0), c!(true));
        assert_eq!(c!(false) | c!(0), c!(false));
    }

    #[test]
    fn not() {
        assert_eq!(!c!(false), c!(true));
        assert_eq!(!c!(true), c!(false));
        assert_eq!(!c!(v:x), !c!(v:x));
        assert!(!c!(v:x) != c!(v:x));
    }

    #[test]
    fn bool_expr() {
        assert_eq!(c!(false) & c!(false) | c!(true), c!(true));
        assert_eq!(c!(false) | c!(false) & c!(true), c!(false));
        assert_eq!(c!(false) & c!(true) | c!(false), c!(false));
        assert_eq!(c!(true) & (c!(false) | c!(true)), c!(true));
    }
}
