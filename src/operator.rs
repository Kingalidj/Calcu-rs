use std::{collections::HashMap, fmt};

use crate::{
    base::{Base, BaseKind},
    numeric::{Integer, Number, NumberKind, ONE, ZERO},
    traits::{CalcursType, Numeric},
};

/// Represents addition in symbolic expressions
///
/// Implemented with a coefficient and a hashmap: \
/// coeff + key1*value1 + key2*value2 + ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Add {
    coeff: NumberKind,
    arg_map: HashMap<BaseKind, NumberKind>,
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.arg_map.iter();

        if let Some((k, v)) = iter.next() {
            write!(f, "{v}{k}")?;
        }

        for (k, v) in iter {
            write!(f, " + {v}{k}")?;
        }

        if !self.coeff.is_zero() {
            write!(f, " + {}", self.coeff)?;
        }

        Ok(())
    }
}

impl std::hash::Hash for Add {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "Add".hash(state);
        self.coeff.hash(state);
        for a in &self.arg_map {
            a.hash(state);
        }
    }
}

impl CalcursType for Add {
    fn base(self) -> Base {
        BaseKind::Add(self).into()
    }
}

impl Add {
    pub fn add<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Base {
        let b1 = b1.base();
        let b2 = b2.base();
        Self::add_kind(b1.kind, b2.kind)
    }

    pub fn add_kind(b1: BaseKind, b2: BaseKind) -> Base {
        let add = Self {
            coeff: Integer::num(0),
            arg_map: Default::default(),
        };
        add.append_basic(b1).append_basic(b2).cleanup()
    }

    fn cleanup(self) -> Base {
        if self.arg_map.is_empty() {
            Number::from(self.coeff).base()
        } else {
            self.base()
        }
    }

    fn append_basic(mut self, b: BaseKind) -> Self {
        use BaseKind as B;

        match b {
            B::Boolean(_) => todo!(),

            B::Number(num) => {
                if !num.is_zero() {
                    self.coeff = self.coeff.add_kind(num.kind);
                }
            }
            B::Mul(mut mul) => {
                let mut coeff = Integer::num(1);
                (mul.coeff, coeff) = (coeff, mul.coeff);
                self.append_term(coeff, BaseKind::Mul(mul));
            }

            B::Add(add) => {
                add.arg_map.into_iter().for_each(|(term, coeff)| {
                    self.append_term(coeff, term);
                });

                self.coeff = self.coeff.add_kind(add.coeff);
            }
            B::Var(_) | B::Dummy => {
                let coeff = ONE.kind.clone();
                self.append_term(coeff, b);
            }
        }

        self
    }

    /// adds the term: coeff*b
    fn append_term(&mut self, coeff: NumberKind, b: BaseKind) {
        if let Some(mut key) = self.arg_map.remove(&b) {
            key = key.add_kind(coeff);

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mul {
    coeff: NumberKind,
    arg_map: HashMap<BaseKind, BaseKind>,
}

impl Mul {
    pub fn mul<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Base {
        let b1 = b1.base();
        let b2 = b2.base();

        Self {
            coeff: Integer::num(1),
            arg_map: Default::default(),
        }
        .append_basic(b1.kind)
        .append_basic(b2.kind)
        .cleanup()
    }

    fn append_basic(mut self, b: BaseKind) -> Self {
        use BaseKind as B;

        match b {
            B::Number(n) => {
                if n.is_zero() {
                    self.coeff = ZERO.kind.clone();
                    self.arg_map.clear();
                } else {
                    self.coeff = self.coeff.mul_kind(n.kind);
                }
            }
            B::Mul(mul) => {
                self.coeff = self.coeff.mul_kind(mul.coeff);
                mul.arg_map
                    .into_iter()
                    .for_each(|(key, val)| self.append_term(key, val))
            }
            B::Var(_) | B::Boolean(_) | B::Add(_) | B::Dummy => {
                let exp = BaseKind::Number(ONE.clone());
                self.append_term(b, exp);
            }
        }
        self
    }

    /// adds the term: b^exp
    fn append_term(&mut self, b: BaseKind, exp: BaseKind) {
        if let Some(key) = self.arg_map.remove(&b) {
            let exp = Add::add_kind(key, exp);
            self.arg_map.insert(b, exp.kind);
        } else {
            self.arg_map.insert(b, exp);
        }
    }

    fn cleanup(self) -> Base {
        if self.coeff.is_zero() || self.arg_map.is_empty() {
            self.coeff.base()
        } else {
            self.base()
        }
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.arg_map.iter();

        if let Some((k, v)) = iter.next() {
            write!(f, "{v}^{k}")?;
        }

        for (k, v) in iter {
            write!(f, " * {v}^{k}")?;
        }

        if !self.coeff.is_one() {
            write!(f, " * {}", self.coeff)?;
        }

        Ok(())
    }
}

impl std::hash::Hash for Mul {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "Mul".hash(state);
        self.coeff.hash(state);
        for a in &self.arg_map {
            a.hash(state);
        }
    }
}

impl CalcursType for Mul {
    fn base(self) -> Base {
        BaseKind::Mul(self).into()
    }
}
