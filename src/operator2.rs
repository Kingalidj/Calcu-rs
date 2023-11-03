use std::collections::HashMap;

use crate::{
    base::Base,
    numeric::{Number, Undefined},
    traits::{Bool, CalcursType, Num},
};

use crate::constants as C;

mod utils {
    pub trait KeyValueFields: Eq {
        type Key;
        type Value;
        fn key(&self) -> &Self::Key;
        fn value(&self) -> &Self::Value;
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct KeyValueEntry<T> {
        entry: T,
    }

    impl<T: KeyValueFields + Eq> KeyValueFields for KeyValueEntry<T> {
        type Key = T::Key;
        type Value = T::Value;

        fn key(&self) -> &Self::Key {
            self.entry.key()
        }

        fn value(&self) -> &Self::Value {
            todo!()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct VecMap<T> {
        vec: Vec<T>,
    }

    impl<T: Default> Default for VecMap<T> {
        fn default() -> Self {
            Self { vec: vec![] }
        }
    }
}

/// coeff + mul1 + mul2 + mul3...
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Add {
    coeff: Number,
    args: Vec<Mul>,
}

pub type Sub = Add;

impl Add {
    /// n1 + n2
    pub fn add(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            coeff: C::ZERO,
            args: Default::default(),
        }
        .base()
    }
}

impl CalcursType for Add {
    fn base(self) -> Base {
        Base::Add(self.into())
    }
}

impl std::fmt::Display for Add {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

type MulCoeff = Number;
type MulArgs = Vec<Pow>;

/// coeff * pow1 * pow2 * pow3...
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Mul {
    coeff: Number,
    args: Vec<Pow>,
}

pub type Div = Mul;

impl Mul {
    pub fn mul(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        let mut mul = Mul {
            coeff: C::ONE,
            args: Default::default(),
        };
        mul.arg(n1.base());
        mul.arg(n2.base());
        mul.reduce()
    }

    pub fn div(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self::mul(n1, Pow::pow(n2, C::MINUS_ONE))
    }

    fn arg(&mut self, b: Base) {
        use Base as B;

        match b {
            B::BooleanAtom(bool) => self.coeff *= bool.to_num(),
            B::Number(n) => self.coeff *= n,

            B::Pow(pow) => self.pow_arg(*pow),
            B::Mul(mul) => {
                self.coeff *= mul.coeff;
                mul.args.into_iter().for_each(|p| self.pow_arg(p));
            }

            B::Var(_) | B::Dummy | B::Add(_) => self.pow_arg(Pow::from_base(b)),
        }
    }

    fn pow_arg(&mut self, pow: Pow) {
        if let Some(p) = self.args.iter_mut().find(|p| p.base == pow.base) {
            // (x^a) * (x^b) = x^(a + b)
            p.exp = Add::add(p.exp, pow.exp);
        } else {
            self.args.push(pow);
        }
    }

    fn reduce_internal(&mut self) {
        if self.coeff == C::ZERO {
            self.args.clear();
            return;
        }

        let mut set = HashMap::new();

        self.args
            .into_iter()
            .map(|mut pow| {
                pow.reduce_internal();
                pow
            })
            .filter(|pow| pow.base != C::ZERO.base())
            .for_each(|mut pow| {
                if let Some(mut exp) = set.remove(&pow.base) {
                    // x^a * x^b = x^(a + b)
                    pow.exp = Add::add(pow.exp, exp);
                    set.insert(pow.base, pow.exp);
                } else {
                    set.insert(pow.base, pow.exp);
                }
            });

        self.args = set
            .into_iter()
            .map(|(base, exp)| Pow { base, exp })
            .collect();
    }

    fn reduce(mut self) -> Base {
        self.reduce_internal();

        match self {
            Mul { coeff: C::ZERO, .. } => C::ZERO.base(),
            Mul { coeff, args } if args.is_empty() => coeff.base(),
            Mul { coeff, args } if args.is_empty() => coeff.base(),
            Mul {
                coeff: C::ONE,
                mut args,
            } if args.len() == 1 => args.pop().unwrap().base(),
            _ => self.base(),
        }
    }

    #[inline]
    fn from_base(b: Base) -> Self {
        Mul {
            coeff: C::ONE,
            args: vec![Pow::from_base(b)],
        }
    }
}

impl CalcursType for Mul {
    fn base(self) -> Base {
        Base::Mul(self.into())
    }
}

impl std::fmt::Display for Mul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

/// base^exp
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

    fn reduce_internal(&mut self) {
        use Base as B;

        if let B::BooleanAtom(atom) = self.base {
            self.base = atom.to_num().base();
        }

        if let B::BooleanAtom(atom) = self.exp {
            self.exp = atom.to_num().base();
        }
    }

    pub fn reduce(self) -> Base {
        use Base as B;

        if let B::BooleanAtom(atom) = self.base {
            self.base = atom.to_num().base();
        }

        if let B::BooleanAtom(atom) = self.exp {
            self.exp = atom.to_num().base();
        }

        match (self.base, self.exp) {
            // 1^x = 1
            (B::Number(C::ONE), _) => C::ONE.base(),
            // x^1 = x
            (x, B::Number(C::ONE)) => x,

            // (_)^0:
            // 0^0 = undefined
            (B::Number(C::ZERO), B::Number(C::ZERO)) => Undefined.base(),
            // x^0 = 1 (x != 0)
            (x, B::Number(C::ZERO)) => C::ONE.base(),
            // 0^x = undefined (x < 0)
            (B::Number(C::ZERO), B::Number(x)) if x.is_neg() => Undefined.base(),
            // 0^x = 0 (x > 0)
            (B::Number(C::ZERO), x) => Undefined.base(),

            // n^-1 = 1/n (n != 0)
            (B::Number(n), B::Number(C::MINUS_ONE)) => C::ONE.div_num(n).base(),

            // (x^y)^z = x^(y*z)
            (B::Pow(pow), z) => Pow::pow(pow.base, Mul::mul(pow.exp, z)),

            _ => self.base(),
        }
    }

    #[inline]
    const fn from_base(b: Base) -> Self {
        Pow {
            base: b,
            exp: C::ONE.base(),
        }
    }
}

impl CalcursType for Pow {
    fn base(self) -> Base {
        Base::Pow(self.into())
    }
}

impl std::fmt::Display for Pow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
