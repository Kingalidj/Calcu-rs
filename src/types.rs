use crate::boolean::BooleanType;
use derive_more::{Display, From};

#[derive(From, Clone, Debug, Display)]
pub enum CalcursType {
    #[from(forward)]
    Typed(Typed),
    Symbol(Symbol),
}

#[derive(From, Copy, Clone, Display, Debug, PartialEq)]
pub struct Symbol {
    pub ident: &'static str,
}

#[derive(From, Display, Debug, Clone)]
#[from(forward)]
pub enum Typed {
    Boolean(BooleanType),
}

/// defines a substitution that constrains the type of the replaced value: e.g a [BooleanType] can only
/// be replaced by a [BooleanType]
pub trait TypedSubstitude: Clone {
    fn subs(&self, _sym: Symbol, _val: Self) -> Self {
        self.clone()
    }
}

pub trait Simplify: Clone {
    fn simplify(&self) -> Self {
        self.clone()
    }
}

impl TypedSubstitude for CalcursType {
    fn subs(&self, sym: Symbol, val: Self) -> Self {
        use CalcursType as CT;
        use Typed as T;
        match (self, val) {
            (CT::Typed(t1), CT::Typed(t2)) => t1.subs(sym, t2).into(),
            (CT::Typed(T::Boolean(b)), CT::Symbol(s)) => b.subs(sym, s.into()).into(),

            (CT::Symbol(_), CT::Typed(t)) => t.into(),
            (CT::Symbol(_), CT::Symbol(s)) => s.into(),
        }
    }
}

impl Simplify for Symbol {}

impl Simplify for Typed {
    fn simplify(&self) -> Self {
        match self {
            Typed::Boolean(b) => b.simplify().into(),
        }
    }
}

impl Simplify for CalcursType {
    fn simplify(&self) -> Self {
        match self {
            CalcursType::Typed(t) => t.simplify().into(),
            CalcursType::Symbol(_) => self.clone(),
        }
    }
}

impl Symbol {
    pub const fn typ(ident: &'static str) -> CalcursType {
        CalcursType::Symbol(Self { ident })
    }
}

impl TypedSubstitude for Typed {
    fn subs(&self, sym: Symbol, val: Self) -> Self {
        use Typed as T;
        match (self, val) {
            (T::Boolean(b1), T::Boolean(b2)) => b1.subs(sym, b2).into(),
        }
    }
}
