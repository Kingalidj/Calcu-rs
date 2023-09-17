use core::fmt;
use std::{marker::PhantomData, ops};

use crate::boolean::{And, BooleanBinOp, BooleanType, Not, Or};
use derive_more::{Display, From};

#[derive(From, Clone, Debug, Display)]
pub enum CalcursType {
    #[from(forward)]
    Typed(Typed),
    Symbol(Symbol),
}

impl ops::Not for CalcursType {
    type Output = Option<CalcursType>;

    fn not(self) -> Self::Output {
        Not::as_calcurs_op(self)
    }
}

impl ops::BitAnd<CalcursType> for CalcursType {
    type Output = Option<CalcursType>;

    fn bitand(self, rhs: CalcursType) -> Self::Output {
        And::as_calcurs_op(self, rhs)
    }
}

impl ops::BitOr<CalcursType> for CalcursType {
    type Output = Option<CalcursType>;

    fn bitor(self, rhs: CalcursType) -> Self::Output {
        Or::as_calcurs_op(self, rhs)
    }
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

pub trait Operator: 'static {
    type OperandType;
}

pub trait Commutativity: Operator {}

pub trait Associativity: Operator {}

pub trait Distributivity<OP: Operator>: Operator {}

pub trait Identity: Operator {
    const IDENTITY: Self::OperandType;
}

pub type PatRef<OP1, OP2> = &'static Pattern<OP1, OP2>;

#[derive(Debug, Clone)]
pub enum Pattern<OP1: 'static, OP2: 'static> {
    Op1(PatRef<OP1, OP2>, PatRef<OP1, OP2>),
    Op2(PatRef<OP1, OP2>, PatRef<OP1, OP2>),
    Item(char),
    __(PhantomData<(OP1, OP2)>),
}

impl<OP1, OP2> fmt::Display for Pattern<OP1, OP2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Pattern as P;
        match self {
            P::Op1(lhs, rhs) => write!(f, "({} op1 {})", lhs, rhs),
            P::Op2(lhs, rhs) => write!(f, "({} op2 {})", lhs, rhs),
            P::Item(c) => write!(f, "{c}"),
            P::__(_) => unreachable!(),
        }
    }
}

pub trait Law<OP1: Operator, OP2: Operator> {
    const FROM: Pattern<OP1, OP2>;
    const INTO: Pattern<OP1, OP2>;
}

// a op1 (b op2 c)
// into
// (a op1 b) op2 (a op1 c)
#[derive(Debug, Clone)]
pub struct DistributivityRule<OP1, OP2> {
    __: PhantomData<(OP1, OP2)>,
}

// OP1 is dist over OP2 && OP1::ElementType == OP2::ElementType
impl<OP2: Operator, OP1: Operator<OperandType = OP2::OperandType> + Distributivity<OP2>>
    Law<OP1, OP2> for DistributivityRule<OP1, OP2>
{
    const FROM: Pattern<OP1, OP2> = Pattern::Op1(
        &Pattern::Item('a'),
        &Pattern::Op2(&Pattern::Item('b'), &Pattern::Item('c')),
    );

    const INTO: Pattern<OP1, OP2> = Pattern::Op2(
        &Pattern::Op1(&Pattern::Item('a'), &Pattern::Item('b')),
        &Pattern::Op1(&Pattern::Item('a'), &Pattern::Item('c')),
    );
}
