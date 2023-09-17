use crate::core::{
    Associativity, CalcursType, Commutativity, Distributivity, Identity, Operator, Simplify,
    Symbol, Typed, TypedSubstitude,
};
use derive_more::{Display, From};
use std::ops;

mod algorithms;

pub const TRUE: BooleanType = BooleanType::True(BooleanTrue);
pub const FALSE: BooleanType = BooleanType::False(BooleanFalse);

#[derive(From, Debug, Display, Clone)]
pub enum BooleanType {
    True(BooleanTrue),
    False(BooleanFalse),
    Symbol(Symbol),

    Not(Box<Not>),
    And(Box<And>),
    Or(Box<Or>),
}

impl TypedSubstitude for BooleanType {
    fn subs(&self, sym: Symbol, val: Self) -> Self {
        use BooleanType as BT;
        match self {
            BT::True(_) => self.clone(),
            BT::False(_) => self.clone(),
            BT::Symbol(s) => match s == &sym {
                true => val,
                false => self.clone(),
            },

            BT::Not(b) => b.val.subs(sym, val),
            BT::And(b) => b.subs(sym, val),
            BT::Or(b) => b.subs(sym, val),
        }
    }
}

impl Simplify for BooleanType {
    fn simplify(&self) -> Self {
        use BooleanType as BT;
        match self {
            BT::True(_) => self.clone(),
            BT::False(_) => self.clone(),
            BT::Symbol(_) => self.clone(),
            BT::Not(b) => b.val.simplify(),
            BT::And(b) => b.simplify(),
            BT::Or(b) => b.simplify(),
        }
    }
}

#[derive(Debug, Display, PartialEq, Copy, Clone)]
#[display(fmt = "true")]
pub struct BooleanTrue;

#[derive(Debug, Display, PartialEq, Copy, Clone)]
#[display(fmt = "false")]
pub struct BooleanFalse;

impl BooleanTrue {
    pub const fn typ() -> CalcursType {
        CalcursType::Typed(Typed::Boolean(BooleanType::True(Self)))
    }
}

impl BooleanFalse {
    pub const fn typ() -> CalcursType {
        CalcursType::Typed(Typed::Boolean(BooleanType::False(Self)))
    }
}

pub trait BooleanOp: Operator<OperandType = BooleanType> + Clone {
    fn simplify(&self) -> BooleanType;
    fn to_bool(self) -> BooleanType;
    fn from_bool(val: &BooleanType) -> Option<&Self>;
}

// assume commutivity for now
// every op should not mutate value
pub trait BooleanBinOp: BooleanOp {
    fn new(lhs: BooleanType, rhs: BooleanType) -> Self;

    fn lhs(&self) -> &BooleanType;
    fn rhs(&self) -> &BooleanType;

    fn subs(&self, sym: Symbol, val: BooleanType) -> BooleanType {
        let lhs = self.lhs().subs(sym, val.clone());
        let rhs = self.rhs().subs(sym, val);
        Self::bool(lhs, rhs)
    }

    fn as_calcurs_op(lhs: CalcursType, rhs: CalcursType) -> Option<CalcursType> {
        use CalcursType as CT;
        use Typed as T;
        Some(
            match (lhs, rhs) {
                // op(bool, bool) => op(bool, bool)
                (CT::Typed(T::Boolean(b1)), CT::Typed(T::Boolean(b2))) => Self::bool(b1, b2),

                // op(sym, sym) => op(bool(sym), bool(sym))
                (CT::Symbol(s1), CT::Symbol(s2)) => Self::bool(s1.into(), s2.into()),

                // op(bool, sym) => op(bool, bool(sym))
                (CT::Typed(T::Boolean(b)), CT::Symbol(s)) => Self::bool(b, s.into()),

                // op(sym, bool) => op(bool, bool(sym))
                (CT::Symbol(s), CT::Typed(T::Boolean(b))) => Self::bool(s.into(), b),
            }
            .into(),
        )
    }

    #[inline]
    fn bool(lhs: BooleanType, rhs: BooleanType) -> BooleanType {
        Self::new(lhs, rhs).to_bool()
    }

    #[inline]
    fn typ(lhs: BooleanType, rhs: BooleanType) -> CalcursType {
        Self::bool(lhs, rhs).into()
    }
}

// fn bool_assoc<OP: BooleanBinOp>(e: &BooleanType) -> Option<BooleanType> {
//     if let Some(b1) = OP::from_bool(e) {
//         let lhs = b1.lhs();
//         let rhs = b1.rhs();
//         match (OP::from_bool(lhs), OP::from_bool(rhs)) {
//             (Some(b2), None) => {
//                 let lhs = lhs.clone();
//                 let inner_lhs = b2.lhs().clone();
//                 let inner_rhs = b2.rhs().clone();
//                 return Some(OP::bool(OP::bool(lhs, inner_lhs), inner_rhs));
//             }
//             _ => (),
//         }
//     }

//     None
// }

// fn bool_assoc_w_comm<OP: BooleanBinOp + Commutativity>(e: &BooleanType) -> BooleanType {
//     if let Some(res) = bool_assoc::<OP>(e) {
//         return res;
//     }

//     // swap and try again
//     let e = <OP as Commutativity>::apply(e);

//     if let Some(res) = bool_assoc::<OP>(&e) {
//         return res;
//     }

//     e.clone()
// }

// fn bool_dist<OP1: BooleanBinOp, OP2: BooleanBinOp>(e: &BooleanType) -> Option<BooleanType> {
//     if let Some(b1) = OP1::from_bool(e) {
//         let lhs = b1.lhs();
//         let rhs = b1.rhs();

//         if let Some(b2) = OP2::from_bool(rhs) {
//             let x = lhs.clone();
//             let y = b2.lhs().clone();
//             let z = b2.rhs().clone();
//             // we now have x op1 (y op2 z)
//             return Some(OP2::bool(OP1::bool(x.clone(), y), OP1::bool(x, z)));
//         }
//     }

//     None
// }

// fn bool_dist_w_comm<OP1: BooleanBinOp + Commutativity, OP2: BooleanBinOp + Commutativity>(
//     e: &BooleanType,
// ) -> BooleanType {
//     if let Some(res) = bool_dist::<OP1, OP2>(e) {
//         return res;
//     }

//     // swap and try again
//     let e = <OP1 as Commutativity>::apply(e);

//     if let Some(res) = bool_dist::<OP1, OP2>(&e) {
//         return res;
//     }

//     e.clone()
// }

#[derive(Debug, Display, Clone)]
#[display(fmt = "({lhs} ∧ {rhs})")]
pub struct And {
    lhs: BooleanType,
    rhs: BooleanType,
}

impl Operator for And {
    type OperandType = BooleanType;
}

impl Identity for And {
    const IDENTITY: Self::OperandType = TRUE;
}

impl Commutativity for And {}
impl Associativity for And {}
impl Distributivity<Or> for And {}

impl BooleanOp for And {
    fn simplify(&self) -> BooleanType {
        use BooleanType as BT;
        let lhs = self.lhs();
        let rhs = self.rhs();

        match (lhs, rhs) {
            (_, BT::False(_)) | (BT::False(_), _) => FALSE,

            (_, BT::True(_)) => lhs.simplify(),
            (BT::True(_), _) => rhs.simplify(),

            (_, _) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                Self::bool(lhs, rhs)
            }
        }
    }

    fn to_bool(self) -> BooleanType {
        Box::new(self).into()
    }

    fn from_bool(val: &BooleanType) -> Option<&Self> {
        match val {
            BooleanType::And(and) => Some(and),
            _ => None,
        }
    }
}

impl BooleanBinOp for And {
    fn new(lhs: BooleanType, rhs: BooleanType) -> Self {
        And { lhs, rhs }
    }

    #[inline]
    fn lhs(&self) -> &BooleanType {
        &self.lhs
    }

    #[inline]
    fn rhs(&self) -> &BooleanType {
        &self.rhs
    }
}

#[derive(Debug, Display, Clone)]
#[display(fmt = "({lhs} ∨ {rhs})")]
pub struct Or {
    lhs: BooleanType,
    rhs: BooleanType,
}

impl Commutativity for Or {}
impl Associativity for Or {}
impl Distributivity<And> for Or {}

impl Operator for Or {
    type OperandType = BooleanType;
}

impl Identity for Or {
    const IDENTITY: Self::OperandType = FALSE;
}

impl BooleanOp for Or {
    fn simplify(&self) -> BooleanType {
        use BooleanType as BT;
        let lhs = self.lhs();
        let rhs = self.rhs();

        match (lhs, rhs) {
            (_, BT::True(_)) | (BT::True(_), _) => TRUE,

            (_, BT::False(_)) => lhs.simplify(),
            (BT::False(_), _) => rhs.simplify(),

            (_, _) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                Self::bool(lhs, rhs)
            }
        }
    }

    fn to_bool(self) -> BooleanType {
        Box::new(self).into()
    }

    fn from_bool(val: &BooleanType) -> Option<&Self> {
        match val {
            BooleanType::Or(or) => Some(or),
            _ => None,
        }
    }
}

impl BooleanBinOp for Or {
    fn new(lhs: BooleanType, rhs: BooleanType) -> Self {
        Self { lhs, rhs }
    }

    fn lhs(&self) -> &BooleanType {
        &self.lhs
    }

    fn rhs(&self) -> &BooleanType {
        &self.rhs
    }
}

#[derive(Debug, Display, Clone)]
#[display(fmt = "(¬{val})")]
pub struct Not {
    val: BooleanType,
}

impl Operator for Not {
    type OperandType = BooleanType;
}

impl BooleanOp for Not {
    fn to_bool(self) -> BooleanType {
        Box::new(self).into()
    }

    fn simplify(&self) -> BooleanType {
        use BooleanType as BT;
        match &self.val {
            BT::True(_) => FALSE,
            BT::False(_) => TRUE,
            BT::Not(v) => v.clone().into(),
            BT::Symbol(_) | BT::And(_) | BT::Or(_) => self.clone().to_bool(),
        }
    }

    fn from_bool(val: &BooleanType) -> Option<&Self> {
        match val {
            BooleanType::Not(not) => Some(not),
            _ => None,
        }
    }
}

impl Not {
    pub fn typ(val: BooleanType) -> CalcursType {
        Self::bool(val).into()
    }

    pub fn bool(val: BooleanType) -> BooleanType {
        Box::new(Self { val }).into()
    }

    pub fn as_calcurs_op(val: CalcursType) -> Option<CalcursType> {
        use CalcursType as CT;
        use Typed as T;

        Some(
            match val {
                CT::Typed(T::Boolean(b)) => Self::bool(b),
                CT::Symbol(s) => Self::bool(s.into()),
            }
            .into(),
        )
    }
}

// bool logic

impl ops::Not for BooleanType {
    type Output = BooleanType;

    fn not(self) -> Self::Output {
        Not::bool(self)
    }
}

impl ops::BitAnd<BooleanType> for BooleanType {
    type Output = BooleanType;

    fn bitand(self, rhs: BooleanType) -> Self::Output {
        And::bool(self, rhs)
    }
}

impl ops::BitOr<BooleanType> for BooleanType {
    type Output = BooleanType;

    fn bitor(self, rhs: BooleanType) -> Self::Output {
        Or::bool(self, rhs)
    }
}
