use std::ops;

use derive_more::{Display, From};

use crate::types::{CalcursType, Simplify, Symbol, Typed, TypedSubstitude};

pub const TRUE: BooleanType = BooleanType::True(BooleanTrue);
pub const FALSE: BooleanType = BooleanType::False(BooleanFalse);

#[derive(From, Debug, Display, Clone)]
pub enum BooleanType {
    True(BooleanTrue),
    False(BooleanFalse),
    Symbol(Symbol),

    And(Box<And>),
    Or(Box<Or>),
}

impl BooleanType {
    pub fn value(&self) -> Option<bool> {
        use BooleanType as BT;
        match self {
            BT::True(_) => Some(true),
            BT::False(_) => Some(false),
            BT::Symbol(_) => None,
            BT::And(v) => v.value(),
            BT::Or(v) => v.value(),
        }
    }
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

// assume commutivity for now
// every op should not mutate value
pub trait BooleanBinOp: Clone {
    fn new_impl(lhs: BooleanType, rhs: BooleanType) -> Self;
    fn to_bool(self) -> BooleanType;

    fn lhs(&self) -> &BooleanType;
    fn rhs(&self) -> &BooleanType;
    fn value(&self) -> Option<bool>;

    fn subs(&self, sym: Symbol, val: BooleanType) -> BooleanType {
        let lhs = self.lhs().subs(sym, val.clone());
        let rhs = self.rhs().subs(sym, val);
        Self::bool_typ(lhs, rhs)
    }

    fn simplify(&self) -> BooleanType;

    fn as_calcurs_op(lhs: CalcursType, rhs: CalcursType) -> Option<CalcursType> {
        use CalcursType as CT;
        use Typed as T;
        Some(
            match (lhs, rhs) {
                // op(bool, bool) => op(bool, bool)
                (CT::Typed(T::Boolean(b1)), CT::Typed(T::Boolean(b2))) => Self::bool_typ(b1, b2),

                // op(sym, sym) => op(bool(sym), bool(sym))
                (CT::Symbol(s1), CT::Symbol(s2)) => Self::bool_typ(s1.into(), s2.into()),

                // op(bool, sym) => op(bool, bool(sym))
                (CT::Typed(T::Boolean(b)), CT::Symbol(s)) => Self::bool_typ(b, s.into()),

                // op(sym, bool) => op(bool, bool(sym))
                (CT::Symbol(s), CT::Typed(T::Boolean(b))) => Self::bool_typ(s.into(), b),
            }
            .into(),
        )
    }

    #[inline]
    fn bool_typ(lhs: BooleanType, rhs: BooleanType) -> BooleanType {
        Self::new_impl(lhs, rhs).to_bool()
    }

    fn typ(lhs: BooleanType, rhs: BooleanType) -> CalcursType {
        let lhs = lhs.simplify();
        let rhs = rhs.simplify();
        Self::bool_typ(lhs, rhs).into()
    }
}

#[derive(Debug, Display, Clone)]
#[display(fmt = "({lhs} ∧ {rhs})")]
pub struct And {
    lhs: BooleanType,
    rhs: BooleanType,
    val: Option<bool>,
}

impl BooleanBinOp for And {
    fn new_impl(lhs: BooleanType, rhs: BooleanType) -> Self {
        let val = match (&lhs.value(), &rhs.value()) {
            (Some(b1), Some(b2)) => Some(*b1 && *b2),
            _ => None,
        };

        And { lhs, rhs, val }
    }

    fn to_bool(self) -> BooleanType {
        Box::new(self).into()
    }

    #[inline]
    fn lhs(&self) -> &BooleanType {
        &self.lhs
    }

    #[inline]
    fn rhs(&self) -> &BooleanType {
        &self.rhs
    }

    #[inline]
    fn value(&self) -> Option<bool> {
        self.val
    }

    fn simplify(&self) -> BooleanType {
        match self.val {
            Some(true) => return TRUE,
            Some(false) => return FALSE,
            None => (),
        }

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
                Self::bool_typ(lhs, rhs)
            }
        }
    }
}

#[derive(Debug, Display, Clone)]
#[display(fmt = "({lhs} ∨ {rhs})")]
pub struct Or {
    lhs: BooleanType,
    rhs: BooleanType,
    val: Option<bool>,
}

impl BooleanBinOp for Or {
    fn new_impl(lhs: BooleanType, rhs: BooleanType) -> Self {
        let val = match (&lhs.value(), &rhs.value()) {
            (Some(b1), Some(b2)) => Some(*b1 || *b2),
            _ => None,
        };

        Self { lhs, rhs, val }
    }

    fn to_bool(self) -> BooleanType {
        Box::new(self).into()
    }

    fn lhs(&self) -> &BooleanType {
        &self.lhs
    }

    fn rhs(&self) -> &BooleanType {
        &self.rhs
    }

    #[inline]
    fn value(&self) -> Option<bool> {
        self.val
    }

    fn simplify(&self) -> BooleanType {
        match self.val {
            Some(true) => return TRUE,
            Some(false) => return FALSE,
            None => (),
        }

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
                Self::bool_typ(lhs, rhs)
            }
        }
    }
}

// calcurs bool logic

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

impl ops::BitAnd<BooleanType> for BooleanType {
    type Output = BooleanType;

    fn bitand(self, rhs: BooleanType) -> Self::Output {
        And::bool_typ(self, rhs)
    }
}

impl ops::BitOr<BooleanType> for BooleanType {
    type Output = BooleanType;

    fn bitor(self, rhs: BooleanType) -> Self::Output {
        Or::bool_typ(self, rhs)
    }
}
