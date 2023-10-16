use std::{collections::HashSet, fmt, ops};

use derive_more::Display;

use crate::{
    base::{Base, BaseKind, SubsDict, Variable, PTR},
    traits::CalcursType,
};

pub const FALSE: Boolean = Boolean {
    kind: BooleanKind::False(False),
};

pub const TRUE: Boolean = Boolean {
    kind: BooleanKind::True(True),
};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub enum BooleanKind {
    True(True),
    False(False),
    Var(Variable),

    And(And),
    Or(Or),
    Not(Not),

    Unknown(PTR<Base>),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub struct Boolean {
    pub kind: BooleanKind,
}

impl From<bool> for Boolean {
    fn from(value: bool) -> Self {
        match value {
            true => TRUE,
            false => FALSE,
        }
    }
}

impl From<BooleanKind> for Boolean {
    fn from(value: BooleanKind) -> Self {
        Boolean { kind: value }
    }
}

impl CalcursType for Boolean {
    fn base(self) -> Base {
        BaseKind::Boolean(self).into()
    }
}

impl Boolean {
    pub fn to_basic(self) -> Base {
        BaseKind::Boolean(self).into()
    }
}

impl BooleanKind {
    pub fn simplify(self) -> Self {
        use BooleanKind as BK;
        match self {
            BK::And(a) => a.simplify(),
            BK::Or(o) => o.simplify(),
            BK::Not(n) => n.simplify(),
            BK::Unknown(b) => BK::Unknown(b.simplify().into()),
            BK::True(_) | BK::False(_) | BK::Var(_) => self,
        }
    }

    pub fn base(self) -> Base {
        BaseKind::Boolean(self.into()).into()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
#[display(fmt = "1")]
pub struct True;

impl CalcursType for True {
    fn base(self) -> Base {
        BooleanKind::True(self).to_basic()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
#[display(fmt = "0")]
pub struct False;

impl CalcursType for False {
    fn base(self) -> Base {
        BooleanKind::False(self).to_basic()
    }
}

impl BooleanKind {
    pub fn from_basic(b: Base) -> Self {
        match b.kind {
            BaseKind::Var(v) => BooleanKind::Var(v),
            BaseKind::Boolean(b) => b.kind,
            BaseKind::Dummy => BooleanKind::Unknown(PTR::new(b)),
            _ => todo!(),
        }
    }

    pub fn subs(self, dict: SubsDict) -> Self {
        match self {
            BooleanKind::Var(ref v) => {
                if dict.borrow().contains_key(v) {
                    let basic = dict.borrow().get(v).unwrap().clone();
                    Self::from_basic(basic)
                } else {
                    self
                }
            }
            BooleanKind::And(a) => a.subs(dict),
            BooleanKind::Or(o) => o.subs(dict),
            BooleanKind::Not(n) => n.subs(dict),
            BooleanKind::Unknown(u) => BooleanKind::Unknown(Box::new(u.kind.subs(dict).into())),
            BooleanKind::True(_) | BooleanKind::False(_) => self,
        }
    }

    pub fn to_basic(self) -> Base {
        BaseKind::Boolean(self.into()).into()
    }
}

/// Logical And. Implemented with a list of [BooleanKind]
///
/// The only simplifications performed by the default [And] are: \
/// (a ∧ b) ∧ (c ∧ d) => a ∧ b ∧ c ∧ d (flatten And) \
/// a ∧ b ∧ c ∧ a => a ∧ b ∧ c,  \
/// a ∧ b ∧ c ∧ false => false,  \
/// a ∧ b ∧ c ∧ true => a ∧ b ∧ c,
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct And {
    args: Vec<BooleanKind>,
}

impl fmt::Display for And {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.args.iter();

        if let Some(first) = iter.next() {
            write!(f, "({}", first)?;
        }

        for e in iter {
            write!(f, " ∧ {}", e)?;
        }

        write!(f, ")")
    }
}

impl std::hash::Hash for And {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "And".hash(state);
        for a in &self.args {
            a.hash(state)
        }
    }
}

impl And {
    pub fn and<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Boolean {
        let mut or = Self { args: Vec::new() };

        or.insert_arg(b1.base());
        or.insert_arg(b2.base());
        or.simplify().into()
    }

    fn insert_arg(&mut self, b: Base) {
        self.args.push(BooleanKind::from_basic(b));
    }

    /// remove duplicates & remove [True] and return if [False] is found
    fn simplify(self) -> BooleanKind {
        let mut set = HashSet::new();

        for e in self.args {
            let e = e.simplify();

            if let BooleanKind::False(_) = e {
                return FALSE.kind;
            } else if let BooleanKind::True(_) = e {
                continue;
            }

            set.insert(e);
        }

        if set.is_empty() {
            TRUE.kind
        } else if set.len() == 1 {
            let mut iter = set.into_iter();
            iter.next().unwrap()
        } else {
            BooleanKind::And(Self {
                args: set.into_iter().collect(),
            })
        }
    }

    pub fn subs(self, dict: SubsDict) -> BooleanKind {
        let args = self
            .args
            .into_iter()
            .map(|x| x.subs(dict.clone()))
            .collect();
        BooleanKind::And(Self { args })
    }
}

impl CalcursType for And {
    fn base(self) -> Base {
        self.simplify().to_basic()
    }
}

/// Logical Or. Implemented with a list of [BooleanKind]
///
/// The only simplifications performed by the default [Or] are: \
/// (a ∨ b) ∨ (c ∨ d) => a ∨ b ∨ c ∨ d (flatten Or) \
/// a ∨ b ∨ c ∨ a => a ∨ b ∨ c,  \
/// a ∨ b ∨ c ∨ true => true,  \
/// a ∨ b ∨ c ∨ false => a ∨ b ∨ c,
///
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Or {
    args: Vec<BooleanKind>,
}

impl std::hash::Hash for Or {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "Or".hash(state);
        for b in &self.args {
            b.hash(state);
        }
    }
}

impl fmt::Display for Or {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.args.iter();

        if let Some(first) = iter.next() {
            write!(f, "({}", first)?;
        }

        for e in iter {
            write!(f, " ∨ {}", e)?;
        }

        write!(f, ")")
    }
}

impl Or {
    pub fn subs(self, dict: SubsDict) -> BooleanKind {
        let args = self
            .args
            .into_iter()
            .map(|x| x.subs(dict.clone()))
            .collect();
        BooleanKind::Or(Self { args })
    }

    pub fn or<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Boolean {
        let mut or = Self { args: Vec::new() };

        or.insert_arg(b1.base());
        or.insert_arg(b2.base());
        or.simplify().into()
    }

    fn insert_arg(&mut self, b: Base) {
        self.args.push(BooleanKind::from_basic(b));
    }

    /// remove duplicates & remove [False] and return if [True] is found
    fn simplify(self) -> BooleanKind {
        let mut set = HashSet::new();

        for e in self.args {
            let e = e.simplify();
            if let BooleanKind::True(_) = e {
                return TRUE.kind;
            } else if let BooleanKind::False(_) = e {
                continue;
            }

            set.insert(e);
        }

        if set.is_empty() {
            return FALSE.kind;
        } else if set.len() == 1 {
            let mut iter = set.into_iter();
            return iter.next().unwrap();
        }

        BooleanKind::Or(Self {
            args: set.into_iter().collect(),
        })
    }
}

impl CalcursType for Or {
    fn base(self) -> Base {
        BooleanKind::Or(self).to_basic()
    }
}

/// Logical Not. Implemented with a [PTR] to a [Boolean]
///
/// The only simplifications performed by the default [Not] are: \
/// ¬ false => true
/// ¬ true  => false
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Not {
    val: PTR<BooleanKind>,
}

impl fmt::Display for Not {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "¬{}", self.val)
    }
}

impl Not {
    pub fn subs(self, dict: SubsDict) -> BooleanKind {
        let val = self.val.subs(dict);
        BooleanKind::Not(Not { val: val.into() })
    }

    pub fn not<T: CalcursType>(b: T) -> Boolean {
        let not = Self {
            val: PTR::new(BooleanKind::from_basic(b.base())),
        };

        not.simplify().into()
    }

    fn simplify(self) -> BooleanKind {
        let val = self.val.simplify();

        match val {
            BooleanKind::True(_) => FALSE.kind,
            BooleanKind::False(_) => TRUE.kind,
            BooleanKind::Not(not) => *not.val,
            BooleanKind::Var(_)
            | BooleanKind::And(_)
            | BooleanKind::Or(_)
            | BooleanKind::Unknown(_) => BooleanKind::Not(Not { val: val.into() }),
        }
    }
}

impl<T: CalcursType> ops::BitAnd<T> for Base {
    type Output = Base;

    fn bitand(self, rhs: T) -> Self::Output {
        And::and(self, rhs.base()).to_basic()
    }
}

impl<T: CalcursType> ops::BitOr<T> for Base {
    type Output = Base;

    fn bitor(self, rhs: T) -> Self::Output {
        Or::or(self, rhs.base()).to_basic()
    }
}

impl ops::Not for Base {
    type Output = Base;

    fn not(self) -> Self::Output {
        Not::not(self).to_basic()
    }
}

impl CalcursType for Not {
    fn base(self) -> Base {
        BooleanKind::Not(self).to_basic()
    }
}

#[cfg(test)]
mod boolean_types {

    use crate::*;
    use boolean::*;

    #[test]
    fn and() {
        let and = And::and(And::and(TRUE, Variable::new("x")), FALSE).to_basic();
        assert_eq!(and, FALSE.base());

        let and = And::and(And::and(TRUE, Variable::new("x")), Variable::new("x")).to_basic();
        assert_eq!(and, BooleanKind::Var(Variable::new("x")).to_basic())
    }

    #[test]
    fn or() {
        let or = Or::or(Or::or(FALSE, Variable::new("x")), TRUE).to_basic();
        assert_eq!(or, TRUE.base());

        let or = Or::or(Or::or(False, Variable::new("x")), Variable::new("x")).to_basic();
        assert_eq!(or, BooleanKind::Var(Variable::new("x")).to_basic())
    }

    #[test]
    fn subs() {
        let x = Variable::new("x").base();
        let y = Variable::new("y").base();
        let z = Variable::new("z").base();

        let expr = (x.clone() & y.clone()) | z.clone();
        let eval = expr.subs("x", y.clone()).subs("y", True).subs("z", False);
        assert_eq!(eval.simplify(), TRUE.base());

        let expr = !((x.clone() & y.clone()) | z.clone());
        let eval = expr.subs("x", y.clone()).subs("y", True).subs("z", False);
        assert_eq!(eval.simplify(), FALSE.base());
    }
}
