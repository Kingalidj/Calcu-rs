use std::{collections::HashSet, fmt, ops};

use derive_more::Display;

use crate::{
    cast_ref, early_ret, get_ref_impl, ArgSet, Basic, BasicKind, CalcursType, SubsDict, TypeID,
    Variable, PTR,
};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub struct Boolean {
    pub kind: BooleanKind,
}

impl From<bool> for Boolean {
    fn from(value: bool) -> Self {
        BooleanKind::from(value).into()
    }
}

impl From<BooleanKind> for Boolean {
    fn from(value: BooleanKind) -> Self {
        Boolean { kind: value }
    }
}

impl CalcursType for Boolean {
    const ID: TypeID = TypeID::Boolean;

    fn to_basic(self) -> Basic {
        BasicKind::Boolean(self).into()
    }
}

impl Boolean {
    get_ref_impl!(kind);

    pub fn to_basic(self) -> Basic {
        BasicKind::Boolean(self).into()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub enum BooleanKind {
    True(True),
    False(False),
    Var(Variable),

    And(And),
    Or(Or),
    Not(Not),

    Unknown(PTR<Basic>),
}

impl BooleanKind {
    pub fn simplify(self) -> Self {
        match self {
            BooleanKind::And(a) => a.simplify(),
            BooleanKind::Or(o) => o.simplify(),
            BooleanKind::Not(n) => n.simplify(),
            BooleanKind::Unknown(b) => BooleanKind::Unknown(b.simplify().into()),
            BooleanKind::True(_) | BooleanKind::False(_) | BooleanKind::Var(_) => self,
        }
    }
}

impl From<bool> for BooleanKind {
    fn from(value: bool) -> Self {
        match value {
            true => BooleanKind::True(True),
            false => BooleanKind::False(False),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
#[display(fmt = "1")]
pub struct True;

impl True {
    get_ref_impl!();
}

impl CalcursType for True {
    const ID: TypeID = TypeID::True;

    fn to_basic(self) -> Basic {
        BooleanKind::True(self).to_basic()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
#[display(fmt = "0")]
pub struct False;

impl False {
    get_ref_impl!();
}

impl CalcursType for False {
    const ID: TypeID = TypeID::False;

    fn to_basic(self) -> Basic {
        BooleanKind::False(self).to_basic()
    }
}

impl BooleanKind {
    #[inline(always)]
    pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
        use BooleanKind as BK;
        match self {
            BK::True(b) => b.get_ref::<T>(),
            BK::False(b) => b.get_ref::<T>(),
            BK::Var(b) => b.get_ref::<T>(),
            BK::And(b) => b.get_ref::<T>(),
            BK::Or(b) => b.get_ref::<T>(),
            BK::Not(b) => b.get_ref::<T>(),
            BK::Unknown(b) => b.get_ref::<T>(),
        }
    }

    pub fn from_basic(b: Basic) -> Self {
        match b.kind {
            BasicKind::Var(v) => BooleanKind::Var(v),
            BasicKind::Boolean(b) => b.kind,
            BasicKind::Dummy => BooleanKind::Unknown(PTR::new(b)),
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

    pub fn to_basic(self) -> Basic {
        BasicKind::Boolean(self.into()).into()
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
    args: ArgSet<BooleanKind>,
}

impl fmt::Display for And {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.args.iter();

        if let Some(first) = iter.next() {
            write!(f, "({}", first)?;
        }

        while let Some(e) = iter.next() {
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
    get_ref_impl!();

    pub fn and<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Boolean {
        let mut or = Self {
            args: ArgSet::new(),
        };

        or.insert_arg(b1.to_basic());
        or.insert_arg(b2.to_basic());
        or.simplify().into()
    }

    fn insert_arg(&mut self, b: Basic) {
        self.args.push(BooleanKind::from_basic(b));
    }

    /// remove duplicates & remove [True] and return if [False] is found
    fn simplify(self) -> BooleanKind {
        let mut set = HashSet::new();

        for e in self.args {
            let e = e.simplify();

            if let BooleanKind::False(_) = e {
                return false.into();
            } else if let BooleanKind::True(_) = e {
                continue;
            }

            set.insert(e);
        }

        if set.is_empty() {
            return true.into();
        } else if set.len() == 1 {
            let mut iter = set.into_iter();
            return iter.next().unwrap().into();
        }

        BooleanKind::And(Self {
            args: set.into_iter().collect(),
        })
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
    const ID: TypeID = TypeID::And;

    fn to_basic(self) -> Basic {
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
    args: ArgSet<BooleanKind>,
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

        while let Some(e) = iter.next() {
            write!(f, " ∨ {}", e)?;
        }

        write!(f, ")")
    }
}

impl Or {
    get_ref_impl!();

    pub fn subs(self, dict: SubsDict) -> BooleanKind {
        let args = self
            .args
            .into_iter()
            .map(|x| x.subs(dict.clone()))
            .collect();
        BooleanKind::Or(Self { args })
    }

    pub fn or<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Boolean {
        let mut or = Self {
            args: ArgSet::new(),
        };

        or.insert_arg(b1.to_basic());
        or.insert_arg(b2.to_basic());
        or.simplify().into()
    }

    fn insert_arg(&mut self, b: Basic) {
        self.args.push(BooleanKind::from_basic(b));
    }

    /// remove duplicates & remove [False] and return if [True] is found
    fn simplify(self) -> BooleanKind {
        let mut set = HashSet::new();

        for e in self.args {
            let e = e.simplify();
            if let BooleanKind::True(_) = e {
                return true.into();
            } else if let BooleanKind::False(_) = e {
                continue;
            }

            set.insert(e);
        }

        if set.is_empty() {
            return false.into();
        } else if set.len() == 1 {
            let mut iter = set.into_iter();
            return iter.next().unwrap().into();
        }

        BooleanKind::Or(Self {
            args: set.into_iter().collect(),
        })
    }
}

impl CalcursType for Or {
    const ID: TypeID = TypeID::Or;

    fn to_basic(self) -> Basic {
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
    get_ref_impl!();

    pub fn subs(self, dict: SubsDict) -> BooleanKind {
        let val = self.val.subs(dict);
        BooleanKind::Not(Not { val: val.into() })
    }

    pub fn not<T: CalcursType>(b: T) -> Boolean {
        let not = Self {
            val: PTR::new(BooleanKind::from_basic(b.to_basic())),
        };

        not.simplify().into()
    }

    fn simplify(self) -> BooleanKind {
        let val = self.val.simplify();

        match val {
            BooleanKind::True(_) => false.into(),
            BooleanKind::False(_) => true.into(),
            BooleanKind::Not(not) => *not.val,
            BooleanKind::Var(_)
            | BooleanKind::And(_)
            | BooleanKind::Or(_)
            | BooleanKind::Unknown(_) => BooleanKind::Not(Not { val: val.into() }),
        }
    }
}

impl<T: CalcursType> ops::BitAnd<T> for Basic {
    type Output = Basic;

    fn bitand(self, rhs: T) -> Self::Output {
        And::and(self, rhs.to_basic()).to_basic()
    }
}

impl<T: CalcursType> ops::BitOr<T> for Basic {
    type Output = Basic;

    fn bitor(self, rhs: T) -> Self::Output {
        Or::or(self, rhs.to_basic()).to_basic()
    }
}

impl ops::Not for Basic {
    type Output = Basic;

    fn not(self) -> Self::Output {
        Not::not(self).to_basic()
    }
}

impl CalcursType for Not {
    const ID: TypeID = TypeID::Not;

    fn to_basic(self) -> Basic {
        BooleanKind::Not(self).to_basic()
    }
}

pub const FALSE: Basic = Basic {
    kind: BasicKind::Boolean(Boolean {
        kind: BooleanKind::False(False),
    }),
};
pub const TRUE: Basic = Basic {
    kind: BasicKind::Boolean(Boolean {
        kind: BooleanKind::True(True),
    }),
};

#[cfg(test)]
mod boolean_types {

    use crate::*;
    use boolean::*;

    #[test]
    fn ast_getter() {
        const _: bool = TRUE.is::<Basic>();
        assert!(TRUE.is::<Basic>());
        assert!(TRUE.is::<Boolean>());
        assert!(TRUE.is::<True>());

        assert!(!FALSE.is::<Variable>());
        assert!(!FALSE.is::<And>());
        assert!(!FALSE.is::<Not>());
        assert!(!FALSE.is::<Or>());
        assert!(!FALSE.is::<True>());
    }

    #[test]
    fn and() {
        let and = And::and(And::and(TRUE, Variable::new("x")), FALSE).to_basic();
        assert!(and.is::<False>());

        let and = And::and(And::and(TRUE, Variable::new("x")), Variable::new("x")).to_basic();
        assert!(and.is::<Variable>())
    }

    #[test]
    fn or() {
        let or = Or::or(Or::or(FALSE, Variable::new("x")), TRUE).to_basic();
        assert!(or.is::<True>());

        let or = Or::or(Or::or(False, Variable::new("x")), Variable::new("x")).to_basic();
        assert!(or.is::<Variable>())
    }

    #[test]
    fn subs() {
        let x = Variable::new("x").to_basic();
        let y = Variable::new("y").to_basic();
        let z = Variable::new("z").to_basic();

        let expr = (x.clone() & y.clone()) | z.clone();
        let eval = expr.subs("x", y.clone()).subs("y", True).subs("z", False);
        assert!(eval.simplify().is::<True>());

        let expr = !((x.clone() & y.clone()) | z.clone());
        let eval = expr.subs("x", y.clone()).subs("y", True).subs("z", False);
        assert!(eval.simplify().is::<False>());
    }
}
