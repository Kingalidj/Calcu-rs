use fmt::{Debug, Display, Formatter};
use std::{fmt, ops};

//use calcu_rs::scalar::{Float, Infinity};
use calcu_rs::{
    e_graph,
    e_graph::GraphExpr,
    operator::{Diff, Pow, Prod, Quot, Sum},
    pattern::Item,
    rational::Rational,
    util::*,
};

/// implemented by every symbolic math type
pub trait CalcursType: Clone + Debug + Into<Expr> {
    fn desc(&self) -> Item;
}

/// contains one or multiple expressions of type [Expr]
pub trait Construct: CalcursType {
    //This operator returns false when [other] is identical
    // to some complete subexpression of [self] and otherwise returns true
    //fn free_of(&self, other: &Expr) -> bool;
    //fn contains(&self, other: &Expr) -> bool {
    //    !self.free_of(other)
    //}

    // Checks if expression is a general polynomial expression (GPE) in [vars]
    // note that: \
    // every sub-expresion must also be a GPE in [vars] \
    // 0.is_polynomial(...) -> true \
    // (y^2 + y).is_polynomial(x) -> true
    //fn is_polynomial_in(&self, vars: &[Expr]) -> bool {
    //    for v in self.operands() {
    //        if !(v.is_polynomial_in(vars)) {
    //            return false;
    //        }
    //    }
    //    true
    //}

    // returns all generalized variables in the expression (e.g x, but also sin(x))
    // normally you should call [variables]
    //fn all_variables(&self) -> Vec<Expr>;

    // returns all unique generalized variables in the expression (e.g x, but also sin(x))
    //fn variables(&self) -> Vec<Expr> {
    //    let vars = self.variables();
    //    let unique_vars: HashSet<Expr> = vars.into_iter().collect();
    //    unique_vars.into_iter().collect()
    //}

    fn simplify(self) -> Expr;
}

pub type PTR<T> = Box<T>;

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expr {
    Rational(Rational),
    Symbol(Symbol),

    Sum(Sum),
    Prod(Prod),
    Pow(PTR<Pow>),

    /// only used if the result is provenly undefined, e.g 0 / 0
    ///
    /// not the same as [f64::NAN]
    Undefined,

    PlaceHolder(&'static str),
}

macro_rules! impl_from_for_expr {
    ($typ:ident) => {
        impl From<$typ> for Expr {
            #[inline(always)]
            fn from(value: $typ) -> Expr {
                Expr::$typ(value.into())
            }
        }
    };
}

impl_from_for_expr!(Rational);
impl_from_for_expr!(Sum);
impl_from_for_expr!(Prod);
impl_from_for_expr!(Pow);

impl Expr {
    pub fn pow(self, other: impl CalcursType) -> Expr {
        Pow::pow(self, other)
    }

    pub fn operands(&self) -> Vec<&Expr> {
        use Expr as E;
        match self {
            E::Rational(_) | E::Symbol(_) | E::Undefined | E::PlaceHolder(_) => vec![],
            E::Sum(sum) => sum.operands.iter().collect(),
            E::Prod(prod) => prod.operands.iter().collect(),
            E::Pow(pow) => vec![&pow.base, &pow.exponent],
        }
    }

    pub fn operands_mut(&mut self) -> &mut [Expr] {
        use Expr as E;
        match self {
            E::Rational(_) | E::Symbol(_) | E::Undefined | E::PlaceHolder(_) => &mut [],
            E::Sum(sum) => sum.operands.as_mut_slice(),
            E::Prod(prod) => prod.operands.as_mut_slice(),
            E::Pow(pow) => pow.operands_mut(),
        }
    }
}

// basic checks, e.g if Float, then not int
#[inline]
fn debug_desc_check(i: Item) {
    use Item as I;
    if i.is(I::Zero) {
        debug_assert!(i.is_not(I::Pos));
        debug_assert!(i.is_not(I::Neg));
    } else if i.is(I::Scalar) {
        debug_assert!(i.is(I::Pos) ^ i.is(I::Neg));
    }

    if i.is(I::Float) {
        debug_assert!(i.is_not(I::Integer))
    }
}

impl CalcursType for Expr {
    fn desc(&self) -> Item {
        use Expr as E;
        let d = match self {
            E::Symbol(s) => s.desc(),
            E::Rational(r) => r.desc(),
            E::Sum(a) => a.desc(),
            E::Prod(m) => m.desc(),
            E::Pow(p) => p.desc(),
            E::Undefined => Item::Undef,
            E::PlaceHolder(_) => panic!(),
        };
        debug_desc_check(d);
        d
    }
}

impl Construct for Expr {
    #[inline]
    fn simplify(mut self) -> Expr {
        use Expr as E;

        let simplified = GraphExpr::analyse(
            &self,
            Duration::from_millis(100),
            &GraphExpr::scalar_rules(),
            e_graph::GraphExprCostFn,
        );

        self = simplified;

        match self {
            E::Symbol(_) | E::Rational(_) | E::Undefined => self,

            E::Sum(sum) => sum.simplify(),
            E::Prod(prod) => prod.simplify(),
            E::Pow(pow) => pow.simplify(),
            E::PlaceHolder(_) => panic!(),
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Symbol {
    pub name: String,
}
impl Symbol {
    pub fn new<I: Into<String>>(name: I) -> Self {
        Self { name: name.into() }
    }
}

impl CalcursType for Symbol {
    fn desc(&self) -> Item {
        Item::Symbol
    }
}
impl From<Symbol> for Expr {
    #[inline(always)]
    fn from(value: Symbol) -> Expr {
        Expr::Symbol(value)
    }
}
impl CalcursType for &Symbol {
    fn desc(&self) -> Item {
        Item::Symbol
    }
}
impl From<&Symbol> for Expr {
    #[inline(always)]
    fn from(_value: &Symbol) -> Expr {
        panic!("only used for derivative")
    }
}

impl ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Self) -> Self::Output {
        Sum::add(self, rhs)
    }
}
impl ops::AddAssign for Expr {
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs + rhs
            let mut lhs: Expr = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Sum::add(lhs, rhs);
        }
    }
}
impl ops::Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Self) -> Self::Output {
        Diff::sub(self, rhs)
    }
}
impl ops::SubAssign for Expr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Diff::sub(self.clone(), rhs);
    }
}
impl ops::Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Self) -> Self::Output {
        Prod::mul(self, rhs)
    }
}
impl ops::MulAssign for Expr {
    fn mul_assign(&mut self, rhs: Self) {
        // self *= rhs => self = self * rhs
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs * rhs
            let mut lhs = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Prod::mul(lhs, rhs);
        }
    }
}
impl ops::Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Self::Output {
        Rational::MINUS_ONE * self
    }
}
impl ops::Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Self) -> Self::Output {
        Quot::div(self, rhs)
    }
}
impl ops::DivAssign for Expr {
    fn div_assign(&mut self, rhs: Self) {
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs / rhs
            let mut lhs = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Quot::div(lhs, rhs);
        }
    }
}

impl<T: Into<String>> From<T> for Symbol {
    fn from(value: T) -> Self {
        Symbol { name: value.into() }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use Expr as E;
        match self {
            E::Symbol(v) => write!(f, "{v}"),
            E::Rational(r) => write!(f, "{r}"),
            E::Sum(a) => write!(f, "({a})"),
            E::Prod(m) => write!(f, "({m})"),
            E::Pow(p) => write!(f, "{p}"),

            E::Undefined => write!(f, "undefined"),
            E::PlaceHolder(ph) => write!(f, "{ph}"),
        }
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}
impl Debug for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}
impl Debug for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use Expr as E;
        match self {
            E::Symbol(v) => write!(f, "{:?}", v),
            E::Rational(r) => write!(f, "{:?}", r),

            E::Sum(a) => write!(f, "{:?}", a),
            E::Prod(m) => write!(f, "{:?}", m),
            E::Pow(p) => write!(f, "{:?}", p),

            E::Undefined => write!(f, "undefined"),
            E::PlaceHolder(ph) => write!(f, "{:?}", ph),
        }
    }
}
