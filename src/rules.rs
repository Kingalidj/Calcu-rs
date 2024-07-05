use crate::egraph::{merge_option, Analysis, Construct, CostFunction, DidMerge, EGraph, Subst};
use crate::*;
use calcu_rs::egraph::RecExpr;
use std::{
    fmt::{Display, Formatter},
    ops,
};
use malachite::num::arithmetic::traits::{CeilingLogBase, FloorLogBase2};

trait RuleCondition<A: Analysis>: Fn(&mut EGraph<A>, ID, &Subst) -> bool {}
impl<A: Analysis, F: Fn(&mut EGraph<A>, ID, &Subst) -> bool> RuleCondition<A> for F {}

fn check_node<A: Analysis>(var: &str, cond: impl Fn(&Node) -> bool) -> impl RuleCondition<A> {
    let var = GlobalSymbol::from(var);
    move |eg: &mut EGraph<A>, _, subst: &Subst| eg[subst[var]].nodes.iter().any(&cond)
}

#[inline]
fn not_undef(var: &str) -> impl RuleCondition<ExprFold> {
    let var = GlobalSymbol::from(var);
    move |eg: &mut EGraph<ExprFold>, _, subst: &Subst| {
        matches!(eg[subst[var]].data, Some(FoldData::Undef))
    }
}

#[inline]
fn not_trivial(var: &str) -> impl RuleCondition<ExprFold> {
    let var = GlobalSymbol::from(var);
    move |eg: &mut EGraph<ExprFold>, _, subst: &Subst| {
        if let Some(fd) = &eg[subst[var]].data {
            matches!(fd, FoldData::Undef)
                || matches!(
                    fd,
                    FoldData::Monomial(Monomial {
                        coeff: Rational::ZERO,
                        ..
                    })
                )
        } else {
            false
        }
    }
}

define_rules!(scalar_rules:
    additive identity:              ?a + 0           -> ?a,
    commutative add:                ?a + ?b          -> ?b + ?a,
    associative add:                ?a + (?b + ?c)   -> (?a + ?b) + ?c,
    additive inverse:               ?a + -1 * ?a     -> 0 if not_undef("?a"),

    multiplicative identity:        ?a * 1           -> ?a,
    multiplicative absorber:        ?a * 0           -> 0 if not_undef("?a"),
    commutative mul:                ?a * ?b          -> ?b * ?a,
    associative mul:                ?a * (?b * ?c)   -> (?a * ?b) * ?c,
    multiplicative inverse:         ?a * ?a^-1       -> 1 if not_trivial("?a"),

    distributivity:                 ?a * (?b + ?c)  <-> ?a * ?b + ?a * ?c,

    product of powers:              ?a^?b * ?a^?c   <-> ?a^(?b + ?c),
    power of product:               (?a * ?b)^?c    <-> ?a^?c * ?b^?c,
    power of power:                 (?a^?b)^?c      <-> ?a^(?b*?c),
    binomial theorem n=2:           (?a + ?b)^2     <-> ?a^2 + 2*?a*?b + ?b^2,
);

#[derive(Debug)]
pub struct ExprFold;

#[derive(Debug, Clone)]
pub enum FoldData {
    Undef,
    Monomial(Monomial),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PowProd {
    vars: HashMap<Symbol, Rational, BuildU64Hasher>,
}

/// c * v_1^e_1 * ... * v_n^e_n
///
#[derive(Debug, Clone, PartialEq, Eq)]
struct Monomial {
    /// c
    coeff: Rational,
    /// v_i^e_i
    vars: HashMap<Symbol, Rational>,
}

impl Monomial {
    fn mul_var(&mut self, var: Symbol, exp: Rational) {
        self.vars
            .entry(var)
            .and_modify(|e| *e += exp.clone())
            .or_insert(exp);
    }

    fn is_const(&self) -> bool {
        self.vars.iter().all(|(_, e)| e.is_zero())
    }

    fn write_to_graph(self, eg: &mut EGraph<ExprFold>) -> ID {
        if self.coeff.is_zero() {
            return eg.add(Node::Rational(self.coeff));
        }

        let ids: Vec<_> = self
            .vars
            .into_iter()
            .filter(|(_, e)| !e.is_zero())
            .map(|(var, exp)| {
                if exp == Rational::ONE {
                    eg.add(Node::Var(var))
                } else {
                    let exp = eg.add(Node::Rational(exp));
                    let var = eg.add(Node::Var(var));
                    eg.add(Node::Pow([var, exp]))
                }
            })
            .collect();

        let poly = ids
            .into_iter()
            .reduce(|lhs, rhs| eg.add(Node::Mul([lhs, rhs])));

        match poly {
            Some(poly) if self.coeff.is_one() => poly,
            Some(poly) => {
                let coeff = eg.add(Node::Rational(self.coeff));
                eg.add(Node::Mul([coeff, poly]))
            }
            None => eg.add(Node::Rational(self.coeff)),
        }
    }
}

impl Display for Monomial {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.coeff)?;
        self.vars.iter().for_each(|(v, e)| {
            write!(f, " * {v}^{e}").unwrap();
        });
        Ok(())
    }
}

impl ops::Add<&Self> for Monomial {
    type Output = Option<Self>;

    fn add(mut self, rhs: &Self) -> Self::Output {
        if rhs.coeff.is_zero() {
            return Some(self);
        }

        if self.vars == rhs.vars {
            self.coeff += &rhs.coeff;
            Some(self)
        } else {
            None
        }
    }
}
impl ops::Mul<Self> for Monomial {
    type Output = Monomial;

    fn mul(self, rhs: Self) -> Self::Output {
        let coeff = self.coeff * rhs.coeff;

        if coeff.is_zero() {
            return Self::from(coeff);
        }

        let mut vars = self.vars;
        let mut other_vars = rhs.vars;
        if vars.len() < other_vars.len() {
            std::mem::swap(&mut vars, &mut other_vars);
        }
        let mut p = Self { coeff, vars };

        other_vars.into_iter().for_each(|(var, exp)| {
            p.mul_var(var, exp);
        });
        p
    }
}
impl Pow<&Self> for Monomial {
    type Output = Option<FoldData>;

    fn pow(mut self, exp: &Self) -> Self::Output {
        let bc = &self.coeff;
        let ec = &exp.coeff;
        let const_base = self.is_const();
        let const_exp = exp.is_const();

        if const_base && self.coeff.is_one() {
            // 1^f(x) = 1
            return Some(FoldData::Monomial(self));
        } else if !const_exp {
            // c^f(x) not storable in [FoldData] (for now)
            return None;
        } else if bc.is_zero() && (ec.is_zero() || ec.is_neg()) {
            // 0^0 = 0^x (if x < 0) = undef
            return Some(FoldData::Undef);
        }

        if ec == &Rational::ZERO {
            // f(x)^0 -> 1
            return Some(FoldData::Monomial(Monomial::from(Rational::ONE)));
        } else if ec == &Rational::ONE {
            // f(x)^1 -> f(x)
            return Some(FoldData::Monomial(self));
        }

        let exp = exp.coeff.clone();
        self.coeff = self.coeff.clone().pow_basic(exp.clone())?;
        self.vars.values_mut().for_each(|e| *e *= exp.clone());
        Some(FoldData::Monomial(self))
    }
}

impl From<Rational> for Monomial {
    fn from(coeff: Rational) -> Self {
        Self {
            coeff,
            vars: Default::default(),
        }
    }
}
impl From<Symbol> for Monomial {
    fn from(var: Symbol) -> Self {
        let mut p = Self::from(Rational::ONE);
        p.mul_var(var, Rational::ONE);
        p
    }
}

#[cfg(debug_assertions)]
impl Eq for FoldData {}
#[cfg(debug_assertions)]
impl PartialEq for FoldData {
    fn eq(&self, other: &Self) -> bool {
        match (self.clone(), other.clone()) {
            (FoldData::Monomial(mut a), FoldData::Monomial(mut b)) => {
                if a.coeff.is_zero() {
                    a.vars.clear();
                }
                if b.coeff.is_zero() {
                    b.vars.clear();
                }
                a.vars.retain(|_v, e| !e.is_zero());
                b.vars.retain(|_v, e| !e.is_zero());
                a == b
            }
            (FoldData::Undef, FoldData::Undef) => true,
            _ => false,
        }
    }
}

impl ops::Add<&Self> for FoldData {
    type Output = Option<Self>;

    fn add(self, rhs: &Self) -> Self::Output {
        match (self, rhs) {
            (FoldData::Undef, _) | (_, FoldData::Undef) => Some(FoldData::Undef),
            (FoldData::Monomial(m1), FoldData::Monomial(m2)) => (m1 + m2).map(FoldData::Monomial),
        }
    }
}
impl ops::Mul for FoldData {
    type Output = Option<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (FoldData::Undef, _) | (_, FoldData::Undef) => FoldData::Undef,
            (FoldData::Monomial(m1), FoldData::Monomial(m2)) => FoldData::Monomial(m1 * m2),
        }
        .into()
    }
}
impl Pow<&Self> for FoldData {
    type Output = Option<Self>;

    fn pow(self, rhs: &Self) -> Self::Output {
        match (self, rhs) {
            (FoldData::Undef, _) | (_, FoldData::Undef) => Some(FoldData::Undef),
            (FoldData::Monomial(m1), FoldData::Monomial(m2)) => m1.pow(m2),
        }
    }
}

impl Display for FoldData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FoldData::Undef => write!(f, "undef"),
            FoldData::Monomial(m) => write!(f, "{}", m),
        }
    }
}

impl Analysis for ExprFold {
    type Data = Option<FoldData>;

    fn make(eg: &mut EGraph<Self>, node: &Node) -> Self::Data {
        let x = |i: &ID| eg[*i].data.as_ref();

        let binop = |op_symbol: &'static str,
                     lhs: &ID,
                     rhs: &ID,
                     op_fn: fn(&FoldData, &FoldData) -> Self::Data| {
            let lhs = x(lhs)?;
            let rhs = x(rhs)?;
            let res = op_fn(lhs, rhs);
            if op_symbol != "^" {
                return res;
            }
            if let Some(res) = &res {
                debug!("Analysis::make: [{lhs}] {op_symbol} [{rhs}] => {res}");
            } else {
                debug!("Analysis::make: [{lhs}] {op_symbol} [{rhs}] => none");
            }
            res
        };

        match node {
            Node::Undef => Some(FoldData::Undef),
            Node::Rational(r) => Some(FoldData::Monomial(r.clone().into())),
            Node::Var(s) => Some(FoldData::Monomial((*s).into())),
            Node::Add([lhs, rhs]) => binop("+", lhs, rhs, |l, r| l.clone() + r),
            Node::Mul([lhs, rhs]) => binop("*", lhs, rhs, |l, r| l.clone() * r.clone()),
            Node::Pow([lhs, rhs]) => binop("^", lhs, rhs, |l, r| l.clone().pow(r)),
        }
    }

    // does nothing, because when eclasses merge, their data should be equal
    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        merge_option(a, b, |to, from| {
            #[cfg(debug_assertions)]
            debug_assert_eq!(*to, from, "from: {} to: {}", from, to);
            DidMerge(false, false)
        })
    }

    // TODO: if fold results in 0, only one branch gets folded,
    // e.g {Mul[0, -1]} -> {0, Mul[self, -1]} leads to -1 * -1 = 1
    fn modify(eg: &mut EGraph<Self>, id: ID) {
        if let Some(p) = eg[id].data.clone() {
            let x = match p {
                FoldData::Undef => eg.add(Node::Undef),
                FoldData::Monomial(p) => p.write_to_graph(eg),
            };
            let root = eg.id_to_node(x).clone();
            eg.union(x, id);

            // if we can fold into an atom we only keep the atom
            if root.is_atom() {
                let eclass = &mut eg[id];
                eclass.nodes.clear();
                eclass.nodes.push(root);
            }
        }
    }
}

#[derive(Debug)]
pub struct ExprCost<'a> {
    pub egraph: &'a EGraph<ExprFold>,
}

impl CostFunction for ExprCost<'_> {
    type Cost = u64;

    fn cost<C>(&mut self, enode: &Node, mut costs: C) -> Self::Cost
    where
        C: FnMut(ID) -> (Self::Cost, Node),
    {
        let base_cost = match enode {
            Node::Undef => 1,
            Node::Var(_) | Node::Rational(_) => 1,

            Node::Add([lhs, rhs]) => {
                let lhs = costs(*lhs).1;
                let rhs = costs(*rhs).1;
                if lhs == rhs {
                    4
                } else {
                    2
                }
            },
            Node::Mul([lhs, rhs]) => {
                let lhs = costs(*lhs).1;
                let rhs = costs(*rhs).1;
                if lhs == rhs {
                    6
                } else {
                    3
                }
            },
            Node::Pow([lhs, rhs]) => {
                //let lhs = costs(*lhs).1;
                let rhs = costs(*rhs).1;
                if let Node::Rational(_) = rhs {
                    2
                } else if let Node::Pow(_) = rhs {
                    8
                } else {
                    4
                }
            },
        };
        enode.fold(base_cost, |sum, i| sum.saturating_add(costs(i).0))
    }
}

#[cfg(test)]
mod test_rules {
    use super::*;

    macro_rules! cmp {
        ($lhs: expr, $rhs: expr) => {{
            assert_eq!($lhs, $rhs, "{} != {}", $lhs.fmt_ast(), $rhs.fmt_ast());
        }}
    }

    macro_rules! r {
        ($lhs: expr, $rhs: expr) => {{
            let start = Instant::now();
            let lhs = $lhs;
            let rhs = $rhs;
            let res = lhs.clone().apply_rules(ExprFold, &scalar_rules());
            cmp!(res, rhs);
            println!(
                "{:6.2} ms: {} -> {}",
                start.elapsed().as_secs_f64() * 1000f64,
                lhs.fmt_ast(),
                res.fmt_ast(),
            );
        }};
    }

    macro_rules! e {
        ($($tt:tt)*) => {
            expr!($($tt)*)
        }
    }

    #[test]
    fn test_scalar_rules() {
        init_logger();
        let mut c = ExprContext::new();
        r!(e!(c: a + 0), e!(c: a));
        r!(e!(c: 0 + a), e!(c: a));
        r!(e!(c: 1 + 2), e!(c: 3));
        r!(e!(c: a + a), e!(c: 2 * a));
        r!(e!(c: a + a + a), e!(c: 3 * a));
        r!(e!(c: a * a), e!(c: a ^ 2));
        r!(e!(c: a * a * a), e!(c: a ^ 3));
        r!(e!(c: (a * b) + (a * b)), e!(c: 2 * a * b));
        r!(e!(c: (x * x + x) / x), e!(c: x + 1));
        r!(e!(c: (x * x + x) / (1 / x)), e!(c: x ^ 3 + x ^ 2));
        r!(e!(c: x ^ 2 + 2 * x * y + y ^ 2), e!(c: (x + y) ^ 2));
        r!(e!(c: (x ^ 2 + 2 * x * y + y ^ 2)^2), e!(c: (x + y) ^ 4));

        r!(e!(c: x + 0), e!(c: x));
        r!(e!(c: 0 + x), e!(c: x));
        r!(e!(c: x + x), e!(c: 2 * x));

        r!(e!(c: x - x), e!(c: 0));
        r!(e!(c: 0 - x), e!(c: -x));
        r!(e!(c: x - 0), e!(c: x));
        r!(e!(c: 3 - 2), e!(c: 1));

        r!(e!(c: x * 0), e!(c: 0));
        r!(e!(c: 0 * x), e!(c: 0));
        r!(e!(c: x * 1), e!(c: x));
        r!(e!(c: 1 * x), e!(c: x));
        r!(e!(c: x * x), e!(c: x ^ 2));

        r!(e!(c: 0 ^ 0), e!(c: undef));
        r!(e!(c: 0 ^ 1), e!(c: 0));
        r!(e!(c: 0 ^ 314), e!(c: 0));
        r!(e!(c: 1 ^ 0), e!(c: 1));
        r!(e!(c: 314 ^ 0), e!(c: 1));
        r!(e!(c: 314 ^ 1), e!(c: 314));
        r!(e!(c: x ^ 1), e!(c: x));
        r!(e!(c: 1 ^ x), e!(c: 1));
        r!(e!(c: 1 ^ 314), e!(c: 1));
        r!(e!(c: 3 ^ 3), e!(c: 27));

        r!(e!(c: a - b), e!(c: a + (-1 * b)));
        r!(e!(c: a / b), e!(c: a * b ^ -1));
        r!(e!(c: -a -b), e!(c: -(a + b)));
    }
}
