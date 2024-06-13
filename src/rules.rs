use crate::*;
use calcu_rs::egraph::{merge_option, Analysis, DidMerge, EGraph};
use std::fmt::{Display, Formatter};
use std::ops;

define_rules!(scalar_rules:
    commutative add:                ?a + ?b -> ?b + ?a,
    associative add:                ?a + (?b + ?c) -> (?a + ?b) + ?c,

    commutative mul:                ?a * ?b -> ?b * ?a,
    associative mul:                ?a * (?b * ?c) -> (?a * ?b) * ?c,

    multiplication distributivity:  ?a * (?b + ?c) <-> ?a * ?b + ?a * ?c,

    power multiplication:           ?a^?b * ?a^?c <-> ?a^(?b + ?c),
    power distributivity 1:         (?a * ?b)^?c <-> ?a^?c * ?b^?c,
    power distributivity 2:         (?a^?b)^?c <-> ?a^(?b*?c),
    power distributivity 3:         (?a + ?b)^2 <-> ?a^2 + 2*?a*?b + ?b^2,
);

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ExprFold;

#[derive(Debug, Clone)]
pub enum FoldData {
    Undef,
    Monomial(Monomial),
}

/// c * v_1^e_1 * ... * v_n^e_n
///
#[derive(Debug, Clone, PartialEq, Eq)]
struct Monomial {
    /// c
    coeff: Rational,
    /// v_i^e_i
    vars: HashMap<String, Rational>,
}

impl Monomial {
    fn mul_var(&mut self, var: String, exp: Rational) {
        self.vars
            .entry(var.clone())
            .and_modify(|mut e| *e += exp.clone())
            .or_insert(exp);
    }

    fn pow(mut self, exp: &Self) -> Option<FoldData> {
        if self.coeff.is_zero() && exp.coeff.is_zero() | self.coeff.is_zero() && exp.coeff.is_neg()
        {
            return Some(FoldData::Undef);
        }

        let exp = exp.coeff.clone();
        self.coeff = self.coeff.clone().pow_basic(exp.clone())?;
        self.vars.values_mut().for_each(|e| *e *= exp.clone());
        Some(FoldData::Monomial(self))
    }

    fn write_to_graph(self, eg: &mut EGraph<ExprFold>) -> ID {
        if self.coeff.is_zero() {
            return eg.add(Node::Rational(self.coeff));
        }

        let ids: Vec<_> = self.vars
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
            write!(f, " * {}^{}", v, e).unwrap();
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

    fn mul(mut self, mut rhs: Self) -> Self::Output {
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

impl From<Rational> for Monomial {
    fn from(coeff: Rational) -> Self {
        Self {
            coeff,
            vars: Default::default(),
        }
    }
}
impl From<String> for Monomial {
    fn from(var: String) -> Self {
        let mut p = Self::from(Rational::ONE);
        p.mul_var(var, Rational::ONE);
        p
    }
}

impl Eq for FoldData {}
impl PartialEq for FoldData {
    fn eq(&self, other: &Self) -> bool {
        if let (FoldData::Monomial(mut a), FoldData::Monomial(mut b)) =
            (self.clone(), other.clone())
        {
            if a.coeff.is_zero() {
                a.vars.clear();
            }
            if b.coeff.is_zero() {
                b.vars.clear();
            }
            a.vars.retain(|v, e| !e.is_zero());
            b.vars.retain(|v, e| !e.is_zero());
            a == b
        } else {
            self == other
        }
    }
}

impl FoldData {
    pub fn pow(self, rhs: &Self) -> Option<Self> {
        match (self, rhs) {
            (FoldData::Undef, _) | (_, FoldData::Undef) => Some(FoldData::Undef),
            (FoldData::Monomial(m1), FoldData::Monomial(m2)) => m1.pow(m2),
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

        match node {
            Node::Undef => Some(FoldData::Undef),
            Node::Rational(r) => Some(FoldData::Monomial(r.clone().into())),
            Node::Var(s) => Some(FoldData::Monomial(s.clone().into())),
            Node::Add([lhs, rhs]) => x(lhs)?.clone() + x(rhs)?,
            Node::Mul([lhs, rhs]) => x(lhs)?.clone() * x(rhs)?.clone(),
            Node::Pow([lhs, rhs]) => x(lhs)?.clone().pow(x(rhs)?),
        }
    }

    // does nothing, because when eclasses merge, their data should be equal
    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        merge_option(a, b, |to, from| {
            debug_assert_eq!(*to, from, "from: {:?} to: {:?}", from, to);
            DidMerge(false, false)
        })
    }

    fn modify(eg: &mut EGraph<Self>, id: ID) {
        if let Some(p) = eg[id].data.clone() {
            let x = match p {
                FoldData::Undef => eg.add(Node::Undef),
                FoldData::Monomial(p) => p.write_to_graph(eg),
            };
            eg.union(x, id);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ExprCost;

impl egraph::CostFunction for ExprCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Node, mut costs: C) -> Self::Cost
    where
        C: FnMut(ID) -> Self::Cost,
    {
        let op_cost = match enode {
            Node::Undef => 0,
            Node::Rational(_) | Node::Var(_) | Node::Pow(_) => 2,
            Node::Mul(_) => 4,
            Node::Add(_) => 8,
        };
        egraph::Construct::fold(enode, op_cost, |sum, i| sum.saturating_add(costs(i)))
    }
}

#[cfg(test)]
mod test_rules {
    use super::*;

    macro_rules! cmp {
        ($lhs:expr, $rhs:expr) => {{
            let lhs = $lhs;
            let rhs = $rhs;
            assert!(lhs.cmp_full(&rhs), "{} != {}", lhs, rhs);
        }};
    }

    macro_rules! run {
        ($lhs: expr, $rhs: expr) => {
            let res = $lhs.apply_rules(&scalar_rules());
            cmp!(res, $rhs);
        };
    }

    #[test]
    fn test_scalar_rules() {
        run!(expr!(a + 0), expr!(a));
        run!(expr!(0 + a), expr!(a));
        run!(expr!(1 + 2), expr!(3));
        run!(expr!(a + a), expr!(2 * a));
        run!(expr!(a + a + a), expr!(3 * a));
        run!(expr!(a * a), expr!(a ^ 2));
        run!(expr!(a * a * a), expr!(a ^ 3));
        run!(expr!((x * x + x) / x), expr!(x + 1));
        run!(expr!((x * x + x) / (1 / x)), expr!(x ^ 3 + x ^ 2));
        run!(expr!(x ^ 2 + 2 * x * y + y ^ 2), expr!((x + y) ^ 2));
    }
}
