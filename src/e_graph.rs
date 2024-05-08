use std::fmt;
use std::str::FromStr;
use calcu_rs::expression::{Construct, Expr, Symbol};
use calcu_rs::prelude::Rational;

pub trait NodeFromExpr: Sized + egg::Language {
    fn from_expr(expr: &Expr, children: Vec<egg::Id>) -> Result<Self, &'static str>;
}


pub trait EExpr: egg::Language + NodeFromExpr {
    fn build<L: NodeFromExpr>(e: &Expr) -> Result<egg::RecExpr<L>, &'static str>;
}

impl<E: EExpr> From<&Expr> for egg::Pattern<E> {
    fn from(value: &Expr) -> Self {
        egg::Pattern::new(E::build::<egg::ENodeOrVar<E>>(value).unwrap())
    }
}

impl<L: EExpr> NodeFromExpr for egg::ENodeOrVar<L> {
    fn from_expr(expr: &Expr, children: Vec<egg::Id>) -> Result<Self, &'static str> {
        if let Expr::PlaceHolder(ph) = expr {
            Ok(egg::ENodeOrVar::Var(egg::Var::from_str(*ph).unwrap()))
        } else {
            L::from_expr(expr, children).map(egg::ENodeOrVar::ENode)
        }
    }
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Debug, Hash, Clone)]
pub enum EggExpr {
    Rational(Rational),
    Symbol(Symbol),

    Add([egg::Id; 2]),
    Mul([egg::Id; 2]),
}

macro_rules! rw {
    ( $name:tt; ($($lhs:tt)+) => ($($rhs:tt)+)) =>
    {{
        let searcher = egg::Pattern::from(&$crate::calc!($($lhs)+));
        let applier = egg::Pattern::from(&$crate::calc!($($rhs)+));
        //println!("{} => {}", searcher, applier);
        egg::Rewrite::new(stringify!($name).to_string(), searcher, applier).unwrap()
    }};
    }

impl EggExpr {
    pub fn make_rules() -> Vec<egg::Rewrite<EggExpr, ()>> {
        vec![
            rw!(commute_add; (_a + _b) => (_b + _a)),
            rw!(commute_mul; (_a * _b) => (_b * _a)),
            rw!(add_0; (_a + 0) => (_a)),
            rw!(mul_0; (_a * 0) => (0)),
            rw!(mul_1; (_a * 1) => (_a)),
            rw!(test; (_a * _b + _a * _c) => (_a * (_b + _c)))
        ]
    }
}


impl NodeFromExpr for EggExpr {
    fn from_expr(expr: &Expr, children: Vec<egg::Id>) -> Result<Self, &'static str> {
        use Expr as E;
        match expr {
            E::Sum(_) => {
                return if children.len() != 2 {
                    Err("Expected 2 child ids for Add")
                } else {
                    Ok(EggExpr::Add(egg::LanguageChildren::from_vec(children)))
                }
            }
            E::Prod(_) => {
                return if children.len() != 2 {
                    Err("Expected 2 child ids for Mul")
                } else {
                    Ok(EggExpr::Mul(egg::LanguageChildren::from_vec(children)))
                }
            }
            E::Pow(_) => {}
            _ => {
                if !children.is_empty() {
                    return Err("provided children id's for non recursive variant");
                }
            }

        }

        match expr {
            E::Rational(r) => {
                return Ok(EggExpr::Rational(r.clone()));
            }
            E::Symbol(s) => {
                return Ok(EggExpr::Symbol(s.clone()));
            }
            E::Float(_) => {}
            E::Infinity(_) => {}
            E::Undefined => {}
            _ => {}
        }
        panic!("unhandled variant");
    }
}

impl EExpr for EggExpr {
    fn build<L: NodeFromExpr>(e: &Expr) -> Result<egg::RecExpr<L>, &'static str> {
        fn build_from_expr<L: NodeFromExpr>(
            e: &Expr,
            expr: &mut egg::RecExpr<L>,
        ) -> Result<egg::Id, &'static str> {
            let ops = e.operands();
            if ops.len() == 1 {
                // TODO: unary check
                return Ok(expr.add(L::from_expr(ops.get(0).unwrap(), vec![])?));
            }

            let op_expr = e;
            let n1 = build_from_expr(ops.get(0).unwrap(), expr)?;
            let n2 = build_from_expr(ops.get(1).unwrap(), expr)?;
            let mut node = expr.add(L::from_expr(op_expr, vec![n1, n2])?);

            for i in 2..ops.len() {
                println!("{:?}", ops.get(i).unwrap());
                let n = build_from_expr(ops.get(i).unwrap(), expr)?;
                node = expr.add(L::from_expr(op_expr, vec![node, n])?);
            }
            Ok(node)
        }

        let mut expr = egg::RecExpr::default();
        build_from_expr(e, &mut expr)?;
        Ok(expr)
    }
}

impl egg::Language for EggExpr {

    #[inline(always)]
    fn matches(&self, other: &Self) -> bool {
        use EggExpr as E;
        std::mem::discriminant(self) == std::mem::discriminant(other) &&
            match (self, other) {
                (E::Rational(data1), E::Rational(data2)) => data1 == data2,
                (E::Symbol(data1), E::Symbol(data2)) => data1 == data2,
                (E::Mul(l), E::Mul(r))
                | (E::Add(l), E::Add(r)) => true,

                _ => false
            }
    }

    fn children(&self) -> &[egg::Id] {
        use EggExpr as E;
        match self {
            E::Rational(_) | E::Symbol(_) => &[],
            E::Add(ids)
            | E::Mul(ids) => ids
        }
    }

    fn children_mut(&mut self) -> &mut [egg::Id] {
        use EggExpr as E;
        match self {
            E::Rational(_) | E::Symbol(_) => &mut [],
            E::Add(ids)
            | E::Mul(ids) => ids
        }
    }
}

impl fmt::Display for EggExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use EggExpr as E;
        match self {
            E::Symbol(data) => fmt::Display::fmt(data, f),
            E::Rational(data) => fmt::Display::fmt(data, f),
            E::Add(..) => f.write_str("+"),
            E::Mul(..) => f.write_str("*"),
        }
    }
}

