use calcu_rs::{prelude::{Pow, Prod, Rational, Sum}, expression::{Expr, Symbol}, define_rules};

use std::fmt;
use std::str::FromStr;

pub trait GraphFromExpr: Sized + egg::Language {
    fn from_expr(expr: &Expr, children: &[egg::Id]) -> Result<Self, &'static str>;
}

pub trait GraphExpression: egg::Language + GraphFromExpr {
    fn build<L: GraphFromExpr>(e: &Expr) -> Result<egg::RecExpr<L>, &'static str>;
}

impl<E: GraphExpression> From<&Expr> for egg::Pattern<E> {
    fn from(value: &Expr) -> Self {
        egg::Pattern::new(E::build::<egg::ENodeOrVar<E>>(value).unwrap())
    }
}

impl<L: GraphExpression> GraphFromExpr for egg::ENodeOrVar<L> {
    fn from_expr(expr: &Expr, children: &[egg::Id]) -> Result<Self, &'static str> {
        if let Expr::PlaceHolder(ph) = expr {
            Ok(egg::ENodeOrVar::Var(egg::Var::from_str(ph).unwrap()))
        } else {
            L::from_expr(expr, children).map(egg::ENodeOrVar::ENode)
        }
    }
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Debug, Hash, Clone)]
pub enum GraphExpr {
    Rational(Rational),
    Symbol(Symbol),

    Add([egg::Id; 2]),
    Mul([egg::Id; 2]),
    Pow([egg::Id; 2]),
}

//macro_rules! rw {
//    ( $name:ident; [$($lhs:tt)+] => [$($rhs:tt)+]) =>
//    {{
//        let searcher = egg::Pattern::from(&$crate::calc!($($lhs)+));
//        let applier = egg::Pattern::from(&$crate::calc!($($rhs)+));
//        //println!("{} => {}", searcher, applier);
//        egg::Rewrite::new(stringify!($name).to_string(), searcher, applier).unwrap()
//    }};
//}

impl GraphExpr {
    //pub fn make_rules() -> Vec<egg::Rewrite<GraphExpr, ()>> {
    //    vec![
    //        rw!(commutative_add; [?a + ?b]         => [?b + ?a]),
    //        rw!(commutative_mul; [?a * ?b]         => [?b * ?a]),
    //        rw!(distributive;    [?a * (?b + ?c)]  => [?a * ?b + ?a * ?c]),
    //        rw!(add_0;           [?a + 0]          => [?a]),
    //        rw!(mul_0;           [?a * 0]          => [0]),
    //        rw!(mul_1;           [?a * 1]          => [?a]),
    //        rw!(pow_0;           [?a^0]            => [1]),
    //        rw!(pow_1;           [?a^1]            => [?a]),
    //    ]
    //}

    define_rules!(basic_rules:

        commutative add: ?a + ?b -> ?b + ?a,
        commutative mul: ?a * ?b -> ?b * ?a,
        distributive:    ?a * (?b + ?c) <-> ?a * ?b + ?a * ?c,
        add identity:    ?a + 0 -> ?a,
        mul identity:    ?a * 1 -> ?a,
        mul zero:        ?a * 0 -> 0,
        pow zero:        ?a^0 -> 1,
        pow one:         ?a^1 -> ?a,
    );
}

#[inline(always)]
fn array_ref_to_array<const N: usize, T: Copy>(arr_ref: &[T]) -> [T; N] {
    let mut arr: [T; N] = unsafe { std::mem::zeroed() };
    assert_eq!(arr_ref.len(), N);
    arr[..].copy_from_slice(arr_ref);
    arr
}

impl GraphFromExpr for GraphExpr {
    fn from_expr(expr: &Expr, children: &[egg::Id]) -> Result<Self, &'static str> {
        use Expr as E;
        match expr {
            E::Sum(_) => {
                if children.len() != 2 {
                    Err("Expected 2 child ids for Add")
                } else {
                    Ok(GraphExpr::Add(array_ref_to_array(children)))
                }
            }
            E::Prod(_) => {
                if children.len() != 2 {
                    Err("Expected 2 child ids for Mul")
                } else {
                    Ok(GraphExpr::Mul(array_ref_to_array(children)))
                }
            }
            E::Pow(_) => {
                if children.len() != 2 {
                    Err("Expected 2 child ids for Pow")
                } else {
                    Ok(GraphExpr::Pow(array_ref_to_array(children)))
                }
            }
            E::Rational(r) => Ok(GraphExpr::Rational(*r)),
            E::Symbol(s) => Ok(GraphExpr::Symbol(s.clone())),
            E::Float(_) => todo!(),
            E::Infinity(_) => todo!(),
            E::Undefined => todo!(),
            Expr::PlaceHolder(_) => panic!("placeholder should be handled as ENodeOrVar"),
        }
    }
}

impl GraphExpression for GraphExpr {
    fn build<L: GraphFromExpr>(e: &Expr) -> Result<egg::RecExpr<L>, &'static str> {
        fn build_from_expr<L: GraphFromExpr>(
            e: &Expr,
            expr: &mut egg::RecExpr<L>,
        ) -> Result<egg::Id, &'static str> {
            let ops = e.operands();

            if ops.is_empty() {
                return Ok(expr.add(L::from_expr(e, &[])?));
            } else if ops.len() == 1 {
                return Ok(expr.add(L::from_expr(ops.first().unwrap(), &[])?));
            }

            let op_expr = e;
            let n1 = build_from_expr(ops.first().unwrap(), expr)?;
            let n2 = build_from_expr(ops.get(1).unwrap(), expr)?;
            let mut node = expr.add(L::from_expr(op_expr, &[n1, n2])?);

            for i in 2..ops.len() {
                let n = build_from_expr(ops.get(i).unwrap(), expr)?;
                node = expr.add(L::from_expr(op_expr, &[node, n])?);
            }
            Ok(node)
        }

        let mut expr = egg::RecExpr::default();
        build_from_expr(e, &mut expr)?;
        Ok(expr)
    }
}

impl egg::Language for GraphExpr {
    #[inline(always)]
    fn matches(&self, other: &Self) -> bool {
        use GraphExpr as E;
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && match (self, other) {
                (E::Rational(data1), E::Rational(data2)) => data1 == data2,
                (E::Symbol(data1), E::Symbol(data2)) => data1 == data2,

                (E::Mul(_), E::Mul(_)) | (E::Add(_), E::Add(_)) | (E::Pow(_), E::Pow(_)) => true,

                _ => false,
            }
    }

    fn children(&self) -> &[egg::Id] {
        use GraphExpr as E;
        match self {
            E::Rational(_) | E::Symbol(_) => &[],
            E::Add(ids) | E::Mul(ids) | E::Pow(ids) => ids,
        }
    }

    fn children_mut(&mut self) -> &mut [egg::Id] {
        use GraphExpr as E;
        match self {
            E::Rational(_) | E::Symbol(_) => &mut [],
            E::Add(ids) | E::Mul(ids) | E::Pow(ids) => ids,
        }
    }
}

impl From<&egg::RecExpr<GraphExpr>> for Expr {
    fn from(e: &egg::RecExpr<GraphExpr>) -> Self {
        use Expr as E;
        use GraphExpr as EE;

        let mut exprs = Vec::with_capacity(e.as_ref().len());

        for (i, n) in e.as_ref().iter().enumerate() {
            match n {
                EE::Rational(r) => {
                    exprs.push(E::Rational(*r));
                }
                EE::Symbol(s) => {
                    exprs.push(E::Symbol(s.clone()));
                }
                EE::Add([lhs, rhs]) => {
                    let lhs = exprs.get(usize::from(*lhs)).unwrap().clone();
                    let rhs = exprs.get(usize::from(*rhs)).unwrap().clone();
                    exprs.insert(i, Sum::add(lhs, rhs));
                }
                EE::Mul([lhs, rhs]) => {
                    let lhs = exprs.get(usize::from(*lhs)).unwrap().clone();
                    let rhs = exprs.get(usize::from(*rhs)).unwrap().clone();
                    exprs.insert(i, Prod::mul(lhs, rhs));
                }
                EE::Pow([lhs, rhs]) => {
                    let lhs = exprs.get(usize::from(*lhs)).unwrap().clone();
                    let rhs = exprs.get(usize::from(*rhs)).unwrap().clone();
                    exprs.insert(i, Pow::pow(lhs, rhs));
                }
            }
        }

        exprs.pop().unwrap()
    }
}

impl From<egg::RecExpr<GraphExpr>> for Expr {
    fn from(e: egg::RecExpr<GraphExpr>) -> Self {
        Expr::from(&e)
    }
}

impl fmt::Display for GraphExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use GraphExpr as E;
        match self {
            E::Symbol(data) => fmt::Display::fmt(data, f),
            E::Rational(data) => fmt::Display::fmt(data, f),
            E::Add(..) => f.write_str("+"),
            E::Mul(..) => f.write_str("*"),
            E::Pow(..) => f.write_str("^"),
        }
    }
}
