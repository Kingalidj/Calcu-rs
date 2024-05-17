use crate::{
    define_rules,
    expression::{Expr, Symbol},
    prelude::{Pow, Prod, Rational, Sum},
};

use crate::scalar::Scalar;
use crate::pattern::Item;
use std::fmt;
use std::fmt::Formatter;
use std::str::FromStr;
use calcu_rs::expression::PTR;
use crate::operator::{Diff, Quot};

pub trait GraphFromExpr: Sized + egg::Language {
    fn from_expr(expr: &Expr, children: &[Id]) -> Result<Self, &'static str>;
}

pub trait GraphExpression: egg::Language + GraphFromExpr {
    type Analyser;
    fn build<L: GraphFromExpr>(e: &Expr) -> Result<egg::RecExpr<L>, &'static str>;
}

impl<E: GraphExpression> From<&Expr> for egg::Pattern<E> {
    fn from(value: &Expr) -> Self {
        egg::Pattern::new(E::build::<egg::ENodeOrVar<E>>(value).unwrap())
    }
}

impl<L: GraphExpression> GraphFromExpr for egg::ENodeOrVar<L> {
    fn from_expr(expr: &Expr, children: &[Id]) -> Result<Self, &'static str> {
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
    Undefined,

    Add([Id; 2]),
    Sub([Id; 2]),
    Mul([Id; 2]),
    Div([Id; 2]),
    Pow([Id; 2]),
}

type EGraph = egg::EGraph<GraphExpr, ExprFolding>;
type Id = egg::Id;

#[derive(Debug, Copy, Clone, Default)]
pub struct ExprFolding;
impl egg::Analysis<GraphExpr> for ExprFolding {
    /// Option<(result of fold, pattern that was folded)>
    type Data = Option<(Scalar, String)>;

    fn make(egraph: &EGraph, enode: &GraphExpr) -> Self::Data {
        //let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0);
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0.clone());
        use GraphExpr as GE;

        let make_binop = |sym: &'static str, op: fn(Scalar, Scalar) -> Scalar, a: &Id, b: &Id| {
            let lhs = x(a)?;
            let rhs = x(b)?;
            let op_str = format!("{} {} {}", lhs, sym, rhs);
            let res = op(lhs, rhs);
            let expl = format!("{} -> {}", op_str, res);
            Some((res, expl))
        };

        match enode {
            GE::Rational(r) => Some((r.clone().into(), String::new())),
            GE::Undefined => Some((Scalar::Undefined, String::new())),
            GE::Add([a, b]) => make_binop("+", |a, b| a + b, a, b),
            GE::Sub([a, b]) => make_binop("-", |a, b| a - b, a, b),
            GE::Mul([a, b]) => make_binop("*", |a, b| a * b, a, b),
            GE::Div([a, b]) => make_binop("/", |a, b| a / b, a, b),
            GE::Pow([a, b]) => {
                let lhs = x(a)?;
                let rhs = x(b)?;
                let op_str =  format!("{}^{}", lhs, rhs);
                let res = lhs.pow(rhs)?;
                let expl = format!("{} -> {}", op_str, res);
                Some((res, expl))
            },
            GE::Symbol(_) => return None,
        }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> egg::DidMerge {
        egg::merge_option(a, b, |to, from| {
            debug_assert_eq!(to.0, from.0, "from: {} to: {}", from.1, to.1);
            //println!("from: {}, to: {}", from.0, to.0);
            egg::DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, from: Id) {
        let (res, merge_expl) = match &mut egraph[from].data {
            None => return,
            Some((s, from)) => {
                let merge_expl = format!("{} -> {}", from, s);
                (s.clone(), merge_expl)
            }
        };


        let ge = match res {
            Scalar::Rational(r) => GraphExpr::Rational(r),
            Scalar::Undefined => GraphExpr::Undefined,
        };

        let to = egraph.add(ge);

        if egraph.are_explanations_enabled() {
            egraph.union_trusted(from, to, merge_expl.as_str());
        } else {
            egraph.union(from, to);
        }
    }
}

pub struct GraphExprCostFn;
impl egg::CostFunction<GraphExpr> for GraphExprCostFn {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &GraphExpr, mut costs: C) -> Self::Cost
        where C: FnMut(egg::Id) -> Self::Cost
    {
        use GraphExpr as GE;
        let op_cost = match enode {
            GE::Rational(_)
            | GE::Symbol(_)
            | GE::Undefined => 1,
            | GE::Pow(_) => 2,
            GE::Mul(_)
            | GE::Div(_) => 4,
            GE::Add(_)
            | GE::Sub(_) => 8,
        };
        egg::Language::fold(enode, op_cost, |sum, i| sum + costs(i))
    }
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

trait RuleCondition: Fn(&mut EGraph, Id, &egg::Subst) -> bool {}
impl<F: Fn(&mut EGraph, Id, &egg::Subst) -> bool> RuleCondition for F {}

//fn expr_is_any<const N: usize>(var: &str, itms: [Item; N]) -> impl RuleCondition {
//    let var = var.parse().unwrap();
//    move |egraph, _, subst| {
//        egraph[subst[var]].data
//            .as_ref()
//            .map_or_else(
//                || false,
//                |s| {
//                    let d = s.0.desc();
//                    for i in itms {
//                        if d.is(i) {
//                            return true;
//                        }
//                    }
//                    false
//                }
//            )
//    }
//}

#[inline(always)]
fn check_expr_desc(var: &str, check_fn: impl Fn(Item) -> bool) -> impl RuleCondition {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        let id = subst.get(var);
        if id.is_none() {
            // fallthrough
            return true;
        }
        egraph[*id.unwrap()].data
            .as_ref()
            .map_or_else(
                || check_fn(Item::Symbol),
                |s| check_fn(s.0.desc())
            )
    }
}

#[inline(always)]
fn expr_is<const N: usize>(var: &str, itms: [Item; N]) -> impl RuleCondition {
    check_expr_desc(var, move |d| {
        for i in itms {
            if d.is_not(i) {
                return false;
            }
        }
        true
    })
}
#[inline(always)]
fn expr_is_not<const N: usize>(var: &str, itms: [Item; N]) -> impl RuleCondition {
    check_expr_desc(var, move |d| {
        for i in itms {
            if d.is(i) {
                return false;
            }
        }
        true
    })
}

macro_rules! cond {
    (? $a: ident is $i:expr) => {
        {
            expr_is(concat!("?",stringify!(a)), $i)
        }
    };
    (? $a: ident not $i:expr) => {
        expr_is_not(concat!("?",stringify!(a)), $i)
    }
}

use Item as I;

impl GraphExpr {
    define_rules!(scalar_rules:
          additive identity:       ?a + 0 -> ?a,
          commutative add:         ?a + ?b -> ?b + ?a,
          associative add:         ?a + (?b + ?c) -> (?a + ?b) + ?c,
          subtraction:             ?a - ?b -> ?a + (-1 * ?b),
          subtraction cancle:      ?a - ?a -> 0       if cond!(?a not [I::Undef]),

          multiplication identity: ?a * 1 -> ?a,
          multiplication absorber: ?a * 0 -> 0        if cond!(?a not [I::Undef]),
          commutative mul:         ?a * ?b -> ?b * ?a,
          associative mul:         ?a * (?b * ?c) -> (?a * ?b) * ?c,
          // turn all divisions into multiplications
          division:                ?a / ?b -> ?a * ?b^(-1)     if cond!(?b not [I::Zero, I::Undef]),
          division cancle:         ?a / ?a -> 1                if cond!(?a not [I::Zero, I::Undef]),

          multiplication distributivity 1:    ?a * (?b + ?c) <-> ?a * ?b + ?a * ?c,
          multiplication distributivity 2:    ?n * ?a + ?a -> (?n + 1) * ?a,
          multiplication distributivity 3:    ?a + ?a -> 2 * ?a,

          power identity:          ?a^1 -> ?a,
          power absorber:          ?a^0 -> 1                        if cond!(?a not [I::Zero]),
          power multiplication 1:  ?a^?b * ?a^?c -> ?a^(?b + ?c),
          power multiplication 2:  ?a^?b * ?a -> ?a^(?b + 1)        if cond!(?a not [I::Zero]),
          power multiplication 3:  ?a * ?a -> ?a^2,

          power distributivity 1:  (?a * ?b)^?c -> ?a^?c * ?b^?c,
          power distributivity 2:  (?a^?b)^?c -> ?a^(?b * ?c),
          power distributivity 3:  (?a + ?b)^2 <-> ?a^2 + 2*?a*?b + ?b^2,
    );

    #[inline]
    pub fn analyse<CF: egg::CostFunction<Self>>(
        expr: &Expr,
        time_limit: std::time::Duration,
        rules: &[egg::Rewrite<Self, <Self as GraphExpression>::Analyser>],
        cost_fn: CF,
    ) -> Expr {
        let expr = Self::build(expr).unwrap();
        let runner = egg::Runner::default()
            .with_explanations_enabled()
            .with_expr(&expr)
            .with_time_limit(time_limit)
            .run(rules);

        //runner.egraph.dot().to_png("graph.png").unwrap();

        let extractor = egg::Extractor::new(&runner.egraph, cost_fn);
        let (_bc, be) = extractor.find_best(runner.roots[0]);

        //let mut expl = runner.explain_equivalence(&expr, &be);
        //println!("{:#?}", expl.explanation_trees);
        //let expl_graph = ExplanationGraph { explanation: expl, egraph: &runner.egraph };
        //println!("{}", expl_graph);

            //.to_dot("graph.dot").unwrap();

        Expr::from(be)


    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
enum ExplType {
    Term(GraphExpr),

    Fold(String),
    BinOpRule {
        name: String,
        lhs: PTR<Explanation>,
        rhs: PTR<Explanation>,
    },
    Steps(Vec<Explanation>),
}

impl Explanation {
    fn fmt_self(&self, f: &mut Formatter<'_>, egraph: &EGraph) -> fmt::Result {
        let id = egraph.lookup(self.node.clone()).unwrap();
        let expr = Expr::from(&egraph.id_to_expr(id));
        match &self.typ {
            ExplType::Term(e) => writeln!(f, "term: {}", e),
            ExplType::Fold(s) => writeln!(f, "fold: {}: {}", s, expr),
            ExplType::BinOpRule {name, lhs, rhs } => {
                writeln!(f, "{}: {}", name, expr)?;
                write!(f, "lhs: ")?;
                lhs.fmt_self(f, egraph)?;
                write!(f, "rhs: ")?;
                rhs.fmt_self(f, egraph)?;
                writeln!(f, "")

                //writeln!(f, "{}: {}", name, expr)?;
                //lhs.fmt_self(f, egraph)?;
                //writeln!(f, "")?;
                //rhs.fmt_self(f, egraph)?;
                //writeln!(f, "")
            }
            ExplType::Steps(steps) => {
                for s in steps {
                    s.fmt_self(f, egraph)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ExplanationGraph<'a> {
    egraph: &'a EGraph,
    explanation: Explanation,
}

impl fmt::Display for ExplanationGraph<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.explanation.fmt_self(f, &self.egraph)
    }
}


#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct Explanation {
    node: GraphExpr,
    typ: ExplType,
}

fn fmt_expl_tree(tree: &egg::TreeExplanation<GraphExpr>) -> Explanation {
    let mut steps = Vec::new();
    for e in tree {
        steps.push(fmt_tree_term(e.as_ref()));
    }
    let node = steps.last().unwrap().node.clone();
    let typ = ExplType::Steps(steps);
    Explanation { node, typ }
}

fn fmt_tree_term(term: &egg::TreeTerm<GraphExpr>) -> Explanation {
    use GraphExpr as GE;
    let node = term.node.clone();
    let typ = match node {
        GE::Rational(_)
        | GE::Symbol(_)
        | GE::Undefined => ExplType::Term(node.clone()),
        GE::Add(_)
        | GE::Sub(_)
        | GE::Mul(_)
        | GE::Div(_)
        | GE::Pow(_) => {
            assert_eq!(term.child_proofs.len(), 2);
            let lhs = fmt_expl_tree(term.child_proofs.get(0).unwrap());
            let rhs = fmt_expl_tree(term.child_proofs.get(1).unwrap());

            let mut name = String::new();
            if let Some(rn) = term.forward_rule {
                name.push_str(&rn.to_string());
            }
            if let Some(rn) = term.backward_rule {
                if !name.is_empty() {
                    name.push_str(", ");
                }
                name.push_str(&rn.to_string());
            }

            if name.is_empty() {
                name = node.to_string();
            }

            if let (ExplType::Term(_), ExplType::Term(_)) = (&lhs.typ, &rhs.typ) {
                ExplType::Fold(name)
            } else {
                ExplType::BinOpRule {
                    name,
                    lhs: lhs.into(),
                    rhs: rhs.into(),
                }
            }
        }
    };
    Explanation { node, typ }
}

#[inline(always)]
fn array_ref_to_array<const N: usize, T: Copy>(arr_ref: &[T]) -> [T; N] {
    let mut arr: [T; N] = unsafe { std::mem::zeroed() };
    assert_eq!(arr_ref.len(), N);
    arr.copy_from_slice(arr_ref);
    arr
}

impl GraphFromExpr for GraphExpr {
    fn from_expr(expr: &Expr, children: &[Id]) -> Result<Self, &'static str> {
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
            E::Rational(r) => Ok(GraphExpr::Rational(r.clone())),
            E::Symbol(s) => Ok(GraphExpr::Symbol(s.clone())),
            E::Undefined => Ok(GraphExpr::Undefined),
            E::PlaceHolder(_) => panic!("placeholder should be handled as ENodeOrVar"),
        }
    }
}

impl GraphExpression for GraphExpr {
    type Analyser = ExprFolding;

    fn build<L: GraphFromExpr>(e: &Expr) -> Result<egg::RecExpr<L>, &'static str> {
        fn build_from_expr<L: GraphFromExpr>(
            e: &Expr,
            expr: &mut egg::RecExpr<L>,
        ) -> Result<Id, &'static str> {
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

                (E::Mul(_), E::Mul(_))
                | (E::Add(_), E::Add(_))
                | (E::Pow(_), E::Pow(_))
                | (E::Undefined, E::Undefined) => true,

                _ => false,
            }
    }

    fn children(&self) -> &[Id] {
        use GraphExpr as E;
        match self {
            E::Rational(_) | E::Symbol(_) | E::Undefined => &[],
            E::Add(ids) | E::Mul(ids) | E::Pow(ids) | E::Sub(ids) | E::Div(ids)=> ids,
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        use GraphExpr as E;
        match self {
            E::Rational(_) | E::Symbol(_) | E::Undefined => &mut [],
            E::Add(ids) | E::Mul(ids) | E::Pow(ids) | E::Sub(ids) | E::Div(ids)=> ids,
        }
    }
}

impl From<&egg::RecExpr<GraphExpr>> for Expr {
    fn from(e: &egg::RecExpr<GraphExpr>) -> Self {
        use Expr as E;
        use GraphExpr as GE;

        let mut expr = Vec::with_capacity(e.as_ref().len());

        for (i, n) in e.as_ref().iter().enumerate() {
            let binop = |op: fn(lhs: Expr, rhs: Expr) -> Expr, lhs: &Id, rhs: &Id, exprs: &mut Vec<Expr>| {
                let lhs = exprs.get(usize::from(*lhs)).unwrap().clone();
                let rhs = exprs.get(usize::from(*rhs)).unwrap().clone();
                exprs.insert(i, op(lhs, rhs));
            };
            match n {
                GE::Rational(r) => {
                    expr.push(E::Rational(r.clone()));
                }
                GE::Symbol(s) => {
                    expr.push(E::Symbol(s.clone()));
                }
                GE::Undefined => expr.push(E::Undefined),
                GE::Add([lhs, rhs]) => binop(Sum::sum, lhs, rhs, &mut expr),
                GE::Sub([lhs, rhs]) => binop(Diff::diff, lhs, rhs, &mut expr),
                GE::Mul([lhs, rhs]) => binop(Prod::prod, lhs, rhs, &mut expr),
                GE::Div([lhs, rhs]) => binop(Quot::quot, lhs, rhs, &mut expr),
                GE::Pow([lhs, rhs]) => binop(Pow::pow, lhs, rhs, &mut expr),
            }
        }

        expr.pop().unwrap()
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
            E::Sub(..) => f.write_str("-"),
            E::Mul(..) => f.write_str("*"),
            E::Div(..) => f.write_str("-"),
            E::Pow(..) => f.write_str("^"),
            E::Undefined => f.write_str("undefined"),
        }
    }
}
