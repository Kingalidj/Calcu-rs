use crate::egraph::{merge_option, RecExpr, Rewrite};
use crate::*;
use calcu_rs::egraph::{Analysis, DidMerge, EGraph};
use indexmap::IndexMap;
use std::{
    any::TypeId,
    collections::VecDeque,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
#[repr(transparent)]
pub struct ID(pub(crate) NonMaxU32);

impl Hash for ID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.get() as u64);
    }
}

impl ID {
    #[inline(always)]
    pub(crate) const fn val(self) -> usize {
        self.0.get() as usize
    }

    /// because rust has no private implementations of public types
    #[inline(always)]
    pub(crate) const fn new(val: usize) -> Self {
        ID(NonMaxU32::new(val as u32))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Node {
    Rational(Rational),
    // todo: symboltable
    Var(String),
    Undef,

    Add([ID; 2]),
    Mul([ID; 2]),
    Pow([ID; 2]),
}

impl Node {
    pub const MINUS_ONE: Self = Node::Rational(Rational::MINUS_ONE);
    pub const ZERO: Self = Node::Rational(Rational::ZERO);
    pub const ONE: Self = Node::Rational(Rational::ONE);
    pub const TWO: Self = Node::Rational(Rational::TWO);

    pub(crate) fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::Rational(r1), Node::Rational(r2)) => r1 == r2,
            (Node::Var(s1), Node::Var(s2)) => s1 == s2,
            (Node::Add(_), Node::Add(_))
            | (Node::Mul(_), Node::Mul(_))
            | (Node::Pow(_), Node::Pow(_)) => true,
            _ => false,
        }
    }

    pub(crate) const fn oprnd_ids(&self) -> &[ID] {
        match self {
            Node::Rational(_) | Node::Var(_) | Node::Undef => &[],
            Node::Add(ids) | Node::Mul(ids) | Node::Pow(ids) => ids,
        }
    }
    pub(crate) fn oprnd_ids_mut(&mut self) -> &mut [ID] {
        match self {
            Node::Rational(_) | Node::Var(_) | Node::Undef => &mut [],
            Node::Add(ids) | Node::Mul(ids) | Node::Pow(ids) => ids,
        }
    }

    pub const fn is_atom(&self) -> bool {
        self.oprnd_ids().is_empty()
    }
}

// wip:
// Expression represented with a tree. Every non-leaf node hold IDs, the IDs are mapped to indices into [ExprTree::nodes]
//
// The id of a node n is defined by its index into the set [ExprTree::nodes]. BUT retrieving the node of a given id will not always return that node.
// in short: id = expr.add_node(n); m = expr[id]; m != n
//
// This allows for permanently remapping a node, e.g: Add(a, a) -> Mul(2, a)
//
// todo: use global IndexSet
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Expr {
    pub(crate) root: Option<ID>,
    pub(crate) nodes: Vec<Node>,
    //pub(crate) nodes: IndexSet<Node>,
    //pub(crate) id_to_indx: HashMap<ID, Index, BuildU64Hasher>,
    // todo use set instead of nodes
}

impl Expr {
    pub fn add_node(&mut self, node: Node) -> ID {
        debug_assert!(
            node.oprnd_ids()
                .iter()
                .all(|id| id.val() < self.nodes.len()),
            "node {:?} has children not in this expr: {:?}",
            node,
            self
        );
        self.nodes.push(node);
        ID::new(self.nodes.len() - 1)
    }

    pub fn set_root(&mut self, root_id: ID) {
        self.root = Some(root_id);
    }
    pub fn make_root(&mut self, node: Node) -> ID {
        let root_id = self.add_node(node);
        self.set_root(root_id);
        root_id
    }

    pub fn root_id(&self) -> ID {
        assert!(self.root.is_some(), "root not found");
        self.root.unwrap()
    }
    pub fn root(&self) -> &Node {
        assert!(self.root.is_some(), "root not found");
        &self[self.root_id()]
    }

    pub fn compact(&mut self) {
        let mut ids = hashmap_with_capacity::<ID, ID>(self.nodes.len());
        let mut set = IndexSet::default();
        for (i, mut node) in self.nodes.drain(..).enumerate() {
            // ids[id] should exist if we iterate in correct order
            node.oprnd_ids_mut().iter_mut().for_each(|id| *id = ids[id]);
            let new_id = set.insert_full(node).0;
            ids.insert(ID::new(i), ID::new(new_id));
        }
        self.nodes.extend(set);
    }

    /// removes duplicate nodes and only keeps nodes connected to root.
    /// if root is not available just remove duplicates
    pub fn cleanup(&mut self) {
        if self.root.is_none() {
            self.compact();
            return;
        }

        let root_id = self.root_id();
        // map old ids to new ones
        let mut ids = hashmap_with_capacity::<ID, ID>(self.nodes.len());
        let mut set = IndexSet::default();
        let mut visited = vec![false; self.nodes.len()];

        let mut stack = VecDeque::default();
        stack.push_back(root_id);

        let mut bfs_order_ids = Vec::default();

        while let Some(id) = stack.pop_front() {
            if visited[id.val()] {
                continue;
            }
            visited[id.val()] = true;
            bfs_order_ids.push(id);

            let n = &self[id];
            n.oprnd_ids().iter().for_each(|id| stack.push_back(*id));
        }

        // reverse bfs -> children of node should already exist
        for id in bfs_order_ids.into_iter().rev() {
            let mut n = self[id].clone();

            n.oprnd_ids_mut().iter_mut().for_each(|id| *id = ids[id]);
            let (indx, _) = set.insert_full(n);
            ids.insert(id, ID::new(indx));
        }

        self.nodes = set.into_iter().collect();
        self.root = Some(ids[&root_id]);

        // ensure the operands appear before the operator
        if cfg!(debug_assertions) {
            self.nodes.iter().enumerate().for_each(|(id, n)| {
                n.oprnd_ids()
                    .iter()
                    .for_each(|op_id| debug_assert!(op_id.val() < id))
            })
        }
    }

    fn cmp_nodes(l_expr: &Expr, r_expr: &Expr, l: ID, r: ID) -> bool {
        let lhs = &l_expr[l];
        let rhs = &r_expr[r];

        match (lhs, rhs) {
            (Node::Rational(r1), Node::Rational(r2)) => r1 == r2,
            (Node::Var(s1), Node::Var(s2)) => s1 == s2,
            (Node::Add([l1, l2]), Node::Add([r1, r2]))
            | (Node::Mul([l1, l2]), Node::Mul([r1, r2]))
            | (Node::Pow([l1, l2]), Node::Pow([r1, r2])) => {
                // l1 * l2 = r1 * r2 = r2 * r1
                (Self::cmp_nodes(l_expr, r_expr, *l1, *r1)
                    && Self::cmp_nodes(l_expr, r_expr, *l2, *r2))
                    || (Self::cmp_nodes(l_expr, r_expr, *l1, *r2)
                        && Self::cmp_nodes(l_expr, r_expr, *l2, *r1))
            }
            _ => false,
        }
    }

    pub fn cmp_full(&self, other: &Self) -> bool {
        let l_root = self.root_id();
        let r_root = other.root_id();
        Self::cmp_nodes(self, other, l_root, r_root)
    }

    fn simplify_sub(&mut self, lhs_id: ID, min_one: ID, rhs_id: ID) -> Option<Node> {
        let lhs = &self[lhs_id];
        let rhs = &self[rhs_id];

        match (lhs, rhs) {
            (Node::Rational(r1), Node::Rational(ref r2)) => Node::Rational(r1.clone() - r2),
            // x - 0 = x
            (lhs, &Node::ZERO) => lhs.clone(),
            // 0 - x = -x
            (&Node::ZERO, _) => Node::Mul([min_one, rhs_id]),
            // x - x = 0
            //(Node::Symbol(s1), Node::Symbol(s2)) if s1 == s2 => Node::ZERO,
            (lhs, rhs) if lhs == rhs => Node::ZERO,
            _ => return None,
        }
        .into()
    }

    fn simplify_add(&mut self, lhs_id: ID, rhs_id: ID) -> Option<Node> {
        if let Some((lhs, min_one, rhs)) = self.is_sub(Node::Add([lhs_id, rhs_id])) {
            return self.simplify_sub(lhs, min_one, rhs);
        }

        let lhs = &self[lhs_id];
        let rhs = &self[rhs_id];

        match (lhs, rhs) {
            (Node::Rational(r1), Node::Rational(ref r2)) => Node::Rational(r1.clone() + r2),
            // x + 0 = 0 + x = x
            (lhs, &Node::ZERO) => lhs.clone(),
            (&Node::ZERO, rhs) => rhs.clone(),
            // x + x = 2 * x
            //(Node::Symbol(s1), Node::Symbol(s2)) if s1 == s2 => {
            //    let two = self.add_node(Node::TWO);
            //    Node::Mul([two, lhs_id])
            //}
            (lhs, rhs) if lhs == rhs => {
                let two = self.add_node(Node::TWO);
                Node::Mul([two, lhs_id])
            }
            _ => return None,
        }
        .into()
    }

    fn simplify_mul(&mut self, lhs_id: ID, rhs_id: ID) -> Option<Node> {
        let lhs = &self[lhs_id];
        let rhs = &self[rhs_id];

        match (lhs, rhs) {
            (Node::Rational(r1), Node::Rational(r2)) => Node::Rational(r1.clone() * r2),

            // x * 0 = 0 * x = 0
            (&Node::ZERO, _) | (_, &Node::ZERO) => Node::ZERO,

            // x * 1 = 1 * x = x
            (lhs, &Node::ONE) => lhs.clone(),
            (&Node::ONE, rhs) => rhs.clone(),

            // todo: generic x, e.g if lhs == rhs?
            // x * x = x^2
            //(Node::Symbol(s1), Node::Symbol(s2)) if s1 == s2 => {
            (lhs, rhs) if lhs == rhs => {
                let two = self.add_node(Node::TWO);
                Node::Pow([lhs_id, two])
            }

            _ => return None,
        }
        .into()
    }

    fn simplify_pow(&mut self, lhs_id: ID, rhs_id: ID) -> Option<Node> {
        let lhs = &self[lhs_id];
        let rhs = &self[rhs_id];

        match (lhs, rhs) {
            (&Node::ZERO, &Node::ZERO) => {
                unimplemented!("0^0 undefined");
            }
            (&Node::ZERO, Node::Rational(r)) if r.is_neg() => {
                unimplemented!("0^0 undefined");
            }
            // 0^a = 0 if Re(a) > 0
            // rem: complex number -> 0^i = undef
            (&Node::ZERO, Node::Rational(r)) if r.is_pos() => Node::ZERO,
            // a^0 = 1 if a != 0
            (Node::Rational(r), &Node::ZERO) if !r.is_zero() => Node::ONE,
            // 1^x = 1
            (&Node::ONE, _) => Node::ONE,
            // x^1 = x
            (lhs, &Node::ONE) => lhs.clone(),

            // a^b
            (Node::Rational(a), Node::Rational(b)) => {
                let a_id = lhs_id;
                let b_id = rhs_id;
                // a^b = pow * a^rest
                let (pow, rest) = a.clone().pow(b.clone());

                if rest == Rational::ZERO {
                    Node::Rational(pow)
                } else {
                    let pow = self.add_node(Node::Rational(pow));
                    let rest = self.add_node(Node::Rational(rest));
                    let a_pow_rest = self.add_node(Node::Pow([a_id, rest]));
                    Node::Mul([pow, a_pow_rest])
                }
            }
            _ => return None,
        }
        .into()
    }

    fn is_rational(&self, id: ID, r1: &Rational) -> bool {
        match &self[id] {
            Node::Rational(r2) => r1 == r2,
            _ => false,
        }
    }

    // check if node matches: Add[a, Mul[-1, b]]
    // if found returns Some(lhs_id, min_one_id, rhs_id)
    fn is_sub(&self, n: Node) -> Option<(ID, ID, ID)> {
        let [lhs, mul] = match n {
            Node::Add(ids) => ids,
            _ => return None,
        };
        let [min_one, rhs] = match self[mul] {
            Node::Mul(ids) => ids,
            _ => return None,
        };
        if !self.is_rational(min_one, &Rational::MINUS_ONE) {
            return None;
        }

        Some((lhs, min_one, rhs))
    }

    // if node is div return Some(lhs_id, min_one_id, rhs_id)
    fn is_div(&self, n: Node) -> Option<(ID, ID, ID)> {
        // lhs * pow
        let [lhs, pow] = match n {
            Node::Mul(ids) => ids,
            _ => return None,
        };
        // lhs * (rhs^min_one)
        let [rhs, min_one] = match self[pow] {
            Node::Pow(ids) => ids,
            _ => return None,
        };
        if !self.is_rational(min_one, &Rational::MINUS_ONE) {
            return None;
        }

        Some((lhs, min_one, rhs))
    }

    pub fn simplify(&mut self) -> bool {
        let root_id = ID::new(self.nodes.len() - 1);
        self.simplify_node(root_id)
    }

    pub fn simplify_node(&mut self, id: ID) -> bool {
        let n = self[id].clone();

        let oprnds_simplified = !n
            .oprnd_ids()
            .iter()
            .map(|id| self.simplify_node(*id))
            .any(|simplified| simplified);

        let node_simplified = match n {
            Node::Rational(_) => None,
            Node::Var(_) => None,
            Node::Undef => None,

            Node::Add([rhs, lhs]) => self.simplify_add(rhs, lhs),
            Node::Mul([rhs, lhs]) => self.simplify_mul(rhs, lhs),
            Node::Pow([rhs, lhs]) => self.simplify_pow(rhs, lhs),
        }
        .map_or(false, |simplified_node| {
            self[id] = simplified_node;
            true
        });

        oprnds_simplified || node_simplified
    }

    pub fn apply_rules<A: Analysis + Default>(mut self, rules: &[Rewrite<A>]) -> Self {
        self.cleanup();
        let rec = RecExpr::from(self);
        let mut runner = egraph::Runner::default()
            //.with_time_limit(Duration::from_secs(1))
            .with_iter_limit(3)
            .with_expr(&rec)
            .run(rules);
        //runner.egraph.dot().to_png("graph.png").unwrap();
        let extractor = egraph::Extractor::new(&runner.egraph, ExprCost);
        let (_, be) = extractor.find_best(runner.roots[0]);

        let root = ID::new(be.nodes.len() - 1);
        Self {
            root: Some(root),
            nodes: be.nodes,
        }
    }
}

impl std::ops::Index<ID> for Expr {
    type Output = Node;

    fn index(&self, index: ID) -> &Self::Output {
        &self.nodes[index.val()]
    }
}
impl std::ops::IndexMut<ID> for Expr {
    fn index_mut(&mut self, index: ID) -> &mut Self::Output {
        &mut self.nodes[index.val()]
    }
}
impl Debug for ID {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Display for ID {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Node::Rational(n) => write!(f, "{n}"),
            Node::Var(s) => write!(f, "{s}"),
            Node::Undef => write!(f, "undef"),
            Node::Add(_) => write!(f, "+"),
            Node::Mul(_) => write!(f, "*"),
            Node::Pow(_) => write!(f, "^"),
        }
    }
}

fn dbg_fmt_graph(graph: &Expr, n: &Node, f: &mut Formatter<'_>) -> fmt::Result {
    match n {
        Node::Rational(r) => write!(f, "{}", r),
        Node::Var(s) => write!(f, "{}", s),
        Node::Undef => write!(f, "undef"),
        Node::Add(_) => write!(f, "Add"),
        Node::Mul(_) => write!(f, "Mul"),
        Node::Pow(_) => write!(f, "Pow"),
    }?;

    if !n.is_atom() {
        write!(f, "[")?;
        let ids = n.oprnd_ids();
        for i in 0..ids.len() - 1 {
            let id = ids[i];
            dbg_fmt_graph(graph, &graph[id], f)?;
            write!(f, ", ")?;
        }
        let last_id = ids[ids.len() - 1];
        dbg_fmt_graph(graph, &graph[last_id], f)?;
        write!(f, "]")?;
    }

    Ok(())
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.nodes.is_empty() {
            return write!(f, "[]");
        };
        let root = self.root();
        if self.nodes.len() == 1 {
            return write!(f, "{:?}", root);
        }
        dbg_fmt_graph(self, root, f)
    }
}

#[cfg(test)]
mod test_expressions {
    use super::*;
    use egraph::*;

    macro_rules! eq {
        ($lhs:expr, $rhs:expr) => {{
            let lhs = $lhs;
            let rhs = $rhs;
            assert_eq!(lhs, rhs, "{} != {}", lhs, rhs);
        }};
    }

    macro_rules! ne {
        ($lhs:expr, $rhs:expr) => {{
            let lhs = $lhs;
            let rhs = $rhs;
            assert_ne!(lhs, rhs, "{} == {}", lhs, rhs);
        }};
    }

    #[test]
    fn test_expr_macro() {
        let mut x = Expr::default();
        x.make_root(Node::Var("x".into()));

        let mut zero = Expr::default();
        zero.make_root(Node::Rational(0.into()));

        let mut add_expr = Expr::default();
        let lhs = add_expr.add_node(Node::Rational(0.into()));
        let rhs = add_expr.add_node(Node::Var("x".into()));
        let add = Node::Add([lhs, rhs]);
        add_expr.make_root(add);

        let mut mul_expr = Expr::default();
        let lhs = mul_expr.add_node(Node::Rational(0.into()));
        let rhs = mul_expr.add_node(Node::Var("x".into()));
        let mul = Node::Mul([lhs, rhs]);
        mul_expr.make_root(mul);

        eq!(expr!(x), x);
        eq!(expr!(0), zero);
        eq!(expr!(0 + x), add_expr);
        eq!(expr!(0 * x), mul_expr);
        ne!(expr!(x * 1), expr!(x + 1));
    }

    macro_rules! check_simplify {
        ($expr:expr, $result:expr) => {{
            let mut expr = $expr;
            let mut res = $result;
            expr.simplify();
            assert!(expr.cmp_full(&res), "simplify({}) != {}", expr, res);
            expr.cleanup();
            assert!(expr.cmp_full(&res), "clean(simplify({})) != {}", expr, res);
            res.cleanup();
            assert!(
                expr.cmp_full(&res),
                "clean(simplify({})) != clean({})",
                expr,
                res
            );
        }};
    }

    #[test]
    fn test_cleanup() {
        let mut expr = expr!((a * b) + (a * b));
        // should remove duplicate nodes -> allows equal id simplification
        expr.cleanup();
        expr.simplify();
        assert!(expr.cmp_full(&expr!(2 * (a * b))));
    }

    #[test]
    fn test_simplify() {
        // add
        check_simplify!(expr!(x + 0), expr!(x));
        check_simplify!(expr!(0 + x), expr!(x));
        check_simplify!(expr!(x + x), expr!(2 * x));

        // sub
        check_simplify!(expr!(x - x), expr!(0));
        check_simplify!(expr!(0 - x), expr!(-x));
        check_simplify!(expr!(x - 0), expr!(x));
        check_simplify!(expr!(3 - 2), expr!(1));

        // mul
        check_simplify!(expr!(x * 0), expr!(0));
        check_simplify!(expr!(0 * x), expr!(0));
        check_simplify!(expr!(x * 1), expr!(x));
        check_simplify!(expr!(1 * x), expr!(x));
        check_simplify!(expr!(x * x), expr!(x ^ 2));

        // pow
        check_simplify!(expr!(0 ^ 1), expr!(0));
        check_simplify!(expr!(0 ^ 314), expr!(0));
        check_simplify!(expr!(1 ^ 0), expr!(1));
        check_simplify!(expr!(314 ^ 0), expr!(1));
        check_simplify!(expr!(314 ^ 1), expr!(314));
        check_simplify!(expr!(x ^ 1), expr!(x));
        check_simplify!(expr!(1 ^ x), expr!(1));
        check_simplify!(expr!(1 ^ 314), expr!(1));
        check_simplify!(expr!(3 ^ 3), expr!(27));

        check_simplify!(expr!(a - b), expr!(a + (-1 * b)));
        check_simplify!(expr!(a / b), expr!(a * b ^ -1));
        //check_simplify!(expr!(41^(321/43)), expr!(194754273881 * 41^(20/43)))
    }

    fn test_frac_pow() {
        let mut lhs = expr!(2 ^ 10);
        lhs.simplify();
        eq!(lhs, expr!(1024));

        //let mut lhs = expr!(321^(321 / 43));
        //lhs.simplify();
        //let mut lhs = ExprTree::default();
        //let exp = lhs.add_node(Node::Rational(Rational::from((321u64, 43u64))));
        //let base = lhs.add_node(Node::Rational(Rational::from(41)));
        //lhs.make_root(Node::Pow([base, exp]));

        //let mut rhs = ExprTree::default();
        //let exp = rhs.add_node(Node::Rational(Rational::from((20u64, 43u64))));
        //let base = lhs.add_node(Node::Rational(Rational::from(41)));
        //let rhs_pow = rhs.add_node(Node::Pow([base, exp]));
        //let pow_res = rhs.add_node(Node::Rational(Rational::from(194754273881i64)));
        //rhs.make_root(Node::Mul([pow_res, rhs_pow]));

        //check_simplify!(lhs, rhs);
    }

    macro_rules! cmp_pat_expr {
        ($pat: expr, $expr: expr) => {{
            let pat = $pat;
            let expr = $expr;
            let expr_pat = PatternAst::from(RecExpr::from(expr.clone()));
            assert_eq!(pat, expr_pat, "expr({:?}) != pat({:?})", expr, pat);
        }};
    }

    #[test]
    fn test_pat_macro() {
        pat!(?x + ?a + 2 * ?c);
        cmp_pat_expr!(pat!(x + x), expr!(x + x));
        cmp_pat_expr!(pat!(-1 * x), expr!(-1 * x));
        cmp_pat_expr!(pat!(x / a + 1), expr!(x * a ^ -1 + 1));
    }

    define_rules!(test_rules:
        r0: ?a + ?b -> ?b + ?a,
        r1: a + b -> c,
        r2: ?x + 0 -> ?x,
        r3: 1 + 2 -> 3,
        r4: ?a * ?b -> ?b * ?a,
        r5: ?a * ?b / ?a -> ?b,
    );

    fn find_best<const N: usize>(mut expr: Expr, rules: &[Rewrite<ExprFold>; N]) -> Expr {
        expr.cleanup();
        let rec = RecExpr::from(expr);
        let runner = Runner::default().with_expr(&rec).run(rules);
        let extractor = Extractor::new(&runner.egraph, ExprCost);
        let (_, be) = extractor.find_best(runner.roots[0]);

        let root = ID::new(be.nodes.len() - 1);
        Expr {
            root: Some(root),
            nodes: be.nodes,
        }
    }

    #[test]
    fn test_defined_rules() {
        let r = &test_rules();
        eq!(find_best(expr!(a + b), r), expr!(c));
        eq!(find_best(expr!(b + a), r), expr!(c));
        eq!(find_best(expr!(x + 0), r), expr!(x));
        eq!(find_best(expr!(0 + x), r), expr!(x));
        eq!(find_best(expr!(42 + 0), r), expr!(42));
        eq!(find_best(expr!(0 + 42), r), expr!(42));
        eq!(find_best(expr!(1 + 2), r), expr!(3));
        eq!(find_best(expr!(2 + 1), r), expr!(3));
    }
}
