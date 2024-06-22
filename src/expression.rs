use crate::{
    egraph::{merge_option, RecExpr, Rewrite},
    *,
};
use calcu_rs::egraph::{Analysis, Construct, DidMerge, EGraph};
use indexmap::IndexMap;
use std::cell::Ref;
use std::{
    any::TypeId,
    cell::{OnceCell, RefCell},
    cmp::Ordering,
    collections,
    collections::VecDeque,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    io,
    ops::Index,
    rc,
    rc::Rc,
};
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
#[repr(transparent)]
pub struct ID(pub(crate) u32);

impl Hash for ID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0 as u64);
    }
}

impl ID {
    pub const MAX: ID = ID::new(usize::MAX);

    #[inline(always)]
    pub(crate) const fn val(self) -> usize {
        self.0 as usize
    }

    /// because rust has no private implementations of public types
    #[inline(always)]
    pub(crate) const fn new(val: usize) -> Self {
        ID(val as u32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Node {
    Rational(Rational),
    Var(Symbol),
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

pub type NodeSet = IndexSet<Node>;
//pub type NodeSetRef = rc::Weak<IndexSet<Node>>;

pub struct ExprContext {
    pub(crate) symbols: SymbolTable,
    pub(crate) nodes: RefCell<NodeSet>,
}

impl ExprContext {
    pub fn new() -> Self {
        Self {
            symbols: SymbolTable::new(),
            nodes: RefCell::new(IndexSet::default()),
        }
    }

    pub fn insert(&self, mut n: Node) -> ID {
        match n {
            Node::Add(ref mut ids) | Node::Mul(ref mut ids) => ids.sort_unstable(),
            Node::Rational(_) | Node::Var(_) | Node::Undef | Node::Pow(_) => {}
        }
        let (indx, _) = self.nodes.borrow_mut().insert_full(n);
        ID::new(indx)
    }

    pub fn get_node(&self, id: ID) -> Ref<Node> {
        Ref::map(self.nodes.borrow(), |nodes| nodes.get_index(id.val()).unwrap())
    }

    pub fn make_expr_id(&self, root_id: ID) -> Expr {
        debug_assert!(self.nodes.borrow().get_index(root_id.val()).is_some());
        Expr {
            cntxt: self,
                root: self.get_node(root_id).clone(),
                root_id,
        }
    }

    pub fn make_expr(&mut self, n: Node) -> Expr {
        let root_id = self.insert(n);
        Expr {
            root_id,
            root: self.get_node(root_id).clone(),
            cntxt: self,
        }
    }

    pub fn var<S: AsRef<str>>(&self, s: S) -> Node {
        Node::Var(self.symbols.insert(s.as_ref()))
    }

    pub fn var_str(&self, s: &Symbol) -> &str {
        self.symbols.get(s)
    }

    pub fn to_dot_to_png(&self, name: &str) -> io::Result<()> {
        let enodes: Vec<_> = self
            .nodes.borrow()
            .clone()
            .into_iter()
            .enumerate()
            .map(|(indx, node)| (node, ID::new(indx)))
            .collect();
        let egraph = EGraph::from_enodes(enodes, ());
        egraph.dot(&self.symbols).to_png(name)
    }
}

impl Index<Symbol> for ExprContext {
    type Output = str;

    fn index(&self, index: Symbol) -> &Self::Output {
        self.symbols.get(&index)
    }
}

impl Debug for ExprContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExprContext")
            .field("nodes", &self.nodes)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct Expr<'a> {
    //pub(crate) nodes: &'a NodeSet ,
    cntxt: &'a ExprContext,
    root_id: ID,
    root: Node,
}

impl Hash for Expr<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.root.hash(state)
    }
}

impl Eq for Expr<'_> {}
impl PartialEq for Expr<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

impl Ord for Expr<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.root.cmp(&other.root)
    }
}
impl PartialOrd for Expr<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.root.partial_cmp(&other.root)
    }
}

impl Construct for Expr<'_> {
    type Discriminant = std::mem::Discriminant<Node>;

    fn discriminant(&self) -> Self::Discriminant {
        std::mem::discriminant(&self.root)
    }

    fn matches(&self, other: &Self) -> bool {
        self.root.matches(&other.root)
    }

    fn operands(&self) -> &[ID] {
        self.root.oprnd_ids()
    }

    fn operands_mut(&mut self) -> &mut [ID] {
        self.root.oprnd_ids_mut()
    }
}

impl ID {
    pub fn with_cntxt(self, cntxt: &ExprContext) -> Expr<'_> {
        let root_id = self;
        let root = cntxt.get_node(root_id).clone();
        Expr {
            root_id,
            root,
            cntxt
        }
    }
}

impl<'a> Expr<'a> {
    pub fn id(&self) -> ID {
        self.root_id
    }

    pub fn root(&self) -> &Node {
        &self.root
    }

    pub fn root_id(&self) -> ID {
        self.root_id
    }

    pub fn from_id(root_id: ID, cntxt: &'a ExprContext) -> Self {
        let root = cntxt.get_node(root_id).clone();
        Self {
            cntxt,
            root_id, root
        }
    }

    pub fn apply_rules<A>(
        self,
        analysis: A,
        rules: &[Rewrite<A>],
    ) -> Expr<'a>
    where
        A: Analysis,
    {
        let start = Instant::now();
        let mut runner = egraph::Runner::<A, ()>::new(analysis)
            .with_explanations_enabled()
            .with_time_limit(Duration::from_millis(500))
            .with_iter_limit(5)
            .with_expr(&self)
            .run(rules);
        log::info!("apply_rules time: {} ms", start.elapsed().as_millis());
        //runner.egraph.dot(&self.cntxt.symbols).to_png("egraph.png").unwrap();

        let extractor = egraph::Extractor::new(&runner.egraph, ExprCost);
        let (_, be) = extractor.find_best2(runner.roots[0], self.cntxt);

        be
    }

    pub fn get_node(&self, id: ID) -> Ref<Node> {
        self.cntxt.get_node(id)
    }

    pub fn extract_nodes(&self) -> Vec<Node> {
        // map old ids to new ones
        let mut ids = hashmap_with_capacity::<ID, ID>(self.cntxt.nodes.borrow().len());
        let mut nodes = Vec::default();
        let mut stack = VecDeque::default();
        let mut bfs_order_ids = Vec::default();

        stack.push_back(self.root_id());
        while let Some(id) = stack.pop_front() {
            bfs_order_ids.push(id);
            let n = &self.get_node(id);
            n.oprnd_ids().iter().for_each(|id| stack.push_back(*id));
        }

        // reverse bfs -> children of node should already exist
        bfs_order_ids.into_iter().rev().for_each(|id| {
            let mut n = self.get_node(id).clone();
            n.oprnd_ids_mut().iter_mut().for_each(|id| *id = ids[id]);
            nodes.push(n);
            let new_id = ID::new(nodes.len() - 1);
            ids.insert(id, new_id);
        });

        // ensure the operands appear before the operator
        if cfg!(debug_assertions) {
            nodes.iter().enumerate().for_each(|(id, n)| {
                n.oprnd_ids()
                    .iter()
                    .for_each(|op_id| debug_assert!(op_id.val() < id))
            })
        }

        nodes
    }
}

impl Index<Symbol> for Expr<'_> {
    type Output = str;
    fn index(&self, index: Symbol) -> &Self::Output {
        &self.cntxt[index]
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

pub struct NodeFmt<'a> {
    node: &'a Node,
    symbol_table: &'a SymbolTable,
}
impl Debug for NodeFmt<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.node {
            Node::Rational(n) => write!(f, "{n}"),
            Node::Var(s) => write!(f, "{}", self.symbol_table.get(s)),
            Node::Undef => write!(f, "undef"),
            Node::Add(_) => write!(f, "+"),
            Node::Mul(_) => write!(f, "*"),
            Node::Pow(_) => write!(f, "^"),
        }
    }
}
impl Display for NodeFmt<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl Node {
    pub fn fmt_symbols<'a>(&'a self, symbol_table: &'a SymbolTable) -> NodeFmt {
        NodeFmt {
            node: self,
            symbol_table,
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Node::Rational(n) => write!(f, "{n}"),
            Node::Var(s) => write!(f, "var({s})"),
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
        Node::Var(s) => write!(f, "{}", &graph[*s]),
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
            dbg_fmt_graph(graph, &graph.get_node(id), f)?;
            write!(f, ", ")?;
        }
        let last_id = ids[ids.len() - 1];
        dbg_fmt_graph(graph, &graph.get_node(last_id), f)?;
        write!(f, "]")?;
    }

    Ok(())
}

impl Display for Expr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.cntxt.nodes.borrow().is_empty() {
            return write!(f, "[]");
        };
        let root = &self.root;
        if self.cntxt.nodes.borrow().len() == 1 {
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
        let mut c = ExprContext::new();
        let var = c.var("x");
        let x = c.insert(var);

        let zero = c.insert(Node::Rational(0.into()));

        let lhs = c.insert(Node::Rational(0.into()));
        let var = c.var("x");
        let rhs = c.insert(var);
        let add_expr = c.insert(Node::Add([lhs, rhs]));

        let lhs = c.insert(Node::Rational(0.into()));
        let var = c.var("x");
        let rhs = c.insert(var);
        let mul_expr = c.insert(Node::Mul([lhs, rhs]));

        eq!(expr!(c: x).root_id, x);
        eq!(expr!(c: 0).root_id, zero);
        eq!(expr!(c: 0 + x).root_id, add_expr);
        eq!(expr!(c: 0 * x).root_id, mul_expr);
        ne!(expr!(c: x * 1).root_id, expr!(c: x + 1).root_id);
    }

    #[test]
    fn test_frac_pow() {
        //check_simplify!(expr!(2^10), expr!(1024))
        //eq!(expr!(2^10), expr!(1024));

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

    //macro_rules! cmp_pat_expr {
    //    ($pat: expr, $expr: expr) => {{
    //        let pat = $pat;
    //        let expr = $expr;
    //        let expr_pat = PatternAst::from(RecExpr::from(expr.clone()));
    //        assert_eq!(pat, expr_pat, "expr({:?}) != pat({:?})", expr, pat);
    //    }};
    //}

    //#[test]
    //fn test_pat_macro() {
    //    pat!(?x + ?a + 2 * ?c);
    //    cmp_pat_expr!(pat!(x + x), expr!(x + x));
    //    cmp_pat_expr!(pat!(-1 * x), expr!(-1 * x));
    //    cmp_pat_expr!(pat!(x / a + 1), expr!(x * a ^ -1 + 1));
    //}
}
