use crate::{
    egraph::{Analysis, Construct, EGraph, Rewrite},
    *,
};
use std::{
    cell::{Ref, RefCell},
    cmp::Ordering,
    collections::VecDeque,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    io,
    ops::{self, Deref, Index},
};

/// Most often used to store an index into an array of Nodes
#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
#[repr(transparent)]
pub struct ID(pub(crate) u32);

impl Hash for ID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0 as u64);
    }
}

impl ID {
    pub const MAX: ID = ID::new(u32::MAX as usize);

    #[inline(always)]
    pub(crate) const fn val(self) -> usize {
        self.0 as usize
    }

    /// because rust has no private implementations of public types
    #[inline(always)]
    pub(crate) const fn new(val: usize) -> Self {
        // is ok because if we overflow we have other problems
        // u32::MAX * sizeof(Rational) ~= 240 gb
        debug_assert!(val <= u32::MAX as usize);
        ID(val as u32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Node {
    // Primitives
    Rational(Rational),
    Var(Symbol),
    Undef,

    // Operators
    Add([ID; 2]),
    Mul([ID; 2]),
    Pow([ID; 2]),
}

impl Node {
    pub const MINUS_TWO: Self = Node::Rational(Rational::MINUS_TWO);
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

impl Default for ExprContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprContext {
    pub fn new() -> Self {
        Self {
            symbols: SymbolTable::new(),
            nodes: RefCell::new(IndexSet::default()),
        }
    }

    fn insert_node_impl(&self, mut n: Node) -> ID {
        match &mut n {
            Node::Add(ids) | Node::Mul(ids) => ids.sort_unstable(),
            Node::Rational(_) | Node::Var(_) | Node::Undef | Node::Pow(_) => {}
        }
        let (indx, _) = self.nodes.borrow_mut().insert_full(n);
        ID::new(indx)
    }

    pub fn insert_mut(&mut self, n: Node) -> ID {
        ID::new(self.nodes.get_mut().insert_full(n).0)
    }

    pub fn insert(&self, n: Node) -> ID {
        self.insert_node_impl(n)
    }

    pub fn get_node(&self, id: ID) -> Ref<Node> {
        Ref::map(self.nodes.borrow(), |nodes| {
            nodes.get_index(id.val()).unwrap()
        })
    }

    pub fn get_rational(&self, id: ID) -> Ref<Rational> {
        Ref::map(self.get_node(id), |n| {
            if let Node::Rational(r) = n {
                r
            } else {
                panic!("get_rational on non-rational node")
            }
        })
    }

    pub fn make_expr_id(&self, root_id: ID) -> Expr {
        debug_assert!(self.nodes.borrow().get_index(root_id.val()).is_some());
        Expr {
            cntxt: self,
            root: self.get_node(root_id).clone(),
            id: root_id,
        }
    }

    pub fn make_expr(&self, n: Node) -> Expr {
        let root_id = self.insert(n);
        Expr {
            id: root_id,
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

    pub fn undef(&mut self) -> Expr {
        let root = Node::Undef;
        let id = self.insert_mut(root.clone());
        Expr { id, root, cntxt: self }
    }
    pub fn rational(&mut self, r: Rational) -> Expr {
        let root = Node::Rational(r);
        let id = self.insert_mut(root.clone());
        Expr { id, root, cntxt: self }
    }
    pub fn add(&mut self, lhs: Expr, rhs: Expr) -> Expr {
        let mut root = Node::Add([lhs.id(), rhs.id()]);
        root.oprnd_ids_mut().sort_unstable();
        let id = self.insert_mut(root.clone());
        Expr { id, root, cntxt: self }
    }
    pub fn sub(&mut self, lhs: Expr, rhs: Expr) -> Expr {
        let nodes = self.nodes.get_mut();
        let mut get_id = |n| ID::new(nodes.insert_full(n).0);
        let min_one = get_id(Node::Rational(Rational::MINUS_ONE));
        let min_rhs = get_id(Node::Mul([min_one, rhs.id()]));
        let root = Node::Add([lhs.id(), min_rhs]);
        let id = get_id(root.clone());
        Expr { id, root, cntxt: self }
    }
    pub fn mul(&mut self, lhs: Expr, rhs: Expr) -> Expr {
        let mut root = Node::Mul([lhs.id(), rhs.id()]);
        root.oprnd_ids_mut().sort_unstable();
        let id = self.insert_mut(root.clone());
        Expr { id, root, cntxt: self }
    }
    pub fn div(&mut self, lhs: Expr, rhs: Expr) -> Expr {
        let nodes = self.nodes.get_mut();
        //let mut id = |n| ID::new(nodes.insert_full(n).0);
        let min_one = self.insert_mut(Node::Rational(Rational::MINUS_ONE));
        let div_rhs = self.insert_mut(Node::Pow([rhs.id(), min_one]));
        let root = Node::Mul([lhs.id(), div_rhs]);
        let id = self.insert_mut(root.clone());
        Expr { id, root, cntxt: self }
    }
    pub fn pow(&mut self, lhs: Expr, rhs: Expr) -> Expr {
        let root = Node::Pow([lhs.id(), rhs.id()]);
        let id = self.insert_mut(root.clone());
        Expr { id, root, cntxt: self }
    }

    pub fn is_rational(&self, n: &Node, r: &Rational) -> bool {
        if let Node::Rational(rational) = &n {
            rational == r
        } else {
            false
        }
    }

    /// Check if we have the ast: Add([LHS, MUL([-1, RHS])])
    pub fn is_sub(&self, n: &Node) -> Option<(ID, ID)> {
        let (lhs, mul) = if let Node::Add([lhs, rhs]) = n {
            (*lhs, *rhs)
        } else {
            return None;
        };

        let (mut min_one, mut rhs) = if let Node::Mul([lhs, rhs]) = &*self.get_node(mul) {
            (*lhs, *rhs)
        } else {
            return None;
        };

        if self.is_rational(&self.get_node(min_one), &Rational::MINUS_ONE) {
            return Some((lhs, rhs));
        }
        std::mem::swap(&mut min_one, &mut rhs);
        if self.is_rational(&self.get_node(min_one), &Rational::MINUS_ONE) {
            return Some((lhs, rhs));
        }

        None
    }

    /// Check if we have the ast: MUL([LHS, DIV([-1, RHS])])
    pub fn is_div(&self, n: &Node) -> Option<(ID, ID)> {
        let (lhs, mul) = if let Node::Mul([lhs, rhs]) = n {
            (*lhs, *rhs)
        } else {
            return None;
        };

        let (rhs, min_one) = if let Node::Pow([lhs, rhs]) = &*self.get_node(mul) {
            (*lhs, *rhs)
        } else {
            return None;
        };

        if self.is_rational(&self.get_node(min_one), &Rational::MINUS_ONE) {
            Some((lhs, rhs))
        } else {
            None
        }
    }

    pub fn fmt_id(&self, id: ID) -> fmt_ast::FmtAst<'_> {
        use f::FmtAst as E;
        use fmt_ast as f;
        let n = self.get_node(id);
        match n.deref() {
            Node::Rational(_) => E::Atom(f::Atom::Rational(self.get_rational(id))),
            Node::Var(v) => E::Atom(f::Atom::Var(self.var_str(v))),
            Node::Undef => E::Atom(f::Atom::Undefined),
            //n @ Node::Add([lhs, rhs]) if let Some((l, r)) = self.is_sub(n) => self.fmt_id(*lhs) - self.fmt_id(*rhs),
            //n @ Node::Mul([lhs, rhs]) if self.is_div(n) => self.fmt_id(*lhs) / self.fmt_id(*rhs),
            n @ Node::Add([lhs, rhs]) => {
                if let Some((lhs, rhs)) = self.is_sub(n) {
                    self.fmt_id(lhs) - self.fmt_id(rhs)
                } else {
                    self.fmt_id(*lhs) + self.fmt_id(*rhs)
                }
            }
            n @ Node::Mul([lhs, rhs]) => {
                if let Some((lhs, rhs)) = self.is_div(n) {
                    self.fmt_id(lhs) / self.fmt_id(rhs)
                } else {
                    self.fmt_id(*lhs) * self.fmt_id(*rhs)
                }
            }
            Node::Pow([lhs, rhs]) => self.fmt_id(*lhs).pow(self.fmt_id(*rhs)),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_dot_to_png(&self, name: &str) -> io::Result<()> {
        let enodes: Vec<_> = self
            .nodes
            .borrow()
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
    id: ID,
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
        self.id == other.id && std::ptr::eq(self.cntxt, other.cntxt)
    }
}

impl Ord for Expr<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.root.cmp(&other.root)
    }
}
impl PartialOrd for Expr<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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
            id: root_id,
            root,
            cntxt,
        }
    }
}


impl<'a> Expr<'a> {
    pub fn id(&self) -> ID {
        self.id
    }

    pub fn root(&self) -> &Node {
        &self.root
    }

    pub fn from_id(root_id: ID, cntxt: &'a ExprContext) -> Self {
        let root = cntxt.get_node(root_id).clone();
        Self {
            cntxt,
            id: root_id,
            root,
        }
    }

    pub fn apply_rules/*<A>*/(self, analysis: ExprFold, rules: &[Rewrite<ExprFold>]) -> Expr<'a>
    //where
    //    A: Analysis + Debug,
    {
        let start = Instant::now();
        let runner = egraph::Runner::<ExprFold, ()>::new(analysis)
            .with_explanations_enabled()
            .with_time_limit(Duration::from_millis(500))
            .with_expr(&self)
            .run(rules);
        info!("apply_rules time: {} ms", start.elapsed().as_millis());

        //#[cfg(not(test))]
        //if runner.egraph.total_number_of_nodes() <= 200 {
        //    runner
        //        .egraph
        //        .dot(&self.cntxt.symbols)
        //        .to_png("egraph.png")
        //        .unwrap();
        //}

        let extractor = egraph::Extractor::new(&runner.egraph, ExprCost { egraph: &runner.egraph });
        let (cost, be) = extractor.find_best2(runner.roots[0], self.cntxt);
        //extractor.dbg_node_cost(runner.roots[0]);

        //let one = self.cntxt.make_expr(Node::ZERO);
        //println!("explanation: {}", runner.explain_existance(&one).get_flat_string());

        //let mut expl = runner.explain_equivalence(&self, &be);
        //println!("{}", expl.get_flat_string());
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

        stack.push_back(self.id());
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

    pub fn fmt_ast(&self) -> FmtAst {
        self.cntxt.fmt_id(self.id)
    }
}

//impl<'a> ops::Add<&'a Expr<'a>> for &Expr<'a> {
//    type Output = Expr<'a>;
//    #[inline]
//    fn add(self, rhs: Self) -> Self::Output {
//        assert_eq!(self.cntxt as *const _, rhs.cntxt as *const _);
//        self.cntxt.make_expr(Node::Add([self.id(), rhs.id()]))
//    }
//}
//impl<'a> ops::Sub<&'a Expr<'a>> for &Expr<'a> {
//    type Output = Expr<'a>;
//    #[inline]
//    fn sub(self, rhs: Self) -> Self::Output {
//        assert_eq!(self.cntxt as *const _, rhs.cntxt as *const _);
//        let c = self.cntxt;
//        let min_one = c.insert(Node::Rational(Rational::MINUS_ONE));
//        let min_rhs = c.insert(Node::Mul([min_one, rhs.id()]));
//        c.make_expr(Node::Add([self.id(), min_rhs]))
//    }
//}
//impl<'a> ops::Mul<&'a Expr<'a>> for &Expr<'a> {
//    type Output = Expr<'a>;
//    #[inline]
//    fn mul(self, rhs: Self) -> Self::Output {
//        assert_eq!(self.cntxt as *const _, rhs.cntxt as *const _);
//        self.cntxt.make_expr(Node::Mul([self.id(), rhs.id()]))
//    }
//}
//impl<'a> ops::Div<&'a Expr<'a>> for &Expr<'a> {
//    type Output = Expr<'a>;
//    #[inline]
//    fn div(self, rhs: Self) -> Self::Output {
//        assert_eq!(self.cntxt as *const _, rhs.cntxt as *const _);
//        let c = self.cntxt;
//        let min_one = c.insert(Node::Rational(Rational::MINUS_ONE));
//        let div_rhs = c.insert(Node::Pow([rhs.id(), min_one]));
//        c.make_expr(Node::Mul([self.id(), div_rhs]))
//    }
//}
//impl<'a> Pow<&'a Expr<'a>> for &Expr<'a> {
//    type Output = Expr<'a>;
//    #[inline]
//    fn pow(self, rhs: Self) -> Self::Output {
//        assert_eq!(self.cntxt as *const _, rhs.cntxt as *const _);
//        self.cntxt.make_expr(Node::Pow([self.id(), rhs.id()]))
//    }
//}

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

    /*
    #[test]
    fn test_expr_macro() {
        let mut c = ExprContext::new();
        let var = c.var("x");
        let x = c.insert(var.id());

        let zero = c.insert(Node::Rational(0.into()));

        let lhs = c.insert(Node::Rational(0.into()));
        let var = c.var("x");
        let rhs = c.insert(var);
        let add_expr = c.insert(Node::Add([lhs, rhs]));

        let lhs = c.insert(Node::Rational(0.into()));
        let var = c.var("x");
        let rhs = c.insert(var);
        let mul_expr = c.insert(Node::Mul([lhs, rhs]));

        eq!(expr!(c: x).id, x);
        eq!(expr!(c: 0).id, zero);
        eq!(expr!(c: 0 + x).id, add_expr);
        eq!(expr!(c: 0 * x).id, mul_expr);
        ne!(expr!(c: x * 1).id, expr!(c: x + 1).id);
    }
     */
}
