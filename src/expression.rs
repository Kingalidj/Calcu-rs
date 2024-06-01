use crate::*;
use std::fmt::{self, Debug, Display, Formatter};

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ID(pub(crate) u32);

impl ID {
    #[inline(always)]
    pub(crate) fn indx(self) -> usize {
        self.0 as usize
    }

    /// because rust has no private implementations of public types
    #[inline(always)]
    pub(crate) fn new(val: usize) -> Self {
        ID(val as u32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Node {
    Rational(Rational),
    // todo: symboltable
    Symbol(String),

    Add([ID; 2]),
    Mul([ID; 2]),
    Pow([ID; 2]),
}

impl Node {
    pub const ZERO: Self = Node::Rational(Rational::ZERO);
    pub const ONE: Self = Node::Rational(Rational::ONE);
    pub const MINUS_ONE: Self = Node::Rational(Rational::MINUS_ONE);

    pub(crate) fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::Rational(r1), Node::Rational(r2)) => r1 == r2,
            (Node::Symbol(s1), Node::Symbol(s2)) => s1 == s2,
            _ => false,
        }
    }

    pub(crate) const fn oprnd_ids(&self) -> &[ID] {
        match self {
            Node::Rational(_) | Node::Symbol(_) => &[],
            Node::Add(ids) | Node::Mul(ids) | Node::Pow(ids) => ids,
        }
    }
    pub(crate) fn oprnd_ids_mut(&mut self) -> &mut [ID] {
        match self {
            Node::Rational(_) | Node::Symbol(_) => &mut [],
            Node::Add(ids) | Node::Mul(ids) | Node::Pow(ids) => ids,
        }
    }

    pub const fn is_atom(&self) -> bool {
        self.oprnd_ids().is_empty()
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprGraph {
    pub(crate) nodes: Vec<Node>,
}

impl ExprGraph {
    pub fn compact(mut self) -> Self {
        let mut ids = hashmap_with_capacity::<ID, ID>(self.nodes.len());
        let mut set = IndexSet::default();
        for (i, mut node) in self.nodes.drain(..).enumerate() {
            node.oprnd_ids_mut().iter_mut().for_each(|id| *id = ids[id]);
            let new_id = set.insert_full(node).0;
            ids.insert(ID::new(i), ID::new(new_id));
        }
        self.nodes.extend(set);
        self
    }

    pub fn add_raw(&mut self, node: Node) -> ID {
        debug_assert!(
            node.oprnd_ids().iter().all(|id| id.indx() < self.nodes.len()),
            "node {:?} has children not in this expr: {:?}",
            node,
            self
        );
        self.nodes.push(node.clone());
        ID::new(self.nodes.len() - 1)
    }

    pub fn add(&mut self, node: Node) -> Expression {
        let id = self.add_raw(node.clone());
        Expression { node, id }
    }
}

impl std::ops::Index<ID> for ExprGraph {
    type Output = Node;

    fn index(&self, index: ID) -> &Self::Output {
        self.nodes.get(index.indx()).unwrap()
    }
}
impl std::ops::IndexMut<ID> for ExprGraph {
    fn index_mut(&mut self, index: ID) -> &mut Self::Output {
        self.nodes.get_mut(index.indx()).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Expression {
    pub node: Node, 
    pub id: ID,
}

impl Debug for ID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::fmt::Display for ID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Node::Rational(n) => write!(f, "{n}"),
            Node::Symbol(s) => write!(f, "{s}"),
            Node::Add(_) => write!(f, "+"),
            Node::Mul(_) => write!(f, "*"),
            Node::Pow(_) => write!(f, "^"),
        }
    }
}

fn dbg_fmt_graph(graph: &ExprGraph, n: &Node, f: &mut Formatter<'_>) -> fmt::Result {
    match n {
        Node::Rational(r) => write!(f, "{}", r),
        Node::Symbol(s) => write!(f, "{}", s),
        Node::Add(_) => write!(f, "Add"),
        Node::Mul(_) => write!(f, "Mul"),
        Node::Pow(_) => write!(f, "Pow"),
    }?;

    if !n.is_atom() {
        write!(f, "[")?;
        let ids = n.oprnd_ids();
        for i in 0..ids.len()-1 {
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

impl Debug for ExprGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.nodes.is_empty() {
            return write!(f, "[]")
        };
        let last = self.nodes.last().unwrap();
        if self.nodes.len() == 1 {
            return write!(f, "{:?}", last)
        }
        dbg_fmt_graph(self, last, f)
    }
}
