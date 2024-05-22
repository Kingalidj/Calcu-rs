use hashbrown::{HashSet, HashMap};
use smallvec::SmallVec;

use crate::rational::Rational;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Id(u32);
impl From<usize> for Id {
    fn from(value: usize) -> Self {
        Id(value as u32)
    }
}
impl From<Id> for usize {
    fn from(value: Id) -> Self {
        value.0 as usize
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Hash)]
pub enum Expr {
    Rational(Rational),
    Variable(String),
    // (n1 op (n2 op ... (n(m-1) op nm)))
    Binop(Operator, SmallVec<[Id; 4]>),
}

impl Expr {
    pub fn operands(&self) -> &[Id] {
        match self {
            Expr::Rational(_) | Expr::Variable(_) => &[],
            Expr::Binop(_, ids) => ids.as_slice(),
        }
    }
    pub fn operands_mut(&mut self) -> &mut [Id] {
        match self {
            Expr::Rational(_) | Expr::Variable(_) => &mut [],
            Expr::Binop(_, ids) => ids.as_mut_slice(),
        }
    }
    pub fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Rational(r1), Expr::Rational(r2)) => r1 == r2,
            (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
            (Expr::Binop(op1, _), Expr::Binop(op2, _)) => op1 == op2,
            _ => false,
        }
    }
}

// canonicalization:
// union_find(a) == union_find(b), iff a == b
// eclass id a is canonical if find(a) == a
// enode n is canonical if n = canonicalize(n)
//      where canonicalize(f(a1, a2, ...)) = f(find(a1), find(a2), ...).

#[derive(Debug, Clone, Default)]
pub struct EGraph {
    // original node rep. by each non-canonical id
    nodes: Vec<Expr>,
    // each enode's Id, not the Id of the eclass
    // enodes in memo are canonicalized at each rebuild, but
    // unions can cause them to become out of date
    memo: HashMap<Expr, Id>,

    pending: Vec<Id>,
    analysis_pending: UniqueQueue<Id>,

    pub(crate) classes: HashMap<Id, EClass>,
    pub(crate) classes_by_op: HashMap<Expr, HashSet<Id>>,

    pub(crate) clean: bool,
}
impl EGraph {
    pub fn id_to_node(&self, id: Id) -> &Expr {
        &self.nodes[usize::from(id)]
    }

    pub fn classes(&self) -> impl ExactSizeIterator<Item = &EClass> {
        self.classes.values()
    }
    pub fn classes_mut(&mut self) -> impl ExactSizeIterator<Item = &mut EClass> {
        self.classes.values_mut()
    }
    pub fn is_empty(&self) -> bool {
        self.memo.is_empty()
    }
    pub fn total_size(&self) -> usize {
        self.memo.len()
    }
    pub fn total_num_of_nodes(&self) -> usize {
        self.classes().map(|c| c.len()).sum()
    }
    pub fn num_of_classes(&self) -> usize {
        self.classes.len()
    }
}

#[derive(Debug, Clone)]
pub struct EClass {
    pub id: Id,
    pub nodes: Vec<Expr>,
    pub(crate) parents: Vec<Id>,
}
impl EClass {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Expr> {
        self.nodes.iter()
    }
    // lifetime: iterator lives as long as self
    pub fn parents<'a>(&'a self) -> impl ExactSizeIterator<Item = Id> + 'a {
        self.parents.iter().copied()
    }
}

#[derive(Debug, Clone, Default)]
pub struct UnionFind {
    parents: Vec<Id>,
}

impl UnionFind {
    pub fn make_set(&mut self) -> Id {
        let id = Id::from(self.parents.len());
        self.parents.push(id);
        id
    }
    pub fn size(&self) -> usize {
        self.parents.len()
    }
    pub fn find(&self, mut current: Id) -> Id {
        while current != self.parent(current) {
            current = self.parent(current)
        }
        current
    }
    pub fn find_mut(&mut self, mut current: Id) -> Id {
        while current != self.parent(current) {
            let grandparent = self.parent(self.parent(current));
            *self.parent_mut(current) = grandparent;
            current = grandparent;
        }
        current
    }
    pub fn union(&mut self, root1: Id, root2: Id) -> Id {
        //TODO: shorter set?
        *self.parent_mut(root2) = root1;
        root1
    }
    fn parent(&self, query: Id) -> Id {
        self.parents[usize::from(query)]
    }
    fn parent_mut(&mut self, query: Id) -> &mut Id {
        &mut self.parents[usize::from(query)]
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-1", derive(Serialize, Deserialize))]
pub(crate) struct UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    set: hashbrown::HashSet<T>,
    queue: std::collections::VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: hashbrown::HashSet::default(),
            queue: std::collections::VecDeque::new(),
        }
    }
}

impl<T> UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    pub fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter.into_iter() {
            self.insert(t);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let res = self.queue.pop_front();
        res.as_ref().map(|t| self.set.remove(t));
        res
    }

    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }
}

impl<T> IntoIterator for UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    type Item = T;

    type IntoIter = <std::collections::VecDeque<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.queue.into_iter()
    }
}

impl<A> FromIterator<A> for UniqueQueue<A>
where
    A: Eq + std::hash::Hash + Clone,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let mut queue = UniqueQueue::default();
        for t in iter {
            queue.insert(t);
        }
        queue
    }
}
