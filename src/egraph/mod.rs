mod machine;

use std::borrow::Cow;
use std::sync::Arc;
use hashbrown::{HashSet, HashMap};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;
use crate::egraph::machine::Subst;

use crate::rational::Rational;

pub type Symbol = symbol_table::GlobalSymbol;
pub type Var = Symbol;

pub type Duration = std::time::Duration;
pub type Instant = std::time::Instant;

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
impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
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

    pub fn fold<F, T>(&self, init: T, mut f: F) -> T
    where
        F: FnMut(T, Id) -> T,
        T: Clone,
    {
        let mut acc = init;
        self.for_each(|id| acc = f(acc.clone(), id));
        acc
    }
    /// Returns true if the predicate is true on all operands.
    /// Does not short circuit.
    pub fn all<F: FnMut(Id) -> bool>(&self, mut f: F) -> bool {
        self.fold(true, |acc, id| acc && f(id))
    }

    /// Runs a given function on each child `Id`.
    pub fn for_each<F: FnMut(Id)>(&self, f: F) {
        self.operands().iter().copied().for_each(f)
    }

    /// Runs a given function on each child `Id`, allowing mutation of that `Id`.
    pub fn for_each_mut<F: FnMut(&mut Id)>(&mut self, f: F) {
        self.operands_mut().iter_mut().for_each(f)
    }

    /// Runs a falliable function on each child, stopping if the function returns
    /// an error.
    pub fn try_for_each<E, F>(&self, mut f: F) -> Result<(), E>
        where
            F: FnMut(Id) -> Result<(), E>,
            E: Clone,
    {
        self.fold(Ok(()), |res, id| res.and_then(|_| f(id)))
    }

    pub fn update_operands<F: FnMut(Id) -> Id>(&mut self, mut f: F) {
        self.for_each_mut(|id| *id = f(*id))
    }

    pub fn map_operands<F: FnMut(Id) -> Id>(mut self, f: F) -> Self {
        self.update_operands(f);
        self
    }

    pub fn len(&self) -> usize {
        self.fold(0, |len, _| len + 1)
    }

    pub fn is_leaf(&self) -> bool {
        self.all(|_| false)
    }

    fn build_recexpr<F>(&self, mut get_node: F) -> RecExpr
    where
        F: FnMut(Id) -> Self,
    {
        self.try_build_recexpr::<_, std::convert::Infallible>(|id| Ok(get_node(id))).unwrap()
    }

    fn try_build_recexpr<F, Err>(&self, mut get_node: F) -> Result<RecExpr, Err>
    where
        F: FnMut(Id) -> Result<Self, Err>,
    {
        let mut set = IndexSet::<Self>::default();
        let mut ids = HashMap::<Id, Id>::default();
        let mut todo = self.operands().to_vec();

        while let Some(id) = todo.last().copied() {
            if ids.contains_key(&id) {
                todo.pop();
                continue;
            }

            let node = get_node(id)?;

            let mut ids_has_all_children = true;
            for child in node.operands() {
                if !ids.contains_key(child) {
                    ids_has_all_children = false;
                    todo.push(*child);
                }
            }

            if ids_has_all_children {
                let node = node.map_operands(|id| ids[&id]);
                let new_id = set.insert_full(node).0;
                ids.insert(id, Id::from(new_id));
                todo.pop();
            }
        }

        let mut nodes: Vec<Self> = set.into_iter().collect();
        nodes.push(self.clone().map_operands(|id| ids[&id]));
        Ok(RecExpr { nodes })
    }
}

pub struct Runner {
    pub egraph: EGraph,
    pub roots: Vec<Id>,
    pub iterations: Vec<()>,
    pub stop_reason: Option<StopReason>,

    iter_limit: usize,
    node_limit: usize,
    time_limit: Duration,
    start_time: Option<Instant>,

    rewriter: Rewriter,
}

enum StopReason {
    Saturated,
    IterationLimit(usize),
    NodeLimit(usize),
    TimeLimit(f64),
}

impl Runner {
    pub fn new() -> Self {
        Self {
            egraph: EGraph::default(),
            roots: vec![],
            iterations: vec![],
            stop_reason: None,

            iter_limit: 30,
            node_limit: 10_000,
            time_limit: Duration::from_secs(5),
            start_time: None,
            rewriter: Rewriter::default(),
        }
    }

    pub fn run(mut self, rules: &[Rewrite]) -> Self {
        //TODO: check_rules
        self.egraph.rebuild();
        loop {
            let iter = self.run_one(rules);
        }
        todo!()
    }

    fn run_one(&mut self, rules: &[Rewrite]) -> Result<(), StopReason> {
        self.start_time.get_or_insert_with(Instant::now);

        self.check_limits()?;

        let n_nodes = self.egraph.total_size();
        let n_classes = self.egraph.num_of_classes();

        let i = self.iterations.len();
        let start_time = Instant::now();

        let mut matches = Vec::new();
        //let mut applied = IndexMap::default();

        rules.iter().try_for_each(|rw| {
            let ms = self.rewriter.search_rewrite(i, &self.egraph, rw);
            matches.push(ms);
            self.check_limits()
        })?;

        //rules.iter().try_for_each(|rw| {
        //    let ms = self.scheduler.seaerch_rwrite(i, &self.egraph, rw);
        //    matches.push(ms);
        //    self.check_limits()
        //})
        todo!()
    }

    fn check_limits(&self) -> Result<(), StopReason> {
        let elapsed = self.start_time.unwrap().elapsed();
        if elapsed > self.time_limit {
            return Err(StopReason::TimeLimit(elapsed.as_secs_f64()));
        }

        let size = self.egraph.total_size();
        if size > self.node_limit {
            return Err(StopReason::NodeLimit(size));
        }

        if self.iterations.len() >= self.iter_limit {
            return Err(StopReason::IterationLimit(self.iterations.len()))
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rewriter {
    default_match_limit: usize,
    default_ban_length: usize,
    stats: IndexMap<Symbol, RuleStats>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleStats {
    times_applied: usize,
    banned_until: usize,
    times_banned: usize,
    match_limit: usize,
    ban_length: usize,
}

impl Rewriter {
    fn search_rewrite(&mut self, iter: usize, egraph: &EGraph, rewrite: &Rewrite) -> Vec<()> {
        let stats = self.get_rule_stats(rewrite.name);

        if iter < stats.banned_until {
            return vec![];
        }

        let threashold = stats.match_limit.checked_shl(stats.times_banned as u32).unwrap();
        todo!()
        //let matches = rewrite.search_with_limit(egraph, threashold.saturating_add(1));
    }
    fn get_rule_stats(&mut self, name: Symbol) -> &mut RuleStats {
        self.stats.entry(name).or_insert(RuleStats {
            times_applied: 0,
            banned_until: 0,
            times_banned: 0,
            match_limit: self.default_match_limit,
            ban_length: self.default_ban_length,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Rewrite {
    name: Symbol,
    searcher: Arc<Pattern>,
    applier: Arc<Pattern>,
}

impl Rewrite {
    pub fn new(name: impl Into<Symbol>, searcher: Pattern, applier: Pattern) -> Result<Self, String> {
        let name = name.into();
        let searcher = Arc::new(searcher);
        let applier = Arc::new(applier);

        let bound_vars = searcher.vars();
        for v in applier.vars() {
            if !bound_vars.contains(&v) {
                return Err(format!("Rewrite {} refers to unbound var {}", name, v));
            }
        }

        Ok(Self {
            name, applier, searcher
        })
    }
}

// canonicalization:
// union_find(a) == union_find(b), iff a == b
// eclass id a is canonical if find(a) == a
// enode n is canonical if n = canonicalize(n)
//      where canonicalize(f(a1, a2, ...)) = f(find(a1), find(a2), ...).

#[derive(Debug, Clone, Default)]
pub struct EGraph {
    analysis: Analysis,

    // original node rep. by each non-canonical id
    nodes: Vec<Expr>,
    // each enode's Id, not the Id of the eclass
    // enodes in memo are canonicalized at each rebuild, but
    // unions can cause them to become out of date
    memo: HashMap<Expr, Id>,

    pending: Vec<Id>,
    analysis_pending: UniqueQueue<Id>,

    pub(crate) union_find: UnionFind,

    pub(crate) classes: HashMap<Id, EClass>,
    pub(crate) classes_by_op: HashMap<std::mem::Discriminant<Expr>, HashSet<Id>>,

    pub(crate) clean: bool,
}
impl EGraph {

    pub fn rebuild(&mut self) -> usize {
        let old_hc_size = self.memo.len();
        let old_n_eclasses = self.num_of_classes();
        let start = Instant::now();

        let n_unions = self.process_unions();
        let trimmed_nodes = self.rebuild_classes();

        let elapsed = start.elapsed();

        //debug_assert!(self.check_memo());
        #[cfg(debug_assertions)]
        self.check_memo();

        self.clean = true;
        n_unions
    }
    fn process_unions(&mut self) -> usize {
        let mut n_unions = 0;

        while !self.pending.is_empty() || !self.analysis_pending.is_empty() {
            while let Some(class_id) = self.pending.pop() {
                let mut node = self.nodes[usize::from(class_id)].clone();
                node.update_operands(|id| self.find_mut(id));
                if let Some(memo_class) = self.memo.insert(node, class_id) {
                    let modified = self.perform_union(memo_class, class_id);
                    n_unions += modified as usize;
                }
            }

            while let Some(class_id) = self.analysis_pending.pop() {
                let node = self.nodes[usize::from(class_id)].clone();
                let class_id = self.find_mut(class_id);
                let node_data = Analysis::make(self, &node);
                let class = self.classes.get_mut(&class_id).unwrap();

                let merged = self.analysis.merge(class, &node_data);
                if merged.0 {
                    self.analysis_pending.extend(class.parents.iter().copied());
                    //TODO: modify
                }
            }
        }

        debug_assert!(self.pending.is_empty());
        debug_assert!(self.analysis_pending.is_empty());

        n_unions
    }

    fn rebuild_classes(&mut self) -> usize {
        let c_by_op = &mut self.classes_by_op;
        c_by_op.values_mut().for_each(|ids| ids.clear());

        let mut trimmed = 0;
        let uf = &mut self.union_find;

        for class in self.classes.values_mut() {
            let old_len = class.len();
            class.nodes.iter_mut().for_each(|n| n.update_operands(|id| uf.find_mut(id)));
            class.nodes.sort_unstable();
            class.nodes.dedup();

            trimmed += old_len - class.nodes.len();

            let mut add = |n: &Expr| {
                c_by_op.entry(std::mem::discriminant(n))
                    .or_default()
                    .insert(class.id)
            };

            let mut nodes = class.nodes.iter();
            if let Some(mut prev) = nodes.next() {
                add(prev);
                for n in nodes {
                    if !prev.matches(n) {
                        add(n);
                        prev = n;
                    }
                }
            }
        }

        for ids in c_by_op.values_mut() {
            let unique: HashSet<Id> = ids.iter().copied().collect();
            assert_eq!(ids.len(), unique.len());
        }

        trimmed
    }

    fn perform_union(&mut self, i1: Id, i2: Id) -> bool {
        // TODO: pre_union
        self.clean = false;
        let mut id1 = self.find_mut(i1);
        let mut id2 = self.find_mut(i2);
        if id1 == id2 {
            // todo explain alt_rewrite
            return false;
        }

        let c1_parents = self.classes[&id1].parents.len();
        let c2_parents = self.classes[&id2].parents.len();
        if c1_parents < c2_parents {
            std::mem::swap(&mut id1, &mut id2);
        }

        // TODO:
        //if let Some(explain) = &mut self.explain {
        //
        //}
        self.union_find.union(id1, id2);

        debug_assert_ne!(id1, id2);
        let c2 = self.classes.remove(&id2).unwrap();
        let c1 = self.classes.get_mut(&id1).unwrap();
        debug_assert_eq!(id1, c1.id);

        self.pending.extend(c2.parents.iter().cloned());
        let merged = self.analysis.merge(c1, &c2);

        if merged.0 {
            self.analysis_pending.extend(c1.parents.iter().cloned());
        }
        if merged.1 {
            self.analysis_pending.extend(c2.parents.iter().cloned());
        }

        concat_vec(&mut c1.nodes, c2.nodes);
        concat_vec(&mut c1.parents, c2.parents);

        true
    }

    fn check_memo(&self) {
        let mut test_memo = HashMap::<&Expr, Id>::default();

        for (&id, class) in self.classes.iter() {
            assert_eq!(class.id, id);
            for node in &class.nodes {
                // class should exist only once
                if let Some(old) = test_memo.insert(node, id) {
                    assert_eq!(
                        self.find(old),
                        self.find(id),
                        "found unexpected equivalence for {:?}\n{:?}\nvs\n{:?}",
                        node,
                        self[self.find(id)].nodes,
                        self[self.find(old)].nodes,
                    );
                }
            }
        }

        // all classes should be canonicalized
        // for all nodes: self.find(node_id) = class_id
        for (n, e) in test_memo {
            assert_eq!(e, self.find(e));
            assert_eq!(
                Some(e),
                self.memo.get(n).map(|id| self.find(*id)),
                "Entry for {:?} at {:?} in test_memo was incorrect", n, e
            );
        }
    }

    pub fn lookup(&self, enode: impl std::borrow::BorrowMut<Expr>) -> Option<Id> {
        // call find again because nodes were canonicalized
        self.lookup_internal(enode).map(|id| self.find(id))
    }
    fn lookup_internal(&self, mut enode: impl std::borrow::BorrowMut<Expr>) -> Option<Id> {
        let enode = enode.borrow_mut();
        enode.update_operands(|id| self.find(id));
        self.memo.get(enode).copied()
    }

    pub fn id_to_node(&self, id: Id) -> &Expr {
        &self.nodes[usize::from(id)]
    }

    pub fn find(&self, id: Id) -> Id {
        self.union_find.find(id)
    }
    fn find_mut(&mut self, id: Id) -> Id {
        self.union_find.find_mut(id)
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

impl std::ops::Index<Id> for EGraph {
    type Output = EClass;
    fn index(&self, id: Id) -> &Self::Output {
        let id = self.find(id);
        self.classes.get(&id).unwrap_or_else(|| panic!("Invalid id {id}"))
    }
}
impl std::ops::IndexMut<Id> for EGraph {
    fn index_mut(&mut self, id: Id) -> &mut Self::Output {
        let id = self.find(id);
        self.classes.get_mut(&id).unwrap_or_else(|| panic!("Invalid id {id}"))
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    pub ast: PatternAst,
    program: machine::Program,
}
impl Pattern {
    pub fn vars(&self) -> Vec<Var> {
        let mut vars = vec![];
        for n in &self.ast.nodes {
            if let ENodeOrVar::Var(v) = n {
                if !vars.contains(v) {
                    vars.push(*v)
                }
            }
        }
        vars
    }

    pub fn search_with_limit(&self, egraph: &EGraph, limit: usize) -> Vec<SearchMatches> {
        //match self.ast.nodes.as_ref().last().unwrap() {

        //}
        todo!()
    }
}

pub struct SearchMatches<'a> {
    pub eclass: Id,
    pub subst: Vec<Subst>,
    pub ast: Option<Cow<'a, PatternAst>>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternAst {
    nodes: Vec<ENodeOrVar>,
}
impl PatternAst {
    pub(crate) fn extract(&self, new_root: Id) -> Self {
        self[new_root].build_recexpr(|id| self[id].clone())
    }
}
impl std::ops::Index<Id> for PatternAst {
    type Output = ENodeOrVar;
    fn index(&self, index: Id) -> &Self::Output {
        self.nodes.get(usize::from(index)).expect("index out of bounds")
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecExpr {
    nodes: Vec<Expr>,
}
impl RecExpr {
    pub(crate) fn extract(&self, new_root: Id) -> Self {
        self[new_root].build_recexpr(|id| self[id].clone())
    }
}
impl std::ops::Index<Id> for RecExpr {
    type Output = Expr;
    fn index(&self, index: Id) -> &Self::Output {
        self.nodes.get(usize::from(index)).expect("index out of bounds")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ENodeOrVar {
    ENode(Expr),
    Var(Var),
}
impl ENodeOrVar {
}


#[derive(Debug, Clone, Copy, Default)]
pub struct Analysis;

impl Analysis {
    fn make(egraph: &EGraph, enode: &Expr) ->  EClass {
        todo!()
    }
    fn merge(&mut self, to: &mut EClass, from: &EClass) -> (bool, bool) {
        todo!()
    }
    fn allow_ematching_cycles(&self) -> bool { true }
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
pub(crate) struct UniqueQueue<T>
    where
        T: Eq + std::hash::Hash + Clone,
{
    set: HashSet<T>,
    queue: std::collections::VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
    where
        T: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: HashSet::default(),
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

pub(crate) fn concat_vec<T>(to: &mut Vec<T>, mut from: Vec<T>) {
    if to.len() < from.len() {
        std::mem::swap(to, &mut from);
    }
    to.extend(from);
}
