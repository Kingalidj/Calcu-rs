use crate::egraph::*;
use construct::Analysis;
use std::{
    borrow::BorrowMut,
    fmt::{self, Debug},
};

use log::*;

/** A data structure to keep track of equalities between expressions.

In `egg`, the main types associated with e-graphs are
[`EGraph`], [`EClass`], [`construct`], and [`ID`].

Many methods of [`EGraph`] deal with [`ID`]s, which represent e-classes.
Because eclasses are frequently merged, many [`ID`]s will refer to the
same e-class.

You can use the `egraph[id]` syntax to get an [`EClass`] from an [`ID`]

[`add`]: EGraph::add()
[`union`]: EGraph::union()
[`rebuild`]: EGraph::rebuild()
[equivalence relation]: https://en.wikipedia.org/wiki/Equivalence_relation
[congruence relation]: https://en.wikipedia.org/wiki/Congruence_relation
[extract]: Extractor
**/
#[derive(Clone)]
pub struct EGraph<A: Analysis> {
    /// The `Analysis` given when creating this `EGraph`.
    pub analysis: A,
    /// The `Explain` used to explain equivalences in this `EGraph`.
    pub(crate) explain: Option<Explain>,
    unionfind: EClassUnion,
    /// Stores the original node represented by each non-canonical id
    nodes: Vec<Node>,
    /// Stores each enode's `Id`, not the `Id` of the eclass.
    /// Enodes in the memo are canonicalized at each rebuild, but after rebuilding new
    /// unions can cause them to become out of date.
    memo: HashMap<Node, ID>,

    /// Nodes which need to be processed for rebuilding. The `Id` is the `Id` of the enode,
    /// not the canonical id of the eclass.
    pending: Vec<ID>,
    analysis_pending: UniqueQueue<ID>,
    pub(crate) classes: HashMap<ID, EClass<A::Data>>,
    pub(crate) classes_by_op: HashMap<<Node as Construct>::Discriminant, HashSet<ID>>,
    /// Whether or not reading operation are allowed on this e-graph.
    /// Mutating operations will set this to `false`, and
    /// [`EGraph::rebuild`] will set it to true.
    /// Reading operations require this to be `true`.
    /// Only manually set it if you know what you're doing.
    pub clean: bool,
}

impl<A: Analysis + Default> Default for EGraph<A> {
    fn default() -> Self {
        Self::new(A::default())
    }
}

// manual debug impl to avoid L: Construct bound on EGraph defn
impl<A: Analysis> Debug for EGraph<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("EGraph")
            .field("memo", &self.memo)
            .field("classes", &self.classes)
            .finish()
    }
}

impl<A: Analysis> EGraph<A> {
    /// Creates a new, empty `EGraph` with the given `Analysis`
    pub fn new(analysis: A) -> Self {
        Self {
            analysis,
            classes: Default::default(),
            unionfind: Default::default(),
            nodes: Default::default(),
            clean: false,
            explain: None,
            pending: Default::default(),
            memo: Default::default(),
            analysis_pending: Default::default(),
            classes_by_op: Default::default(),
        }
    }

    /// Returns an iterator over the eclasses in the egraph.
    pub fn classes(&self) -> impl ExactSizeIterator<Item = &EClass<A::Data>> {
        self.classes.values()
    }

    /// Returns an mutating iterator over the eclasses in the egraph.
    pub fn classes_mut(&mut self) -> impl ExactSizeIterator<Item = &mut EClass<A::Data>> {
        self.classes.values_mut()
    }

    /// Returns `true` if the egraph is empty
    pub fn is_empty(&self) -> bool {
        self.memo.is_empty()
    }

    /// Returns the number of enodes in the `EGraph`.
    ///
    /// Actually returns the size of the hashcons index.
    pub fn total_size(&self) -> usize {
        self.memo.len()
    }

    /// Iterates over the classes, returning the total number of nodes.
    pub fn total_number_of_nodes(&self) -> usize {
        self.classes().map(|c| c.len()).sum()
    }

    /// Returns the number of eclasses in the egraph.
    pub fn number_of_classes(&self) -> usize {
        self.classes.len()
    }

    /// Enable explanations for this `EGraph`.
    /// This allows the egraph to explain why two expressions are
    /// equivalent with the [`explain_equivalence`](EGraph::explain_equivalence) function.
    pub fn with_explanations_enabled(mut self) -> Self {
        if self.explain.is_some() {
            return self;
        }
        if self.total_size() > 0 {
            panic!("Need to set explanations enabled before adding any expressions to the egraph.");
        }
        self.explain = Some(Explain::new());
        self
    }

    /// By default, egg runs a greedy algorithm to reduce the size of resulting explanations (without complexity overhead).
    /// Use this function to turn this algorithm off.
    pub fn without_explanation_length_optimization(mut self) -> Self {
        if let Some(explain) = &mut self.explain {
            explain.optimize_explanation_lengths = false;
            self
        } else {
            panic!("Need to set explanations enabled before setting length optimization.");
        }
    }

    /// By default, egg runs a greedy algorithm to reduce the size of resulting explanations (without complexity overhead).
    /// Use this function to turn this algorithm on again if you have turned it off.
    pub fn with_explanation_length_optimization(mut self) -> Self {
        if let Some(explain) = &mut self.explain {
            explain.optimize_explanation_lengths = true;
            self
        } else {
            panic!("Need to set explanations enabled before setting length optimization.");
        }
    }

    /// Make a copy of the egraph with the same nodes, but no unions between them.
    pub fn copy_without_unions(&self, analysis: A) -> Self {
        if self.explain.is_none() {
            panic!("Use runner.with_explanations_enabled() or egraph.with_explanations_enabled() before running to get a copied egraph without unions");
        }
        let mut egraph = Self::new(analysis);
        for node in &self.nodes {
            egraph.add(node.clone());
        }

        egraph
    }

    /// Performs the union between two egraphs.
    pub fn egraph_union(&mut self, other: &EGraph<A>) {
        let right_unions = other.get_union_equalities();
        for (left, right, why) in right_unions {
            self.union_instantiations(
                &other.id_to_pattern(left).ast,
                &other.id_to_pattern(right).ast,
                &Default::default(),
                why,
            );
        }
        self.rebuild();
    }

    pub(crate) fn from_enodes(enodes: Vec<(Node, ID)>, analysis: A) -> Self {
        let mut egraph = Self::new(analysis);
        let mut ids: HashMap<ID, ID> = Default::default();

        loop {
            let mut did_something = false;

            for (enode, id) in &enodes {
                let valid = enode.oprnd_ids().iter().all(|c| ids.contains_key(c));
                if !valid {
                    continue;
                }

                let mut enode = enode.clone().map_operands(|c| ids[&c]);

                if egraph.lookup(&mut enode).is_some() {
                    continue;
                }

                let added = egraph.add(enode);
                if let Some(existing) = ids.get(id) {
                    egraph.union(*existing, added);
                } else {
                    ids.insert(*id, added);
                }

                did_something = true;
            }

            if !did_something {
                break;
            }
        }

        egraph
    }

    /// A intersection algorithm between two egraphs.
    /// The intersection is correct for all terms that are equal in both egraphs.
    /// Be wary, though, because terms which are not represented in both egraphs
    /// are not captured in the intersection.
    /// The runtime of this algorithm is O(|E1| * |E2|), where |E1| and |E2| are the number of enodes in each egraph.
    pub fn egraph_intersect(&self, other: &EGraph<A>, analysis: A) -> EGraph<A> {
        let mut product_map: HashMap<(ID, ID), ID> = Default::default();
        let mut enodes = vec![];

        for class1 in self.classes() {
            for class2 in other.classes() {
                self.intersect_classes(other, &mut enodes, class1.id, class2.id, &mut product_map);
            }
        }

        Self::from_enodes(enodes, analysis)
    }

    fn get_product_id(class1: ID, class2: ID, product_map: &mut HashMap<(ID, ID), ID>) -> ID {
        if let Some(id) = product_map.get(&(class1, class2)) {
            *id
        } else {
            let id = ID::new(product_map.len());
            product_map.insert((class1, class2), id);
            id
        }
    }

    fn intersect_classes(
        &self,
        other: &EGraph<A>,
        res: &mut Vec<(Node, ID)>,
        class1: ID,
        class2: ID,
        product_map: &mut HashMap<(ID, ID), ID>,
    ) {
        let res_id = Self::get_product_id(class1, class2, product_map);
        for node1 in &self.classes[&class1].nodes {
            for node2 in &other.classes[&class2].nodes {
                if node1.matches(node2) {
                    let children1 = node1.oprnd_ids();
                    let children2 = node2.oprnd_ids();
                    let mut new_node = node1.clone();
                    let children = new_node.operands_mut();
                    for (i, (child1, child2)) in children1.iter().zip(children2.iter()).enumerate()
                    {
                        let prod = Self::get_product_id(
                            self.canon_id(*child1),
                            other.canon_id(*child2),
                            product_map,
                        );
                        children[i] = prod;
                    }

                    res.push((new_node, res_id));
                }
            }
        }
    }

    /// Pick a representative term for a given Id.
    ///
    /// Calling this function on an uncanonical `Id` returns a representative based on the how it
    /// was obtained (see [`add_uncanoncial`](EGraph::add_uncanonical),
    /// [`add_expr_uncanonical`](EGraph::add_expr_uncanonical))
    pub fn id_to_expr(&self, id: ID) -> RecExpr<Node> {
        let mut res = Default::default();
        let mut cache = Default::default();
        self.id_to_expr_internal(&mut res, id, &mut cache);
        res
    }

    fn id_to_expr_internal(
        &self,
        res: &mut RecExpr<Node>,
        node_id: ID,
        cache: &mut HashMap<ID, ID>,
    ) -> ID {
        if let Some(existing) = cache.get(&node_id) {
            return *existing;
        }
        let new_node = self
            .id_to_node(node_id)
            .clone()
            .map_operands(|child| self.id_to_expr_internal(res, child, cache));
        let res_id = res.add(new_node);
        cache.insert(node_id, res_id);
        res_id
    }

    /// Like [`id_to_expr`](EGraph::id_to_expr) but only goes one layer deep
    pub fn id_to_node(&self, id: ID) -> &Node {
        &self.nodes[id.val()]
    }

    /// Like [`id_to_expr`](EGraph::id_to_expr), but creates a pattern instead of a term.
    /// When an eclass listed in the given substitutions is found, it creates a variable.
    /// Otherwise it behaves like [`id_to_expr`](EGraph::id_to_expr).
    pub fn id_to_pattern(&self, id: ID) -> Pattern {
        let mut res = Default::default();
        let mut cache = Default::default();
        self.id_to_pattern_internal(&mut res, id, &mut cache);
        Pattern::new(res)
    }

    // todo: take out, without cache
    // todo: linearize
    fn id_to_pattern_internal(
        &self,
        res: &mut PatternAst,
        node_id: ID,
        cache: &mut HashMap<ID, ID>,
    ) -> ID {
        if let Some(existing) = cache.get(&node_id) {
            return *existing;
        }
        let res_id = {
            let new_node = self
                .id_to_node(node_id)
                .clone()
                .map_operands(|child| self.id_to_pattern_internal(res, child, cache));
            res.add(ENodeOrVar::ENode(new_node))
        };
        cache.insert(node_id, res_id);
        res_id
    }

    /// Get all the unions ever found in the egraph in terms of enode ids.
    pub fn get_union_equalities(&self) -> UnionEqualities {
        if let Some(explain) = &self.explain {
            explain.get_union_equalities()
        } else {
            panic!("Use runner.with_explanations_enabled() or egraph.with_explanations_enabled() before running to get union equalities");
        }
    }

    /// Disable explanations for this `EGraph`.
    pub fn with_explanations_disabled(mut self) -> Self {
        self.explain = None;
        self
    }

    /// Check if explanations are enabled.
    pub fn are_explanations_enabled(&self) -> bool {
        self.explain.is_some()
    }

    /// Get the number of congruences between nodes in the egraph.
    /// Only available when explanations are enabled.
    pub fn get_num_congr(&mut self) -> usize {
        if let Some(explain) = &mut self.explain {
            explain
                .with_nodes(&self.nodes)
                .get_num_congr::<A>(&self.classes, &self.unionfind)
        } else {
            panic!("Use runner.with_explanations_enabled() or egraph.with_explanations_enabled() before running to get explanations.")
        }
    }

    /// Get the number of nodes in the egraph used for explanations.
    pub fn get_explanation_num_nodes(&mut self) -> usize {
        if let Some(explain) = &mut self.explain {
            explain.with_nodes(&self.nodes).get_num_nodes()
        } else {
            panic!("Use runner.with_explanations_enabled() or egraph.with_explanations_enabled() before running to get explanations.")
        }
    }

    /// When explanations are enabled, this function
    /// produces an [`Explanation`] describing why two expressions are equivalent.
    ///
    /// The [`Explanation`] can be used in it's default tree form or in a less compact
    /// flattened form. Each of these also has a s-expression string representation,
    /// given by [`get_flat_string`](Explanation::get_flat_string) and [`get_string`](Explanation::get_string).
    pub fn explain_equivalence(&mut self, left_expr: &Expr, right_expr: &Expr) -> Explanation {
        let left = self.add_expr_uncanonical(left_expr);
        let right = self.add_expr_uncanonical(right_expr);

        self.explain_id_equivalence(left, right)
    }

    /// Equivalent to calling [`explain_equivalence`](EGraph::explain_equivalence)`(`[`id_to_expr`](EGraph::id_to_expr)`(left),`
    /// [`id_to_expr`](EGraph::id_to_expr)`(right))` but more efficient
    ///
    /// This function picks representatives using [`id_to_expr`](EGraph::id_to_expr) so choosing
    /// `Id`s returned by functions like [`add_uncanonical`](EGraph::add_uncanonical) is important
    /// to control explanations
    pub fn explain_id_equivalence(&mut self, left: ID, right: ID) -> Explanation {
        if self.canon_id(left) != self.canon_id(right) {
            panic!(
                "Tried to explain equivalence between non-equal terms {:?} and {:?}",
                self.id_to_expr(left),
                self.id_to_expr(left)
            );
        }
        if let Some(explain) = &mut self.explain {
            explain.with_nodes(&self.nodes).explain_equivalence::<A>(
                left,
                right,
                &mut self.unionfind,
                &self.classes,
            )
        } else {
            panic!("Use runner.with_explanations_enabled() or egraph.with_explanations_enabled() before running to get explanations.")
        }
    }

    /// When explanations are enabled, this function
    /// produces an [`Explanation`] describing how the given expression came
    /// to be in the egraph.
    ///
    /// The [`Explanation`] begins with some expression that was added directly
    /// into the egraph and ends with the given `expr`.
    /// Note that this function can be called again to explain any intermediate terms
    /// used in the output [`Explanation`].
    pub fn explain_existance(&mut self, expr: &Expr) -> Explanation {
        let id = self.add_expr_uncanonical(expr);
        self.explain_existance_id(id)
    }

    /// Equivalent to calling [`explain_existance`](EGraph::explain_existance)`(`[`id_to_expr`](EGraph::id_to_expr)`(id))`
    /// but more efficient
    fn explain_existance_id(&mut self, id: ID) -> Explanation {
        if let Some(explain) = &mut self.explain {
            explain.with_nodes(&self.nodes).explain_existance(id)
        } else {
            panic!("Use runner.with_explanations_enabled() or egraph.with_explanations_enabled() before running to get explanations.")
        }
    }

    /// Return an [`Explanation`] for why a pattern appears in the egraph.
    pub fn explain_existance_pattern(
        &mut self,
        pattern: &PatternAst,
        subst: &Subst,
    ) -> Explanation {
        let id = self.add_instantiation_noncanonical(pattern, subst);
        if let Some(explain) = &mut self.explain {
            explain.with_nodes(&self.nodes).explain_existance(id)
        } else {
            panic!("Use runner.with_explanations_enabled() or egraph.with_explanations_enabled() before running to get explanations.")
        }
    }

    /// Get an explanation for why an expression matches a pattern.
    pub fn explain_matches(
        &mut self,
        left_expr: &Expr,
        right_pattern: &PatternAst,
        subst: &Subst,
    ) -> Explanation {
        let left = self.add_expr_uncanonical(left_expr);
        let right = self.add_instantiation_noncanonical(right_pattern, subst);

        if self.canon_id(left) != self.canon_id(right) {
            panic!(
                "Tried to explain equivalence between non-equal terms {:?} and {:?}",
                left_expr, right_pattern
            );
        }
        if let Some(explain) = &mut self.explain {
            explain.with_nodes(&self.nodes).explain_equivalence::<A>(
                left,
                right,
                &mut self.unionfind,
                &self.classes,
            )
        } else {
            panic!("Use runner.with_explanations_enabled() or egraph.with_explanations_enabled() before running to get explanations.");
        }
    }

    /// Canonicalizes an eclass id.
    ///
    /// This corresponds to the `find` operation on the egraph's
    /// underlying unionfind data structure.
    ///
    /// todo: revert back to find?
    pub fn canon_id(&self, id: ID) -> ID {
        self.unionfind.root(id)
    }

    /// This is private, but internals should use this whenever
    /// possible because it does path compression.
    fn canon_id_mut(&mut self, id: ID) -> ID {
        self.unionfind.root_mut(id)
    }

    /// Creates a [Dot] to visualize this egraph
    ///
    pub fn dot<'a>(&'a self, symbols: &'a SymbolTable) -> egraph::Dot<A> {
        Dot {
            egraph: self,
            symbols,
            config: vec![],
            use_anchors: true,
        }
    }
}

/// Given an `Id` using the `egraph[id]` syntax, retrieve the e-class.
impl<A: Analysis> std::ops::Index<ID> for EGraph<A> {
    type Output = EClass<A::Data>;
    fn index(&self, id: ID) -> &Self::Output {
        let id = self.canon_id(id);
        self.classes
            .get(&id)
            .unwrap_or_else(|| panic!("Invalid id {}", id))
    }
}

/// Given an `Id` using the `&mut egraph[id]` syntax, retrieve a mutable
/// reference to the e-class.
impl<A: Analysis> std::ops::IndexMut<ID> for EGraph<A> {
    fn index_mut(&mut self, id: ID) -> &mut Self::Output {
        let id = self.canon_id_mut(id);
        self.classes
            .get_mut(&id)
            .unwrap_or_else(|| panic!("Invalid id {}", id))
    }
}

impl<A: Analysis> EGraph<A> {
    /// Adds a [`RecExpr`] to the [`EGraph`], returning the id of the RecExpr's eclass.
    pub fn add_expr(&mut self, expr: &Expr) -> ID {
        let id = self.add_expr_uncanonical(expr);
        self.canon_id(id)
    }

    /// Similar to [`add_expr`](EGraph::add_expr) but the `Id` returned may not be canonical
    ///
    /// Calling [`id_to_expr`](EGraph::id_to_expr) on this `Id` return a copy of `expr` when explanations are enabled
    pub fn add_expr_uncanonical(&mut self, expr: &Expr) -> ID {
        let nodes = expr.clone().extract_nodes();
        let mut new_ids = Vec::with_capacity(nodes.len());
        let mut new_node_q = Vec::with_capacity(nodes.len());
        for node in nodes {
            let new_node = node.clone().map_operands(|i| new_ids[i.val()]);
            let size_before = self.unionfind.size();
            let next_id = self.add_uncanonical(new_node);
            if self.unionfind.size() > size_before {
                new_node_q.push(true);
            } else {
                new_node_q.push(false);
            }
            if let Some(explain) = &mut self.explain {
                node.for_each_oprnd(|child| {
                    // Set the existance reason for new nodes to their parent node.
                    if new_node_q[child.val()] {
                        explain.set_existance_reason(new_ids[child.val()], next_id);
                    }
                });
            }
            new_ids.push(next_id);
        }
        *new_ids.last().unwrap()
    }

    /// Adds a [`Pattern`] and a substitution to the [`EGraph`], returning
    /// the eclass of the instantiated pattern.
    pub fn add_instantiation(&mut self, pat: &PatternAst, subst: &Subst) -> ID {
        let id = self.add_instantiation_noncanonical(pat, subst);
        self.canon_id(id)
    }

    /// Similar to [`add_instantiation`](EGraph::add_instantiation) but the `Id` returned may not be
    /// canonical
    ///
    /// Like [`add_uncanonical`](EGraph::add_uncanonical), when explanations are enabled calling
    /// Calling [`id_to_expr`](EGraph::id_to_expr) on this `Id` return an correspond to the
    /// instantiation of the pattern
    fn add_instantiation_noncanonical(&mut self, pat: &PatternAst, subst: &Subst) -> ID {
        let nodes = pat.as_ref();
        let mut new_ids = Vec::with_capacity(nodes.len());
        let mut new_node_q = Vec::with_capacity(nodes.len());
        for node in nodes {
            match node {
                ENodeOrVar::Var(var) => {
                    let id = self.canon_id(subst[*var]);
                    new_ids.push(id);
                    new_node_q.push(false);
                }
                ENodeOrVar::ENode(node) => {
                    let new_node = node.clone().map_operands(|i| new_ids[i.val()]);
                    let size_before = self.unionfind.size();
                    let next_id = self.add_uncanonical(new_node);
                    if self.unionfind.size() > size_before {
                        new_node_q.push(true);
                    } else {
                        new_node_q.push(false);
                    }

                    if let Some(explain) = &mut self.explain {
                        node.for_each_oprnd(|child| {
                            if new_node_q[child.val()] {
                                explain.set_existance_reason(new_ids[child.val()], next_id);
                            }
                        });
                    }
                    new_ids.push(next_id);
                }
            }
        }
        *new_ids.last().unwrap()
    }

    /// Lookup the eclass of the given enode.
    ///
    /// You can pass in either an owned enode or a `&mut` enode,
    /// in which case the enode's children will be canonicalized.
    ///
    pub fn lookup<B>(&self, mut enode: B) -> Option<ID>
    where
        B: BorrowMut<Node>,
    {
        //self.lookup_internal(enode).map(|id| self.eclass_id(id))
        let enode = enode.borrow_mut();
        enode.update_operands(|id| self.canon_id(id));
        match self.memo.get(enode).copied() {
            None => None,
            Some(id) => self.canon_id(id).into(),
        }
    }

    fn lookup_internal<B>(&self, mut enode: B) -> Option<ID>
    where
        B: BorrowMut<Node>,
    {
        let enode = enode.borrow_mut();
        enode.update_operands(|id| self.canon_id(id));
        self.memo.get(enode).copied()
    }

    /// Lookup the eclass of the given [`RecExpr`].
    ///
    /// Equivalent to the last value in [`EGraph::lookup_expr_ids`].
    pub fn lookup_expr(&self, expr: &RecExpr<Node>) -> Option<ID> {
        self.lookup_expr_ids(expr)
            .and_then(|ids| ids.last().copied())
    }

    /// Lookup the eclasses of all the nodes in the given [`RecExpr`].
    pub fn lookup_expr_ids(&self, expr: &RecExpr<Node>) -> Option<Vec<ID>> {
        let nodes = expr.as_ref();
        let mut new_ids = Vec::with_capacity(nodes.len());
        for node in nodes {
            let node = node.clone().map_operands(|i| new_ids[i.val()]);
            let id = self.lookup(node)?;
            new_ids.push(id)
        }
        Some(new_ids)
    }

    /// Adds an enode to the [`EGraph`].
    ///
    /// When adding an enode, to the egraph, [`add`] it performs
    /// _hashconsing_ (sometimes called interning in other contexts).
    ///
    /// Hashconsing ensures that only one copy of that enode is in the egraph.
    /// If a copy is in the egraph, then [`add`] simply returns the id of the
    /// eclass in which the enode was found.
    ///
    /// Like [`union`](EGraph::union), this modifies the e-graph.
    ///
    /// [`add`]: EGraph::add()
    pub fn add(&mut self, enode: Node) -> ID {
        let id = self.add_uncanonical(enode);
        self.canon_id(id)
    }

    /// Similar to [`add`](EGraph::add) but the `Id` returned may not be canonical
    ///
    /// When explanations are enabled calling [`id_to_expr`](EGraph::id_to_expr) on this `Id` will
    /// correspond to the parameter `enode`
    pub fn add_uncanonical(&mut self, mut enode: Node) -> ID {
        let original = enode.clone();
        if let Some(existing_id) = self.lookup_internal(&mut enode) {
            let id = self.canon_id(existing_id);
            // when explanations are enabled, we need a new representative for this expr
            if let Some(explain) = self.explain.as_mut() {
                if let Some(existing_explain) = explain.uncanon_memo.get(&original) {
                    *existing_explain
                } else {
                    let new_id = self.unionfind.init_class();
                    explain.add(original.clone(), new_id, new_id);
                    debug_assert_eq!(ID::new(self.nodes.len()), new_id);
                    self.nodes.push(original);
                    self.unionfind.union(id, new_id);
                    explain.union(existing_id, new_id, Justification::Congruence, true);
                    new_id
                }
            } else {
                existing_id
            }
        } else {
            let id = self.make_new_eclass(enode, original.clone());
            if let Some(explain) = self.explain.as_mut() {
                explain.add(original, id, id);
            }

            // now that we updated explanations, run the analysis for the new eclass
            A::modify(self, id);
            self.clean = false;
            id
        }
    }

    /// This function makes a new eclass in the egraph (but doesn't touch explanations)
    fn make_new_eclass(&mut self, enode: Node, original: Node) -> ID {
        let id = self.unionfind.init_class();
        log::trace!("...adding {:?} to class {}", enode, id);
        let class = EClass {
            id,
            nodes: vec![enode.clone()],
            data: A::make(self, &original),
            parents: Default::default(),
        };

        debug_assert_eq!(ID::new(self.nodes.len()), id);
        self.nodes.push(original);

        // add this enode to the parent lists of its children
        enode.for_each_oprnd(|child| {
            self[child].parents.push(id);
        });

        // TODO is this needed?
        self.pending.push(id);

        self.classes.insert(id, class);
        assert!(self.memo.insert(enode, id).is_none());

        id
    }

    /// Checks whether two [`RecExpr`]s are equivalent.
    /// Returns a list of id where both expression are represented.
    /// In most cases, there will none or exactly one id.
    ///
    pub fn equivs(&self, expr1: &RecExpr<Node>, expr2: &RecExpr<Node>) -> Vec<ID> {
        let pat1 = Pattern::from(expr1.as_ref());
        let pat2 = Pattern::from(expr2.as_ref());
        let matches1 = pat1.search(self);
        trace!("Matches1: {:?}", matches1);

        let matches2 = pat2.search(self);
        trace!("Matches2: {:?}", matches2);

        let mut equiv_eclasses = Vec::new();

        for m1 in &matches1 {
            for m2 in &matches2 {
                if self.canon_id(m1.eclass) == self.canon_id(m2.eclass) {
                    equiv_eclasses.push(m1.eclass)
                }
            }
        }

        equiv_eclasses
    }

    /// Given two patterns and a substitution, add the patterns
    /// and union them.
    ///
    /// When explanations are enabled [`with_explanations_enabled`](Runner::with_explanations_enabled), use
    /// this function instead of [`union`](EGraph::union).
    ///
    /// Returns the id of the new eclass, along with
    /// a `bool` indicating whether a union occured.
    pub fn union_instantiations(
        &mut self,
        from_pat: &PatternAst,
        to_pat: &PatternAst,
        subst: &Subst,
        rule_name: impl Into<GlobalSymbol>,
    ) -> (ID, bool) {
        let id1 = self.add_instantiation_noncanonical(from_pat, subst);
        let size_before = self.unionfind.size();
        let id2 = self.add_instantiation_noncanonical(to_pat, subst);
        let rhs_new = self.unionfind.size() > size_before;

        let did_union = self.perform_union(
            id1,
            id2,
            Some(Justification::Rule(rule_name.into())),
            rhs_new,
        );
        (self.canon_id(id1), did_union)
    }

    /// Unions two e-classes, using a given reason to justify it.
    ///
    /// This function picks representatives using [`id_to_expr`](EGraph::id_to_expr) so choosing
    /// `Id`s returned by functions like [`add_uncanonical`](EGraph::add_uncanonical) is important
    /// to control explanations
    pub fn union_trusted(&mut self, from: ID, to: ID, reason: impl Into<GlobalSymbol>) -> bool {
        self.perform_union(from, to, Some(Justification::Rule(reason.into())), false)
    }

    /// Unions two eclasses given their ids.
    ///
    /// The given ids need not be canonical.
    /// The returned `bool` indicates whether a union is necessary,
    /// so it's `false` if they were already equivalent.
    ///
    /// When explanations are enabled, this function behaves like [`EGraph::union_trusted`],
    ///  and it lists the call site as the proof reason.
    /// You should prefer [`union_instantiations`](EGraph::union_instantiations) when
    ///  you want the proofs to always be meaningful.
    /// Alternatively you can use [`EGraph::union_trusted`] using uncanonical `Id`s obtained from
    ///  functions like [`EGraph::add_uncanonical`]
    /// See [`explain_equivalence`](Runner::explain_equivalence) for a more detailed
    /// explanation of the feature.
    #[track_caller]
    pub fn union(&mut self, id1: ID, id2: ID) -> bool {
        if self.explain.is_some() {
            let caller = std::panic::Location::caller();
            self.union_trusted(id1, id2, caller.to_string().as_str())
        } else {
            self.perform_union(id1, id2, None, false)
        }
    }

    fn perform_union(
        &mut self,
        enode_id1: ID,
        enode_id2: ID,
        rule: Option<Justification>,
        any_new_rhs: bool,
    ) -> bool {
        A::pre_union(self, enode_id1, enode_id2, &rule);

        self.clean = false;
        let mut id1 = self.canon_id_mut(enode_id1);
        let mut id2 = self.canon_id_mut(enode_id2);
        if id1 == id2 {
            if let Some(Justification::Rule(_)) = rule {
                if let Some(explain) = &mut self.explain {
                    explain.alternate_rewrite(enode_id1, enode_id2, rule.unwrap());
                }
            }
            return false;
        }
        // make sure class2 has fewer parents
        let class1_parents = self.classes[&id1].parents.len();
        let class2_parents = self.classes[&id2].parents.len();
        if class1_parents < class2_parents {
            std::mem::swap(&mut id1, &mut id2);
        }

        if let Some(explain) = &mut self.explain {
            explain.union(enode_id1, enode_id2, rule.unwrap(), any_new_rhs);
        }

        // make id1 the new root
        self.unionfind.union(id1, id2);

        assert_ne!(id1, id2);
        let class2 = self.classes.remove(&id2).unwrap();
        let class1 = self.classes.get_mut(&id1).unwrap();
        assert_eq!(id1, class1.id);

        self.pending.extend(class2.parents.iter().copied());
        let did_merge = self.analysis.merge(&mut class1.data, class2.data);
        if did_merge.0 {
            self.analysis_pending.extend(class1.parents.iter().copied());
        }
        if did_merge.1 {
            self.analysis_pending.extend(class2.parents.iter().copied());
        }

        concat_vecs(&mut class1.nodes, class2.nodes);
        concat_vecs(&mut class1.parents, class2.parents);

        A::modify(self, id1);
        true
    }

    /// Update the analysis data of an e-class.
    ///
    /// This also propagates the changes through the e-graph,
    /// so [`Analysis::make`] and [`Analysis::merge`] will get
    /// called for other parts of the e-graph on rebuild.
    pub fn set_analysis_data(&mut self, id: ID, new_data: A::Data) {
        let id = self.canon_id_mut(id);
        let class = self.classes.get_mut(&id).unwrap();
        class.data = new_data;
        self.analysis_pending.extend(class.parents.iter().copied());
        A::modify(self, id)
    }

    /// Returns a more debug-able representation of the egraph.
    ///
    /// [`EGraph`]s implement [`Debug`], but it ain't pretty. It
    /// prints a lot of stuff you probably don't care about.
    /// This method returns a wrapper that implements [`Debug`] in a
    /// slightly nicer way, just dumping enodes in each eclass.
    ///
    /// [`Debug`]: std::fmt::Debug
    pub fn dump(&self) -> impl Debug + '_ {
        EGraphDump(self)
    }
}

impl<A: Analysis> EGraph<A> {
    /// Panic if the given eclass doesn't contain the given patterns
    ///
    /// Useful for testing.
    pub fn check_goals(&self, id: ID, goals: &[Pattern]) {
        let (cost, best) = Extractor::new(self, AstSize).find_best(id);
        println!("End ({}): {:?}", cost, best.pretty(80));

        for (i, goal) in goals.iter().enumerate() {
            println!("Trying to prove goal {}: {}", i, goal.pretty(40));
            let matches = goal.search_eclass(self, id);
            if matches.is_none() {
                let best = Extractor::new(self, AstSize).find_best(id).1;
                panic!(
                    "Could not prove goal {}:\n\
                     {}\n\
                     Best thing found:\n\
                     {}",
                    i,
                    goal.pretty(40),
                    best.pretty(40),
                );
            }
        }
    }
}

// All the rebuilding stuff
impl<A: Analysis> EGraph<A> {
    #[inline(never)]
    fn rebuild_classes(&mut self) -> usize {
        let mut classes_by_op = std::mem::take(&mut self.classes_by_op);
        classes_by_op.values_mut().for_each(|ids| ids.clear());

        let mut trimmed = 0;
        let uf = &mut self.unionfind;

        for class in self.classes.values_mut() {
            let old_len = class.len();
            class
                .nodes
                .iter_mut()
                .for_each(|n| n.update_operands(|id| uf.root_mut(id)));
            class.nodes.sort_unstable();
            class.nodes.dedup();

            trimmed += old_len - class.nodes.len();

            let mut add = |n: &Node| {
                classes_by_op
                    .entry(n.discriminant())
                    .or_default()
                    .insert(class.id)
            };

            // we can go through the ops in order to dedup them, becaue we
            // just sorted them
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

        #[cfg(debug_assertions)]
        for ids in classes_by_op.values_mut() {
            let unique: HashSet<ID> = ids.iter().copied().collect();
            assert_eq!(ids.len(), unique.len());
        }

        self.classes_by_op = classes_by_op;
        trimmed
    }

    #[inline(never)]
    fn check_memo(&self) -> bool {
        let mut test_memo = HashMap::new();

        for (&id, class) in self.classes.iter() {
            assert_eq!(class.id, id);
            for node in &class.nodes {
                if let Some(old) = test_memo.insert(node, id) {
                    assert_eq!(
                        self.canon_id(old),
                        self.canon_id(id),
                        "Found unexpected equivalence for {:?}\n{:?}\nvs\n{:?}",
                        node,
                        self[self.canon_id(id)].nodes,
                        self[self.canon_id(old)].nodes,
                    );
                }
            }
        }

        for (n, e) in test_memo {
            assert_eq!(e, self.canon_id(e));
            assert_eq!(
                Some(e),
                self.memo.get(n).map(|id| self.canon_id(*id)),
                "Entry for {:?} at {} in test_memo was incorrect",
                n,
                e
            );
        }

        true
    }

    #[inline(never)]
    fn process_unions(&mut self) -> usize {
        let mut n_unions = 0;

        while !self.pending.is_empty() || !self.analysis_pending.is_empty() {
            while let Some(class) = self.pending.pop() {
                let mut node = self.nodes[class.val()].clone();
                node.update_operands(|id| self.canon_id_mut(id));
                if let Some(memo_class) = self.memo.insert(node, class) {
                    let did_something = self.perform_union(
                        memo_class,
                        class,
                        Some(Justification::Congruence),
                        false,
                    );
                    n_unions += did_something as usize;
                }
            }

            while let Some(class_id) = self.analysis_pending.pop() {
                let node = self.nodes[class_id.val()].clone();
                let class_id = self.canon_id_mut(class_id);
                let node_data = A::make(self, &node);
                let class = self.classes.get_mut(&class_id).unwrap();

                let did_merge = self.analysis.merge(&mut class.data, node_data);
                if did_merge.0 {
                    self.analysis_pending.extend(class.parents.iter().copied());
                    A::modify(self, class_id)
                }
            }
        }

        assert!(self.pending.is_empty());
        assert!(self.analysis_pending.is_empty());

        n_unions
    }

    /// Restores the egraph invariants of congruence and enode uniqueness.
    ///
    /// As mentioned
    /// [in the tutorial](tutorials/_01_background/index.html#invariants-and-rebuilding),
    /// `egg` takes a lazy approach to maintaining the egraph invariants.
    /// The `rebuild` method allows the user to manually restore those
    /// invariants at a time of their choosing. It's a reasonably
    /// fast, linear-ish traversal through the egraph.
    ///
    /// After modifying an e-graph with [`add`](EGraph::add) or
    /// [`union`](EGraph::union), you must call `rebuild` to restore
    /// invariants before any query operations, otherwise the results
    /// may be stale or incorrect.
    ///
    /// This will set [`EGraph::clean`] to `true`.
    pub fn rebuild(&mut self) -> usize {
        let old_hc_size = self.memo.len();
        let old_n_eclasses = self.number_of_classes();

        let start = Instant::now();

        let n_unions = self.process_unions();
        let trimmed_nodes = self.rebuild_classes();

        let elapsed = start.elapsed();
        info!(
            concat!(
                "REBUILT! in {}.{:03}s\n",
                "  Old: hc size {}, eclasses: {}\n",
                "  New: hc size {}, eclasses: {}\n",
                "  unions: {}, trimmed nodes: {}"
            ),
            elapsed.as_secs(),
            elapsed.subsec_millis(),
            old_hc_size,
            old_n_eclasses,
            self.memo.len(),
            self.number_of_classes(),
            n_unions,
            trimmed_nodes,
        );

        debug_assert!(self.check_memo());
        self.clean = true;
        n_unions
    }

    pub(crate) fn check_each_explain(&mut self, rules: &[&Rewrite<A>]) -> bool {
        if let Some(explain) = &mut self.explain {
            explain.with_nodes(&self.nodes).check_each_explain(rules)
        } else {
            panic!("Can't check explain when explanations are off");
        }
    }
}

/// An equivalence class of enodes.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct EClass<D> {
    /// This eclass's id.
    pub id: ID,
    /// The equivalent enodes in this equivalence class.
    pub nodes: Vec<Node>,
    /// The analysis data associated with this eclass.
    ///
    /// Modifying this field will _not_ cause changes to propagate through the e-graph.
    /// Prefer [`EGraph::set_analysis_data`] instead.
    pub data: D,
    /// The original Ids of parent enodes.
    pub(crate) parents: Vec<ID>,
}

impl<D> EClass<D> {
    /// Returns `true` if the `eclass` is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the number of enodes in this eclass.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Iterates over the enodes in this eclass.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Node> {
        self.nodes.iter()
    }

    /// Iterates over the non-canonical ids of parent enodes of this eclass.
    pub fn parents(&self) -> impl ExactSizeIterator<Item = ID> + '_ {
        self.parents.iter().copied()
    }

    /// Iterates over the childless enodes in this eclass.
    pub fn leaves(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|&n| n.is_leaf())
    }

    /// Asserts that the childless enodes in this eclass are unique.
    pub fn assert_unique_leaves(&self) {
        let mut leaves = self.leaves();
        if let Some(first) = leaves.next() {
            assert!(
                leaves.all(|l| l == first),
                "Different leaves in eclass {}: {:?}",
                self.id,
                self.leaves().collect::<HashSet<_>>()
            );
        }
    }
}

/// just UnionFind to manage EClass Equivalencies
#[derive(Debug, Clone, Default)]
pub(crate) struct EClassUnion {
    pub(crate) parents: Vec<ID>,
}

impl EClassUnion {
    /// creates a new class with a root id, but no children
    pub fn init_class(&mut self) -> ID {
        let id = ID::new(self.parents.len());
        self.parents.push(id);
        id
    }

    pub fn size(&self) -> usize {
        self.parents.len()
    }

    fn parent(&self, query: ID) -> ID {
        self.parents[query.val()]
    }

    fn parent_mut(&mut self, query: ID) -> &mut ID {
        &mut self.parents[query.val()]
    }

    /// all equivalent EClasses have equal roots
    pub fn root(&self, mut current: ID) -> ID {
        while current != self.parent(current) {
            current = self.parent(current)
        }
        current
    }

    // same as [root] but will also compress the tree
    pub fn root_mut(&mut self, mut current: ID) -> ID {
        while current != self.parent(current) {
            let grandparent = self.parent(self.parent(current));
            *self.parent_mut(current) = grandparent;
            current = grandparent;
        }
        current
    }

    /// Given two leader ids, unions the two eclasses making root1 the leader.
    pub fn union(&mut self, root1: ID, root2: ID) -> ID {
        *self.parent_mut(root2) = root1;
        root1
    }
}

struct EGraphDump<'a, A: Analysis>(&'a EGraph<A>);

impl<'a, A: Analysis> Debug for EGraphDump<'a, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ids: Vec<ID> = self.0.classes().map(|c| c.id).collect();
        ids.sort();
        for id in ids {
            let mut nodes = self.0[id].nodes.clone();
            nodes.sort();
            writeln!(f, "{} ({:?}): {:?}", id, self.0[id].data, nodes)?
        }
        Ok(())
    }
}
