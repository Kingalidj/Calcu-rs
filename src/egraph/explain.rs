use crate::egraph::*;

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, VecDeque},
    fmt::{self, Debug, Display, Formatter},
    ops::{Deref, DerefMut},
    rc::Rc,
};

use num_bigint::BigUint;
use num_traits::identities::{One, Zero};

type ProofCost = BigUint;

const CONGRUENCE_LIMIT: usize = 2;
const GREEDY_NUM_ITERS: usize = 2;

/// A justification for a union, either via a rule or congruence.
/// A direct union with a justification is also stored as a rule.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Justification {
    /// Justification by a rule with this name.
    Rule(GlobalSymbol),
    /// Justification by congruence.
    Congruence,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct Connection {
    next: ID,
    current: ID,
    justification: Justification,
    is_rewrite_forward: bool,
}

#[derive(Debug, Clone)]
struct ExplainNode {
    // neighbors includes parent connections
    neighbors: Vec<Connection>,
    parent_connection: Connection,
    // it was inserted because of:
    // 1) it's parent is inserted (points to parent enode)
    // 2) a rewrite instantiated it (points to adjacent enode)
    // 3) it was inserted directly (points to itself)
    // if 1 is true but it's also adjacent (2) then either works and it picks 2
    existance_node: ID,
}

#[derive(Debug, Clone)]
pub struct Explain {
    explainfind: Vec<ExplainNode>,
    pub uncanon_memo: HashMap<Node, ID>,
    /// By default, egg uses a greedy algorithm to find shorter explanations when they are extracted.
    pub optimize_explanation_lengths: bool,
    // For a given pair of enodes in the same eclass,
    // stores the length of the shortest found explanation
    // and the Id of the neighbor for retrieving
    // the explanation.
    // Invariant: The distance is always <= the unoptimized distance
    // That is, less than or equal to the result of `distance_between`
    shortest_explanation_memo: HashMap<(ID, ID), (ProofCost, ID)>,
}

pub(crate) struct ExplainNodes<'a> {
    explain: &'a mut Explain,
    nodes: &'a [Node],
}

#[derive(Default)]
struct DistanceMemo {
    parent_distance: Vec<(ID, ProofCost)>,
    common_ancestor: HashMap<(ID, ID), ID>,
    tree_depth: HashMap<ID, ProofCost>,
}

/// Explanation trees are the compact representation showing
/// how one term can be rewritten to another.
///
/// Each [`TreeTerm`] has child [`TreeExplanation`]
/// justifying a transformation from the initial child to the final child term.
/// Children [`TreeTerm`] can be shared, thus re-using explanations.
/// This sharing can be checked via Rc pointer equality.
///
/// See [`TreeTerm`] for more details on how to
/// interpret each term.
pub type TreeExplanation = Vec<Rc<TreeTerm>>;

/// FlatExplanation are the simpler, expanded representation
/// showing one term being rewritten to another.
/// Each step contains a full `FlatTerm`. Each flat term
/// is connected to the previous by exactly one rewrite.
///
/// See [`FlatTerm`] for more details on how to find this rewrite.
pub type FlatExplanation = Vec<FlatTerm>;

/// A vector of equalities based on enode ids. Each entry represents
/// two enode ids that are equal and why.
pub type UnionEqualities = Vec<(ID, ID, GlobalSymbol)>;

// given two adjacent nodes and the direction of the proof
type ExplainCache = HashMap<(ID, ID), Rc<TreeTerm>>;
type NodeExplanationCache = HashMap<ID, Rc<TreeTerm>>;

/** A data structure representing an explanation that two terms are equivalent.

There are two representations of explanations, each of which can be
represented as s-expressions in strings.
See [`Explanation`] for more details.
**/
pub struct Explanation {
    /// The tree representation of the explanation.
    pub explanation_trees: TreeExplanation,
    flat_explanation: Option<FlatExplanation>,
}

impl Display for Explanation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = self.get_sexpr().to_string();
        f.write_str(&s)
    }
}

impl Explanation {
    /// Get each flattened term in the explanation as an s-expression string.
    ///
    /// The s-expression format mirrors the format of each [`FlatTerm`].
    /// Each expression after the first will be annotated in one location with a rewrite.
    /// When a term is being re-written it is wrapped with "(Rewrite=> rule-name expression)"
    /// or "(Rewrite<= rule-name expression)".
    /// "Rewrite=>" indicates that the previous term is rewritten to the current term
    /// and "Rewrite<=" indicates that the current term is rewritten to the previous term.
    /// The name of the rule or the reason provided to [`union_instantiations`](super::EGraph::union_instantiations).
    pub fn get_flat_string(&mut self) -> String {
        self.get_flat_strings().join("\n")
    }

    /// Get each the tree-style explanation as an s-expression string.
    ///
    /// The s-expression format mirrors the format of each [`TreeTerm`].
    /// When a child contains an explanation, the explanation is wrapped with
    /// "(Explanation ...)".
    /// When a term is being re-written it is wrapped with "(Rewrite=> rule-name expression)"
    /// or "(Rewrite<= rule-name expression)".
    /// "Rewrite=>" indicates that the previous term is rewritten to the current term
    /// and "Rewrite<=" indicates that the current term is rewritten to the previous term.
    /// The name of the rule or the reason provided to [`union_instantiations`](super::EGraph::union_instantiations).
    pub fn get_string(&self) -> String {
        self.to_string()
    }

    /// Get the tree-style explanation as an s-expression string
    /// with let binding to enable sharing of subproofs.
    ///
    /// The following explanation shows that `(+ x (+ x (+ x x))) = (* 4 x)`.
    /// Steps such as factoring are shared via the let bindings.
    pub fn get_string_with_let(&self) -> String {
        let mut s = "".to_string();
        pretty_print(&mut s, &self.get_sexpr_with_let(), 100, 0).unwrap();
        s
    }

    /// Get each term in the explanation as a string.
    /// See [`get_string`](Explanation::get_string) for the format of these strings.
    pub fn get_flat_strings(&mut self) -> Vec<String> {
        self.make_flat_explanation()
            .iter()
            .map(|e| e.to_string())
            .collect()
    }

    fn get_sexpr(&self) -> SExpr {
        let mut items = vec![SExpr::String("Explanation".to_string())];
        for e in self.explanation_trees.iter() {
            items.push(e.get_sexpr());
        }

        SExpr::List(items)
    }

    /// Get the size of this explanation tree in terms of the number of rewrites
    /// in the let-bound version of the tree.
    pub fn get_tree_size(&self) -> ProofCost {
        let mut seen = Default::default();
        let mut seen_adjacent = Default::default();
        let mut sum: ProofCost = BigUint::zero();
        for e in self.explanation_trees.iter() {
            sum += self.tree_size(&mut seen, &mut seen_adjacent, e);
        }
        sum
    }

    fn tree_size(
        &self,
        seen: &mut HashSet<*const TreeTerm>,
        seen_adjacent: &mut HashSet<(ID, ID)>,
        current: &Rc<TreeTerm>,
    ) -> ProofCost {
        if !seen.insert(&**current as *const TreeTerm) {
            return BigUint::zero();
        }
        let mut my_size: ProofCost = BigUint::zero();
        if current.forward_rule.is_some() {
            my_size += 1_u32;
        }
        if current.backward_rule.is_some() {
            my_size += 1_u32;
        }
        assert!(my_size.is_zero() || my_size.is_one());
        if my_size.is_one() {
            if !seen_adjacent.insert((current.current, current.last)) {
                return BigUint::zero();
            } else {
                seen_adjacent.insert((current.last, current.current));
            }
        }

        for child_proof in &current.child_proofs {
            for child in child_proof {
                my_size += self.tree_size(seen, seen_adjacent, child);
            }
        }
        my_size
    }

    fn get_sexpr_with_let(&self) -> SExpr {
        let mut shared: HashSet<*const TreeTerm> = Default::default();
        let mut to_let_bind = vec![];
        for term in &self.explanation_trees {
            self.find_to_let_bind(term.clone(), &mut shared, &mut to_let_bind);
        }

        let mut bindings: HashMap<*const TreeTerm, SExpr> = Default::default();
        let mut generated_bindings: Vec<(SExpr, SExpr)> = Default::default();
        for to_bind in to_let_bind {
            if bindings.get(&(&*to_bind as *const TreeTerm)).is_none() {
                let name = SExpr::String("v_".to_string() + &generated_bindings.len().to_string());
                let ast = to_bind.get_sexpr_with_bindings(&bindings);
                generated_bindings.push((name.clone(), ast));
                bindings.insert(&*to_bind as *const TreeTerm, name);
            }
        }

        let mut items = vec![SExpr::String("Explanation".to_string())];
        for e in self.explanation_trees.iter() {
            if let Some(existing) = bindings.get(&(&**e as *const TreeTerm)) {
                items.push(existing.clone());
            } else {
                items.push(e.get_sexpr_with_bindings(&bindings));
            }
        }

        let mut result = SExpr::List(items);

        for (name, expr) in generated_bindings.into_iter().rev() {
            let let_expr = SExpr::List(vec![name, expr]);
            result = SExpr::List(vec![SExpr::String("let".to_string()), let_expr, result]);
        }

        result
    }

    // for every subterm which is shared in
    // multiple places, add it to to_let_bind
    fn find_to_let_bind(
        &self,
        term: Rc<TreeTerm>,
        shared: &mut HashSet<*const TreeTerm>,
        to_let_bind: &mut Vec<Rc<TreeTerm>>,
    ) {
        if !term.child_proofs.is_empty() {
            if shared.insert(&*term as *const TreeTerm) {
                for proof in &term.child_proofs {
                    for child in proof {
                        self.find_to_let_bind(child.clone(), shared, to_let_bind);
                    }
                }
            } else {
                to_let_bind.push(term);
            }
        }
    }
}

impl Explanation {
    /// Construct a new explanation given its tree representation.
    pub fn new(explanation_trees: TreeExplanation) -> Explanation {
        Explanation {
            explanation_trees,
            flat_explanation: None,
        }
    }

    /// Construct the flat representation of the explanation and return it.
    pub fn make_flat_explanation(&mut self) -> &FlatExplanation {
        if self.flat_explanation.is_some() {
            return self.flat_explanation.as_ref().unwrap();
        } else {
            self.flat_explanation = Some(TreeTerm::flatten_proof(&self.explanation_trees));
            self.flat_explanation.as_ref().unwrap()
        }
    }

    /// Check the validity of the explanation with respect to the given rules.
    /// This only is able to check rule applications when the rules are implement `get_pattern_ast`.
    pub fn check_proof<'a, R, N: Analysis + 'a>(&mut self, rules: R)
    where
        R: IntoIterator<Item = &'a Rewrite<N>>,
    {
        let rules: Vec<&Rewrite<N>> = rules.into_iter().collect();
        let rule_table = Explain::make_rule_table(rules.as_slice());
        self.make_flat_explanation();
        let flat_explanation = self.flat_explanation.as_ref().unwrap();
        assert!(!flat_explanation[0].has_rewrite_forward());
        assert!(!flat_explanation[0].has_rewrite_backward());
        for i in 0..flat_explanation.len() - 1 {
            let current = &flat_explanation[i];
            let next = &flat_explanation[i + 1];

            let has_forward = next.has_rewrite_forward();
            let has_backward = next.has_rewrite_backward();
            assert!(has_forward ^ has_backward);

            if has_forward {
                assert!(self.check_rewrite_at(current, next, &rule_table, true));
            } else {
                assert!(self.check_rewrite_at(current, next, &rule_table, false));
            }
        }
    }

    fn check_rewrite_at<A: Analysis>(
        &self,
        current: &FlatTerm,
        next: &FlatTerm,
        table: &HashMap<GlobalSymbol, &Rewrite<A>>,
        is_forward: bool,
    ) -> bool {
        if is_forward && next.forward_rule.is_some() {
            let rule_name = next.forward_rule.as_ref().unwrap();
            if let Some(rule) = table.get(rule_name) {
                Explanation::check_rewrite(current, next, rule)
            } else {
                // give up when the rule is not provided
                true
            }
        } else if !is_forward && next.backward_rule.is_some() {
            let rule_name = next.backward_rule.as_ref().unwrap();
            if let Some(rule) = table.get(rule_name) {
                Explanation::check_rewrite(next, current, rule)
            } else {
                true
            }
        } else {
            for (left, right) in current.children.iter().zip(next.children.iter()) {
                if !self.check_rewrite_at(left, right, table, is_forward) {
                    return false;
                }
            }
            true
        }
    }

    // if the rewrite is just patterns, then it can check it
    fn check_rewrite<'a, A: Analysis>(
        current: &'a FlatTerm,
        next: &'a FlatTerm,
        rewrite: &Rewrite<A>,
    ) -> bool {
        if let Some(lhs) = rewrite.searcher.get_pattern_ast() {
            if let Some(rhs) = rewrite.applier.get_pattern_ast() {
                let rewritten = current.rewrite(lhs, rhs);
                if &rewritten != next {
                    return false;
                }
            }
        }
        true
    }
}

/// An explanation for a term and its equivalent children.
/// Each child is a proof transforming the initial child into the final child term.
/// The initial term is given by taking each first sub-term
/// in each [`child_proofs`](TreeTerm::child_proofs) recursively.
/// The final term is given by all of the final terms in each [`child_proofs`](TreeTerm::child_proofs).
///
/// If [`forward_rule`](TreeTerm::forward_rule) is provided, then this TreeTerm's initial term
/// can be derived from the previous
/// TreeTerm by applying the rule.
/// Similarly, if [`backward_rule`](TreeTerm::backward_rule) is provided,
/// then the previous TreeTerm's final term is given by applying the rule to this TreeTerm's initial term.
///
/// TreeTerms are flattened by first flattening [`child_proofs`](TreeTerm::child_proofs), then wrapping
/// the flattened proof with this TreeTerm's node.
#[derive(Debug, Clone)]
pub struct TreeTerm {
    /// A node representing this TreeTerm's operator. The children of the node should be ignored.
    pub node: Node,
    /// A rule rewriting this TreeTerm's initial term back to the last TreeTerm's final term.
    pub backward_rule: Option<GlobalSymbol>,
    /// A rule rewriting the last TreeTerm's final term to this TreeTerm's initial term.
    pub forward_rule: Option<GlobalSymbol>,
    /// A list of child proofs, each transforming the initial term to the final term for that child.
    pub child_proofs: Vec<TreeExplanation>,

    last: ID,
    current: ID,
}

impl TreeTerm {
    /// Construct a new TreeTerm given its node and child_proofs.
    pub fn new(node: Node, child_proofs: Vec<TreeExplanation>) -> TreeTerm {
        TreeTerm {
            node,
            backward_rule: None,
            forward_rule: None,
            child_proofs,
            current: ID::new(0),
            last: ID::new(0),
        }
    }

    fn flatten_proof(proof: &[Rc<TreeTerm>]) -> FlatExplanation {
        let mut flat_proof: FlatExplanation = vec![];
        for tree in proof {
            let mut explanation = tree.flatten_explanation();

            if !flat_proof.is_empty()
                && !explanation[0].has_rewrite_forward()
                && !explanation[0].has_rewrite_backward()
            {
                let last = flat_proof.pop().unwrap();
                explanation[0].combine_rewrites(&last);
            }

            flat_proof.extend(explanation);
        }

        flat_proof
    }

    /// Get a FlatTerm representing the first term in this proof.
    pub fn get_initial_flat_term(&self) -> FlatTerm {
        FlatTerm {
            node: self.node.clone(),
            backward_rule: self.backward_rule,
            forward_rule: self.forward_rule,
            children: self
                .child_proofs
                .iter()
                .map(|child_proof| child_proof[0].get_initial_flat_term())
                .collect(),
        }
    }

    /// Get a FlatTerm representing the final term in this proof.
    pub fn get_last_flat_term(&self) -> FlatTerm {
        FlatTerm {
            node: self.node.clone(),
            backward_rule: self.backward_rule,
            forward_rule: self.forward_rule,
            children: self
                .child_proofs
                .iter()
                .map(|child_proof| child_proof[child_proof.len() - 1].get_last_flat_term())
                .collect(),
        }
    }

    /// Construct the [`FlatExplanation`] for this TreeTerm.
    pub fn flatten_explanation(&self) -> FlatExplanation {
        let mut proof = vec![];
        let mut child_proofs = vec![];
        let mut representative_terms = vec![];
        for child_explanation in &self.child_proofs {
            let flat_proof = TreeTerm::flatten_proof(child_explanation);
            representative_terms.push(flat_proof[0].remove_rewrites());
            child_proofs.push(flat_proof);
        }

        proof.push(FlatTerm::new(
            self.node.clone(),
            representative_terms.clone(),
        ));

        for (i, child_proof) in child_proofs.iter().enumerate() {
            // replace first one to preserve the rule annotation
            proof.last_mut().unwrap().children[i] = child_proof[0].clone();

            for child in child_proof.iter().skip(1) {
                let mut children = vec![];
                for (j, rep_term) in representative_terms.iter().enumerate() {
                    if j == i {
                        children.push(child.clone());
                    } else {
                        children.push(rep_term.clone());
                    }
                }

                proof.push(FlatTerm::new(self.node.clone(), children));
            }
            representative_terms[i] = child_proof.last().unwrap().remove_rewrites();
        }

        proof[0].backward_rule = self.backward_rule;
        proof[0].forward_rule = self.forward_rule;

        proof
    }
}

/// A single term in an flattened explanation.
/// After the first term in a [`FlatExplanation`], each term
/// will be annotated with exactly one [`backward_rule`](FlatTerm::backward_rule) or one
/// [`forward_rule`](FlatTerm::forward_rule). This can appear in children [`FlatTerm`]s,
/// indicating that the child is being rewritten.
///
/// When [`forward_rule`](FlatTerm::forward_rule) is provided, the previous FlatTerm can be rewritten
/// to this FlatTerm by applying the rule.
/// When [`backward_rule`](FlatTerm::backward_rule) is provided, the previous FlatTerm is given by applying
/// the rule to this FlatTerm.
/// Rules are either the string of the name of the rule or the reason provided to
/// [`union_instantiations`](super::EGraph::union_instantiations).
///
#[derive(Debug, Clone, Eq)]
pub struct FlatTerm {
    /// The node representing this FlatTerm's operator.
    /// The children of the node should be ignored.
    pub node: Node,
    /// A rule rewriting this FlatTerm back to the last FlatTerm.
    pub backward_rule: Option<GlobalSymbol>,
    /// A rule rewriting the last FlatTerm to this FlatTerm.
    pub forward_rule: Option<GlobalSymbol>,
    /// The children of this FlatTerm.
    pub children: FlatExplanation,
}

impl Display for FlatTerm {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = self.get_sexpr().to_string();
        write!(f, "{}", s)
    }
}

impl PartialEq for FlatTerm {
    fn eq(&self, other: &FlatTerm) -> bool {
        if !self.node.matches(&other.node) {
            return false;
        }

        for (child1, child2) in self.children.iter().zip(other.children.iter()) {
            if !child1.eq(child2) {
                return false;
            }
        }
        true
    }
}

impl FlatTerm {
    /// Remove the rewrite annotation from this flatterm, if any.
    pub fn remove_rewrites(&self) -> FlatTerm {
        FlatTerm::new(
            self.node.clone(),
            self.children
                .iter()
                .map(|child| child.remove_rewrites())
                .collect(),
        )
    }

    fn combine_rewrites(&mut self, other: &FlatTerm) {
        if other.forward_rule.is_some() {
            assert!(self.forward_rule.is_none());
            self.forward_rule = other.forward_rule;
        }

        if other.backward_rule.is_some() {
            assert!(self.backward_rule.is_none());
            self.backward_rule = other.backward_rule;
        }

        for (left, right) in self.children.iter_mut().zip(other.children.iter()) {
            left.combine_rewrites(right);
        }
    }
}

impl Default for Explain {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatTerm {
    /// Convert this FlatTerm to an S-expression.
    /// See [`get_flat_string`](Explanation::get_flat_string) for the format of these expressions.
    pub fn get_string(&self) -> String {
        self.get_sexpr().to_string()
    }

    fn get_sexpr(&self) -> SExpr {
        let op = SExpr::String(self.node.to_string());
        let mut expr = if self.node.is_leaf() {
            op
        } else {
            let mut vec = vec![op];
            for child in &self.children {
                vec.push(child.get_sexpr());
            }
            SExpr::List(vec)
        };

        if let Some(rule_name) = &self.backward_rule {
            expr = SExpr::List(vec![
                SExpr::String("Rewrite<=".to_string()),
                SExpr::String((*rule_name).to_string()),
                expr,
            ]);
        }

        if let Some(rule_name) = &self.forward_rule {
            expr = SExpr::List(vec![
                SExpr::String("Rewrite=>".to_string()),
                SExpr::String((*rule_name).to_string()),
                expr,
            ]);
        }

        expr
    }
}

impl Display for TreeTerm {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut buf = String::new();
        let width = 80;
        pretty_print(&mut buf, &self.get_sexpr(), width, 1).unwrap();
        write!(f, "{}", buf)
    }
}

impl TreeTerm {
    /// Convert this TreeTerm to an S-expression.
    fn get_sexpr(&self) -> SExpr {
        self.get_sexpr_with_bindings(&Default::default())
    }

    fn get_sexpr_with_bindings(&self, bindings: &HashMap<*const TreeTerm, SExpr>) -> SExpr {
        let op = SExpr::String(self.node.to_string());
        let mut expr = if self.node.is_leaf() {
            op
        } else {
            let mut vec = vec![op];
            for child in &self.child_proofs {
                assert!(!child.is_empty());
                if child.len() == 1 {
                    if let Some(existing) = bindings.get(&(&*child[0] as *const TreeTerm)) {
                        vec.push(existing.clone());
                    } else {
                        vec.push(child[0].get_sexpr_with_bindings(bindings));
                    }
                } else {
                    let mut child_expressions = vec![SExpr::String("Explanation".to_string())];
                    for child_explanation in child.iter() {
                        if let Some(existing) =
                            bindings.get(&(&**child_explanation as *const TreeTerm))
                        {
                            child_expressions.push(existing.clone());
                        } else {
                            child_expressions
                                .push(child_explanation.get_sexpr_with_bindings(bindings));
                        }
                    }
                    vec.push(SExpr::List(child_expressions));
                }
            }
            SExpr::List(vec)
        };

        if let Some(rule_name) = &self.backward_rule {
            expr = SExpr::List(vec![
                SExpr::String("Rewrite<=".to_string()),
                SExpr::String((*rule_name).to_string()),
                expr,
            ]);
        }

        if let Some(rule_name) = &self.forward_rule {
            expr = SExpr::List(vec![
                SExpr::String("Rewrite=>".to_string()),
                SExpr::String((*rule_name).to_string()),
                expr,
            ]);
        }

        expr
    }
}

impl FlatTerm {
    /// Construct a new FlatTerm given a node and its children.
    pub fn new(node: Node, children: FlatExplanation) -> FlatTerm {
        FlatTerm {
            node,
            backward_rule: None,
            forward_rule: None,
            children,
        }
    }

    /// Rewrite the FlatTerm by matching the lhs and substituting the rhs.
    /// The lhs must be guaranteed to match.
    pub fn rewrite(&self, lhs: &PatternAst, rhs: &PatternAst) -> FlatTerm {
        let lhs_nodes = lhs.as_ref();
        let rhs_nodes = rhs.as_ref();
        let mut bindings = Default::default();
        self.make_bindings(lhs_nodes, lhs_nodes.len() - 1, &mut bindings);
        FlatTerm::from_pattern(rhs_nodes, rhs_nodes.len() - 1, &bindings)
    }

    /// Checks if this term or any child has a [`forward_rule`](FlatTerm::forward_rule).
    pub fn has_rewrite_forward(&self) -> bool {
        self.forward_rule.is_some()
            || self
                .children
                .iter()
                .any(|child| child.has_rewrite_forward())
    }

    /// Checks if this term or any child has a [`backward_rule`](FlatTerm::backward_rule).
    pub fn has_rewrite_backward(&self) -> bool {
        self.backward_rule.is_some()
            || self
                .children
                .iter()
                .any(|child| child.has_rewrite_backward())
    }

    fn from_pattern(
        pattern: &[ENodeOrVar],
        location: usize,
        bindings: &HashMap<GlobalSymbol, &FlatTerm>,
    ) -> FlatTerm {
        match &pattern[location] {
            ENodeOrVar::Var(var) => (*bindings.get(var).unwrap()).clone(),
            ENodeOrVar::ENode(node) => {
                let children = node.fold(vec![], |mut acc, child| {
                    acc.push(FlatTerm::from_pattern(pattern, child.val(), bindings));
                    acc
                });
                FlatTerm::new(node.clone(), children)
            }
        }
    }

    fn make_bindings<'a>(
        &'a self,
        pattern: &[ENodeOrVar],
        location: usize,
        bindings: &mut HashMap<GlobalSymbol, &'a FlatTerm>,
    ) {
        match &pattern[location] {
            ENodeOrVar::Var(var) => {
                if let Some(existing) = bindings.get(var) {
                    if existing != &self {
                        panic!(
                            "Invalid proof: binding for variable {:?} does not match between {:?} \n and \n {:?}",
                            var, existing, self);
                    }
                } else {
                    bindings.insert(*var, self);
                }
            }
            ENodeOrVar::ENode(node) => {
                // The node must match the rewrite or the proof is invalid.
                assert!(node.matches(&self.node));
                let mut counter = 0;
                node.for_each_oprnd(|child| {
                    self.children[counter].make_bindings(pattern, child.val(), bindings);
                    counter += 1;
                });
            }
        }
    }
}

// Make sure to use push_increase instead of push when using priority queue
#[derive(Clone, Eq, PartialEq)]
struct HeapState<I> {
    cost: ProofCost,
    item: I,
}
// The priority queue depends on `Ord`.
// Explicitly implement the trait so the queue becomes a min-heap
// instead of a max-heap.
impl<I: Eq + PartialEq> Ord for HeapState<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Notice that the we flip the ordering on costs.
        // In case of a tie we compare positions - this step is necessary
        // to make implementations of `PartialEq` and `Ord` consistent.
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| self.cost.cmp(&other.cost))
    }
}

// `PartialOrd` needs to be implemented as well.
impl<I: Eq + PartialEq> PartialOrd for HeapState<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Explain {
    fn make_rule_table<'a, A>(rules: &[&'a Rewrite<A>]) -> HashMap<GlobalSymbol, &'a Rewrite<A>> {
        let mut table: HashMap<GlobalSymbol, &'a Rewrite<A>> = Default::default();
        for r in rules {
            table.insert(r.name, r);
        }
        table
    }
    pub fn new() -> Self {
        Explain {
            explainfind: vec![],
            uncanon_memo: Default::default(),
            shortest_explanation_memo: Default::default(),
            optimize_explanation_lengths: true,
        }
    }

    pub(crate) fn set_existance_reason(&mut self, node: ID, existance_node: ID) {
        self.explainfind[node.val()].existance_node = existance_node;
    }

    pub(crate) fn add(&mut self, node: Node, set: ID, existance_node: ID) -> ID {
        assert_eq!(self.explainfind.len(), set.val());
        self.uncanon_memo.insert(node, set);
        self.explainfind.push(ExplainNode {
            neighbors: vec![],
            parent_connection: Connection {
                justification: Justification::Congruence,
                is_rewrite_forward: false,
                next: set,
                current: set,
            },
            existance_node,
        });
        set
    }

    // reverse edges recursively to make this node the leader
    fn make_leader(&mut self, node: ID) {
        let next = self.explainfind[node.val()].parent_connection.next;
        if next != node {
            self.make_leader(next);
            let node_connection = &self.explainfind[node.val()].parent_connection;
            let pconnection = Connection {
                justification: node_connection.justification.clone(),
                is_rewrite_forward: !node_connection.is_rewrite_forward,
                next: node,
                current: next,
            };
            self.explainfind[next.val()].parent_connection = pconnection;
        }
    }

    pub(crate) fn alternate_rewrite(&mut self, node1: ID, node2: ID, justification: Justification) {
        if node1 == node2 {
            return;
        }
        if let Some((cost, _)) = self.shortest_explanation_memo.get(&(node1, node2)) {
            if cost.is_zero() || cost.is_one() {
                return;
            }
        }

        let lconnection = Connection {
            justification: justification.clone(),
            is_rewrite_forward: true,
            next: node2,
            current: node1,
        };

        let rconnection = Connection {
            justification,
            is_rewrite_forward: false,
            next: node1,
            current: node2,
        };

        self.explainfind[node1.val()].neighbors.push(lconnection);
        self.explainfind[node2.val()].neighbors.push(rconnection);
        self.shortest_explanation_memo
            .insert((node1, node2), (BigUint::one(), node2));
        self.shortest_explanation_memo
            .insert((node2, node1), (BigUint::one(), node1));
    }

    pub(crate) fn union(
        &mut self,
        node1: ID,
        node2: ID,
        justification: Justification,
        new_rhs: bool,
    ) {
        if let Justification::Congruence = justification {
            // assert!(self.node(node1).matches(self.node(node2)));
        }
        if new_rhs {
            self.set_existance_reason(node2, node1)
        }

        self.make_leader(node1);
        self.explainfind[node1.val()].parent_connection.next = node2;

        if let Justification::Rule(_) = justification {
            self.shortest_explanation_memo
                .insert((node1, node2), (BigUint::one(), node2));
            self.shortest_explanation_memo
                .insert((node2, node1), (BigUint::one(), node1));
        }

        let pconnection = Connection {
            justification: justification.clone(),
            is_rewrite_forward: true,
            next: node2,
            current: node1,
        };
        let other_pconnection = Connection {
            justification,
            is_rewrite_forward: false,
            next: node1,
            current: node2,
        };
        self.explainfind[node1.val()]
            .neighbors
            .push(pconnection.clone());
        self.explainfind[node2.val()]
            .neighbors
            .push(other_pconnection);
        self.explainfind[node1.val()].parent_connection = pconnection;
    }
    pub(crate) fn get_union_equalities(&self) -> UnionEqualities {
        let mut equalities = vec![];
        for node in &self.explainfind {
            for neighbor in &node.neighbors {
                if neighbor.is_rewrite_forward {
                    if let Justification::Rule(r) = neighbor.justification {
                        equalities.push((neighbor.current, neighbor.next, r));
                    }
                }
            }
        }
        equalities
    }

    pub(crate) fn with_nodes<'a>(&'a mut self, nodes: &'a [Node]) -> ExplainNodes<'a> {
        ExplainNodes {
            explain: self,
            nodes,
        }
    }
}

impl<'a> Deref for ExplainNodes<'a> {
    type Target = Explain;

    fn deref(&self) -> &Self::Target {
        self.explain
    }
}

impl<'a> DerefMut for ExplainNodes<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.explain
    }
}

impl<'x> ExplainNodes<'x> {
    pub(crate) fn node(&self, node_id: ID) -> &Node {
        &self.nodes[node_id.val()]
    }
    fn node_to_explanation(&self, node_id: ID, cache: &mut NodeExplanationCache) -> Rc<TreeTerm> {
        if let Some(existing) = cache.get(&node_id) {
            existing.clone()
        } else {
            let node = self.node(node_id).clone();
            let children = node.fold(vec![], |mut sofar, child| {
                sofar.push(vec![self.node_to_explanation(child, cache)]);
                sofar
            });
            let res = Rc::new(TreeTerm::new(node, children));
            cache.insert(node_id, res.clone());
            res
        }
    }

    fn node_to_flat_explanation(&self, node_id: ID) -> FlatTerm {
        let node = self.node(node_id).clone();
        let children = node.fold(vec![], |mut sofar, child| {
            sofar.push(self.node_to_flat_explanation(child));
            sofar
        });
        FlatTerm::new(node, children)
    }

    pub fn check_each_explain<A: Analysis>(&self, rules: &[&Rewrite<A>]) -> bool {
        let rule_table = Explain::make_rule_table(rules);
        for i in 0..self.explainfind.len() {
            let explain_node = &self.explainfind[i];

            // check that explanation reasons never form a cycle
            let mut existance = i;
            let mut seen_existance: HashSet<usize> = Default::default();
            loop {
                seen_existance.insert(existance);
                let next = self.explainfind[existance].existance_node.val();
                if existance == next {
                    break;
                }
                existance = next;
                if seen_existance.contains(&existance) {
                    panic!("Cycle in existance!");
                }
            }

            if explain_node.parent_connection.next != ID::new(i) {
                let mut current_explanation = self.node_to_flat_explanation(ID::new(i));
                let mut next_explanation =
                    self.node_to_flat_explanation(explain_node.parent_connection.next);
                if let Justification::Rule(rule_name) =
                    &explain_node.parent_connection.justification
                {
                    if let Some(rule) = rule_table.get(rule_name) {
                        if !explain_node.parent_connection.is_rewrite_forward {
                            std::mem::swap(&mut current_explanation, &mut next_explanation);
                        }
                        if !Explanation::check_rewrite(
                            &current_explanation,
                            &next_explanation,
                            rule,
                        ) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub(crate) fn explain_equivalence<A: Analysis>(
        &mut self,
        left: ID,
        right: ID,
        unionfind: &mut EClassUnion,
        classes: &HashMap<ID, EClass<A::Data>>,
    ) -> Explanation {
        if self.optimize_explanation_lengths {
            self.calculate_shortest_explanations::<A>(left, right, classes, unionfind);
        }

        let mut cache = Default::default();
        let mut enode_cache = Default::default();
        Explanation::new(self.explain_enodes(left, right, &mut cache, &mut enode_cache, false))
    }

    pub(crate) fn explain_existance(&mut self, left: ID) -> Explanation {
        let mut cache = Default::default();
        let mut enode_cache = Default::default();
        Explanation::new(self.explain_enode_existance(
            left,
            self.node_to_explanation(left, &mut enode_cache),
            &mut cache,
            &mut enode_cache,
        ))
    }

    fn common_ancestor(&self, mut left: ID, mut right: ID) -> ID {
        let mut seen_left: HashSet<ID> = Default::default();
        let mut seen_right: HashSet<ID> = Default::default();
        loop {
            seen_left.insert(left);
            if seen_right.contains(&left) {
                return left;
            }

            seen_right.insert(right);
            if seen_left.contains(&right) {
                return right;
            }

            let next_left = self.explainfind[left.val()].parent_connection.next;
            let next_right = self.explainfind[right.val()].parent_connection.next;
            assert!(next_left != left || next_right != right);
            left = next_left;
            right = next_right;
        }
    }

    fn get_connections(&self, mut node: ID, ancestor: ID) -> Vec<Connection> {
        if node == ancestor {
            return vec![];
        }

        let mut nodes = vec![];
        loop {
            let next = self.explainfind[node.val()].parent_connection.next;
            nodes.push(self.explainfind[node.val()].parent_connection.clone());
            if next == ancestor {
                return nodes;
            }
            assert!(next != node);
            node = next;
        }
    }

    fn get_path_unoptimized(&self, left: ID, right: ID) -> (Vec<Connection>, Vec<Connection>) {
        let ancestor = self.common_ancestor(left, right);
        let left_connections = self.get_connections(left, ancestor);
        let right_connections = self.get_connections(right, ancestor);
        (left_connections, right_connections)
    }

    fn get_neighbor(&self, current: ID, next: ID) -> Connection {
        for neighbor in &self.explainfind[current.val()].neighbors {
            if neighbor.next == next {
                if let Justification::Rule(_) = neighbor.justification {
                    return neighbor.clone();
                }
            }
        }
        Connection {
            justification: Justification::Congruence,
            current,
            next,
            is_rewrite_forward: true,
        }
    }

    fn get_path(&self, mut left: ID, right: ID) -> (Vec<Connection>, Vec<Connection>) {
        let mut left_connections = vec![];
        loop {
            if left == right {
                return (left_connections, vec![]);
            }
            if let Some((_, next)) = self.shortest_explanation_memo.get(&(left, right)) {
                left_connections.push(self.get_neighbor(left, *next));
                left = *next;
            } else {
                break;
            }
        }

        let (restleft, right_connections) = self.get_path_unoptimized(left, right);
        left_connections.extend(restleft);
        (left_connections, right_connections)
    }

    fn explain_enode_existance(
        &self,
        node: ID,
        rest_of_proof: Rc<TreeTerm>,
        cache: &mut ExplainCache,
        enode_cache: &mut NodeExplanationCache,
    ) -> TreeExplanation {
        let graphnode = &self.explainfind[node.val()];
        let existance = graphnode.existance_node;
        let existance_node = &self.explainfind[existance.val()];
        // case 1)
        if existance == node {
            return vec![self.node_to_explanation(node, enode_cache), rest_of_proof];
        }

        // case 2)
        if graphnode.parent_connection.next == existance
            || existance_node.parent_connection.next == node
        {
            let mut connection = graphnode.parent_connection.clone();

            if graphnode.parent_connection.next == existance {
                connection.is_rewrite_forward = !connection.is_rewrite_forward;
                std::mem::swap(&mut connection.next, &mut connection.current);
            }
            return self.explain_enode_existance(
                existance,
                self.explain_adjacent(connection, cache, enode_cache, false),
                cache,
                enode_cache,
            );
        }

        // case 3)
        let mut new_rest_of_proof = (*self.node_to_explanation(existance, enode_cache)).clone();
        let mut index_of_child = 0;
        let mut found = false;
        self.node(existance).for_each_oprnd(|child| {
            if found {
                return;
            }
            if child == node {
                found = true;
            } else {
                index_of_child += 1;
            }
        });
        assert!(found);
        new_rest_of_proof.child_proofs[index_of_child].push(rest_of_proof);

        self.explain_enode_existance(existance, Rc::new(new_rest_of_proof), cache, enode_cache)
    }

    fn explain_enodes(
        &self,
        left: ID,
        right: ID,
        cache: &mut ExplainCache,
        node_explanation_cache: &mut NodeExplanationCache,
        use_unoptimized: bool,
    ) -> TreeExplanation {
        let mut proof = vec![self.node_to_explanation(left, node_explanation_cache)];
        let (left_connections, right_connections) = if use_unoptimized {
            self.get_path_unoptimized(left, right)
        } else {
            self.get_path(left, right)
        };

        for (i, connection) in left_connections
            .iter()
            .chain(right_connections.iter().rev())
            .enumerate()
        {
            let mut connection = connection.clone();
            if i >= left_connections.len() {
                connection.is_rewrite_forward = !connection.is_rewrite_forward;
                std::mem::swap(&mut connection.next, &mut connection.current);
            }

            proof.push(self.explain_adjacent(
                connection,
                cache,
                node_explanation_cache,
                use_unoptimized,
            ));
        }
        proof
    }

    fn explain_adjacent(
        &self,
        connection: Connection,
        cache: &mut ExplainCache,
        node_explanation_cache: &mut NodeExplanationCache,
        use_unoptimized: bool,
    ) -> Rc<TreeTerm> {
        let fingerprint = (connection.current, connection.next);

        if let Some(answer) = cache.get(&fingerprint) {
            return answer.clone();
        }

        let term = match connection.justification {
            Justification::Rule(name) => {
                let mut rewritten =
                    (*self.node_to_explanation(connection.next, node_explanation_cache)).clone();
                if connection.is_rewrite_forward {
                    rewritten.forward_rule = Some(name);
                } else {
                    rewritten.backward_rule = Some(name);
                }

                rewritten.current = connection.next;
                rewritten.last = connection.current;

                Rc::new(rewritten)
            }
            Justification::Congruence => {
                // add the children proofs to the last explanation
                let current_node = self.node(connection.current);
                let next_node = self.node(connection.next);
                assert!(current_node.matches(next_node));
                let mut subproofs = vec![];

                for (left_child, right_child) in current_node
                    .ids()
                    .iter()
                    .zip(next_node.ids().iter())
                {
                    subproofs.push(self.explain_enodes(
                        *left_child,
                        *right_child,
                        cache,
                        node_explanation_cache,
                        use_unoptimized,
                    ));
                }
                Rc::new(TreeTerm::new(current_node.clone(), subproofs))
            }
        };

        cache.insert(fingerprint, term.clone());

        term
    }

    fn find_all_enodes(&self, eclass: ID) -> HashSet<ID> {
        let mut enodes = HashSet::default();
        let mut todo = vec![eclass];

        while let Some(current) = todo.pop() {
            if enodes.insert(current) {
                for neighbor in &self.explainfind[current.val()].neighbors {
                    todo.push(neighbor.next);
                }
            }
        }
        enodes
    }

    fn add_tree_depths(&self, node: ID, depths: &mut HashMap<ID, ProofCost>) -> ProofCost {
        if depths.get(&node).is_none() {
            let parent = self.parent(node);
            let depth = if parent == node {
                BigUint::zero()
            } else {
                self.add_tree_depths(parent, depths) + 1_u32
            };

            depths.insert(node, depth);
        }

        depths.get(&node).unwrap().clone()
    }

    fn calculate_tree_depths(&self) -> HashMap<ID, ProofCost> {
        let mut depths = HashMap::default();
        for i in 0..self.explainfind.len() {
            self.add_tree_depths(ID::new(i), &mut depths);
        }
        depths
    }

    fn replace_distance(&mut self, current: ID, next: ID, right: ID, distance: ProofCost) {
        self.shortest_explanation_memo
            .insert((current, right), (distance, next));
    }

    fn populate_path_length(
        &mut self,
        right: ID,
        left_connections: &[Connection],
        distance_memo: &mut DistanceMemo,
    ) {
        self.shortest_explanation_memo
            .insert((right, right), (BigUint::zero(), right));
        for connection in left_connections.iter().rev() {
            let next = connection.next;
            let current = connection.current;
            let next_cost = self
                .shortest_explanation_memo
                .get(&(next, right))
                .unwrap()
                .0
                .clone();
            let dist = self.connection_distance(connection, distance_memo);
            self.replace_distance(current, next, right, next_cost + dist);
        }
    }

    fn distance_between(
        &mut self,
        left: ID,
        right: ID,
        distance_memo: &mut DistanceMemo,
    ) -> ProofCost {
        if left == right {
            return BigUint::zero();
        }
        let ancestor = if let Some(a) = distance_memo.common_ancestor.get(&(left, right)) {
            *a
        } else {
            // fall back on calculating ancestor for top-level query (not from congruence)
            self.common_ancestor(left, right)
        };
        // calculate edges until you are past the ancestor
        self.calculate_parent_distance(left, ancestor, distance_memo);
        self.calculate_parent_distance(right, ancestor, distance_memo);

        // now all three share an ancestor
        let a = self.calculate_parent_distance(ancestor, ID::MAX, distance_memo);
        let b = self.calculate_parent_distance(left, ID::MAX, distance_memo);
        let c = self.calculate_parent_distance(right, ID::MAX, distance_memo);

        assert!(
            distance_memo.parent_distance[ancestor.val()].0
                == distance_memo.parent_distance[left.val()].0
        );
        assert!(
            distance_memo.parent_distance[ancestor.val()].0
                == distance_memo.parent_distance[right.val()].0
        );

        // calculate distance to find upper bound
        b + c - (a << 1)

        //assert_eq!(dist+1, Explanation::new(self.explain_enodes(left, right, &mut Default::default())).make_flat_explanation().len());
    }

    fn congruence_distance(
        &mut self,
        current: ID,
        next: ID,
        distance_memo: &mut DistanceMemo,
    ) -> ProofCost {
        let current_node = self.node(current).clone();
        let next_node = self.node(next).clone();
        let mut cost: ProofCost = BigUint::zero();
        for (left_child, right_child) in current_node
            .ids()
            .iter()
            .zip(next_node.ids().iter())
        {
            cost += self.distance_between(*left_child, *right_child, distance_memo);
        }
        cost
    }

    fn connection_distance(
        &mut self,
        connection: &Connection,
        distance_memo: &mut DistanceMemo,
    ) -> ProofCost {
        match connection.justification {
            Justification::Congruence => {
                self.congruence_distance(connection.current, connection.next, distance_memo)
            }
            Justification::Rule(_) => BigUint::one(),
        }
    }

    fn calculate_parent_distance(
        &mut self,
        enode: ID,
        ancestor: ID,
        distance_memo: &mut DistanceMemo,
    ) -> ProofCost {
        loop {
            let parent = distance_memo.parent_distance[enode.val()].0;
            let dist = distance_memo.parent_distance[enode.val()].1.clone();
            if self.parent(parent) == parent {
                break;
            }

            let parent_parent = distance_memo.parent_distance[parent.val()].0;
            if parent_parent != parent {
                let new_dist = dist + distance_memo.parent_distance[parent.val()].1.clone();
                distance_memo.parent_distance[enode.val()] = (parent_parent, new_dist);
            } else {
                if ancestor == ID::MAX {
                    break;
                }
                if distance_memo.tree_depth.get(&parent).unwrap()
                    <= distance_memo.tree_depth.get(&ancestor).unwrap()
                {
                    break;
                }

                // find the length of one parent connection
                let connection = &self.explainfind[parent.val()].parent_connection;
                let current = connection.current;
                let next = connection.next;
                let cost = match connection.justification {
                    Justification::Congruence => {
                        self.congruence_distance(current, next, distance_memo)
                    }
                    Justification::Rule(_) => BigUint::one(),
                };
                distance_memo.parent_distance[parent.val()] = (self.parent(parent), cost);
            }
        }

        //assert_eq!(distance_memo.parent_distance[usize::from(enode)].1+1,
        //Explanation::new(self.explain_enodes(enode, distance_memo.parent_distance[usize::from(enode)].0, &mut Default::default())).make_flat_explanation().len());

        distance_memo.parent_distance[enode.val()].1.clone()
    }

    fn find_congruence_neighbors<A: Analysis>(
        &self,
        classes: &HashMap<ID, EClass<A::Data>>,
        congruence_neighbors: &mut [Vec<ID>],
        unionfind: &EClassUnion,
    ) {
        let mut counter = 0;
        // add the normal congruence edges first
        for node in &self.explainfind {
            if let Justification::Congruence = node.parent_connection.justification {
                let current = node.parent_connection.current;
                let next = node.parent_connection.next;
                congruence_neighbors[current.val()].push(next);
                congruence_neighbors[next.val()].push(current);
                counter += 1;
            }
        }

        'outer: for eclass in classes.keys() {
            let enodes = self.find_all_enodes(*eclass);
            // find all congruence nodes
            let mut cannon_enodes: HashMap<Node, Vec<ID>> = Default::default();
            for enode in &enodes {
                let cannon = self
                    .node(*enode)
                    .clone()
                    .map_operands(|child| unionfind.root(child));
                if let Some(others) = cannon_enodes.get_mut(&cannon) {
                    for other in others.iter() {
                        congruence_neighbors[enode.val()].push(*other);
                        congruence_neighbors[other.val()].push(*enode);
                    }
                    counter += 1;
                    others.push(*enode);
                } else {
                    counter += 1;
                    cannon_enodes.insert(cannon, vec![*enode]);
                }
                // Don't find every congruence edge because that could be n^2 edges
                if counter > CONGRUENCE_LIMIT * self.explainfind.len() {
                    break 'outer;
                }
            }
        }
    }

    pub fn get_num_congr<A: Analysis>(
        &self,
        classes: &HashMap<ID, EClass<A::Data>>,
        unionfind: &EClassUnion,
    ) -> usize {
        let mut congruence_neighbors = vec![vec![]; self.explainfind.len()];
        self.find_congruence_neighbors::<A>(classes, &mut congruence_neighbors, unionfind);
        let mut count = 0;
        for v in congruence_neighbors {
            count += v.len();
        }

        count / 2
    }

    pub fn get_num_nodes(&self) -> usize {
        self.explainfind.len()
    }

    fn shortest_path_modulo_congruence(
        &mut self,
        start: ID,
        end: ID,
        congruence_neighbors: &[Vec<ID>],
        distance_memo: &mut DistanceMemo,
    ) -> Option<(Vec<Connection>, Vec<Connection>)> {
        let mut todo = BinaryHeap::new();
        todo.push(HeapState {
            cost: BigUint::zero(),
            item: Connection {
                current: start,
                next: start,
                justification: Justification::Congruence,
                is_rewrite_forward: true,
            },
        });

        let mut last = HashMap::new();
        let mut path_cost = HashMap::new();

        'outer: loop {
            if todo.is_empty() {
                break 'outer;
            }
            let state = todo.pop().unwrap();
            let connection = state.item;
            let cost_so_far = state.cost.clone();
            let current = connection.next;

            if last.get(&current).is_some() {
                continue 'outer;
            } else {
                last.insert(current, connection);
                path_cost.insert(current, cost_so_far.clone());
            }

            if current == end {
                break;
            }

            for neighbor in &self.explainfind[current.val()].neighbors {
                if let Justification::Rule(_) = neighbor.justification {
                    let neighbor_cost = cost_so_far.clone() + 1_u32;
                    todo.push(HeapState {
                        item: neighbor.clone(),
                        cost: neighbor_cost,
                    });
                }
            }

            for other in congruence_neighbors[current.val()].iter() {
                let next = other;
                let distance = self.congruence_distance(current, *next, distance_memo);
                let next_cost = cost_so_far.clone() + distance;
                todo.push(HeapState {
                    item: Connection {
                        current,
                        next: *next,
                        justification: Justification::Congruence,
                        is_rewrite_forward: true,
                    },
                    cost: next_cost,
                });
            }
        }

        let total_cost = path_cost.get(&end);

        let left_connections;
        let mut right_connections = vec![];

        // we would like to assert that we found a path better than the normal one
        // but since proof sizes are saturated this is not true
        /*let dist = self.distance_between(start, end, distance_memo);
        if *total_cost.unwrap() > dist {
            panic!(
                "Found cost greater than baseline {} vs {}",
                total_cost.unwrap(),
                dist
            );
        }*/
        if *total_cost.unwrap() >= self.distance_between(start, end, distance_memo) {
            let (a_left_connections, a_right_connections) = self.get_path_unoptimized(start, end);
            left_connections = a_left_connections;
            right_connections = a_right_connections;
        } else {
            let mut current = end;
            let mut connections = vec![];
            while current != start {
                let prev = last.get(&current);
                if let Some(prev_connection) = prev {
                    connections.push(prev_connection.clone());
                    current = prev_connection.current;
                } else {
                    break;
                }
            }
            connections.reverse();
            self.populate_path_length(end, &connections, distance_memo);
            left_connections = connections;
        }

        Some((left_connections, right_connections))
    }

    fn greedy_short_explanations(
        &mut self,
        start: ID,
        end: ID,
        congruence_neighbors: &[Vec<ID>],
        distance_memo: &mut DistanceMemo,
        mut fuel: usize,
    ) {
        let mut todo_congruence = VecDeque::new();
        todo_congruence.push_back((start, end));

        while !todo_congruence.is_empty() {
            let (start, end) = todo_congruence.pop_front().unwrap();
            let eclass_size = self.find_all_enodes(start).len();
            if fuel < eclass_size {
                continue;
            }
            fuel = fuel.saturating_sub(eclass_size);

            let (left_connections, right_connections) = self
                .shortest_path_modulo_congruence(start, end, congruence_neighbors, distance_memo)
                .unwrap();

            //assert!(Explanation::new(self.explain_enodes(start, end, &mut Default::default())).make_flat_explanation().len()-1 <= total_cost);

            for (i, connection) in left_connections
                .iter()
                .chain(right_connections.iter().rev())
                .enumerate()
            {
                let mut next = connection.next;
                let mut current = connection.current;
                if i >= left_connections.len() {
                    std::mem::swap(&mut next, &mut current);
                }
                if let Justification::Congruence = connection.justification {
                    let current_node = self.node(current).clone();
                    let next_node = self.node(next).clone();
                    for (left_child, right_child) in current_node
                        .ids()
                        .iter()
                        .zip(next_node.ids().iter())
                    {
                        todo_congruence.push_back((*left_child, *right_child));
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tarjan_ocla(
        &self,
        enode: ID,
        children: &HashMap<ID, Vec<ID>>,
        common_ancestor_queries: &HashMap<ID, Vec<ID>>,
        black_set: &mut HashSet<ID>,
        unionfind: &mut EClassUnion,
        ancestor: &mut Vec<ID>,
        common_ancestor: &mut HashMap<(ID, ID), ID>,
    ) {
        ancestor[enode.val()] = enode;
        for child in children[&enode].iter() {
            self.tarjan_ocla(
                *child,
                children,
                common_ancestor_queries,
                black_set,
                unionfind,
                ancestor,
                common_ancestor,
            );
            unionfind.union(enode, *child);
            ancestor[unionfind.root(enode).val()] = enode;
        }

        if common_ancestor_queries.get(&enode).is_some() {
            black_set.insert(enode);
            for other in common_ancestor_queries.get(&enode).unwrap() {
                if black_set.contains(other) {
                    let ancestor = ancestor[unionfind.root(*other).val()];
                    common_ancestor.insert((enode, *other), ancestor);
                    common_ancestor.insert((*other, enode), ancestor);
                }
            }
        }
    }

    fn parent(&self, enode: ID) -> ID {
        self.explainfind[enode.val()].parent_connection.next
    }

    fn calculate_common_ancestor<D>(
        &self,
        classes: &HashMap<ID, EClass<D>>,
        congruence_neighbors: &[Vec<ID>],
    ) -> HashMap<(ID, ID), ID> {
        let mut common_ancestor_queries = HashMap::default();
        for (s_int, others) in congruence_neighbors.iter().enumerate() {
            let start = &ID::new(s_int);
            for other in others {
                for (left, right) in self
                    .node(*start)
                    .ids()
                    .iter()
                    .zip(self.node(*other).ids().iter())
                {
                    if left != right {
                        if common_ancestor_queries.get(start).is_none() {
                            common_ancestor_queries.insert(*start, vec![]);
                        }
                        if common_ancestor_queries.get(other).is_none() {
                            common_ancestor_queries.insert(*other, vec![]);
                        }
                        common_ancestor_queries.get_mut(start).unwrap().push(*other);
                        common_ancestor_queries.get_mut(other).unwrap().push(*start);
                    }
                }
            }
        }

        let mut common_ancestor = HashMap::default();
        let mut unionfind = EClassUnion::default();
        let mut ancestor = vec![];
        for i in 0..self.explainfind.len() {
            unionfind.init_class();
            ancestor.push(ID::new(i));
        }
        for (eclass, _) in classes.iter() {
            let enodes = self.find_all_enodes(*eclass);
            let mut children: HashMap<ID, Vec<ID>> = HashMap::default();
            for enode in &enodes {
                children.insert(*enode, vec![]);
            }
            for enode in &enodes {
                if self.parent(*enode) != *enode {
                    children.get_mut(&self.parent(*enode)).unwrap().push(*enode);
                }
            }

            let mut black_set = HashSet::default();

            let mut parent = *enodes.iter().next().unwrap();
            while parent != self.parent(parent) {
                parent = self.parent(parent);
            }
            self.tarjan_ocla(
                parent,
                &children,
                &common_ancestor_queries,
                &mut black_set,
                &mut unionfind,
                &mut ancestor,
                &mut common_ancestor,
            );
        }

        common_ancestor
    }

    fn calculate_shortest_explanations<A: Analysis>(
        &mut self,
        start: ID,
        end: ID,
        classes: &HashMap<ID, EClass<A::Data>>,
        unionfind: &EClassUnion,
    ) {
        let mut congruence_neighbors = vec![vec![]; self.explainfind.len()];
        self.find_congruence_neighbors::<A>(classes, &mut congruence_neighbors, unionfind);
        let mut parent_distance = vec![(ID::new(0), BigUint::zero()); self.explainfind.len()];
        for (i, entry) in parent_distance.iter_mut().enumerate() {
            entry.0 = ID::new(i);
        }

        let mut distance_memo = DistanceMemo {
            parent_distance,
            common_ancestor: self.calculate_common_ancestor(classes, &congruence_neighbors),
            tree_depth: self.calculate_tree_depths(),
        };

        let fuel = GREEDY_NUM_ITERS * self.explainfind.len();
        self.greedy_short_explanations(start, end, &congruence_neighbors, &mut distance_memo, fuel);
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum SExpr {
    /// plain String symbolic-expression
    String(String),
    /// list symbolic-expression
    List(Vec<SExpr>),
    /// empty, trivial symbolic-expression
    Empty,
}

impl Display for SExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            SExpr::String(ref s) => write!(f, "{s}"),
            SExpr::List(ref v) => {
                write!(f, "(")?;
                let l = v.len();
                for (i, x) in v.iter().enumerate() {
                    if i < l - 1 {
                        write!(f, "{} ", x)?;
                    } else {
                        write!(f, "{}", x)?;
                    }
                }
                write!(f, ")")
            }
            SExpr::Empty => Ok(()),
        }
    }
}

pub(crate) fn pretty_print(
    buf: &mut String,
    sexpr: &SExpr,
    width: usize,
    level: usize,
) -> fmt::Result {
    use std::fmt::Write;
    if let SExpr::List(list) = sexpr {
        let indent = sexpr.to_string().len() > width;
        write!(buf, "(")?;

        for (i, val) in list.iter().enumerate() {
            if indent && i > 0 {
                writeln!(buf)?;
                for _ in 0..level {
                    write!(buf, "  ")?;
                }
            }
            pretty_print(buf, val, width, level + 1)?;
            if !indent && i < list.len() - 1 {
                write!(buf, " ")?;
            }
        }

        write!(buf, ")")?;
        Ok(())
    } else {
        // I don't care about quotes
        write!(buf, "{}", sexpr.to_string().trim_matches('"'))
    }
}
