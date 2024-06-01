mod egraph;
mod explain;
mod language;
mod machine;
mod multipattern;
mod pattern;
mod rewrite;
mod run;

pub(crate) use {
    crate::*,
    egraph::EClassUnion,
    explain::Explain
};

pub use {
    calcu_rs::util::*,
    egraph::EClass,
    egraph::EGraph,
    explain::{
        Explanation, FlatExplanation, FlatTerm, Justification, TreeExplanation, TreeTerm,
        UnionEqualities,
    },
    language::*,
    multipattern::*,
    pattern::{ENodeOrVar, Pattern, PatternAst, SearchMatches},
    rewrite::{
        Applier, Condition, ConditionEqual, ConditionalApplier, Rewrite, Searcher, Subst, Var,
    },
    run::*,
};

#[cfg(test)]
fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}
