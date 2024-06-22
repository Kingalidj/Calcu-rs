/*!

This code is a modified fork of the work by [Willsey et al.] <a href="#ref1">[1]</a>.

1. <span id="ref1"></span> [Willsey et al., 2021] Willsey, Max and Nandi, Chandrakana and Wang, Yisu Remy and Flatt, Oliver and Tatlock, Zachary and Panchekha, Pavel. "egg: Fast and Extensible Equality Saturation." Proc. ACM Program. Lang. 5, POPL (January 2021), Article 23. DOI: [10.1145/3434304](https://doi.org/10.1145/3434304)

*/

mod construct;
mod dot;
mod egraph;
mod explain;
mod machine;
mod multipattern;
mod pattern;
mod rewrite;
mod run;

pub(crate) use {crate::*, egraph::EClassUnion, explain::Explain};

pub use {
    calcu_rs::utils::*,
    construct::*,
    dot::Dot,
    egraph::{EClass, EGraph},
    explain::{
        Explanation, FlatExplanation, FlatTerm, Justification, TreeExplanation, TreeTerm,
        UnionEqualities,
    },
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
