use std::marker::PhantomData;

use crate::{can_cast, dyn_cast, And, Basic, Boolean, BooleanAtom, CloneByVal, Not, Or, Symbol};
use paste::paste;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlFlow {
    Return,
    Continue,
}

use ControlFlow as CF;

impl ControlFlow {
    /// return? yes / no
    pub fn ret(b: bool) -> Self {
        match b {
            true => CF::Return,
            false => CF::Continue,
        }
    }

    /// continue? yes / no
    pub fn cont(b: bool) -> Self {
        Self::ret(!b)
    }
}

macro_rules! early_return {
    ($e: expr) => {{
        match $e {
            CF::Return => return CF::Return,
            CF::Continue => CF::Continue,
        }
    }};
}

macro_rules! walk {
        ($vis: ident, $var: ident: $typ: ident $bod: tt) => {
            paste! {
                fn [<walk_ $typ:snake>]<'ast>($vis: &mut impl Visitor<'ast>, $var: &'ast $typ) -> ControlFlow {
                    early_return!($vis.[<visit_ $typ:snake>]($var));
                    $bod
                }
            }
        };
    }

walk! { v, b: Basic {
    use Basic as B;
    match b {
        B::Boolean(b) => walk_boolean(v, b),
        B::Symbol(b) => walk_symbol(v, b),
    }
}}

walk! {v, s: Symbol {
    ControlFlow::Continue
}}

walk! {v, b: Boolean {
    use Boolean as B;
    match b {
        B::And(b) => walk_and(v, b),
        B::Or(b) => walk_or(v, b),
        B::Not(b) => walk_not(v, b),
        B::BooleanAtom(b) => walk_boolean_atom(v, b),
    }
}}

walk! {v, b: BooleanAtom {
    ControlFlow::Continue
}}

walk! {v, a: And {
    early_return!(walk_boolean(v, a.coeff.as_ref()));

    for b in a.args.as_ref() {
        early_return!(walk_or(v, b));
    }

    ControlFlow::Continue
}}

walk! {v, o: Or {
    early_return!(walk_boolean(v, o.coeff.as_ref()));
    early_return!(walk_basic(v, o.value.as_ref()));

    ControlFlow::Continue
}}

walk! {v, n: Not {
    early_return!(walk_basic(v, n.val.as_ref()));
    ControlFlow::Continue
}}

macro_rules! visit_all {
    (&mut $self: ident, $var: ident: $typ: ident $bod: tt) => {
        fn visit_basic(&mut $self, $var: &Basic) -> ControlFlow {
            type $typ = Basic;
            $bod
        }

        fn visit_boolean_atom(&mut $self, $var: &BooleanAtom) -> ControlFlow {
            type $typ = BooleanAtom;
            $bod
        }

        fn visit_boolean(&mut $self, $var: &Boolean) -> ControlFlow {
            type $typ = Boolean;
            $bod
        }

        fn visit_and(&mut $self, $var: &And) -> ControlFlow {
            type $typ = And;
            $bod
        }

        fn visit_or(&mut $self, $var: &Or) -> ControlFlow {
            type $typ = Or;
            $bod
        }

        fn visit_not(&mut $self, $var: &Not) -> ControlFlow {
            type $typ = Not;
            $bod
        }

        fn visit_symbol(&mut $self, $var: &Symbol) -> ControlFlow {
            type $typ = Symbol;
            $bod
        }
    };
}

pub trait Visitor<'ast>: Sized {
    fn visit_basic(&mut self, _b: &'ast Basic) -> ControlFlow {
        ControlFlow::Continue
    }
    fn visit_boolean_atom(&mut self, _b: &'ast BooleanAtom) -> ControlFlow {
        ControlFlow::Continue
    }

    fn visit_symbol(&mut self, _s: &'ast Symbol) -> ControlFlow {
        ControlFlow::Continue
    }

    fn visit_boolean(&mut self, _b: &'ast Boolean) -> ControlFlow {
        ControlFlow::Continue
    }

    fn visit_and(&mut self, _a: &'ast And) -> ControlFlow {
        ControlFlow::Continue
    }

    fn visit_or(&mut self, _o: &'ast Or) -> ControlFlow {
        ControlFlow::Continue
    }

    fn visit_not(&mut self, _n: &'ast Not) -> ControlFlow {
        ControlFlow::Continue
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IsVisitor<T> {
    val: bool,
    tag: PhantomData<T>,
}

impl<T: 'static> IsVisitor<T> {
    pub fn is(b: &Basic) -> bool {
        let mut vis = Self {
            val: false,
            tag: PhantomData,
        };

        walk_basic(&mut vis, b);
        vis.val
    }
}

impl<'ast, T: 'static> Visitor<'ast> for IsVisitor<T> {
    visit_all! { &mut self, _i: Item {
        self.val = can_cast::<Item, T>();
        ControlFlow::ret(self.val)
    }}
}

pub struct DownCastVisitor<T> {
    cast: Option<T>,
}

impl<T: 'static> DownCastVisitor<T> {
    pub fn downcast(b: &Basic) -> Option<T> {
        let mut vis = Self { cast: None };
        walk_basic(&mut vis, b);
        vis.cast
    }
}

impl<'a, T: 'static> Visitor<'a> for DownCastVisitor<T> {
    visit_all! { &mut self, i: Item {
        if can_cast::<T, Item>() {
            self.cast = dyn_cast(i.clone()).ok();
        }

        ControlFlow::Return
    }}
}
