use std::marker::PhantomData;

/// a set
pub trait Set: 'static {
    type Type;
}

/// An Operation on a set S is a function S<sup>n</sup> → S, where n ≥ 0 is called the **arity*
///
/// [further info](https://en.wikipedia.org/wiki/Operation_(mathematics))
pub trait Operation<S: Set>: Sized {}

/// unary [Operation]
pub trait Unary<S: Set, O: Operation<S>> {
    fn _0(&self) -> &S::Type;
}

/// binary [Operation]
pub trait Binary<S: Set, O: Operation<S>> {
    fn _0(&self) -> &S::Type;
    fn _1(&self) -> &S::Type;
}

/// ternary [Operation]
pub trait Ternary<S: Set, O: Operation<S>> {
    fn _0(&self) -> &S::Type;
    fn _1(&self) -> &S::Type;
    fn _2(&self) -> &S::Type;
}

/// A tuple of [Operation] used for [Algebra]
pub trait OperatorList<S: Set> {}

macro_rules! tuple_op_list_impls {
    ( $head:ident $( ,$tail:ident )* $(,)?) => {
        impl<S, $head, $( $tail ),*> OperatorList<S> for ($head, $( $tail ),*)
        where
            S: Set,
            $head: Operation<S>,
            $( $tail: Operation<S> ),*
        {}

        tuple_op_list_impls!($( $tail, )*);
    };

    () => {};
}

impl<S: Set> OperatorList<S> for () {}
tuple_op_list_impls!(A, B, C, D, E, F, G, H, I, J);

/// An Algebra is a pair < S; Ω >, where S is a set and Ω = (ω<sub>1</sub>, ..., ω<sub>n</sub>) is a list of [Operation]s
/// on S
pub trait Algebra<S: Set>: Sized {
    type OPS: OperatorList<S>;
}

/// A neutral element of < S; ⋆ > \
/// e ∈ S: e ⋆ a = a ⋆ e = a ∀ a ∈ S
pub trait Identity<S, OP, A>: Operation<S>
where
    S: Set,
    OP: Operation<S>,
    A: Algebra<S, OPS = (OP,)>,
{
    fn e() -> S::Type;
}

/// left [Inverse], gets auto implemented when Inverse is implemented
pub trait LInverse<S, OP, A>
where
    S: Set,
    OP: Operation<S> + Identity<S, OP, A>,
    A: Algebra<S, OPS = (OP,)>,
{
    fn l_inv(&self) -> S::Type;
}

/// left [Inverse], gets auto implemented when Inverse is implemented
pub trait RInverse<S, OP, A>
where
    S: Set,
    OP: Operation<S> + Identity<S, OP, A>,
    A: Algebra<S, OPS = (OP,)>,
{
    fn r_inv(&self) -> S::Type;
}

/// An inverse element a in an [Algebra] < S; ⋆, e > is
/// an element b ∈ S such that: b ⋆ a = a ⋆ b = e
pub trait Inverse<S, OP, A>
where
    S: Set,
    OP: Operation<S>,
    A: Algebra<S, OPS = (OP,)>,
{
    fn inv(&self) -> S::Type;
}

impl<
        S: Set,
        OP: Operation<S> + Identity<S, OP, A>,
        A: Algebra<S, OPS = (OP,)>,
        T: LInverse<S, OP, A> + RInverse<S, OP, A>,
    > Inverse<S, OP, A> for T
{
    fn inv(&self) -> <S as Set>::Type {
        self.r_inv()
    }
}

/// a ⋆ (b ⋆ c) = (a ⋆ b) ⋆ c
pub trait Associative<S: Set>: Operation<S> {}

/// a ⋆ b = b ⋆ a
pub trait Commutative<S: Set>: Operation<S> {}

/// a ⋆ (b + c) = (a ⋆ b) + (a ⋆ c)
pub trait Distributive<S: Set, OP: Operation<S>>: Operation<S> {}

/// A monoid is an [Algebra] < M; ⋆, e> where ⋆ is an associative and e is the neutral element
pub struct Monoid<S, OP>
where
    S: Set,
    OP: Associative<S> + Identity<S, OP, Self>,
{
    __: PhantomData<(S, OP)>,
}

impl<S, OP> Algebra<S> for Monoid<S, OP>
where
    S: Set,
    OP: Associative<S> + Identity<S, OP, Self>,
{
    type OPS = (OP,);
}

/// A group is an [Algebra] < G; ⋆, ^, e > satisfying the following axioms: \
/// **G1:** ⋆ is associative \
/// **G2:** e is a neutral element (e.g [Identity]) \
/// **G3:** ∀ a ∈ G has an [Inverse] element a ⋆ â = â ⋆ a = e
pub struct Group<S, OP>
where
    S: Set,
    OP: Associative<S> + Inverse<S, OP, Self>,
{
    __: PhantomData<(S, OP)>,
}

impl<S, OP> Algebra<S> for Group<S, OP>
where
    S: Set,
    OP: Associative<S> + Inverse<S, OP, Self>,
{
    type OPS = (OP,);
}
