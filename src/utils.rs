use log::warn;
use std::collections::VecDeque;
use std::hash::{BuildHasherDefault, Hasher};
use std::marker::PhantomData;
use std::{
    fmt::{self, Debug, Display, Formatter},
    iter::FromIterator,
};

/// This is provided by the [`symbol_table`](https://crates.io/crates/symbol_table) crate.
///
/// The internal symbol cache leaks the strings, which should be
/// fine if you only put in things like variable names and identifiers.
pub use symbol_table::GlobalSymbol as GlobSymbol;

pub(crate) use hashmap::*;
pub(crate) type BuildHasher = fxhash::FxBuildHasher;
pub(crate) use paste::paste;

#[cfg(feature = "deterministic")]
mod hashmap {
    pub(crate) type HashMap<K, V> = super::IndexMap<K, V>;
    pub(crate) type HashSet<K> = super::IndexSet<K>;
}
#[cfg(not(feature = "deterministic"))]
mod hashmap {
    use super::BuildHasher;
    pub(crate) type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasher>;
    pub(crate) type HashSet<K> = std::collections::HashSet<K, BuildHasher>;
}

/// use the value itself as hash value
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct U64Hasher(u64);
pub type BuildU64Hasher = BuildHasherDefault<U64Hasher>;

impl Hasher for U64Hasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        panic!("use write_u64 function");
    }

    fn write_u64(&mut self, u: u64) {
        debug_assert_eq!(self.0, 0, "only one write to hasher is allowed");
        self.0 = u;
    }

    fn write_usize(&mut self, u: usize) {
        debug_assert_eq!(self.0, 0, "only one write to hasher is allowed");
        self.0 = u as u64;
    }
}

pub(crate) fn hashmap_with_capacity<K, V>(cap: usize) -> HashMap<K, V> {
    HashMap::with_capacity_and_hasher(cap, <_>::default())
}

pub(crate) type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub(crate) type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

pub(crate) type Instant = quanta::Instant;
pub(crate) type Duration = std::time::Duration;

pub(crate) fn concat_vecs<T>(to: &mut Vec<T>, mut from: Vec<T>) {
    if to.len() < from.len() {
        std::mem::swap(to, &mut from)
    }
    to.extend(from);
}

/// A wrapper that uses display implementation as debug
pub(crate) struct DisplayAsDebug<T>(pub T);

impl<T: Display> Debug for DisplayAsDebug<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

/** A data structure to maintain a queue of unique elements.

Notably, insert/pop operations have O(1) expected amortized runtime complexity.
*/
#[derive(Clone)]
pub(crate) struct UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    set: HashSet<T>,
    queue: VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: HashSet::default(),
            queue: VecDeque::new(),
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

    type IntoIter = <VecDeque<T> as IntoIterator>::IntoIter;

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

macro_rules! non_max {
    ($non_max_ty: ident, $non_zero_ty: ty, $ty: ty) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(transparent)]
        pub struct $non_max_ty($non_zero_ty);

        impl $non_max_ty {
            pub const ZERO: $non_max_ty = Self::new(0);
            pub const ONE: $non_max_ty = Self::new(1);
            pub const TWO: $non_max_ty = Self::new(2);
            pub const MAX: $non_max_ty = Self::new(<$ty>::MAX - 1);

            #[inline(always)]
            pub const fn new(val: $ty) -> Self {
                let non_zero = val ^ <$ty>::MAX;
                assert!(non_zero != 0, "NonZero is Zero");
                Self(unsafe { <$non_zero_ty>::new_unchecked(non_zero) })
            }

            #[inline(always)]
            pub const fn try_new(val: $ty) -> Option<Self> {
                match <$non_zero_ty>::new(val ^ <$ty>::MAX) {
                    None => None,
                    Some(val) => Some(Self(val)),
                }
            }

            #[inline(always)]
            pub const fn get(&self) -> $ty {
                self.0.get() ^ <$ty>::MAX
            }
        }

        impl Default for $non_max_ty {
            #[inline(always)]
            fn default() -> Self {
                Self::new(0)
            }
        }

        impl std::fmt::Display for $non_max_ty {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.get())
            }
        }

        impl std::fmt::Debug for $non_max_ty {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.get())
            }
        }

        impl From<$non_max_ty> for $ty {
            #[inline(always)]
            fn from(value: $non_max_ty) -> Self {
                value.get()
            }
        }
    };

    ($ty: ty) => {
        paste! {
            non_max!{  [<NonMax $ty:camel>], std::num::[<NonZero $ty:camel>], $ty }
        }
    };
}

non_max!(u8);
non_max!(u16);
non_max!(u32);
non_max!(u64);
non_max!(u128);
non_max!(usize);

#[cfg(test)]
mod tests {
    use calcu_rs::egraph::*;

    macro_rules! non_max_test {
        ($ty:ty) => {
            paste! {
                #[test]
                fn [< non_max_ $ty:snake _test >]() {
                    use [< NonMax $ty:camel >] as NonMax;
                    assert_eq!(NonMax::ZERO.get(), 0);
                    assert_eq!(NonMax::ONE.get(), 1);
                    assert_eq!(NonMax::MAX.get(), <$ty>::MAX - 1);
                    assert_eq!(NonMax::new(42).get(), 42);
                }

                #[test]
                #[should_panic]
                fn [< non_max_ $ty:snake _panic >]() {
                    use [< NonMax $ty:camel >] as NonMax;
                    NonMax::new(<$ty>::MAX);
                }
            }
        };
    }

    non_max_test!(u8);
    non_max_test!(u16);
    non_max_test!(u32);
    non_max_test!(u64);
    non_max_test!(u128);
    non_max_test!(usize);

    fn ids(us: impl IntoIterator<Item = usize>) -> Vec<ID> {
        us.into_iter().map(|u| ID::new(u)).collect()
    }

    #[test]
    fn union_find() {
        let n = 10;
        let id = ID::new;

        let mut uf = EClassUnion::default();
        for _ in 0..n {
            uf.init_class();
        }

        // test the initial condition of everyone in their own set
        assert_eq!(uf.parents, ids(0..n));

        // build up one set
        uf.union(id(0), id(1));
        uf.union(id(0), id(2));
        uf.union(id(0), id(3));

        // build up another set
        uf.union(id(6), id(7));
        uf.union(id(6), id(8));
        uf.union(id(6), id(9));

        // this should compress all paths
        for i in 0..n {
            uf.root_mut(id(i));
        }

        // indexes:         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6];
        assert_eq!(uf.parents, ids(expected));
    }
}
