/// returns the "inherited" value
pub trait Inherited<T> {
    fn base(&self) -> &T;
}
