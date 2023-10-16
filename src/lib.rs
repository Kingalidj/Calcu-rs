pub mod base;
pub mod binop;
pub mod boolean;
pub mod numeric;
pub mod traits;

pub mod prelude {
    pub use crate::base::Variable;
    pub use crate::binop::*;
    pub use crate::boolean::*;
    pub use crate::numeric::*;
    pub use crate::traits::*;
}
