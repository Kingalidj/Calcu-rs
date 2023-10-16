pub mod base;
pub mod boolean;
pub mod numeric;
pub mod operator;
pub mod traits;

pub mod prelude {
    pub use crate::base::Variable;
    pub use crate::boolean::*;
    pub use crate::numeric::*;
    pub use crate::operator::*;
    pub use crate::traits::*;
}
