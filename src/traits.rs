use crate::{base::Base, numbers::Sign};

pub trait CalcursType: Clone {
    fn base(self) -> Base;
}

pub trait Numeric {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn sign(&self) -> Sign;

    fn is_pos(&self) -> bool {
        self.sign().is_pos()
    }

    fn is_neg(&self) -> bool {
        self.sign().is_neg()
    }
}
