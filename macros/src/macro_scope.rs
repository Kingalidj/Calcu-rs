use std::{cell::RefCell, rc::Rc};

use syn::{parse::Parse, Attribute, Item, ItemMod, ItemStruct, ItemTrait};

use crate::utils::*;

#[derive(Debug, Clone, PartialEq)]
pub struct MarkedItem<T> {
    pub mark: Attribute,
    pub item: T,
}

pub type SharedMarkedItem<T> = MarkedItem<Rc<RefCell<T>>>;

impl<T> MarkedItem<T> {
    pub fn new(mark: Attribute, item: T) -> Self {
        Self { mark, item }
    }
}

/// Return all Items with the given mark, also removes the mark from the item
///
/// We use a attribute macro as a way to mark items, so that we can further process them in the
/// proc_macros
pub fn get_items_by_mark<'a>(
    items: &[Rc<RefCell<Item>>],
    mark: &'a str,
) -> Vec<MarkedItem<Rc<RefCell<Item>>>> {
    let mut marked_items: Vec<MarkedItem<Rc<RefCell<Item>>>> = vec![];

    use Item as I;

    for item in items {
        let mut i = item.borrow_mut();
        let attrs = match *i {
            I::Struct(ItemStruct { ref mut attrs, .. }) => attrs,
            I::Trait(ItemTrait { ref mut attrs, .. }) => attrs,
            I::Mod(syn::ItemMod { ref mut attrs, .. }) => attrs,
            I::Enum(syn::ItemEnum { ref mut attrs, .. }) => attrs,
            I::Fn(syn::ItemFn { ref mut attrs, .. }) => attrs,
            _ => continue,
        };

        if let Some(indx) = find_attribute(attrs, mark) {
            let a = attrs.remove(indx);
            marked_items.push(MarkedItem::new(a, item.clone()));
        }
    }

    marked_items
}

#[derive(Debug, Clone, Default)]
pub struct MacroScope {
    pub items: Vec<Rc<RefCell<Item>>>,
}

impl Parse for MacroScope {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item: ItemMod = input.parse()?;

        let items = match item.content {
            Some(c) => c.1,
            None => return Ok(Default::default()),
        };

        let items: Vec<_> = items
            .into_iter()
            .map(|item| Rc::new(RefCell::new(item)))
            .collect();

        Ok(Self { items })
    }
}
