use proc_macro2::TokenStream;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{Attribute, Error, Field, Fields, FieldsNamed, Ident, ItemStruct};

macro_rules! cast_item {
    ($item: ident as $type: path $([$($ref_mut: ident)+])?) => {
        if let $type($($($ref_mut)*)? x) = $item {
            x
        } else {
            return Err(Error::new_spanned(
                $item.into_token_stream(),
                format!("Expected {}, found something else", stringify!($type)),
            ));
        }
    };

}

pub(crate) use cast_item;

/// Returns the index of the first [Attribute] with a given name if found
pub fn find_attribute(attrs: &[Attribute], name: &str) -> Option<usize> {
    for (index, struct_attrib) in attrs.iter().enumerate() {
        let path = struct_attrib.path();

        if path.is_ident(name) {
            return Some(index);
        }
    }

    None
}

pub fn import_crate(name: &str) -> TokenStream {
    let found_crate =
        crate_name(name).unwrap_or_else(|_| panic!("{name} is not present in Cargo.toml"));

    match found_crate {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, proc_macro2::Span::call_site());
            quote!( #ident )
        }
    }
}

/// Appends the given field to the struct
///
/// # Errors
///
/// This function will return an error if the struct has no named fields
pub fn append_field(mut strct: ItemStruct, field: Field) -> syn::Result<ItemStruct> {
    match strct.fields {
        Fields::Named(FieldsNamed { ref mut named, .. }) => {
            named.push(field);
            Ok(strct)
        }
        _ => Err(Error::new_spanned(
            strct.fields,
            "Only named fields are supported for adding the base field.",
        )),
    }
}
