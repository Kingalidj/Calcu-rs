use proc_macro2::TokenStream;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{Attribute, Error, Field, Fields, FieldsNamed, Ident, ItemStruct};

macro_rules! cast_item {
    ($item: ident as $type: path $([$($ref_mut: ident)+])?) => {
        cast_item!($item as $type $([$($ref_mut)+])?, format!("Expected {}, found something else", stringify!($type)))
    };

    ($item: ident as $type: path $([$($ref_mut: ident)+])?, $err_msg: expr) => {
        if let $type($($($ref_mut)*)? x) = $item {
            x
        } else {
            return Err(Error::new_spanned(
                $item.into_token_stream(),
                $err_msg
            ));
        }
    };
}

pub(crate) use cast_item;

/// Returns the index of the first [Attribute] that contains a given name if found
pub fn find_attribute(attrs: &[Attribute], name: &str) -> Option<(usize, String)> {
    for (index, struct_attrib) in attrs.iter().enumerate() {
        let path = struct_attrib.path();

        match path.get_ident() {
            Some(ident) => {
                let ident = ident.to_string();
                if ident.contains(name) {
                    return Some((index, ident));
                }
            }
            None => (),
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
pub fn append_struct_field(strct: &mut ItemStruct, field: Field) -> syn::Result<()> {
    match strct.fields {
        Fields::Named(FieldsNamed { ref mut named, .. }) => {
            named.push(field);
            Ok(())
            // Ok(strct)
        }
        _ => Err(Error::new_spanned(
            strct.fields.clone(),
            "Only named fields are supported for adding the base field.",
        )),
    }
}
