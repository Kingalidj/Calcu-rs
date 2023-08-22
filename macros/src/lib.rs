use proc_macro2::{Ident, TokenStream};
use quote::{quote, ToTokens};
use syn::{parse::Parser, parse_macro_input, Field, FieldsNamed};

use proc_macro_crate::{crate_name, FoundCrate};

extern crate proc_macro as proc;

fn import_crate(name: &str) -> TokenStream {
    let found_crate = crate_name(name).expect(&format!("{name} is not present in Cargo.toml"));

    match found_crate {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, proc_macro2::Span::call_site());
            quote!( #ident )
        }
    }
}

/// this macro gets as input a type and will insert a field called base with that type it will also implement the [calcurs_internals::Inherited] trait
#[proc_macro_attribute]
pub fn inherit(attr: proc::TokenStream, item: proc::TokenStream) -> proc::TokenStream {
    let mut item = parse_macro_input!(item as syn::ItemStruct);
    let base_type = parse_macro_input!(attr as syn::Type);

    let struct_name = &item.ident;
    let struct_generics = &item.generics;

    let base = Field::parse_named
        .parse2(quote! {base: #base_type})
        .unwrap();

    let internals = import_crate("internals");

    match item.fields {
        syn::Fields::Named(FieldsNamed { ref mut named, .. }) => named.push(base),
        _ => {
            return syn::Error::new_spanned(
                item.fields,
                "Only named fields are supported for adding the base field.",
            )
            .into_compile_error()
            .into()
        }
    };

    quote! {
        #item

        impl #struct_generics #internals::Inherited<#base_type> for #struct_name #struct_generics {
            fn base(&self)  -> &#base_type {
                &self.base
            }
        }

    }
    .into()
}

#[proc_macro_attribute]
pub fn init_calcrs_macro_scope(_: proc::TokenStream, item: proc::TokenStream) -> proc::TokenStream {
    let input: syn::ItemMod = parse_macro_input!(item as syn::ItemMod);

    let mut stream = TokenStream::new();

    input.content.map(|x| {
        x.1.into_iter()
            .for_each(|item: syn::Item| item.to_tokens(&mut stream))
    });

    stream.into()
}
