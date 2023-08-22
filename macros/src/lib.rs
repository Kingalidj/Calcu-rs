use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{parse::Parser, parse_macro_input, Field, FieldsNamed, Item, ItemStruct, Meta};

use proc_macro_crate::{crate_name, FoundCrate};

extern crate proc_macro as proc;

fn import_crate(name: &str) -> TokenStream {
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
pub fn init_calcurs_macro_scope(
    _: proc::TokenStream,
    item: proc::TokenStream,
) -> proc::TokenStream {
    let input: syn::ItemMod = parse_macro_input!(item as syn::ItemMod);

    let mut stream = TokenStream::new();

    if input.content.is_none() {
        return stream.into();
    }

    let items = input.content.unwrap().1;

    let mut calcurs_types: Vec<Ident> = vec![];

    items.into_iter().for_each(|mut item: Item| {
        let mut calcurs_type = None;

        match item {
            Item::Struct(ItemStruct {
                ref ident,
                ref mut attrs,
                ..
            }) => {
                for (index, attr) in attrs.iter().enumerate() {
                    if let Meta::Path(ref p) = attr.meta {
                        if p.segments.is_empty() {
                            continue;
                        }

                        let macro_name = &p.segments.last().unwrap().ident;

                        if macro_name == "calcurs_type" {
                            calcurs_type = Some(index);
                            calcurs_types.push(ident.clone());
                            break;
                        }
                    }
                }

                if let Some(indx) = calcurs_type {
                    attrs.remove(indx);

                    let inherit_attr = quote!(#[inherit(Base)]);

                    let attr = &syn::Attribute::parse_outer.parse2(inherit_attr).unwrap()[0];
                    attrs.insert(0, attr.clone());
                }
            }
            _ => (),
        }

        item.to_tokens(&mut stream);
    });

    calcurs_types
        .iter()
        .for_each(|typ| println!("{:?}", typ.to_string()));

    stream.into()
}

#[proc_macro_attribute]
pub fn calcurs_type(_: proc::TokenStream, _: proc::TokenStream) -> proc::TokenStream {
    return syn::Error::new(
        Span::call_site().into(),
        "called attribute calcurs_type outside a calcurs_scope",
    )
    .into_compile_error()
    .into();
}
