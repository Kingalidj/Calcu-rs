use std::{cell::RefCell, rc::Rc};

use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, Parser},
    parse_macro_input, Attribute, Error, Field, Fields, FieldsNamed, Item, ItemStruct, Meta,
};

#[allow(unused_imports)]
// needed because of doclink bug
use calcurs_internals::Inherited;

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

/// Appends the given field to the struct
///
/// # Errors
///
/// This function will return an error if the struct has no named fields
fn append_field(mut strct: ItemStruct, field: Field) -> Result<ItemStruct, TokenStream> {
    match strct.fields {
        Fields::Named(FieldsNamed { ref mut named, .. }) => {
            named.push(field);
            Ok(strct)
        }
        _ => Err(Error::new_spanned(
            strct.fields,
            "Only named fields are supported for adding the base field.",
        )
        .into_compile_error()),
    }
}

/// This macro gets as input a type and will insert a field called base with that type it will also implement the [Inherited] trait
#[proc_macro_attribute]
pub fn inherit(attr: proc::TokenStream, item: proc::TokenStream) -> proc::TokenStream {
    let item = parse_macro_input!(item as ItemStruct);
    let base_type = parse_macro_input!(attr as syn::Type);

    let struct_name = &item.ident.clone();
    let struct_generics = &item.generics.clone();

    let base = Field::parse_named
        .parse2(quote! {base: #base_type})
        .unwrap();

    let internals = import_crate("internals");

    let item = append_field(item, base);

    if let Err(err) = item {
        return err.into();
    }

    let item = item.unwrap();

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

/// Returns the index of the first [Attribute] with a given name if found
fn does_attrs_contain(attrs: &[Attribute], name: &str) -> Option<usize> {
    for (index, struct_attr) in attrs.iter().enumerate() {
        if let Meta::Path(ref p) = struct_attr.meta {
            if p.segments.is_empty() {
                continue;
            }

            let macro_name = &p.segments.last().unwrap().ident;

            if macro_name == name {
                return Some(index);
            }
        }
    }

    None
}

/// Return all Items with the given mark, also removes the mark from the item
///
/// We use a attribute macro as a way to mark items, so that we can further process them in the
/// proc_macros
fn get_items_by_mark<'a>(items: &[Rc<RefCell<Item>>], mark: &'a str) -> Vec<Rc<RefCell<Item>>> {
    let mut marked_items: Vec<Rc<RefCell<Item>>> = vec![];

    items
        .iter()
        .for_each(|item: &Rc<RefCell<Item>>| match *item.borrow_mut() {
            Item::Struct(ItemStruct { ref mut attrs, .. }) => {
                if let Some(indx) = does_attrs_contain(attrs, mark) {
                    attrs.remove(indx);
                    marked_items.push(item.clone());
                }
            }
            _ => (),
        });

    marked_items
}

fn impl_calcurs_types(
    types: &[Rc<RefCell<Item>>],
    base: &Item,
) -> Result<TokenStream, TokenStream> {
    let base_type = if let Item::Struct(ItemStruct { ident, .. }) = base {
        ident
    } else {
        return Err(Error::new_spanned(
            base.into_token_stream(),
            "Calcurs_base has to be a struct",
        )
        .into_compile_error());
    };

    let base_field = Field::parse_named
        .parse2(quote! {base: #base_type})
        .unwrap();

    let internals = import_crate("internals");

    let mut impls_stream = TokenStream::new();

    for item in types {
        if let Item::Struct(ref mut s) = *item.borrow_mut() {
            *s = append_field(s.clone(), base_field.clone())?;

            let generics = &s.generics;
            let name = &s.ident;

            let impl_code = quote! {
                impl #generics #internals::Inherited<#base_type> for #name #generics {
                    fn base(&self) -> &#base_type {
                        &self.base
                    }
                }
            };

            impls_stream.extend(impl_code);
        } else {
            return Err(Error::new_spanned(
                item.borrow().clone().into_token_stream(),
                "Only structs can be calcurs_types",
            )
            .into_compile_error());
        }
    }

    Ok(impls_stream)
}

#[derive(Debug, Default, Clone)]
struct MacroScope {
    content: Vec<Item>,
}

impl Parse for MacroScope {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut scope = MacroScope::default();

        let r#mod = syn::ItemMod::parse(input)?;

        if let Some((_, content)) = r#mod.content {
            scope.content = content;
        }

        Ok(scope)
    }
}

/// Modifies the tokens based on the attributes used, e.g: [macro@calcurs_type] \
/// strips the module and inserts the modified [TokenStream]
///
/// a attribute macro + a mod was used, because: \
/// - function-like macros messed with the lsp \
/// - inner attribute macros are unstable
#[proc_macro_attribute]
pub fn init_calcurs_macro_scope(
    _: proc::TokenStream,
    item: proc::TokenStream,
) -> proc::TokenStream {
    // let input: syn::ItemMod = parse_macro_input!(item as syn::ItemMod);

    let input = parse_macro_input!(item as MacroScope);

    let mut stream = TokenStream::new();

    if input.content.is_empty() {
        return stream.into();
    }

    let items = input.content;

    let items: Vec<_> = items
        .into_iter()
        .map(|item| Rc::new(RefCell::new(item)))
        .collect();

    let calcurs_base = get_items_by_mark(&items, "calcurs_base");
    let calcurs_types = get_items_by_mark(&items, "calcurs_type");

    if calcurs_base.len() > 1 {
        return Error::new(Span::call_site(), "Currently only 1 Base is supported")
            .into_compile_error()
            .into();
    }

    if calcurs_types.len() != 0 && calcurs_base.len() != 1 {
        return Error::new_spanned(
            calcurs_types[0].borrow().clone().into_token_stream(),
            "No calcurs_base defined for calcurs_type",
        )
        .into_compile_error()
        .into();
    }
    let calcurs_base = calcurs_base.get(0).unwrap().borrow().clone();

    match impl_calcurs_types(&calcurs_types, &calcurs_base) {
        Ok(s) => stream.extend(s),
        Err(err) => return err.into(),
    }

    items
        .iter()
        .for_each(|item| item.borrow().to_tokens(&mut stream));

    stream.into()
}

/// struct marked with this attribute will be turned into CalcursTypes
///
/// this includes the following actions: \
/// add a base field of type defined by [macro@calcurs_base] and implement [Inherited]
#[proc_macro_attribute]
pub fn calcurs_type(_: proc::TokenStream, _: proc::TokenStream) -> proc::TokenStream {
    return Error::new(
        Span::call_site().into(),
        "called attribute calcurs_type outside a calcurs_scope",
    )
    .into_compile_error()
    .into();
}

/// can only be defined once per calcurs_macro_scope. the [macro@calcurs_type] macro will use the marked
/// struct as the base
#[proc_macro_attribute]
pub fn calcurs_base(_: proc::TokenStream, _: proc::TokenStream) -> proc::TokenStream {
    return Error::new(
        Span::call_site().into(),
        "called attribute calcurs_base outside a calcurs_scope",
    )
    .into_compile_error()
    .into();
}
