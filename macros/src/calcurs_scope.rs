use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap},
    rc::Rc,
};

use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, Parser},
    Attribute, Error, ExprLit, Field, Ident, Item, ItemMod, ItemStruct, ItemTrait,
};

use crate::{
    parsers::{FieldValues, FuncMacroArg},
    utils::*,
};

/// Return all Items with the given mark, also removes the mark from the item
///
/// We use a attribute macro as a way to mark items, so that we can further process them in the
/// proc_macros
fn get_items_by_mark<'a>(
    items: &[Rc<RefCell<Item>>],
    mark: &'a str,
) -> Vec<(Attribute, Rc<RefCell<Item>>)> {
    let mut marked_items: Vec<(Attribute, Rc<RefCell<Item>>)> = vec![];

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
            marked_items.push((a, item.clone()));
        }
    }

    marked_items
}

#[derive(Clone, PartialEq, Hash, Eq, PartialOrd, Ord)]
struct SmallIdent {
    symbol: String,
}

impl std::fmt::Debug for SmallIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.symbol)
    }
}

impl From<Ident> for SmallIdent {
    fn from(value: Ident) -> Self {
        Self {
            symbol: value.to_string(),
        }
    }
}

// fn print_trait_props(props: &HashMap<Ident, HashMap<Ident, ExprLit>>) {
//     props.iter().for_each(|(trt, deps)| {
//         print!("{}:\n", trt);
//         deps.iter()
//             .for_each(|dep| println!("\t{} = {:?}", dep.0, dep.1.lit))
//     });
// }

#[derive(Debug, Clone, PartialEq)]
struct TraitProperties {
    r#trait: Ident,
    properties: HashMap<Ident, ExprLit>,
}

fn build_dep_tree(
    calcurs_props: Vec<(Attribute, Rc<RefCell<Item>>)>,
) -> syn::Result<HashMap<Ident, HashMap<Ident, ExprLit>>> {
    let calcurs_props: Result<Vec<_>, _> = calcurs_props
        .into_iter()
        .map(|(attr, item)| {
            let item = item.borrow().clone();
            let trt = cast_item!(item as Item::Trait);

            Ok((attr, trt))
        })
        .collect();
    let calcurs_props = calcurs_props?;

    let mut properties: HashMap<Ident, HashMap<Ident, ExprLit>> = HashMap::new();
    let mut relations: HashMap<Ident, BTreeSet<Ident>> = HashMap::new();

    // relations init
    for (attrib, trt) in calcurs_props {
        let vals: HashMap<Ident, ExprLit> = attrib
            .parse_args_with(FieldValues::parse_terminated)?
            .into_iter()
            .map(|x| (x.field, x.value))
            .collect();

        properties.insert(trt.ident.clone(), vals);

        use syn::TypeParamBound as TPB;

        let mut sups = BTreeSet::new();
        for sup in trt.supertraits {
            let path = match sup {
                TPB::Trait(syn::TraitBound { path, .. }) => path,
                _ => continue,
            };

            if let Some(ident) = path.get_ident() {
                sups.insert(ident.clone());
            }
        }

        relations.insert(trt.ident, sups);
    }

    let mut dep_trees: HashMap<Ident, BTreeSet<Ident>> = HashMap::new();

    // build tree (overwriting values is ill supported, would need to support saving depth)
    for (item, parents) in &relations {
        let mut deps: BTreeSet<Ident> = BTreeSet::from_iter(parents.clone().into_iter());
        let mut stack = parents.clone();

        while let Some(p) = stack.pop_last() {
            if let Some(parents) = relations.get(&p) {
                let new_deps: Vec<_> = parents
                    .iter()
                    .filter(|p| !deps.contains(&p))
                    .map(|x| x.clone())
                    .collect();

                deps.extend(new_deps.clone());
                stack.extend(new_deps);
            }
        }

        dep_trees.insert(item.clone().into(), deps);
    }

    let mut trait_props: HashMap<Ident, HashMap<Ident, ExprLit>> = HashMap::new();

    for (trt, deps) in dep_trees {
        let mut assigns = match properties.get(&trt) {
            Some(props) => props.clone(),
            None => continue,
        };

        for dep in deps {
            assigns.extend(match properties.get(&dep) {
                Some(props) => props.clone(),
                None => continue,
            });
        }

        trait_props.insert(trt, assigns);
    }

    // print_trait_props(&trait_props);

    Ok(trait_props)
}

fn impl_base_constructor(
    strct: (&Attribute, &ItemStruct),
    default: &FuncMacroArg,
    base_type: &Ident,
    calcurs_traits: &HashMap<Ident, HashMap<Ident, ExprLit>>,
) -> syn::Result<TokenStream> {
    let (attrib, strct) = strct;
    let derived = attrib.parse_args_with(Ident::parse)?;

    let generics = &strct.generics;
    let name = &strct.ident;

    let attribs = match calcurs_traits.get(&derived) {
        Some(attribs) => attribs,
        None => {
            return Err(Error::new(
                derived.span(),
                format!("Calcurs Trait: {} was never defined", derived.to_string()),
            ))
        }
    };

    let (fields, lit): (Vec<_>, Vec<_>) = attribs.iter().unzip();

    let cnst = match default.is_const {
        true => Some(quote!(const)),
        false => None,
    };

    let code = quote! {
        impl #generics #name #generics {
            pub #cnst fn new_base() -> #base_type {
                #base_type {
                    #(#fields: #lit,)*
                    .. #default()
                }
            }
        }
    };

    Ok(code)
}

fn impl_calcurs_types(
    base: (Attribute, ItemStruct),
    types: &[(Attribute, Rc<RefCell<Item>>)],
    traits: HashMap<Ident, HashMap<Ident, ExprLit>>,
) -> syn::Result<TokenStream> {
    let base_type = &base.1.ident;
    let base_attrib = base.0;

    let constructor = match base_attrib.meta {
        syn::Meta::List(syn::MetaList { ref tokens, .. }) => {
            FuncMacroArg::parse.parse2(tokens.clone())?
        }
        _ => {
            return Err(Error::new_spanned(
                base_attrib.into_token_stream(),
                format!("Provide a function that constructs a default base!"),
            ))
        }
    };

    let base_field = Field::parse_named
        .parse2(quote! {base: #base_type})
        .expect("impl_calcurs_types: could not implemnt base field");

    let internals = import_crate("internals");

    let mut stream = TokenStream::new();

    for item in types {
        let strct = &mut *item.1.borrow_mut();
        let strct = cast_item!(strct as Item::Struct[ref mut]);
        let attrib = &item.0;

        *strct = append_field(strct.clone(), base_field.clone())?;

        stream.extend(impl_base_constructor(
            (attrib, strct),
            &constructor,
            base_type,
            &traits,
        )?);

        let generics = &strct.generics;
        let name = &strct.ident;

        let impl_code = quote! {
            impl #generics #internals::Inherited<#base_type> for #name #generics {
                fn base(&self) -> &#base_type {
                    &self.base
                }
            }
        };

        stream.extend(impl_code);
    }

    Ok(stream)
}

fn parse_calcurs_scope(items: &[Rc<RefCell<Item>>]) -> syn::Result<TokenStream> {
    let mut stream = TokenStream::new();

    let calcurs_base = get_items_by_mark(&items, "calcurs_base");
    let calcurs_types = get_items_by_mark(&items, "calcurs_type");
    let calcurs_traits = get_items_by_mark(&items, "calcurs_trait");

    if calcurs_base.len() > 1 {
        return Err(Error::new(
            Span::call_site(),
            "Currently only 1 Base is supported",
        ));
    }

    if calcurs_types.len() != 0 && calcurs_base.len() != 1 {
        return Err(Error::new_spanned(
            calcurs_types[0].1.borrow().clone().into_token_stream(),
            "No calcurs_base defined for calcurs_type",
        ));
    }
    let base_attrib = calcurs_base.get(0).unwrap();
    let attrib = base_attrib.0.clone();
    let base = base_attrib.1.borrow().clone();
    let base = cast_item!(base as Item::Struct);

    // let calcurs_traits = build_calcurs_trait_tree(calcurs_traits)?;
    let calcurs_props = build_dep_tree(calcurs_traits)?;

    stream.extend(impl_calcurs_types(
        (attrib, base),
        &calcurs_types,
        calcurs_props,
    )?);

    Ok(stream)
}

#[derive(Debug, Clone, Default)]
pub struct CalcursMacroScope {
    stream: TokenStream,
}

impl ToTokens for CalcursMacroScope {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(self.stream.clone());
    }
}

impl Parse for CalcursMacroScope {
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

        let mut stream = parse_calcurs_scope(&items)?;

        items
            .iter()
            .for_each(|item| item.borrow().to_tokens(&mut stream));

        Ok(Self { stream })
    }
}
