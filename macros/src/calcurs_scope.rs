use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap},
    rc::Rc,
};

use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, Parser},
    Attribute, Error, ExprLit, Field, Ident, Item, ItemStruct, ItemTrait,
};

use crate::{
    macro_scope::{get_items_by_mark, MacroScope, MarkedItem, SharedMarkedItem},
    parsers::{FieldValues, FuncMacroArg},
    utils::*,
};

struct TraitProperties(HashMap<Ident, ExprLit>);

fn init_property_relations(
    calcurs_props: Vec<(Attribute, ItemTrait)>,
) -> syn::Result<(
    HashMap<Ident, BTreeSet<Ident>>,
    HashMap<Ident, HashMap<Ident, ExprLit>>,
)> {
    let mut properties: HashMap<Ident, HashMap<Ident, ExprLit>> = HashMap::new();
    let mut relations: HashMap<Ident, BTreeSet<Ident>> = HashMap::new();

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

    Ok((relations, properties))
}

fn dependencies_from_relation(
    relations: &HashMap<Ident, BTreeSet<Ident>>,
) -> syn::Result<HashMap<Ident, BTreeSet<Ident>>> {
    let mut dep_trees: HashMap<Ident, BTreeSet<Ident>> = HashMap::new();

    // build tree (overwriting values is ill supported, would need to support saving depth)
    for (item, parents) in relations {
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

    Ok(dep_trees)
}

fn build_dependency_tree(
    calcurs_props: Vec<SharedMarkedItem<Item>>,
) -> syn::Result<HashMap<Ident, TraitProperties>> {
    let calcurs_props: Result<Vec<_>, _> = calcurs_props
        .into_iter()
        .map(|marked_item| {
            let item = marked_item.item.borrow().clone();
            let trt = cast_item!(item as Item::Trait);

            Ok((marked_item.mark, trt))
        })
        .collect();
    let calcurs_props = calcurs_props?;

    let (relations, direct_props) = init_property_relations(calcurs_props)?;

    let dep_trees = dependencies_from_relation(&relations)?;

    let mut trait_props: HashMap<Ident, TraitProperties> = HashMap::new();

    for (trt, deps) in dep_trees {
        let mut assigns = match direct_props.get(&trt) {
            Some(props) => props.clone(),
            None => continue,
        };

        for dep in deps {
            assigns.extend(match direct_props.get(&dep) {
                Some(props) => props.clone(),
                None => continue,
            });
        }

        trait_props.insert(trt, TraitProperties(assigns));
    }

    Ok(trait_props)
}

fn impl_base_constructor(
    strct: (&Attribute, &ItemStruct),
    default: &FuncMacroArg,
    base_type: &Ident,
    calcurs_traits: &HashMap<Ident, TraitProperties>,
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

    let (fields, lit): (Vec<_>, Vec<_>) = attribs.0.iter().unzip();

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
    base: MarkedItem<ItemStruct>,
    types: &[SharedMarkedItem<Item>],
    traits: HashMap<Ident, TraitProperties>,
) -> syn::Result<TokenStream> {
    let base_type = &base.item.ident;
    let base_attrib = base.mark;

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
        let strct = &mut *item.item.borrow_mut();
        let strct = cast_item!(strct as Item::Struct[ref mut]);
        let attrib = &item.mark;

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
    let calcurs_trait_props = get_items_by_mark(&items, "calcurs_trait");

    if calcurs_base.len() > 1 {
        return Err(Error::new(
            Span::call_site(),
            "Currently only 1 Base is supported",
        ));
    }

    if calcurs_types.len() != 0 && calcurs_base.len() != 1 {
        return Err(Error::new_spanned(
            calcurs_types[0].item.borrow().clone().into_token_stream(),
            "No calcurs_base defined for calcurs_type",
        ));
    }
    let base_attrib = calcurs_base.get(0).unwrap();
    let attrib = base_attrib.mark.clone();
    let base = base_attrib.item.borrow().clone();
    let base = cast_item!(base as Item::Struct);

    let calcurs_props = build_dependency_tree(calcurs_trait_props)?;

    stream.extend(impl_calcurs_types(
        MarkedItem::new(attrib, base),
        &calcurs_types,
        calcurs_props,
    )?);

    Ok(stream)
}

#[derive(Debug, Clone, Default)]
pub struct CalcursMacroScope {
    items: Vec<Rc<RefCell<Item>>>,
    gen_stream: TokenStream,
}

impl ToTokens for CalcursMacroScope {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.items
            .iter()
            .for_each(|item| item.borrow().to_tokens(tokens));

        tokens.extend(self.gen_stream.clone());
    }
}

impl Parse for CalcursMacroScope {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let scope: MacroScope = input.parse()?;
        let items = scope.items;

        let gen_stream = parse_calcurs_scope(&items)?;

        Ok(Self { items, gen_stream })
    }
}
