use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap, HashSet},
    rc::Rc,
};

use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, Parser},
    punctuated::Punctuated,
    Error, ExprLit, Ident, Item, ItemStruct, ItemTrait, Token,
};

use crate::{
    macro_scope::{get_items_by_mark_prefix, MacroScope, MarkedItem, SharedMarkedItem},
    parsers::{FieldValues, FuncMacroArg},
    utils::*,
};

#[derive(Debug, Clone, PartialEq, Default)]
struct TraitProperties(HashMap<Ident, ExprLit>);

fn init_property_relations(
    calcurs_props: Vec<MarkedItem<ItemTrait>>,
) -> syn::Result<(TraitRelations, HashMap<Ident, HashMap<Ident, ExprLit>>)> {
    let mut properties: HashMap<Ident, HashMap<Ident, ExprLit>> = HashMap::new();
    let mut relations: TraitRelations = HashMap::new();

    let calcurs_traits: HashSet<_> = calcurs_props
        .iter()
        .map(|item| item.item.ident.clone())
        .collect();

    for MarkedItem {
        mark: attrib,
        item: trt,
    } in calcurs_props
    {
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
                if calcurs_traits.contains(ident) {
                    sups.insert(ident.clone());
                }
            }
        }

        relations.insert(trt.ident, sups);
    }

    Ok((relations, properties))
}

fn dependencies_from_relation(relations: &TraitRelations) -> syn::Result<TraitRelations> {
    let mut dep_trees: TraitRelations = HashMap::new();

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

type TraitPropertieMap = HashMap<Ident, TraitProperties>;
type TraitRelations = HashMap<Ident, BTreeSet<Ident>>;

#[derive(Debug, Clone, PartialEq)]
struct TraitDepTree {
    property_map: TraitPropertieMap,
    relations: TraitRelations,
}

impl TraitDepTree {
    fn new(property_map: TraitPropertieMap, relations: TraitRelations) -> Self {
        Self {
            property_map,
            relations,
        }
    }
}

fn build_dependency_tree(trait_props: Vec<MarkedItem<ItemTrait>>) -> syn::Result<TraitDepTree> {
    let (relations, direct_props) = init_property_relations(trait_props)?;

    let dep_trees = dependencies_from_relation(&relations)?;

    let mut trait_props: TraitPropertieMap = HashMap::new();

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

    Ok(TraitDepTree::new(trait_props, relations))
}

fn impl_diff_debug_fn(base_type: &ItemStruct, default: &FuncMacroArg) -> syn::Result<TokenStream> {
    let fields = match &base_type.fields {
        syn::Fields::Named(syn::FieldsNamed { named, .. }) => {
            named.iter().map(|field| field.ident.clone().unwrap())
        }
        _ => {
            return Err(Error::new(
                base_type.ident.span(),
                "DiffDebug is not defined for structs without named fields",
            ))
        }
    };

    let ident = &base_type.ident;

    Ok(quote! {
        impl #ident {
            fn diff_debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use std::fmt::Write;

                let default = #default();

                let mut diff = String::new();

                #(
                if (self.#fields != default.#fields) {
                    write!(diff, "{} = {:?}, ", stringify!(#fields), self.#fields)?;
                }
                )*

                write!(f, "Base {{{}}}", diff)
            }
        }
    })
}

fn impl_new_base_fn(
    item: &MarkedItem<ItemStruct>,
    default: &FuncMacroArg,
    base_type: &Ident,
    calcurs_traits: &TraitPropertieMap,
) -> syn::Result<TokenStream> {
    let strct = &item.item;
    let attrib = &item.mark;
    let parents = attrib.parse_args_with(StructParents::parse_terminated)?;

    let generics = &strct.generics;
    let name = &strct.ident;

    let mut attribs = TraitProperties::default();

    for derived in parents {
        match calcurs_traits.get(&derived) {
            Some(attrs) => attribs.0.extend(attrs.0.clone()),
            None => {
                return Err(Error::new(
                    derived.span(),
                    format!("Calcurs Trait: {} was never defined", derived.to_string()),
                ))
            }
        };
    }

    let (fields, lit): (Vec<_>, Vec<_>) = attribs.0.iter().unzip();

    let cnst = match default.is_const {
        true => Some(quote!(const)),
        false => None,
    };

    let code = quote! {
        impl #generics #name #generics {
            #cnst fn new_base() -> #base_type {
                #base_type {
                    #(#fields: #lit,)*
                    .. #default()
                }
            }
        }
    };

    Ok(code)
}

type StructParents = Punctuated<Ident, Token![+]>;

fn impl_trait_dependencies(
    calcurs_struct: &MarkedItem<ItemStruct>,
    relations: &TraitRelations,
) -> syn::Result<TokenStream> {
    let item = &calcurs_struct.item;
    let strct_name = &item.ident;
    let generics = &item.generics;
    let mark = &calcurs_struct.mark;

    let trait_names = mark.parse_args_with(StructParents::parse_terminated)?;

    let trait_names: Vec<_> = trait_names.into_iter().collect();

    let mut deps = BTreeSet::new();
    for name in &trait_names {
        match relations.get(&name) {
            Some(d) => deps.extend(d.clone()),
            None => return Ok(quote!()),
        }
    }

    let mut parents = BTreeSet::new();

    while let Some(dep) = deps.pop_last() {
        if parents.contains(&dep) {
            continue;
        }

        parents.insert(dep.clone());

        match relations.get(&dep) {
            Some(par) => deps.extend(par.clone()),
            None => (),
        }
    }

    parents.extend(trait_names);

    Ok(quote! {
        #(impl #generics #parents for #strct_name #generics {} )*
    })
}

fn impl_calcurs_items(
    base: MarkedItem<ItemStruct>,
    types: &[SharedMarkedItem<Item>],
    dep_tree: TraitDepTree,
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

    // let base_field = Field::parse_named
    //     .parse2(quote! {base: #base_type})
    //     .expect("impl_calcurs_items: could not implemnt base field");

    // let internals = import_crate("internals");

    let mut stream = TokenStream::new();

    let s = impl_diff_debug_fn(&base.item, &constructor)?;
    stream.extend(s);

    for typ in types {
        let strct = &mut *typ.item.borrow_mut();
        let strct = cast_item!(strct as Item::Struct[ref mut]);
        let type_mark = &typ.mark;
        let calcurs_struct = MarkedItem::new(type_mark.clone(), strct.clone());

        // append_struct_field(strct, base_field.clone())?;

        let s = impl_trait_dependencies(&calcurs_struct, &dep_tree.relations);
        stream.extend(s);

        let s = impl_new_base_fn(
            &calcurs_struct,
            &constructor,
            base_type,
            &dep_tree.property_map,
        )?;

        stream.extend(s);

        // let generics = &strct.generics;
        // let name = &strct.ident;

        // let impl_code = quote! {
        //     impl #generics #internals::Inherited<#base_type> for #name #generics {
        //         fn base(&self) -> &#base_type {
        //             &self.base
        //         }
        //     }
        // };

        // stream.extend(impl_code);
    }

    Ok(stream)
}

macro_rules! cast_marked_items {
    ($items: ident as $enum: path [$struct: ident]) => {{
        let items: syn::Result<Vec<MarkedItem<$struct>>> = $items
            .into_iter()
            .map(|marked_item| {
                let item = marked_item.item.borrow().clone();
                let trt = cast_item!(item as $enum);

                Ok(MarkedItem::new(marked_item.mark, trt))
            })
            .collect();
        items
    }};
}

fn parse_calcurs_scope(items: &[Rc<RefCell<Item>>]) -> syn::Result<TokenStream> {
    let mut stream = TokenStream::new();

    let mut calcurs_items = get_items_by_mark_prefix(&items, "calcurs");

    let calcurs_base = calcurs_items.remove("calcurs_base").unwrap_or_default();
    let calcurs_types = calcurs_items.remove("calcurs_type").unwrap_or_default();
    let traits = calcurs_items.remove("calcurs_trait").unwrap_or_default();
    let traits = cast_marked_items!(traits as Item::Trait[ItemTrait])?;

    if calcurs_base.len() > 1 {
        return Err(Error::new(Span::call_site(), "Only 1 Base is supported"));
    }
    if calcurs_types.len() != 0 && calcurs_base.len() != 1 {
        return Err(Error::new_spanned(
            calcurs_types[0].item.borrow().clone().into_token_stream(),
            "No calcurs_base defined for calcurs_type",
        ));
    }

    let base = cast_marked_items!(calcurs_base as Item::Struct[ItemStruct])?;
    let base = base.get(0).unwrap().clone();

    let trait_dep_tree = build_dependency_tree(traits)?;

    let s = impl_calcurs_items(base, &calcurs_types, trait_dep_tree)?;

    stream.extend(s);

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
