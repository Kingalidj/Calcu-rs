use calcu_rs::{expr as e, Expr, SymbolicExpr};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn add_ones(n: i64) {
    let mut sum = e!(0);
    for _ in 0..n {
        sum += e!(1);
    }
}

fn derivatives() {
    let d = |e: Expr| {
        e.derivative(black_box(e!(x)))
            .rationalize()
            .expand()
            .factor_out()
            .reduce()
            .reduce()
    };

    d(e!(x ^ 2));
    d(e!(sin(x)));
    d(e!(exp(x)));
    d(e!(x * exp(x)));
    d(e!(ln(x)));
    d(e!(1 / x));
    d(e!(tan(x)));
    d(e!(arc_tan(x)));
    d(e!(x * ln(x) * sin(x)));
    d(e!(x * cos(x) * ln(x) + sin(x) * ln(x) + sin(x)));
    d(e!(x ^ 2));
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("sum 100", |b| b.iter(|| add_ones(black_box(100))));
    c.bench_function("derivatives", |b| b.iter(|| derivatives()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
