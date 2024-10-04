use calcu_rs::{expr as e, Expr};
use criterion::{criterion_group, criterion_main, Criterion};

fn add_ones(n: i64) {
    let mut sum = Expr::zero();
    for _ in 0..n {
        sum += Expr::one();
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("sum 1e4", |b| b.iter(|| add_ones(10_000)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
