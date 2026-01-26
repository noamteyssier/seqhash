use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use seqhash::SeqHash;

const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

fn generate_random_parents(num_parents: usize, seq_len: usize) -> Vec<Vec<u8>> {
    let mut rng = rand::thread_rng();
    (0..num_parents)
        .map(|_| (0..seq_len).map(|_| BASES[rng.gen_range(0..4)]).collect())
        .collect()
}

fn generate_queries(parents: &[Vec<u8>], num_queries: usize, exact_ratio: f64) -> Vec<Vec<u8>> {
    let mut rng = rand::thread_rng();
    let mut queries = Vec::with_capacity(num_queries);
    let seq_len = parents[0].len();

    for _ in 0..num_queries {
        let parent = &parents[rng.gen_range(0..parents.len())];
        let mut query = parent.clone();

        if rng.gen::<f64>() >= exact_ratio {
            // Introduce single mutation
            let pos = rng.gen_range(0..seq_len);
            let original = query[pos];
            let mut new_base = BASES[rng.gen_range(0..4)];
            while new_base == original {
                new_base = BASES[rng.gen_range(0..4)];
            }
            query[pos] = new_base;
        }

        queries.push(query);
    }

    queries
}

fn bench_construction(c: &mut Criterion) {
    let parents = generate_random_parents(50_000, 49);

    c.bench_function("seqhash_construction_50k", |b| {
        b.iter(|| SeqHash::new(black_box(&parents)).unwrap())
    });
}

fn bench_query(c: &mut Criterion) {
    let parents = generate_random_parents(50_000, 49);
    let queries = generate_queries(&parents, 100_000, 0.90);
    let index = SeqHash::new(&parents).unwrap();

    c.bench_function("seqhash_query_100k", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(index.query(q));
            }
        })
    });
}

fn bench_query_single(c: &mut Criterion) {
    let parents = generate_random_parents(50_000, 49);
    let queries = generate_queries(&parents, 1000, 0.90);
    let index = SeqHash::new(&parents).unwrap();

    c.bench_function("seqhash_query_single", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &queries[i % queries.len()];
            i += 1;
            black_box(index.query(q))
        })
    });
}

criterion_group!(benches, bench_construction, bench_query, bench_query_single);
criterion_main!(benches);
