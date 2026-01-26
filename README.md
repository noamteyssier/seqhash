# seqhash

Fast mismatch-tolerant sequence lookup with disambiguation.

`seqhash` is a high-performance Rust library for building indices that support approximate matching of DNA sequences. Given a set of parent sequences, it constructs an index that can determine whether a query sequence matches any parent exactly *or* differs by exactly one substitution—while detecting and rejecting ambiguous cases where a query could map to multiple parents.

## The Problem

In many bioinformatics applications (barcode demultiplexing, guide RNA mapping, subsequence matching), you need to match observed sequences against a reference set while tolerating sequencing errors.
A common requirement is to allow up to one mismatch, but only when the match is unambiguous.

### The Naive Approach

The straightforward solution is to pre-compute all possible single-base mutations for each parent sequence and store them in a hash table:

```
Parent: ACGT
Mutations: CCGT, GCGT, TCGT, NCGT,  (position 0)
           AAGT, AGGT, ATGT, ANGT,  (position 1)
           ACAT, ACCT, ACTT, ACNT,  (position 2)
           ACGA, ACGC, ACGG, ACGN   (position 3)
```

This works, but has drawbacks:

1. **Memory overhead**: Each entry stores a full copy of the sequence plus hash table overhead. With 50,000 parents, 49bp sequences, and ~200 mutations each, you're storing ~10 million sequences.

2. **Construction time**: Allocating and hashing millions of sequence copies is expensive.

3. **Query overhead**: Looking up variable-length keys requires hashing the full sequence and comparing bytes on collision.

### Alternative Approaches

#### Store parents only, generate mutations in-flight

An alternative approach is to only store a hash table of the parent sequences and for each sequence that does not match to generate their mutations on the fly.
This can reduce memory overhead at construction time, but the query time will be slower as it will require generating the mutations on the fly.

This can be minimal for small datasets, or when the positive lookup rate is high, but even for mapping rates in the 90% on normal biological datasets, the overhead of generating mutations at each base can be quite significant.

Unfortunately to handle ambiguous mismatches (i.e. a mutated sequence which has multiple possible parents), this approach requires generating *all* possible mutations every time and not just up to the first matching mutation.

#### Store parents only, calculate hamming distance

Another alternative approach is to store a hash table of the parent sequences and for each sequence that does not match to calculate the hamming distance to an existing parent sequence.

This can easily become impractical as it requires performing a full hamming distance calculation for each sequence *for each parent* and suffers from the same issues as the previous approach with respect to ambiguous mismatches.

### The seqhash Approach

`seqhash` uses a fundamentally different strategy:

1. **Store parents once**: Parent sequences are stored contiguously in a single `Vec<u8>`, indexed by position.

2. **Compact entry encoding**: Instead of storing mutation sequences, we store *metadata about the mutation* in a 64-bit integer:
   - Parent index (32 bits)
   - Mutation position (14 bits)
   - Original base (8 bits)
   - Mutated base (8 bits)
   - Flags for ambiguity/exact-match (2 bits)

3. **Hash the mutation, store the delta**: We hash mutation sequences during construction but only store the compact entry. During query, we verify matches by checking that the query differs from the parent at exactly the expected position with the expected bases.

This design provides:

- **~3× lower memory usage**: Entries are 8 bytes instead of ~80 bytes (sequence + overhead)
- **~4× faster construction**: No allocation per mutation, just compute hash and insert entry
- **~1.5× faster queries**: Fixed-size entries, better cache locality

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
seqhash = "0.1"
```

### Basic Example

```rust
use seqhash::{SeqHash, Match};

let parents: Vec<&[u8]> = vec![
    b"ACGTACGTACGT",
    b"GGGGCCCCAAAA",
    b"TTTTAAAACCCC",
];

let index = SeqHash::new(&parents).unwrap();

// Exact match
assert!(matches!(
    index.query(b"ACGTACGTACGT"),
    Some(Match::Exact { parent_idx: 0 })
));

// Single-base mismatch (T→A at position 11)
assert!(matches!(
    index.query(b"ACGTACGTACGA"),
    Some(Match::Mismatch { parent_idx: 0, pos: 11 })
));

// No match (too many differences)
assert!(index.query(b"GGGGGGGGGGGG").is_none());
```

### Match Results

The `Match` enum tells you how a query matched:

```rust
match index.query(sequence) {
    Some(Match::Exact { parent_idx }) => {
        println!("Exact match to parent {}", parent_idx);
    }
    Some(Match::Mismatch { parent_idx, pos }) => {
        println!("One mismatch from parent {} at position {}", parent_idx, pos);
    }
    None => {
        // No match, or ambiguous (maps to multiple parents)
    }
}
```

### Handling Ambiguity

When a sequence could match multiple parents (e.g., it's one mutation away from two different parents at the same position), `seqhash` returns `None` to avoid incorrect assignments:

```rust
// Two parents that differ only at position 0
let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"TCGTACGT"];
let index = SeqHash::new(&parents).unwrap();

// CCGTACGT is one mutation from both parents → ambiguous
assert!(index.query(b"CCGTACGT").is_none());
assert!(index.is_ambiguous(b"CCGTACGT"));
```

### Builder Configuration

Use `SeqHashBuilder` for more control:

```rust
use seqhash::SeqHashBuilder;

// Exact matching only (no mismatch tolerance)
let index = SeqHashBuilder::default()
    .exact()
    .build(&parents)
    .unwrap();

// Reject sequences containing N bases
let index = SeqHashBuilder::default()
    .exclude_n()
    .build(&parents)
    .unwrap();
```

## Performance Characteristics

Benchmarked with 50,000 parents of 49bp each:

| Metric | seqhash | Naive HashMap | Improvement |
|--------|---------|---------------|-------------|
| Construction | 270ms | 1050ms | **3.9×** |
| Query (single) | 19ns | 29ns | **1.5×** |
| Query (100k batch) | 2.4ms | 4.4ms | **1.8×** |
| Memory | 238 MB | 797 MB | **3.3×** |

## Limitations

- Maximum sequence length: 16,383 bases (14-bit position encoding)
- All parent sequences must have the same length
- Only single-base substitutions are tolerated (no indels)
- Parents must be unique

## License

MIT
