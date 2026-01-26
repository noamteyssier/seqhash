//! Fast mismatch-tolerant sequence lookup with disambiguation.
//!
//! `seqhash` is a high-performance Rust library for building mismatch-tolerant
//! sequence lookup indices. Given a set of parent sequences, it constructs an
//! index that can query whether an input sequence matches any parent exactly
//! OR is exactly one substitution away—while detecting and rejecting ambiguous
//! cases where a sequence could map to multiple parents.
//!
//! # Example
//!
//! ```
//! use seqhash::{SeqHash, Match};
//!
//! let parents: Vec<&[u8]> = vec![
//!     b"ACGTACGTACGT",
//!     b"GGGGCCCCAAAA",
//!     b"TTTTAAAACCCC",
//! ];
//!
//! let index = SeqHash::new(&parents).unwrap();
//!
//! // Exact match
//! assert!(matches!(
//!     index.query(b"ACGTACGTACGT"),
//!     Some(Match::Exact { parent_idx: 0 })
//! ));
//!
//! // Mismatch match (one base different)
//! let query_with_error = b"ACGTACGTACGA"; // T->A at position 11
//! assert!(matches!(
//!     index.query(query_with_error),
//!     Some(Match::Mismatch { parent_idx: 0, pos: 11 })
//! ));
//! ```

use hashbrown::HashMap;

/// Maximum sequence length (14 bits for position encoding).
pub const MAX_SEQ_LEN: usize = 16383;

/// Valid DNA bases.
const VALID_BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

// Entry bit layout:
// Bit 63:    ambiguous flag
// Bit 62:    is_parent flag
// Bits 48-61: mutation position (14 bits)
// Bits 40-47: original base (8 bits)
// Bits 32-39: mutated base (8 bits)
// Bits 0-31:  parent index (32 bits)

const AMBIGUOUS_BIT: u64 = 1 << 63;
const IS_PARENT_BIT: u64 = 1 << 62;
const POSITION_SHIFT: u64 = 48;
const POSITION_MASK: u64 = 0x3FFF; // 14 bits
const ORIGINAL_BASE_SHIFT: u64 = 40;
const MUTATED_BASE_SHIFT: u64 = 32;
const BASE_MASK: u64 = 0xFF;
const PARENT_IDX_MASK: u64 = 0xFFFFFFFF;

/// A successful match result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Match {
    /// Query exactly matches parent.
    Exact { parent_idx: usize },
    /// Query has single-base mismatch from parent.
    Mismatch { parent_idx: usize, pos: usize },
}

impl Match {
    /// Returns the parent index regardless of match type.
    #[inline]
    pub fn parent_idx(&self) -> usize {
        match self {
            Match::Exact { parent_idx } | Match::Mismatch { parent_idx, .. } => *parent_idx,
        }
    }

    /// Returns true if this was an exact match.
    #[inline]
    pub fn is_exact(&self) -> bool {
        matches!(self, Match::Exact { .. })
    }

    /// Returns the mismatch position, if any.
    #[inline]
    pub fn mismatch_pos(&self) -> Option<usize> {
        match self {
            Match::Exact { .. } => None,
            Match::Mismatch { pos, .. } => Some(*pos),
        }
    }
}

/// Errors during index construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqHashError {
    /// No parent sequences provided.
    EmptyParents,
    /// Parent sequences have different lengths.
    InconsistentLength {
        expected: usize,
        found: usize,
        index: usize,
    },
    /// Sequence length exceeds maximum (16383).
    SequenceTooLong { len: usize },
    /// Duplicate parent sequence found.
    DuplicateParent { index: usize, original: usize },
    /// Sequence contains invalid bases (not A, C, G, T).
    InvalidBase { index: usize, pos: usize, base: u8 },
}

impl std::fmt::Display for SeqHashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeqHashError::EmptyParents => write!(f, "no parent sequences provided"),
            SeqHashError::InconsistentLength {
                expected,
                found,
                index,
            } => write!(
                f,
                "parent at index {} has length {} (expected {})",
                index, found, expected
            ),
            SeqHashError::SequenceTooLong { len } => {
                write!(f, "sequence length {} exceeds maximum {}", len, MAX_SEQ_LEN)
            }
            SeqHashError::DuplicateParent { index, original } => {
                write!(
                    f,
                    "parent at index {} is duplicate of parent at index {}",
                    index, original
                )
            }
            SeqHashError::InvalidBase { index, pos, base } => {
                write!(
                    f,
                    "invalid base '{}' at position {} in parent {}",
                    *base as char, pos, index
                )
            }
        }
    }
}

impl std::error::Error for SeqHashError {}

/// Encoded entry in the lookup table.
#[derive(Debug, Clone, Copy)]
struct Entry(u64);

impl Entry {
    /// Create a new entry for a parent sequence.
    #[inline]
    fn new_parent(parent_idx: u32) -> Self {
        Entry(IS_PARENT_BIT | (parent_idx as u64))
    }

    /// Create a new entry for a mismatch.
    #[inline]
    fn new_mismatch(parent_idx: u32, pos: u16, original_base: u8, mutated_base: u8) -> Self {
        Entry(
            ((pos as u64) << POSITION_SHIFT)
                | ((original_base as u64) << ORIGINAL_BASE_SHIFT)
                | ((mutated_base as u64) << MUTATED_BASE_SHIFT)
                | (parent_idx as u64),
        )
    }

    /// Create an ambiguous entry marker.
    #[inline]
    fn ambiguous() -> Self {
        Entry(AMBIGUOUS_BIT)
    }

    /// Check if this entry is marked ambiguous.
    #[inline]
    fn is_ambiguous(self) -> bool {
        (self.0 & AMBIGUOUS_BIT) != 0
    }

    /// Check if this entry represents a parent (exact match).
    #[inline]
    fn is_parent(self) -> bool {
        (self.0 & IS_PARENT_BIT) != 0
    }

    /// Get the parent index.
    #[inline]
    fn parent_idx(self) -> usize {
        (self.0 & PARENT_IDX_MASK) as usize
    }

    /// Get the mutation position.
    #[inline]
    fn position(self) -> usize {
        ((self.0 >> POSITION_SHIFT) & POSITION_MASK) as usize
    }

    /// Get the original base at the mutation position.
    #[inline]
    fn original_base(self) -> u8 {
        ((self.0 >> ORIGINAL_BASE_SHIFT) & BASE_MASK) as u8
    }

    /// Get the mutated base (what the query should have).
    #[inline]
    fn mutated_base(self) -> u8 {
        ((self.0 >> MUTATED_BASE_SHIFT) & BASE_MASK) as u8
    }
}

/// Hash a sequence using fxhash.
#[inline]
fn hash_sequence(seq: &[u8]) -> u64 {
    fxhash::hash64(seq)
}

/// Check if a base is valid (A, C, G, T).
#[inline]
fn is_valid_base(b: u8) -> bool {
    matches!(b, b'A' | b'C' | b'G' | b'T')
}

/// Check if a base is valid, optionally allowing N.
#[inline]
fn is_valid_base_with_n(b: u8, allow_n: bool) -> bool {
    is_valid_base(b) || (allow_n && b == b'N')
}

/// Fast mismatch-tolerant sequence lookup index.
#[derive(Debug)]
pub struct SeqHash {
    /// Contiguous storage of all parent sequences.
    parents: Vec<u8>,
    /// Number of parent sequences.
    num_parents: usize,
    /// Length of each sequence.
    seq_len: usize,
    /// Hash -> Entry lookup table.
    lookup: HashMap<u64, Entry>,
    /// Count of ambiguous sequences detected.
    num_ambiguous: usize,
    /// If true, only exact matches are supported (no mismatch entries).
    exact_only: bool,
}

/// Builder for constructing a [`SeqHash`] index with custom configuration.
///
/// # Example
///
/// ```
/// use seqhash::SeqHashBuilder;
///
/// let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];
///
/// // Build with default settings (allows 1 mismatch, allows N bases)
/// let index = SeqHashBuilder::default().build(&parents).unwrap();
///
/// // Build with exact match only (no mismatch tolerance)
/// let exact_only = SeqHashBuilder::default()
///     .exact()
///     .build(&parents)
///     .unwrap();
///
/// // Build rejecting N bases in sequences
/// let no_n = SeqHashBuilder::default()
///     .exclude_n()
///     .build(&parents)
///     .unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SeqHashBuilder {
    /// If true, only index exact matches (no mismatch entries).
    exact_only: bool,
    /// If true, allow N bases in sequences (skip N positions for mutations).
    allow_n: bool,
}

impl Default for SeqHashBuilder {
    fn default() -> Self {
        SeqHashBuilder {
            exact_only: false,
            allow_n: true,
        }
    }
}

impl SeqHashBuilder {
    /// Configure for exact matching only (no mismatch tolerance).
    ///
    /// When set, the index will only match sequences that exactly match a parent.
    /// This reduces memory usage since no mutation entries are generated.
    pub fn exact(mut self) -> Self {
        self.exact_only = true;
        self
    }

    /// Reject N bases in sequences.
    ///
    /// By default, sequences containing N are accepted (N positions are skipped
    /// when generating mismatch entries). When this is set, sequences containing
    /// N will be rejected with an `InvalidBase` error.
    pub fn exclude_n(mut self) -> Self {
        self.allow_n = false;
        self
    }

    /// Build the [`SeqHash`] index from the given parent sequences.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No parent sequences are provided
    /// - Sequences have inconsistent lengths
    /// - Sequence length exceeds 16383
    /// - Duplicate parent sequences exist
    /// - Sequences contain invalid bases (unless `allow_n()` is set for N)
    pub fn build<S: AsRef<[u8]>>(self, parents: &[S]) -> Result<SeqHash, SeqHashError> {
        SeqHash::build_internal(parents, self.exact_only, self.allow_n)
    }
}

impl SeqHash {
    /// Construct a new index from parent sequences.
    ///
    /// All sequences must be the same length and contain only A, C, G, T, or N.
    /// This uses default settings (allows 1 mismatch, allows N bases).
    ///
    /// For more control, use [`SeqHashBuilder`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No parent sequences are provided
    /// - Sequences have inconsistent lengths
    /// - Sequence length exceeds 16383
    /// - Duplicate parent sequences exist
    /// - Sequences contain invalid bases
    pub fn new<S: AsRef<[u8]>>(parents: &[S]) -> Result<Self, SeqHashError> {
        Self::build_internal(parents, false, true)
    }

    /// Internal build function used by both `new` and `SeqHashBuilder`.
    fn build_internal<S: AsRef<[u8]>>(
        parents: &[S],
        exact_only: bool,
        allow_n: bool,
    ) -> Result<Self, SeqHashError> {
        if parents.is_empty() {
            return Err(SeqHashError::EmptyParents);
        }

        let seq_len = parents[0].as_ref().len();
        if seq_len > MAX_SEQ_LEN {
            return Err(SeqHashError::SequenceTooLong { len: seq_len });
        }

        let num_parents = parents.len();

        // Pre-allocate contiguous parent storage
        let mut parent_data = Vec::with_capacity(num_parents * seq_len);

        // Estimated capacity: parents + ~3*len mutations per parent (if not exact_only)
        let estimated_entries = if exact_only {
            num_parents
        } else {
            num_parents * (1 + 3 * seq_len)
        };
        let mut lookup: HashMap<u64, Entry> = HashMap::with_capacity(estimated_entries);
        let mut num_ambiguous = 0;

        // First pass: validate and store parents, insert parent entries
        for (idx, parent) in parents.iter().enumerate() {
            let seq = parent.as_ref();

            // Check length consistency
            if seq.len() != seq_len {
                return Err(SeqHashError::InconsistentLength {
                    expected: seq_len,
                    found: seq.len(),
                    index: idx,
                });
            }

            // Validate bases
            for (pos, &base) in seq.iter().enumerate() {
                if !is_valid_base_with_n(base, allow_n) {
                    return Err(SeqHashError::InvalidBase {
                        index: idx,
                        pos,
                        base,
                    });
                }
            }

            // Store parent sequence
            parent_data.extend_from_slice(seq);

            // Insert parent into lookup
            let hash = hash_sequence(seq);
            if let Some(existing) = lookup.get(&hash) {
                if existing.is_parent() {
                    // Check if it's actually a duplicate sequence
                    let existing_idx = existing.parent_idx();
                    let existing_seq =
                        &parent_data[existing_idx * seq_len..(existing_idx + 1) * seq_len];
                    if existing_seq == seq {
                        return Err(SeqHashError::DuplicateParent {
                            index: idx,
                            original: existing_idx,
                        });
                    }
                }
                // Hash collision - mark as ambiguous
                lookup.insert(hash, Entry::ambiguous());
                num_ambiguous += 1;
            } else {
                lookup.insert(hash, Entry::new_parent(idx as u32));
            }
        }

        // Second pass: generate all single-base mutations (unless exact_only)
        if !exact_only {
            let mut mutant_seq = vec![0u8; seq_len];

            for parent_idx in 0..num_parents {
                let parent_start = parent_idx * seq_len;
                let parent_seq = &parent_data[parent_start..parent_start + seq_len];

                for pos in 0..seq_len {
                    let original_base = parent_seq[pos];

                    // Skip N positions when generating mutations
                    if original_base == b'N' {
                        continue;
                    }

                    for &new_base in &VALID_BASES {
                        if new_base == original_base {
                            continue;
                        }

                        // Create mutant sequence
                        mutant_seq.copy_from_slice(parent_seq);
                        mutant_seq[pos] = new_base;

                        let hash = hash_sequence(&mutant_seq);

                        match lookup.get(&hash) {
                            None => {
                                // New entry
                                lookup.insert(
                                    hash,
                                    Entry::new_mismatch(
                                        parent_idx as u32,
                                        pos as u16,
                                        original_base,
                                        new_base,
                                    ),
                                );
                            }
                            Some(existing) => {
                                // If collision is with a parent entry, keep the parent
                                // (exact matches always take precedence)
                                // If collision is with another mismatch entry, mark ambiguous
                                if !existing.is_ambiguous() && !existing.is_parent() {
                                    lookup.insert(hash, Entry::ambiguous());
                                    num_ambiguous += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(SeqHash {
            parents: parent_data,
            num_parents,
            seq_len,
            lookup,
            num_ambiguous,
            exact_only,
        })
    }

    /// Query a sequence.
    ///
    /// Returns the match if unambiguous, None if ambiguous or not found.
    #[inline]
    pub fn query(&self, seq: &[u8]) -> Option<Match> {
        if seq.len() != self.seq_len {
            return None;
        }

        let hash = hash_sequence(seq);
        let entry = *self.lookup.get(&hash)?;

        if entry.is_ambiguous() {
            return None;
        }

        if entry.is_parent() {
            // Verify exact match
            let parent_idx = entry.parent_idx();
            let parent_seq = self.get_parent(parent_idx)?;
            if seq == parent_seq {
                Some(Match::Exact { parent_idx })
            } else {
                None // Hash collision, not actual match
            }
        } else {
            // Verify mismatch match
            let parent_idx = entry.parent_idx();
            let pos = entry.position();
            let original_base = entry.original_base();
            let mutated_base = entry.mutated_base();

            let parent_seq = self.get_parent(parent_idx)?;

            // The query should have mutated_base at pos, parent has original_base
            if seq[pos] != mutated_base || parent_seq[pos] != original_base {
                return None;
            }

            // All other positions should match - use slice comparisons for vectorization
            if seq[..pos] != parent_seq[..pos] || seq[pos + 1..] != parent_seq[pos + 1..] {
                return None;
            }

            Some(Match::Mismatch { parent_idx, pos })
        }
    }

    /// Check if a sequence would be ambiguous (maps to multiple parents).
    #[inline]
    pub fn is_ambiguous(&self, seq: &[u8]) -> bool {
        if seq.len() != self.seq_len {
            return false;
        }

        let hash = hash_sequence(seq);
        self.lookup
            .get(&hash)
            .map(|e| e.is_ambiguous())
            .unwrap_or(false)
    }

    /// Get a parent sequence by index.
    #[inline]
    pub fn get_parent(&self, idx: usize) -> Option<&[u8]> {
        if idx >= self.num_parents {
            return None;
        }
        let start = idx * self.seq_len;
        let end = start + self.seq_len;
        Some(&self.parents[start..end])
    }

    /// Number of parent sequences.
    #[inline]
    pub fn num_parents(&self) -> usize {
        self.num_parents
    }

    /// Length of each sequence.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of entries in the lookup table (parents + unambiguous mutations).
    #[inline]
    pub fn num_entries(&self) -> usize {
        self.lookup.len()
    }

    /// Number of ambiguous sequences detected.
    #[inline]
    pub fn num_ambiguous(&self) -> usize {
        self.num_ambiguous
    }

    /// Returns true if this index only supports exact matches.
    #[inline]
    pub fn is_exact_only(&self) -> bool {
        self.exact_only
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_parents() {
        let parents: Vec<&[u8]> = vec![];
        let result = SeqHash::new(&parents);
        assert_eq!(result.unwrap_err(), SeqHashError::EmptyParents);
    }

    #[test]
    fn test_inconsistent_length() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"ACGTACGT"];
        let result = SeqHash::new(&parents);
        assert_eq!(
            result.unwrap_err(),
            SeqHashError::InconsistentLength {
                expected: 4,
                found: 8,
                index: 1
            }
        );
    }

    #[test]
    fn test_invalid_base() {
        // X is always invalid
        let parents: Vec<&[u8]> = vec![b"ACGX"];
        let result = SeqHash::new(&parents);
        assert_eq!(
            result.unwrap_err(),
            SeqHashError::InvalidBase {
                index: 0,
                pos: 3,
                base: b'X'
            }
        );

        // N is now valid by default
        let parents_with_n: Vec<&[u8]> = vec![b"ACGN"];
        assert!(SeqHash::new(&parents_with_n).is_ok());
    }

    #[test]
    fn test_duplicate_parent() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGG", b"ACGT"];
        let result = SeqHash::new(&parents);
        assert_eq!(
            result.unwrap_err(),
            SeqHashError::DuplicateParent {
                index: 2,
                original: 0
            }
        );
    }

    #[test]
    fn test_exact_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"GGGGCCCCAAAA", b"TTTTAAAACCCC"];

        let index = SeqHash::new(&parents).unwrap();

        assert_eq!(index.num_parents(), 3);
        assert_eq!(index.seq_len(), 12);

        // Test exact matches
        assert_eq!(
            index.query(b"ACGTACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );
        assert_eq!(
            index.query(b"GGGGCCCCAAAA"),
            Some(Match::Exact { parent_idx: 1 })
        );
        assert_eq!(
            index.query(b"TTTTAAAACCCC"),
            Some(Match::Exact { parent_idx: 2 })
        );
    }

    #[test]
    fn test_mismatch_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGT"];

        let index = SeqHash::new(&parents).unwrap();

        // T->A at position 11
        let result = index.query(b"ACGTACGTACGA");
        assert_eq!(
            result,
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 11
            })
        );

        // A->G at position 0
        let result = index.query(b"GCGTACGTACGT");
        assert_eq!(
            result,
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 0
            })
        );

        // C->T at position 1
        let result = index.query(b"ATGTACGTACGT");
        assert_eq!(
            result,
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 1
            })
        );
    }

    #[test]
    fn test_no_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGT"];

        let index = SeqHash::new(&parents).unwrap();

        // Two mismatches - should not match
        assert_eq!(index.query(b"GCGTACGTACGA"), None);

        // Completely different
        assert_eq!(index.query(b"TTTTTTTTTTTT"), None);
    }

    #[test]
    fn test_wrong_length_query() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];

        let index = SeqHash::new(&parents).unwrap();

        // Too short
        assert_eq!(index.query(b"ACGT"), None);

        // Too long
        assert_eq!(index.query(b"ACGTACGTACGT"), None);
    }

    #[test]
    fn test_ambiguous_detection() {
        // Create two parents that are one mismatch apart
        // Their shared mutant will be ambiguous
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"TCGTACGT"]; // Differ only at position 0

        let index = SeqHash::new(&parents).unwrap();

        // Both parents should still be findable
        assert_eq!(
            index.query(b"ACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );
        assert_eq!(
            index.query(b"TCGTACGT"),
            Some(Match::Exact { parent_idx: 1 })
        );

        // Mutations that don't overlap should work
        // ACGTACGT with A->G at position 4 = ACGTGCGT
        assert_eq!(
            index.query(b"ACGTGCGT"),
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 4
            })
        );

        // Some sequences should be ambiguous (one mutation from both parents)
        // CCGTACGT is one mutation from both ACGTACGT (A->C at 0) and TCGTACGT (T->C at 0)
        assert!(index.is_ambiguous(b"CCGTACGT"));
        assert_eq!(index.query(b"CCGTACGT"), None);
    }

    #[test]
    fn test_match_methods() {
        let exact = Match::Exact { parent_idx: 5 };
        assert_eq!(exact.parent_idx(), 5);
        assert!(exact.is_exact());
        assert_eq!(exact.mismatch_pos(), None);

        let mismatch = Match::Mismatch {
            parent_idx: 3,
            pos: 7,
        };
        assert_eq!(mismatch.parent_idx(), 3);
        assert!(!mismatch.is_exact());
        assert_eq!(mismatch.mismatch_pos(), Some(7));
    }

    #[test]
    fn test_get_parent() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGG", b"TTTT"];

        let index = SeqHash::new(&parents).unwrap();

        assert_eq!(index.get_parent(0), Some(b"ACGT".as_slice()));
        assert_eq!(index.get_parent(1), Some(b"GGGG".as_slice()));
        assert_eq!(index.get_parent(2), Some(b"TTTT".as_slice()));
        assert_eq!(index.get_parent(3), None);
    }

    #[test]
    fn test_entry_encoding() {
        // Test parent entry
        let entry = Entry::new_parent(12345);
        assert!(entry.is_parent());
        assert!(!entry.is_ambiguous());
        assert_eq!(entry.parent_idx(), 12345);

        // Test mismatch entry
        let entry = Entry::new_mismatch(999, 100, b'A', b'T');
        assert!(!entry.is_parent());
        assert!(!entry.is_ambiguous());
        assert_eq!(entry.parent_idx(), 999);
        assert_eq!(entry.position(), 100);
        assert_eq!(entry.original_base(), b'A');
        assert_eq!(entry.mutated_base(), b'T');

        // Test ambiguous entry
        let entry = Entry::ambiguous();
        assert!(entry.is_ambiguous());
    }

    #[test]
    fn test_hash_function() {
        // Same sequence should produce same hash
        assert_eq!(hash_sequence(b"ACGT"), hash_sequence(b"ACGT"));

        // Different sequences should (usually) produce different hashes
        assert_ne!(hash_sequence(b"ACGT"), hash_sequence(b"ACGA"));
    }

    #[test]
    fn test_num_entries() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];

        let index = SeqHash::new(&parents).unwrap();

        // 1 parent + up to 12 mutations (4 positions * 3 alternatives each)
        // Some might collide if hash collisions occur, but shouldn't for this simple case
        assert!(index.num_entries() >= 1);
        assert!(index.num_entries() <= 13); // 1 + 12
    }

    #[test]
    fn test_all_single_mutations() {
        let parents: Vec<&[u8]> = vec![b"AAAA"];
        let index = SeqHash::new(&parents).unwrap();

        // Test all single mutations from AAAA
        let mutations = [
            (b"CAAA", 0),
            (b"GAAA", 0),
            (b"TAAA", 0),
            (b"ACAA", 1),
            (b"AGAA", 1),
            (b"ATAA", 1),
            (b"AACA", 2),
            (b"AAGA", 2),
            (b"AATA", 2),
            (b"AAAC", 3),
            (b"AAAG", 3),
            (b"AAAT", 3),
        ];

        for (query, expected_pos) in mutations {
            let result = index.query(query);
            assert_eq!(
                result,
                Some(Match::Mismatch {
                    parent_idx: 0,
                    pos: expected_pos
                }),
                "Failed for query {:?}",
                std::str::from_utf8(query)
            );
        }
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            SeqHashError::EmptyParents.to_string(),
            "no parent sequences provided"
        );

        assert_eq!(
            SeqHashError::InconsistentLength {
                expected: 10,
                found: 5,
                index: 3
            }
            .to_string(),
            "parent at index 3 has length 5 (expected 10)"
        );

        assert_eq!(
            SeqHashError::SequenceTooLong { len: 20000 }.to_string(),
            "sequence length 20000 exceeds maximum 16383"
        );

        assert_eq!(
            SeqHashError::DuplicateParent {
                index: 5,
                original: 2
            }
            .to_string(),
            "parent at index 5 is duplicate of parent at index 2"
        );

        assert_eq!(
            SeqHashError::InvalidBase {
                index: 1,
                pos: 4,
                base: b'N'
            }
            .to_string(),
            "invalid base 'N' at position 4 in parent 1"
        );
    }

    #[test]
    fn test_multiple_parents_different_mutations() {
        // Three parents that are well-separated
        let parents: Vec<&[u8]> = vec![
            b"AAAAAAAA", // All A
            b"CCCCCCCC", // All C
            b"GGGGGGGG", // All G
        ];

        let index = SeqHash::new(&parents).unwrap();

        // Each parent's mutations should be unambiguous
        // since they're so different from each other

        // Mutation of first parent
        assert_eq!(
            index.query(b"CAAAAAAA"),
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 0
            })
        );

        // Mutation of second parent
        assert_eq!(
            index.query(b"ACCCCCCC"),
            Some(Match::Mismatch {
                parent_idx: 1,
                pos: 0
            })
        );

        // Mutation of third parent
        assert_eq!(
            index.query(b"AGGGGGGG"),
            Some(Match::Mismatch {
                parent_idx: 2,
                pos: 0
            })
        );
    }

    #[test]
    fn test_is_ambiguous_wrong_length() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        // Wrong length should return false, not panic
        assert!(!index.is_ambiguous(b"AC"));
        assert!(!index.is_ambiguous(b"ACGTACGT"));
    }

    #[test]
    fn test_string_input() {
        // Test that we can use String/&str via AsRef<[u8]>
        let parents: Vec<String> = vec!["ACGTACGT".to_string(), "GGGGCCCC".to_string()];

        let index = SeqHash::new(&parents).unwrap();
        assert_eq!(index.num_parents(), 2);
        assert_eq!(
            index.query(b"ACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );
    }

    #[test]
    fn test_vec_u8_input() {
        let parents: Vec<Vec<u8>> = vec![b"ACGTACGT".to_vec(), b"GGGGCCCC".to_vec()];

        let index = SeqHash::new(&parents).unwrap();
        assert_eq!(index.num_parents(), 2);
        assert_eq!(
            index.query(b"ACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );
    }

    #[test]
    fn test_builder_default() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];

        let index = SeqHashBuilder::default().build(&parents).unwrap();

        assert_eq!(index.num_parents(), 2);
        assert!(!index.is_exact_only());

        // Should support exact matches
        assert_eq!(
            index.query(b"ACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Should support mismatch matches
        assert_eq!(
            index.query(b"ACGTACGA"), // T->A at pos 7
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 7
            })
        );
    }

    #[test]
    fn test_builder_exact_only() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];

        let index = SeqHashBuilder::default().exact().build(&parents).unwrap();

        assert_eq!(index.num_parents(), 2);
        assert!(index.is_exact_only());

        // Should support exact matches
        assert_eq!(
            index.query(b"ACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Should NOT support mismatch matches
        assert_eq!(index.query(b"ACGTACGA"), None);

        // Exact-only index should have fewer entries (just parents)
        assert_eq!(index.num_entries(), 2);
    }

    #[test]
    fn test_builder_exclude_n() {
        // With exclude_n, N should be rejected
        let parents_with_n: Vec<&[u8]> = vec![b"ACGTNCGT"];
        let result = SeqHashBuilder::default().exclude_n().build(&parents_with_n);
        assert_eq!(
            result.unwrap_err(),
            SeqHashError::InvalidBase {
                index: 0,
                pos: 4,
                base: b'N'
            }
        );

        // By default, N should be accepted
        let index = SeqHashBuilder::default().build(&parents_with_n).unwrap();

        assert_eq!(index.num_parents(), 1);

        // Exact match should work
        assert_eq!(
            index.query(b"ACGTNCGT"),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Mismatch at non-N position should work
        assert_eq!(
            index.query(b"GCGTNCGT"), // A->G at pos 0
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 0
            })
        );
    }

    #[test]
    fn test_builder_skips_n_positions() {
        let parents: Vec<&[u8]> = vec![b"ANGT"];

        let index = SeqHashBuilder::default().build(&parents).unwrap();

        // Mutations at non-N positions should be indexed
        assert_eq!(
            index.query(b"GNGT"), // A->G at pos 0
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 0
            })
        );
        assert_eq!(
            index.query(b"ANAT"), // G->A at pos 2
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 2
            })
        );

        // Query with different base at N position should not match
        // (since we don't generate mutations at N positions)
        assert_eq!(index.query(b"AAGT"), None);
        assert_eq!(index.query(b"ACGT"), None);
    }

    #[test]
    fn test_builder_exact_with_n() {
        let parents: Vec<&[u8]> = vec![b"ACNTNC"];

        let index = SeqHashBuilder::default().exact().build(&parents).unwrap();

        assert!(index.is_exact_only());
        assert_eq!(index.num_entries(), 1);

        // Only exact match should work
        assert_eq!(index.query(b"ACNTNC"), Some(Match::Exact { parent_idx: 0 }));
        assert_eq!(index.query(b"GCNTNC"), None);
    }

    #[test]
    fn test_new_allows_n_by_default() {
        // SeqHash::new should allow N bases by default
        let parents: Vec<&[u8]> = vec![b"ACNGT"];
        let index = SeqHash::new(&parents).unwrap();

        assert_eq!(index.num_parents(), 1);
        assert_eq!(index.query(b"ACNGT"), Some(Match::Exact { parent_idx: 0 }));
    }
}
