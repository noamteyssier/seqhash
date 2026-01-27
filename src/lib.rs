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
//!
//! # Case Normalization
//!
//! By default, parent sequences are normalized to uppercase during index construction.
//! This ensures consistent matching regardless of input case:
//!
//! ```
//! use seqhash::SeqHash;
//!
//! // Lowercase input is automatically converted to uppercase
//! let parents: Vec<&[u8]> = vec![b"acgtacgt", b"ggggcccc"];
//! let index = SeqHash::new(&parents).unwrap();
//!
//! // Queries must match the normalized (uppercase) sequences
//! assert!(index.query(b"ACGTACGT").is_some());
//! ```
//!
//! For cases where lowercase bases have special meaning (e.g., soft-masked regions),
//! use [`SeqHashBuilder::keep_case()`] to preserve the original case:
//!
//! ```
//! use seqhash::SeqHashBuilder;
//!
//! let parents: Vec<&[u8]> = vec![b"ACGTacgt"]; // Mixed case preserved
//! let index = SeqHashBuilder::default()
//!     .keep_case()
//!     .build(&parents)
//!     .unwrap();
//!
//! // Only exact case matches will work
//! assert!(index.query(b"ACGTacgt").is_some());
//! assert!(index.query(b"ACGTACGT").is_none());
//! ```
//!
//! > **Note**: Querying always matches *exact* sequences, so if you choose to store lowercase bases, they will be treated as distinct from their uppercase counterparts.
//!
//! # Serialization
//!
//! The `serde` feature enables saving and loading pre-built indices to disk.
//! This is useful when you want to build an index once and reuse it across
//! multiple runs without rebuilding.
//!
//! ```toml
//! [dependencies]
//! seqhash = { version = "0.1", features = ["serde"] }
//! ```
//!
//! ```ignore
//! // Save an index to disk
//! index.save("my_index.seqhash")?;
//!
//! // Load an index from disk
//! let index = SeqHash::load("my_index.seqhash")?;
//! ```
//!
//! The recommended file extension is `.seqhash`. The index is stored in
//! bincode format. With the `serde` feature enabled, you can also serialize
//! to any serde-compatible format (JSON, MessagePack, etc.) directly.

use hashbrown::HashMap;

mod multilen;
mod split;
pub use multilen::{MultiLenMatch, MultiLenSeqHash, MultiLenSeqHashBuilder};
pub use split::{Half, SplitMatch, SplitSeqHash};

/// Maximum sequence length (14 bits for position encoding).
pub const MAX_SEQ_LEN: usize = 16383;

/// Valid DNA bases.
const VALID_BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// Valid DNA bases including N.
const VALID_BASES_WITH_N: [u8; 5] = [b'A', b'C', b'G', b'T', b'N'];

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Match {
    /// Query exactly matches parent.
    Exact { parent_idx: usize },
    /// Query has single-base mismatch from parent.
    Mismatch { parent_idx: usize, pos: usize },
}

impl Match {
    /// Returns the parent index regardless of match type.
    #[inline]
    #[must_use]
    pub fn parent_idx(&self) -> usize {
        match self {
            Match::Exact { parent_idx } | Match::Mismatch { parent_idx, .. } => *parent_idx,
        }
    }

    /// Returns true if this was an exact match.
    #[inline]
    #[must_use]
    pub fn is_exact(&self) -> bool {
        matches!(self, Match::Exact { .. })
    }

    /// Returns the mismatch position, if any.
    #[inline]
    #[must_use]
    pub fn mismatch_pos(&self) -> Option<usize> {
        match self {
            Match::Exact { .. } => None,
            Match::Mismatch { pos, .. } => Some(*pos),
        }
    }

    /// Returns the hamming distance contribution of this match.
    ///
    /// Returns 0 for exact matches, 1 for mismatch matches.
    #[inline]
    #[must_use]
    pub fn hdist(&self) -> usize {
        match self {
            Match::Exact { .. } => 0,
            Match::Mismatch { .. } => 1,
        }
    }
}

/// Errors during index construction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
                "parent at index {index} has length {found} (expected {expected})"
            ),
            SeqHashError::SequenceTooLong { len } => {
                write!(f, "sequence length {len} exceeds maximum {MAX_SEQ_LEN}")
            }
            SeqHashError::DuplicateParent { index, original } => {
                write!(
                    f,
                    "parent at index {index} is duplicate of parent at index {original}"
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Entry(u64);

impl Entry {
    /// Create a new entry for a parent sequence.
    #[inline]
    fn new_parent(parent_idx: u32) -> Self {
        Entry(IS_PARENT_BIT | u64::from(parent_idx))
    }

    /// Create a new entry for a mismatch.
    #[inline]
    fn new_mismatch(parent_idx: u32, pos: u16, original_base: u8, mutated_base: u8) -> Self {
        Entry(
            (u64::from(pos) << POSITION_SHIFT)
                | (u64::from(original_base) << ORIGINAL_BASE_SHIFT)
                | (u64::from(mutated_base) << MUTATED_BASE_SHIFT)
                | u64::from(parent_idx),
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

/// Check if a base is valid (A, C, G, T, case-insensitive).
#[inline]
fn is_valid_base(b: u8) -> bool {
    matches!(b, b'A' | b'C' | b'G' | b'T' | b'a' | b'c' | b'g' | b't')
}

/// Check if a base is valid, optionally allowing N.
#[inline]
fn is_valid_base_with_n(b: u8, allow_n: bool) -> bool {
    is_valid_base(b) || (allow_n && (b == b'N' || b == b'n'))
}

/// Calculate whether two sequences are within a given hamming distance.
#[inline]
fn within_hamming_distance(seq1: &[u8], seq2: &[u8], max_hdist: usize) -> bool {
    if seq1.len() != seq2.len() {
        return false;
    }
    let mut hdist = 0;
    for (a, b) in seq1.iter().zip(seq2.iter()) {
        if a != b {
            hdist += 1;
            if hdist > max_hdist {
                return false;
            }
        }
    }
    true
}

/// Fast mismatch-tolerant sequence lookup index.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// If true, N bases are allowed in sequences.
    allow_n: bool,
    /// If true, sequences are normalized to uppercase.
    normalize_case: bool,
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
/// // Build with default settings (allows 1 mismatch, allows N bases, normalizes case)
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
///
/// // Build preserving lowercase bases (useful when case has special meaning)
/// let keep_case = SeqHashBuilder::default()
///     .keep_case()
///     .build(&parents)
///     .unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SeqHashBuilder {
    /// If true, only index exact matches (no mismatch entries).
    exact_only: bool,
    /// If true, allow N bases in sequences (skip N positions for mutations).
    allow_n: bool,
    /// If true, convert sequences to uppercase before indexing (default: true).
    normalize_case: bool,
}

impl Default for SeqHashBuilder {
    fn default() -> Self {
        SeqHashBuilder {
            exact_only: false,
            allow_n: true,
            normalize_case: true,
        }
    }
}

impl SeqHashBuilder {
    /// Configure for exact matching only (no mismatch tolerance).
    ///
    /// When set, the index will only match sequences that exactly match a parent.
    /// This reduces memory usage since no mutation entries are generated.
    #[must_use]
    pub fn exact(mut self) -> Self {
        self.exact_only = true;
        self
    }

    /// Reject N bases in sequences.
    ///
    /// By default, sequences containing N are accepted (N positions are skipped
    /// when generating mismatch entries). When this is set, sequences containing
    /// N will be rejected with an `InvalidBase` error.
    #[must_use]
    pub fn exclude_n(mut self) -> Self {
        self.allow_n = false;
        self
    }

    /// Preserve the case of input sequences.
    ///
    /// By default, sequences are converted to uppercase before indexing.
    /// When this is set, sequences are kept as-is, preserving lowercase bases.
    /// This is useful when lowercase bases have special meaning in your data.
    #[must_use]
    pub fn keep_case(mut self) -> Self {
        self.normalize_case = false;
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
        SeqHash::build_internal(parents, self.exact_only, self.allow_n, self.normalize_case)
    }
}

impl SeqHash {
    /// Construct a new index from parent sequences.
    ///
    /// All sequences must be the same length and contain only A, C, G, T, or N.
    /// This uses default settings (allows 1 mismatch, allows N bases, normalizes case).
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
        Self::build_internal(parents, false, true, true)
    }

    /// Internal build function used by both `new` and `SeqHashBuilder`.
    fn build_internal<S: AsRef<[u8]>>(
        parents: &[S],
        exact_only: bool,
        allow_n: bool,
        normalize_case: bool,
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

        Self::initialize_parents(
            &mut lookup,
            &mut parent_data,
            &mut num_ambiguous,
            parents,
            seq_len,
            normalize_case,
            allow_n,
        )?;

        // Second pass: generate all single-base mutations (unless exact_only)
        if !exact_only {
            Self::initialize_mutations(
                &mut lookup,
                &mut num_ambiguous,
                &parent_data,
                seq_len,
                num_parents,
                allow_n,
            );
        }

        Ok(SeqHash {
            parents: parent_data,
            num_parents,
            seq_len,
            lookup,
            num_ambiguous,
            exact_only,
            allow_n,
            normalize_case,
        })
    }

    /// Internal function used to initialize the parent data in the lookup table
    fn initialize_parents<S: AsRef<[u8]>>(
        lookup: &mut HashMap<u64, Entry>,
        parent_data: &mut Vec<u8>,
        num_ambiguous: &mut usize,
        parents: &[S],
        seq_len: usize,
        normalize_case: bool,
        allow_n: bool,
    ) -> Result<(), SeqHashError> {
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

            // Normalize case if requested
            let normalized_seq: Vec<u8>;
            let seq_to_use = if normalize_case {
                normalized_seq = seq.to_ascii_uppercase();
                &normalized_seq
            } else {
                seq
            };

            // Validate bases (using normalized sequence)
            for (pos, &base) in seq_to_use.iter().enumerate() {
                if !is_valid_base_with_n(base, allow_n) {
                    return Err(SeqHashError::InvalidBase {
                        index: idx,
                        pos,
                        base,
                    });
                }
            }

            // Store parent sequence (normalized if requested)
            parent_data.extend_from_slice(seq_to_use);

            // Insert parent into lookup (using normalized sequence)
            let hash = hash_sequence(seq_to_use);
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
                *num_ambiguous += 1;
            } else {
                lookup.insert(hash, Entry::new_parent(idx as u32));
            }
        }

        Ok(())
    }

    /// Internal function used to generate all mutational sequences
    fn initialize_mutations(
        lookup: &mut HashMap<u64, Entry>,
        num_ambiguous: &mut usize,
        parent_data: &[u8],
        seq_len: usize,
        num_parents: usize,
        allow_n: bool,
    ) {
        let mut mutant_seq = vec![0u8; seq_len];

        // Choose mutation alphabet based on allow_n setting
        let mutation_bases: &[u8] = if allow_n {
            &VALID_BASES_WITH_N
        } else {
            &VALID_BASES
        };

        for parent_idx in 0..num_parents {
            let parent_start = parent_idx * seq_len;
            let parent_seq = &parent_data[parent_start..parent_start + seq_len];

            for pos in 0..seq_len {
                let original_base = parent_seq[pos];

                for &new_base in mutation_bases {
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
                                *num_ambiguous += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Query a sequence.
    ///
    /// Returns the match if unambiguous, None if ambiguous or not found.
    #[inline]
    #[must_use]
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
    #[must_use]
    pub fn is_ambiguous(&self, seq: &[u8]) -> bool {
        if seq.len() != self.seq_len {
            return false;
        }

        let hash = hash_sequence(seq);
        self.lookup.get(&hash).is_some_and(|e| e.is_ambiguous())
    }

    /// Query at a specific position within a longer sequence.
    ///
    /// Extracts a subsequence of length `seq_len` starting at `pos` and queries it.
    /// Returns `None` if the position is out of bounds or no match is found.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGT"];
    /// let index = SeqHash::new(&parents).unwrap();
    ///
    /// // Target sequence is embedded at position 2
    /// let read = b"NNACGTNN";
    /// assert!(index.query_at(read, 2).is_some());
    /// ```
    #[inline]
    #[must_use]
    pub fn query_at(&self, seq: &[u8], pos: usize) -> Option<Match> {
        let end = pos.checked_add(self.seq_len)?;
        if end > seq.len() {
            return None;
        }
        self.query(&seq[pos..end])
    }

    /// Query at a position with remapping window.
    ///
    /// Tries `pos` first, then alternates `+1, -1, +2, -2, ...` up to `window`.
    /// Returns the first match found, or `None` if no match within the window.
    ///
    /// This is useful when the target position may drift slightly due to small indels.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGT"];
    /// let index = SeqHash::new(&parents).unwrap();
    ///
    /// // Target is at position 3, but we think it's at position 2
    /// let read = b"NNNACGTNN";
    /// assert!(index.query_at_with_remap(read, 2, 2).is_some());
    /// ```
    #[inline]
    #[must_use]
    pub fn query_at_with_remap(&self, seq: &[u8], pos: usize, window: usize) -> Option<Match> {
        self.query_at_with_remap_offset(seq, pos, window)
            .map(|(m, _)| m)
    }

    /// Query at a position with remapping, also returning the offset where match was found.
    ///
    /// Tries `pos` first, then alternates `+1, -1, +2, -2, ...` up to `window`.
    /// Returns the match and offset (0 for direct hit, positive for downstream, negative for upstream).
    ///
    /// This is useful when you want to track position drift statistics.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGT"];
    /// let index = SeqHash::new(&parents).unwrap();
    ///
    /// // Target is at position 3, but we think it's at position 2
    /// let read = b"NNNACGTNN";
    /// let result = index.query_at_with_remap_offset(read, 2, 2);
    /// assert!(matches!(result, Some((_, 1)))); // Found at offset +1
    /// ```
    #[must_use]
    pub fn query_at_with_remap_offset(
        &self,
        seq: &[u8],
        pos: usize,
        window: usize,
    ) -> Option<(Match, isize)> {
        // Try exact position first
        if let Some(m) = self.query_at(seq, pos) {
            return Some((m, 0));
        }

        // Alternate +1, -1, +2, -2, etc.
        for offset in 1..=window {
            if let Some(m) = self.query_at(seq, pos + offset) {
                return Some((m, offset as isize));
            }
            if offset <= pos {
                if let Some(m) = self.query_at(seq, pos - offset) {
                    return Some((m, -(offset as isize)));
                }
            }
        }

        None
    }

    /// Sliding window search from start of sequence.
    ///
    /// Scans through the sequence looking for the first match.
    /// Returns the match and its position in the input sequence.
    ///
    /// This is useful when the target position is unknown.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGT"];
    /// let index = SeqHash::new(&parents).unwrap();
    ///
    /// let read = b"NNNACGTNNN";
    /// let result = index.query_sliding(read);
    /// assert!(matches!(result, Some((_, 3)))); // Found at position 3
    /// ```
    #[must_use]
    pub fn query_sliding(&self, seq: &[u8]) -> Option<(Match, usize)> {
        self.query_sliding_iter(seq).next()
    }

    /// Sliding window search returning an iterator over all matches.
    ///
    /// Scans through the sequence and yields all matches found.
    /// This is useful when a sequence may contain multiple target regions.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGT"];
    /// let index = SeqHash::new(&parents).unwrap();
    ///
    /// let read = b"ACGTNNACGT"; // Two occurrences
    /// let matches: Vec<_> = index.query_sliding_iter(read).collect();
    /// assert_eq!(matches.len(), 2);
    /// assert_eq!(matches[0].1, 0); // First at position 0
    /// assert_eq!(matches[1].1, 6); // Second at position 6
    /// ```
    pub fn query_sliding_iter<'a>(
        &'a self,
        seq: &'a [u8],
    ) -> impl Iterator<Item = (Match, usize)> + 'a {
        let num_positions = if seq.len() >= self.seq_len {
            seq.len() - self.seq_len + 1
        } else {
            0
        };
        (0..num_positions)
            .filter_map(move |pos| self.query(&seq[pos..pos + self.seq_len]).map(|m| (m, pos)))
    }

    /// Get a parent sequence by index.
    #[inline]
    #[must_use]
    pub fn get_parent(&self, idx: usize) -> Option<&[u8]> {
        if idx >= self.num_parents {
            return None;
        }
        let start = idx * self.seq_len;
        let end = start + self.seq_len;
        Some(&self.parents[start..end])
    }

    /// Iterate over all parent sequences.
    #[inline]
    pub fn iter_parents(&self) -> impl Iterator<Item = &[u8]> {
        self.parents.chunks_exact(self.seq_len)
    }

    /// Number of parent sequences.
    #[inline]
    #[must_use]
    pub fn num_parents(&self) -> usize {
        self.num_parents
    }

    /// Length of each sequence.
    #[inline]
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of entries in the lookup table (parents + unambiguous mutations).
    #[inline]
    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.lookup.len()
    }

    /// Number of ambiguous sequences detected.
    #[inline]
    #[must_use]
    pub fn num_ambiguous(&self) -> usize {
        self.num_ambiguous
    }

    /// Returns true if this index only supports exact matches.
    #[inline]
    #[must_use]
    pub fn is_exact_only(&self) -> bool {
        self.exact_only
    }

    /// Returns true if this index allows N bases.
    #[inline]
    #[must_use]
    pub fn allows_n(&self) -> bool {
        self.allow_n
    }

    /// Returns true if this index normalizes sequences to uppercase.
    #[inline]
    #[must_use]
    pub fn normalizes_case(&self) -> bool {
        self.normalize_case
    }

    /// Check if a specific parent is within the specified hamming distance of the query sequence.
    ///
    /// This method calculates the hamming distance between the query and the specified parent.
    /// It returns true if the parent is within the specified distance.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![
    ///     b"ACGTACGT",
    ///     b"GGGGCCCC",
    /// ];
    /// let index = SeqHash::new(&parents).unwrap();
    ///
    /// // Query differs by 1 base from first parent
    /// assert!(index.is_within_hdist(b"ACGTACGA", 0, 1));
    ///
    /// // Query differs by more than 1 base from first parent
    /// assert!(!index.is_within_hdist(b"TTTTTTTT", 0, 1));
    ///
    /// // But it's within 8 bases
    /// assert!(index.is_within_hdist(b"TTTTTTTT", 0, 8));
    /// ```
    #[inline]
    #[must_use]
    pub fn is_within_hdist(&self, query: &[u8], parent_idx: usize, hdist: usize) -> bool {
        if query.len() != self.seq_len {
            return false;
        }

        let parent_seq = match self.get_parent(parent_idx) {
            Some(seq) => seq,
            None => return false, // Invalid parent index
        };

        // check hamming distance
        within_hamming_distance(query, parent_seq, hdist)
    }

    /// Save the index to a file.
    ///
    /// The file will be saved in bincode format. The recommended extension is `.seqhash`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use seqhash::SeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];
    /// let index = SeqHash::new(&parents).unwrap();
    /// index.save("my_index.seqhash").unwrap();
    /// ```
    #[cfg(feature = "serde")]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    /// Load an index from a file.
    ///
    /// The file should be in bincode format, as created by [`SeqHash::save`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use seqhash::SeqHash;
    ///
    /// let index = SeqHash::load("my_index.seqhash").unwrap();
    /// ```
    #[cfg(feature = "serde")]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
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

        // 1 parent + up to 16 mutations (4 positions * 4 alternatives each, including N)
        // Some might collide if hash collisions occur, but shouldn't for this simple case
        assert!(index.num_entries() >= 1);
        assert!(index.num_entries() <= 17); // 1 + 16
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
    fn test_query_at() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGG"];
        let index = SeqHash::new(&parents).unwrap();

        // Exact match at position 2
        let read = b"NNACGTNN";
        assert_eq!(
            index.query_at(read, 2),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Exact match at position 0
        let read = b"ACGTNNNN";
        assert_eq!(
            index.query_at(read, 0),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Exact match at end
        let read = b"NNNNACGT";
        assert_eq!(
            index.query_at(read, 4),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Mismatch match
        let read = b"NNACGANN"; // T->A mismatch
        assert_eq!(
            index.query_at(read, 2),
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 3
            })
        );

        // No match
        let read = b"NNTTTTNN";
        assert_eq!(index.query_at(read, 2), None);

        // Second parent match
        let read = b"NNGGGGNN";
        assert_eq!(
            index.query_at(read, 2),
            Some(Match::Exact { parent_idx: 1 })
        );
    }

    #[test]
    fn test_query_at_bounds() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        let read = b"NNNNACGT"; // 8 bytes, ACGT at position 4

        // Position out of bounds (would read past end)
        assert_eq!(index.query_at(read, 5), None);
        assert_eq!(index.query_at(read, 6), None);
        assert_eq!(index.query_at(read, 100), None);

        // Exactly at the boundary (last valid position)
        assert_eq!(
            index.query_at(read, 4),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Empty sequence
        assert_eq!(index.query_at(b"", 0), None);

        // Sequence shorter than seq_len
        assert_eq!(index.query_at(b"AC", 0), None);
    }

    #[test]
    fn test_query_at_with_remap() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        // Direct hit at expected position
        let read = b"NNACGTNNNN";
        assert_eq!(
            index.query_at_with_remap(read, 2, 3),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Hit at offset +1
        let read = b"NNNACGTNNN";
        assert_eq!(
            index.query_at_with_remap(read, 2, 3),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Hit at offset -1
        let read = b"NACGTNNNNN";
        assert_eq!(
            index.query_at_with_remap(read, 2, 3),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Hit at offset +2
        let read = b"NNNNACGTNN";
        assert_eq!(
            index.query_at_with_remap(read, 2, 3),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Hit at offset -2
        let read = b"ACGTNNNNNN";
        assert_eq!(
            index.query_at_with_remap(read, 2, 3),
            Some(Match::Exact { parent_idx: 0 })
        );

        // No hit within window
        let read = b"NNNNNNACGT";
        assert_eq!(index.query_at_with_remap(read, 2, 2), None);

        // Hit just at edge of window
        let read = b"NNNNNACGTN";
        assert_eq!(
            index.query_at_with_remap(read, 2, 3),
            Some(Match::Exact { parent_idx: 0 })
        );
    }

    #[test]
    fn test_query_at_with_remap_offset() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        // Direct hit - offset should be 0
        let read = b"NNACGTNNNN";
        assert_eq!(
            index.query_at_with_remap_offset(read, 2, 3),
            Some((Match::Exact { parent_idx: 0 }, 0))
        );

        // Hit at offset +1
        let read = b"NNNACGTNNN";
        assert_eq!(
            index.query_at_with_remap_offset(read, 2, 3),
            Some((Match::Exact { parent_idx: 0 }, 1))
        );

        // Hit at offset -1
        let read = b"NACGTNNNNN";
        assert_eq!(
            index.query_at_with_remap_offset(read, 2, 3),
            Some((Match::Exact { parent_idx: 0 }, -1))
        );

        // Hit at offset +2
        let read = b"NNNNACGTNN";
        assert_eq!(
            index.query_at_with_remap_offset(read, 2, 3),
            Some((Match::Exact { parent_idx: 0 }, 2))
        );

        // Hit at offset -2
        let read = b"ACGTNNNNNN";
        assert_eq!(
            index.query_at_with_remap_offset(read, 2, 3),
            Some((Match::Exact { parent_idx: 0 }, -2))
        );

        // Mismatch with offset tracking
        let read = b"NNNACGANN"; // T->A mismatch at offset +1
        let result = index.query_at_with_remap_offset(read, 2, 3);
        assert_eq!(
            result,
            Some((
                Match::Mismatch {
                    parent_idx: 0,
                    pos: 3
                },
                1
            ))
        );

        // No hit within window
        let read = b"NNNNNNACGT";
        assert_eq!(index.query_at_with_remap_offset(read, 2, 2), None);
    }

    #[test]
    fn test_query_at_with_remap_prefers_direct_hit() {
        // When there could be multiple matches at different offsets,
        // direct hit (offset 0) should be returned
        let parents: Vec<&[u8]> = vec![b"AAAA"];
        let index = SeqHash::new(&parents).unwrap();

        // All A's - matches at multiple positions
        let read = b"AAAAAAAAAA";

        // Should return offset 0 (direct hit)
        let result = index.query_at_with_remap_offset(read, 3, 3);
        assert_eq!(result, Some((Match::Exact { parent_idx: 0 }, 0)));
    }

    #[test]
    fn test_query_at_with_remap_edge_cases() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        // pos=0, can't go negative
        let read = b"NACGTNNNN";
        let result = index.query_at_with_remap_offset(read, 0, 3);
        assert_eq!(result, Some((Match::Exact { parent_idx: 0 }, 1)));

        // pos=0, direct hit
        let read = b"ACGTNNNN";
        let result = index.query_at_with_remap_offset(read, 0, 3);
        assert_eq!(result, Some((Match::Exact { parent_idx: 0 }, 0)));

        // Window of 0 means only direct hit
        let read = b"NACGTNNNN";
        let result = index.query_at_with_remap_offset(read, 0, 0);
        assert_eq!(result, None);

        let read = b"ACGTNNNN";
        let result = index.query_at_with_remap_offset(read, 0, 0);
        assert_eq!(result, Some((Match::Exact { parent_idx: 0 }, 0)));
    }

    #[test]
    fn test_query_sliding() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGG"];
        let index = SeqHash::new(&parents).unwrap();

        // Match at beginning
        let read = b"ACGTNNNN";
        assert_eq!(
            index.query_sliding(read),
            Some((Match::Exact { parent_idx: 0 }, 0))
        );

        // Match in middle
        let read = b"NNNACGTNNN";
        assert_eq!(
            index.query_sliding(read),
            Some((Match::Exact { parent_idx: 0 }, 3))
        );

        // Match at end
        let read = b"NNNNACGT";
        assert_eq!(
            index.query_sliding(read),
            Some((Match::Exact { parent_idx: 0 }, 4))
        );

        // Second parent exact match at position 0
        let read = b"GGGGTTTT";
        assert_eq!(
            index.query_sliding(read),
            Some((Match::Exact { parent_idx: 1 }, 0))
        );

        // Mismatch match
        let read = b"NNACGANN"; // T->A
        assert_eq!(
            index.query_sliding(read),
            Some((
                Match::Mismatch {
                    parent_idx: 0,
                    pos: 3
                },
                2
            ))
        );

        // No match
        let read = b"TTTTTTTT";
        assert_eq!(index.query_sliding(read), None);

        // Sequence too short
        let read = b"AC";
        assert_eq!(index.query_sliding(read), None);

        // Exact length match
        let read = b"ACGT";
        assert_eq!(
            index.query_sliding(read),
            Some((Match::Exact { parent_idx: 0 }, 0))
        );
    }

    #[test]
    fn test_query_sliding_returns_first_match() {
        let parents: Vec<&[u8]> = vec![b"AAAA"];
        let index = SeqHash::new(&parents).unwrap();

        // Multiple possible matches, should return first one
        let read = b"AAAAAAAAA";
        let result = index.query_sliding(read);
        assert_eq!(result, Some((Match::Exact { parent_idx: 0 }, 0)));
    }

    #[test]
    fn test_query_sliding_empty() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        assert_eq!(index.query_sliding(b""), None);
    }

    #[test]
    fn test_query_sliding_iter_multiple_matches() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        // Two exact matches
        let read = b"ACGTNNACGT";
        let matches: Vec<_> = index.query_sliding_iter(read).collect();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0], (Match::Exact { parent_idx: 0 }, 0));
        assert_eq!(matches[1], (Match::Exact { parent_idx: 0 }, 6));
    }

    #[test]
    fn test_query_sliding_iter_mixed_matches() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        // One exact, one mismatch
        let read = b"ACGTNNACGA"; // ACGT at 0, ACGA (T->A mismatch) at 6
        let matches: Vec<_> = index.query_sliding_iter(read).collect();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0], (Match::Exact { parent_idx: 0 }, 0));
        assert_eq!(
            matches[1],
            (
                Match::Mismatch {
                    parent_idx: 0,
                    pos: 3
                },
                6
            )
        );
    }

    #[test]
    fn test_query_sliding_iter_no_matches() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        let read = b"TTTTTTTTTT";
        let matches: Vec<_> = index.query_sliding_iter(read).collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_query_sliding_iter_empty_seq() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        let matches: Vec<_> = index.query_sliding_iter(b"").collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_query_sliding_iter_short_seq() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        let matches: Vec<_> = index.query_sliding_iter(b"AC").collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_query_sliding_iter_lazy() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = SeqHash::new(&parents).unwrap();

        // Many matches, but only take first 2
        let read = b"ACGTACGTACGTACGT";
        let matches: Vec<_> = index.query_sliding_iter(read).take(2).collect();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].1, 0);
        assert_eq!(matches[1].1, 4);
    }

    #[test]
    fn test_query_sliding_iter_multiple_parents() {
        let parents: Vec<&[u8]> = vec![b"AAAA", b"GGGG"];
        let index = SeqHashBuilder::default().exact().build(&parents).unwrap();

        // With exact-only mode, only exact matches are found
        let read = b"AAAACCGGGG";
        let matches: Vec<_> = index.query_sliding_iter(read).collect();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0], (Match::Exact { parent_idx: 0 }, 0));
        assert_eq!(matches[1], (Match::Exact { parent_idx: 1 }, 6));
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
    fn test_builder_generates_n_mutations() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];

        let index = SeqHashBuilder::default().build(&parents).unwrap();

        // Should generate N mutations at each position
        assert_eq!(
            index.query(b"NCGT"), // A->N at pos 0
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 0
            })
        );
        assert_eq!(
            index.query(b"ANGT"), // C->N at pos 1
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 1
            })
        );
        assert_eq!(
            index.query(b"ACNT"), // G->N at pos 2
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 2
            })
        );
        assert_eq!(
            index.query(b"ACGN"), // T->N at pos 3
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 3
            })
        );
    }

    #[test]
    fn test_builder_exclude_n_no_n_mutations() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];

        let index = SeqHashBuilder::default()
            .exclude_n()
            .build(&parents)
            .unwrap();

        // Should NOT generate N mutations
        assert_eq!(index.query(b"NCGT"), None);
        assert_eq!(index.query(b"ANGT"), None);
        assert_eq!(index.query(b"ACNT"), None);
        assert_eq!(index.query(b"ACGN"), None);

        // But regular mutations should still work
        assert_eq!(
            index.query(b"GCGT"), // A->G at pos 0
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 0
            })
        );
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

    #[test]
    fn test_case_normalization_default() {
        // By default, sequences should be normalized to uppercase
        let parents: Vec<&[u8]> = vec![b"acgtacgt", b"ggggcccc"];
        let index = SeqHash::new(&parents).unwrap();

        assert!(index.normalizes_case());
        assert_eq!(index.num_parents(), 2);

        // Stored sequences should be uppercase
        assert_eq!(index.get_parent(0), Some(b"ACGTACGT".as_slice()));
        assert_eq!(index.get_parent(1), Some(b"GGGGCCCC".as_slice()));

        // Query with uppercase should work
        assert_eq!(
            index.query(b"ACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Query with uppercase mismatch should work
        assert_eq!(
            index.query(b"ACGTACGA"), // T->A at pos 7
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 7
            })
        );
    }

    #[test]
    fn test_keep_case() {
        // With keep_case, sequences should preserve lowercase
        let parents: Vec<&[u8]> = vec![b"acgtACGT", b"GGGGcccc"];
        let index = SeqHashBuilder::default()
            .keep_case()
            .build(&parents)
            .unwrap();

        assert!(!index.normalizes_case());
        assert_eq!(index.num_parents(), 2);

        // Stored sequences should preserve case
        assert_eq!(index.get_parent(0), Some(b"acgtACGT".as_slice()));
        assert_eq!(index.get_parent(1), Some(b"GGGGcccc".as_slice()));

        // Query with exact case should work
        assert_eq!(
            index.query(b"acgtACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Query with different case should NOT work
        assert_eq!(index.query(b"ACGTACGT"), None);

        // Mismatch with correct case should work
        assert_eq!(
            index.query(b"acgtACGA"), // T->A at pos 7
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 7
            })
        );
    }

    #[test]
    fn test_case_normalization_with_builder() {
        // Using builder default should normalize case
        let parents: Vec<&[u8]> = vec![b"acgt", b"gggg"];
        let index = SeqHashBuilder::default().build(&parents).unwrap();

        assert!(index.normalizes_case());
        assert_eq!(index.get_parent(0), Some(b"ACGT".as_slice()));
        assert_eq!(index.get_parent(1), Some(b"GGGG".as_slice()));
    }

    #[test]
    fn test_case_normalization_mixed_case_parents() {
        // Mixed case input should be normalized to uppercase
        let parents: Vec<&[u8]> = vec![b"AcGt", b"gGcC"];
        let index = SeqHash::new(&parents).unwrap();

        assert_eq!(index.get_parent(0), Some(b"ACGT".as_slice()));
        assert_eq!(index.get_parent(1), Some(b"GGCC".as_slice()));

        // Exact match with uppercase
        assert_eq!(index.query(b"ACGT"), Some(Match::Exact { parent_idx: 0 }));
        assert_eq!(index.query(b"GGCC"), Some(Match::Exact { parent_idx: 1 }));
    }

    #[test]
    fn test_keep_case_exact_only() {
        // Combining keep_case with exact_only
        let parents: Vec<&[u8]> = vec![b"acgtACGT"];
        let index = SeqHashBuilder::default()
            .keep_case()
            .exact()
            .build(&parents)
            .unwrap();

        assert!(!index.normalizes_case());
        assert!(index.is_exact_only());

        // Should match exact case
        assert_eq!(
            index.query(b"acgtACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );

        // Should not match different case
        assert_eq!(index.query(b"ACGTACGT"), None);

        // Should not support mismatches
        assert_eq!(index.query(b"acgtACGA"), None);
    }

    #[test]
    fn test_case_normalization_with_n() {
        // Normalization should work with N bases
        let parents: Vec<&[u8]> = vec![b"acgtn"];
        let index = SeqHash::new(&parents).unwrap();

        assert_eq!(index.get_parent(0), Some(b"ACGTN".as_slice()));
        assert_eq!(index.query(b"ACGTN"), Some(Match::Exact { parent_idx: 0 }));
    }

    #[test]
    fn test_keep_case_with_lowercase_validation() {
        // When keeping case, lowercase letters should still be valid DNA bases
        let parents: Vec<&[u8]> = vec![b"acgt"];
        let index = SeqHashBuilder::default()
            .keep_case()
            .build(&parents)
            .unwrap();

        // Should validate lowercase as valid bases
        assert_eq!(index.num_parents(), 1);
        assert_eq!(index.get_parent(0), Some(b"acgt".as_slice()));
    }

    #[test]
    fn test_is_within_hdist() {
        let parents: Vec<&[u8]> = vec![
            b"ACGTACGT", // parent 0
            b"GGGGCCCC", // parent 1
            b"TTTTAAAA", // parent 2
        ];
        let index = SeqHash::new(&parents).unwrap();

        // Test exact match (distance 0)
        assert!(index.is_within_hdist(b"ACGTACGT", 0, 0));
        assert!(index.is_within_hdist(b"ACGTACGT", 0, 1));

        // Test single mismatch
        assert!(!index.is_within_hdist(b"ACGTACGA", 0, 0)); // 1 mismatch, hdist 0
        assert!(index.is_within_hdist(b"ACGTACGA", 0, 1)); // 1 mismatch, hdist 1
        assert!(index.is_within_hdist(b"ACGTACGA", 0, 2)); // 1 mismatch, hdist 2

        // Test multiple mismatches
        assert!(!index.is_within_hdist(b"ACGTACAA", 0, 1)); // 2 mismatches, hdist 1
        assert!(index.is_within_hdist(b"ACGTACAA", 0, 2)); // 2 mismatches, hdist 2

        // Test completely different sequence
        assert!(!index.is_within_hdist(b"GGGGGGGG", 0, 3)); // 8 mismatches, hdist 3
        assert!(index.is_within_hdist(b"GGGGGGGG", 0, 8)); // 8 mismatches, hdist 8

        // Test different parents
        assert!(index.is_within_hdist(b"GGGGCCCC", 1, 0)); // exact match to parent 1
        assert!(!index.is_within_hdist(b"GGGGCCCC", 0, 5)); // 6 mismatches from parent 0, hdist 5
        assert!(index.is_within_hdist(b"GGGGCCCC", 0, 6)); // 6 mismatches from parent 0, hdist 6

        // Test invalid parent index
        assert!(!index.is_within_hdist(b"ACGTACGT", 99, 0));

        // Test wrong query length
        assert!(!index.is_within_hdist(b"ACGT", 0, 10));
        assert!(!index.is_within_hdist(b"ACGTACGTACGT", 0, 10));
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn test_seqhash_roundtrip_json() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC", b"TTTTAAAA"];
        let index = SeqHash::new(&parents).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&index).unwrap();

        // Deserialize back
        let restored: SeqHash = serde_json::from_str(&json).unwrap();

        // Verify all properties match
        assert_eq!(restored.num_parents(), index.num_parents());
        assert_eq!(restored.seq_len(), index.seq_len());
        assert_eq!(restored.num_entries(), index.num_entries());
        assert_eq!(restored.num_ambiguous(), index.num_ambiguous());
        assert_eq!(restored.is_exact_only(), index.is_exact_only());
        assert_eq!(restored.allows_n(), index.allows_n());

        // Verify queries work correctly
        assert_eq!(
            restored.query(b"ACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );
        assert_eq!(
            restored.query(b"GCGTACGT"), // A->G at pos 0
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 0
            })
        );

        // Verify parent retrieval
        for i in 0..index.num_parents() {
            assert_eq!(restored.get_parent(i), index.get_parent(i));
        }
    }

    #[test]
    fn test_seqhash_roundtrip_bincode() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"GGGGCCCCAAAA"];
        let index = SeqHash::new(&parents).unwrap();

        // Serialize to bincode
        let bytes = bincode::serialize(&index).unwrap();

        // Deserialize back
        let restored: SeqHash = bincode::deserialize(&bytes).unwrap();

        // Verify queries work
        assert_eq!(
            restored.query(b"ACGTACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );
        assert_eq!(
            restored.query(b"ACGTACGTACGA"), // T->A at pos 11
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 11
            })
        );
    }

    #[test]
    fn test_seqhash_exact_only_roundtrip() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];
        let index = SeqHashBuilder::default().exact().build(&parents).unwrap();

        let bytes = bincode::serialize(&index).unwrap();
        let restored: SeqHash = bincode::deserialize(&bytes).unwrap();

        assert!(restored.is_exact_only());
        assert_eq!(
            restored.query(b"ACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );
        // Mismatch should not work in exact-only mode
        assert_eq!(restored.query(b"GCGTACGT"), None);
    }

    #[test]
    fn test_match_serde() {
        let exact = Match::Exact { parent_idx: 42 };
        let json = serde_json::to_string(&exact).unwrap();
        let restored: Match = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, exact);

        let mismatch = Match::Mismatch {
            parent_idx: 7,
            pos: 13,
        };
        let json = serde_json::to_string(&mismatch).unwrap();
        let restored: Match = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, mismatch);
    }

    #[test]
    fn test_error_serde() {
        let errors = vec![
            SeqHashError::EmptyParents,
            SeqHashError::InconsistentLength {
                expected: 10,
                found: 5,
                index: 2,
            },
            SeqHashError::SequenceTooLong { len: 20000 },
            SeqHashError::DuplicateParent {
                index: 3,
                original: 1,
            },
            SeqHashError::InvalidBase {
                index: 0,
                pos: 5,
                base: b'X',
            },
        ];

        for error in errors {
            let json = serde_json::to_string(&error).unwrap();
            let restored: SeqHashError = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, error);
        }
    }

    #[test]
    fn test_save_and_load() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"GGGGCCCCAAAA", b"TTTTAAAACCCC"];
        let index = SeqHash::new(&parents).unwrap();

        // Create a temporary file path
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_index.seqhash");

        // Save the index
        index.save(&file_path).unwrap();

        // Load the index
        let loaded = SeqHash::load(&file_path).unwrap();

        // Verify all properties match
        assert_eq!(loaded.num_parents(), index.num_parents());
        assert_eq!(loaded.seq_len(), index.seq_len());
        assert_eq!(loaded.num_entries(), index.num_entries());
        assert_eq!(loaded.num_ambiguous(), index.num_ambiguous());
        assert_eq!(loaded.is_exact_only(), index.is_exact_only());
        assert_eq!(loaded.allows_n(), index.allows_n());

        // Verify queries work
        assert_eq!(
            loaded.query(b"ACGTACGTACGT"),
            Some(Match::Exact { parent_idx: 0 })
        );
        assert_eq!(
            loaded.query(b"ACGTACGTACGA"), // T->A at pos 11
            Some(Match::Mismatch {
                parent_idx: 0,
                pos: 11
            })
        );

        // Verify parent data
        for i in 0..index.num_parents() {
            assert_eq!(loaded.get_parent(i), index.get_parent(i));
        }

        // Clean up
        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = SeqHash::load("/nonexistent/path/to/file.seqhash");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn test_load_invalid_file() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("invalid.seqhash");

        // Write invalid data
        std::fs::write(&file_path, b"not valid bincode data").unwrap();

        let result = SeqHash::load(&file_path);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidData);

        // Clean up
        std::fs::remove_file(&file_path).ok();
    }
}
