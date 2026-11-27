//! A quick hash for maps and sets keyed by our own numbers: addresses of
//! our objects, title addresses, slot indices. The standard maps' SipHash
//! guards against keys chosen to collide, which these are not; hashing and
//! hash map lookups were about a tenth of the submitting thread's running
//! time (run ab-all1), and every range a list refreshes is looked up in a
//! set by three words. `VICTORY_QUICK_HASH=0` hashes with SipHash as before.

use std::hash::{BuildHasher, Hasher};

pub(crate) type Map<K, V> = std::collections::HashMap<K, V, Build>;
pub(crate) type Set<K> = std::collections::HashSet<K, Build>;

/// Builds [`Quick`] hashers.
#[derive(Clone, Copy, Default)]
pub(crate) struct Build;

impl BuildHasher for Build {
    type Hasher = Quick;

    fn build_hasher(&self) -> Quick {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| !crate::switch::off("VICTORY_QUICK_HASH")) {
            Quick::Words(SEED)
        } else {
            Quick::Sip(std::collections::hash_map::DefaultHasher::new())
        }
    }
}

/// A multiply per word and a mix at the end, or SipHash (the switch).
pub(crate) enum Quick {
    Words(u64),
    Sip(std::collections::hash_map::DefaultHasher),
}

/// An odd constant with its bits spread (2^64 over the golden ratio).
const K: u64 = 0x9e37_79b9_7f4a_7c15;

/// Where a hash starts (not 0, which words of 0 would leave 0).
const SEED: u64 = 0x243f_6a88_85a3_08d3;

impl Quick {
    fn word(&mut self, word: u64) {
        match self {
            Quick::Words(h) => *h = (h.rotate_left(5) ^ word).wrapping_mul(K),
            Quick::Sip(sip) => sip.write_u64(word),
        }
    }
}

impl Hasher for Quick {
    fn write(&mut self, bytes: &[u8]) {
        if let Quick::Sip(sip) = self {
            return sip.write(bytes);
        }
        let (chunks, rest) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.word(u64::from_le_bytes(*chunk));
        }
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            // The length too: a run of zeros is not its shorter self.
            self.word(u64::from_le_bytes(last) ^ (rest.len() as u64) << 56);
        }
    }

    fn write_u8(&mut self, n: u8) {
        self.word(u64::from(n));
    }

    fn write_u32(&mut self, n: u32) {
        self.word(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        self.word(n);
    }

    fn write_usize(&mut self, n: usize) {
        self.word(n as u64);
    }

    /// Mixed so that the low bits (a table's bucket) and the high ones (its
    /// tags) both take from every bit: the keys are aligned addresses, whose
    /// low bits are zero and stay zero through the multiplies.
    fn finish(&self) -> u64 {
        match self {
            Quick::Words(h) => {
                let h = (h ^ h >> 32).wrapping_mul(0xd6e8_feb8_6659_fd93);
                h ^ h >> 32
            }
            Quick::Sip(sip) => sip.finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash<T: std::hash::Hash>(key: T) -> u64 {
        Build.hash_one(key)
    }

    /// Aligned addresses, the keys these maps mostly have, spread over a
    /// table's buckets (the low bits of the hash) and its tags (the top
    /// seven): no bucket of 1,024 takes more than a few of 4,096 keys that
    /// differ only above bit 12.
    #[test]
    fn aligned_addresses_spread_over_buckets_and_tags() {
        let (mut buckets, mut tags) = ([0u32; 1024], [0u32; 128]);
        for i in 0..4096u64 {
            let h = hash(0x7f00_0000_0000usize + (i as usize) * 4096);
            buckets[(h & 1023) as usize] += 1;
            tags[(h >> 57) as usize] += 1;
        }
        assert!(buckets.iter().all(|&n| n <= 20), "{:?}", buckets.iter().max());
        assert!(tags.iter().all(|&n| (8..=64).contains(&n)), "{:?} {:?}", tags.iter().min(), tags.iter().max());
    }

    /// Keys of several words differ by any one of them, and by their order.
    #[test]
    fn every_word_of_a_key_counts() {
        let base = hash((1usize, 2usize, 3usize));
        assert_ne!(base, hash((1usize, 2usize, 4usize)));
        assert_ne!(base, hash((0usize, 2usize, 3usize)));
        assert_ne!(base, hash((2usize, 1usize, 3usize)));
        assert_ne!(hash((0usize, 0usize)), hash((0usize, 0usize, 0usize)));
        assert_ne!(hash([0u8; 3]), hash([0u8; 5]));
    }

    /// A map with it holds what it was given, slot indices and tuples alike.
    #[test]
    fn maps_and_sets_hold_their_keys() {
        let mut map: Map<u32, u32> = Map::default();
        let mut set: Set<(usize, usize, usize)> = Set::default();
        for i in 0..100_000u32 {
            map.insert(i, !i);
            set.insert((i as usize * 64, 0x1000, i as usize));
        }
        assert_eq!((map.len(), set.len()), (100_000, 100_000));
        assert!((0..100_000u32).all(|i| map.get(&i) == Some(&!i)));
        assert!((0..100_000usize).all(|i| set.contains(&(i * 64, 0x1000, i))));
        assert!(!set.contains(&(64, 0x1000, 0)));
    }
}
