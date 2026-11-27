//! Equality of byte ranges for our own comparisons of title memory with
//! its copies (`VICTORY_FAST_COMPARE=1`; off by default): SSE2, 64 bytes a
//! step, the four 16-byte differences ORed together and one `pmovmskb`
//! every 128 bytes. Under Wine slice equality goes to the runtime's
//! `memcmp`, which works a byte or a word at a time and orders the bytes,
//! where these comparisons need only equal or not. SSE2, not AVX2: x86-64
//! always has it, and Rosetta 2 translates it to NEON one for one (it runs
//! no AVX2 code faster). Only the callers in this crate use it; the
//! process's `memcmp` is left as it is. After SlaktarnVI / avatarmanz
//! e823263 (`fastmem.rs`, AVX2 there).

use std::sync::OnceLock;

/// Whether [`eq`] takes the SSE2 path (`VICTORY_FAST_COMPARE` `1`, `on`,
/// `true` or `yes`). Read once.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse(std::env::var("VICTORY_FAST_COMPARE").ok().as_deref()))
}

/// `VICTORY_FAST_COMPARE`'s value. Pure, for testing.
fn parse(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes")
    })
}

/// `a == b`, by SSE2 with the switch on.
#[inline]
pub(crate) fn eq(a: &[u8], b: &[u8]) -> bool {
    if on() { fast_eq(a, b) } else { a == b }
}

/// `a == b` by SSE2 where there is one (x86-64), else slice equality.
#[inline]
pub(crate) fn fast_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 64 {
        // SAFETY: SSE2 is part of x86-64; both slices hold `a.len()` bytes.
        return unsafe { eq_sse2(a.as_ptr(), b.as_ptr(), a.len()) };
    }
    a == b
}

/// Whether the `len` bytes (at least 64) at `a` and `b` are equal.
///
/// # Safety
/// Both must be readable for `len` bytes.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eq_sse2(a: *const u8, b: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128,
        _mm_setzero_si128, _mm_xor_si128,
    };
    debug_assert!(len >= 64);
    // SAFETY (the whole function): reads within the `len` bytes at each.
    unsafe {
        // The XOR of 64 bytes at `at`, ORed into 16.
        let block = |at: usize| -> __m128i {
            let d = |k: usize| {
                let x = _mm_loadu_si128(a.add(at + 16 * k).cast());
                let y = _mm_loadu_si128(b.add(at + 16 * k).cast());
                _mm_xor_si128(x, y)
            };
            _mm_or_si128(_mm_or_si128(d(0), d(1)), _mm_or_si128(d(2), d(3)))
        };
        let zero = _mm_setzero_si128();
        let clean = |x: __m128i| _mm_movemask_epi8(_mm_cmpeq_epi8(x, zero)) == 0xFFFF;
        let mut at = 0;
        while at + 128 <= len {
            if !clean(_mm_or_si128(block(at), block(at + 64))) {
                return false;
            }
            at += 128;
        }
        if at + 64 <= len {
            if !clean(block(at)) {
                return false;
            }
            at += 64;
        }
        // The rest: the last 64 bytes, over some compared already.
        at == len || clean(block(len - 64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_off_unless_set() {
        assert!(!parse(None));
        assert!(!parse(Some("0")));
        assert!(!parse(Some("")));
        assert!(parse(Some("1")));
        assert!(parse(Some(" On ")));
    }

    /// Every length to 600 and some long ones, with a difference at each
    /// byte in turn (and none): the same answers as slice equality.
    #[test]
    fn it_answers_as_slice_equality() {
        let mut lengths: Vec<usize> = (0..=600).collect();
        lengths.extend([4095, 4096, 4097, 65536, 65536 + 63]);
        for len in lengths {
            let a: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            let mut b = a.clone();
            assert!(fast_eq(&a, &b), "{len} bytes, equal");
            assert_eq!(eq(&a, &b), a == b);
            let at: Vec<usize> = if len <= 600 {
                (0..len).collect()
            } else {
                vec![0, 1, 63, 64, 127, 128, len / 2, len - 65, len - 64, len - 1]
            };
            for i in at {
                for bit in [0x01u8, 0x80] {
                    b[i] ^= bit;
                    assert!(!fast_eq(&a, &b), "{len} bytes, byte {i} differs");
                    assert_eq!(eq(&a, &b), a == b);
                    b[i] ^= bit;
                }
            }
        }
        assert!(!fast_eq(&[0u8; 64], &[0u8; 65]), "lengths differ");
    }

    /// Unaligned starts on both sides.
    #[test]
    fn unaligned_slices_compare_alike() {
        let a: Vec<u8> = (0..5000).map(|i| (i % 251) as u8).collect();
        let b = a.clone();
        for (x, y) in [(1, 1), (3, 3), (7, 7)] {
            for len in [64, 100, 1000, 4096] {
                assert!(fast_eq(&a[x..x + len], &b[y..y + len]));
                let mut c = b.clone();
                c[y + len - 1] ^= 1;
                assert!(!fast_eq(&a[x..x + len], &c[y..y + len]));
            }
        }
    }

    /// Times both on equal 4 KiB and 64 KiB pieces (the copy-back's): run
    /// with `--ignored --nocapture`, under Wine for the numbers that count.
    #[test]
    #[ignore]
    fn fast_compare_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        for len in [256usize, 4096, 65536] {
            let a = vec![0x5Au8; len];
            let b = a.clone();
            let rounds = (256 << 20) / len;
            let time = |f: &dyn Fn(&[u8], &[u8]) -> bool| {
                let start = Instant::now();
                let mut n = 0;
                for _ in 0..rounds {
                    n += usize::from(f(black_box(&a), black_box(&b)));
                }
                assert_eq!(n, rounds);
                start.elapsed().as_secs_f64() * 1e9 / rounds as f64
            };
            let plain = time(&|x, y| x == y);
            let fast = time(&|x, y| fast_eq(x, y));
            eprintln!(
                "{len} bytes equal: slice equality {plain:.0} ns, SSE2 {fast:.0} ns ({:.1}x)",
                plain / fast
            );
        }
    }
}
