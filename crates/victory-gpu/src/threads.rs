//! Dispatches counted in threads: the title's `DispatchX(x, y, z, flags)`
//! with [`THREAD_DIMENSIONS`] in its flags.
//!
//! The flags are taken for the console GPU's dispatch initiator bits: on its
//! AMD hardware bit 5 of `COMPUTE_DISPATCH_INITIATOR` is
//! `USE_THREAD_DIMENSIONS` (the dimensions count threads; the last thread
//! group of an axis may be partial), as the open drivers' register headers
//! name it. That reading is another fork's (DexNom/VIctory bc3cd4b0, then
//! Belobug 36a19d16 and avatarmanz 810efe58): taken for groups, a
//! "rehydration" stage of the ray-tracing structure builder ran up to 64
//! times its threads there and stored over the structures after its own.
//! Nothing in this tree had looked at the flags before, so a run's
//! "DispatchX flags" and "a dispatch counted in threads" lines (diagnostics)
//! are what says the title sets the bit, and on which programs.
//!
//! The host dispatches groups: [`counted`] gives the groups that hold the
//! threads and, per axis, the count where the last group is partial. A
//! program made for it (`victory_shader::emit::Options::thread_counts`)
//! takes those counts in user data and its threads past them leave
//! (`victory_shader::emit::DISPATCH_THREADS`); `State::dispatch` sets them.

#![cfg_attr(not(windows), allow(dead_code))]

/// `DispatchX`'s flag: x, y and z count threads, not thread groups.
pub(crate) const THREAD_DIMENSIONS: u32 = 0x20;

/// What a dispatch counted in threads is on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Counted {
    /// The groups of the program's size that hold the threads.
    pub groups: [u32; 3],
    /// Per axis: the count of threads where the last group is partial (its
    /// threads from that number on must not run), else 0 (whole groups).
    pub limits: [u32; 3],
}

/// The dispatch of `threads` threads of a program whose groups are `size`
/// threads. (An axis the program declares 0 threads on counts as 1.)
pub(crate) fn counted(threads: [u32; 3], size: [u32; 3]) -> Counted {
    let size = size.map(|n| n.max(1));
    Counted {
        groups: std::array::from_fn(|i| threads[i].div_ceil(size[i])),
        limits: std::array::from_fn(|i| if threads[i].is_multiple_of(size[i]) { 0 } else { threads[i] }),
    }
}

impl Counted {
    /// The axes whose last group is partial and for which the program has
    /// no count word (`words`: the axes it has one for): the threads of
    /// those groups past the count run, which the console's do not.
    pub(crate) fn uncut(&self, words: [bool; 3]) -> [bool; 3] {
        std::array::from_fn(|i| self.limits[i] != 0 && !words[i])
    }
}

/// Whether [`THREAD_DIMENSIONS`] is followed: `VICTORY_THREAD_DIMENSIONS=0`
/// takes such counts for groups, as before. Read once, here.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_THREAD_DIMENSIONS"))
}

/// Whether programs are made to leave out the threads past a count
/// (`victory_shader::emit::Options::thread_counts`): `VICTORY_THREAD_CUT=0`
/// makes them as before, and a partial last group then runs whole; so does
/// `VICTORY_THREAD_DIMENSIONS=0`, with which no count is ever set. Part of
/// the translation: the shader cache keys on it. Read once, here.
pub(crate) fn cut() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() && !crate::switch::off("VICTORY_THREAD_CUT"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_are_counted_in_the_programs_groups() {
        // The other forks' cases (DexNom bc3cd4b0, avatarmanz 810efe58).
        assert_eq!(counted([8847, 1, 1], [64, 1, 1]).groups, [139, 1, 1]);
        assert_eq!(counted([74, 1, 1], [64, 1, 1]).groups, [2, 1, 1]);
        assert_eq!(counted([32, 1, 1], [32, 1, 1]).groups, [1, 1, 1]);
        assert_eq!(counted([0, 12, 9], [64, 4, 4]).groups, [0, 3, 3]);
        assert_eq!(counted([u32::MAX, 1, 1], [64, 1, 1]).groups, [1 << 26, 1, 1]);
        // Taken for groups, 8,847 threads were 566,208: 64 times as many.
        assert_eq!(8847 * 64, 566_208);
    }

    #[test]
    fn a_partial_last_group_is_cut_at_the_count() {
        // 8,847 = 138 * 64 + 15: the last group's threads 15 to 63 must not run.
        assert_eq!(counted([8847, 1, 1], [64, 1, 1]).limits, [8847, 0, 0]);
        assert_eq!(counted([74, 3, 1], [64, 1, 1]).limits, [74, 0, 0]);
        // Whole groups: nothing to cut.
        assert_eq!(counted([128, 1, 1], [64, 1, 1]), Counted { groups: [2, 1, 1], limits: [0; 3] });
        assert_eq!(counted([64, 8, 4], [64, 8, 4]), Counted { groups: [1; 3], limits: [0; 3] });
        // Each axis by itself.
        assert_eq!(
            counted([17, 16, 5], [8, 8, 4]),
            Counted { groups: [3, 2, 2], limits: [17, 0, 5] }
        );
        // No thread on an axis: no group, and nothing to cut.
        assert_eq!(counted([0, 1, 1], [64, 1, 1]), Counted { groups: [0, 1, 1], limits: [0; 3] });
    }

    #[test]
    fn the_threads_that_run_are_those_below_the_counts() {
        // What a program does with the limits: a thread runs where its
        // number on each axis is below a limit that is not 0.
        for (threads, size) in [([74, 1, 1], [64, 1, 1]), ([17, 16, 5], [8, 8, 4]), ([128, 3, 2], [64, 2, 2])] {
            let c = counted(threads, size);
            let mut ran = 0u64;
            for g in 0..c.groups.iter().product::<u32>() * size.iter().product::<u32>() {
                // Any numbering of the host's threads: by axis.
                let all: [u32; 3] = std::array::from_fn(|i| c.groups[i] * size[i]);
                let at = [g % all[0], g / all[0] % all[1], g / (all[0] * all[1])];
                if (0..3).all(|i| c.limits[i] == 0 || at[i] < c.limits[i]) {
                    assert!((0..3).all(|i| at[i] < threads[i]), "{at:?} of {threads:?}");
                    ran += 1;
                }
            }
            assert_eq!(ran, threads.iter().map(|&n| u64::from(n)).product::<u64>(), "{threads:?}");
        }
    }

    #[test]
    fn a_zero_size_counts_as_one() {
        assert_eq!(counted([5, 1, 1], [0, 0, 0]), Counted { groups: [5, 1, 1], limits: [0; 3] });
    }

    #[test]
    fn a_partial_group_without_a_count_word_is_not_cut() {
        let c = counted([74, 9, 1], [64, 4, 1]);
        assert_eq!(c.limits, [74, 9, 0]);
        assert_eq!(c.uncut([true, true, false]), [false; 3]);
        assert_eq!(c.uncut([true, false, false]), [false, true, false]);
        assert_eq!(c.uncut([false; 3]), [true, true, false]);
        // Whole groups need no word.
        assert_eq!(counted([128, 8, 1], [64, 4, 1]).uncut([false; 3]), [false; 3]);
    }
}
